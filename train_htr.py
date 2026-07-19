import os
import time
import argparse
import logging
import contextlib
import ssl
import random
import glob
import xml.etree.ElementTree as ET

# Bypass SSL certificate verification on macOS Python
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import albumentations as A
from torch.optim import AdamW
from transformers import (
    XLMRobertaTokenizer,
    ViTImageProcessor,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    get_linear_schedule_with_warmup
)
import datasets
import jiwer

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("HTR_Trainer")

# Preprocessing OpenCV helper
def preprocess_opencv_image(img_np):
    """
    Applies production-grade image preprocessing:
    - Grayscale conversion
    - Illumination/shadow removal
    - Bilateral filter denoising
    - Contrast normalization
    - Deskewing
    """
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()

    # 1. Shadow removal / background illumination correction
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated, 21)
    diff = cv2.absdiff(gray, bg_img)
    normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # 2. Denoising using Bilateral Filter to preserve edges
    denoised = cv2.bilateralFilter(normalized, 9, 75, 75)

    # 3. Deskewing (orientation correction)
    coords = np.column_stack(np.where(denoised < 127))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        if abs(angle) > 0.5:
            (h, w) = denoised.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            denoised = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return denoised

# Albumentations Augmentations
train_transforms = A.Compose([
    A.Rotate(limit=10, p=0.4, border_mode=cv2.BORDER_REPLICATE),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.3, border_mode=cv2.BORDER_REPLICATE),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(var_limit=(10.0, 30.0), p=0.2)
])

# Bounding box line crop loader for Bentham Layout XMLs
def load_local_bentham():
    bentham_gt_dir = "data/bentham/BenthamDatasetR0-GT/PAGE"
    bentham_img_dir = "data/bentham/BenthamDatasetR0-Images/Images/Pages"
    samples = []
    
    if not os.path.exists(bentham_gt_dir) or not os.path.exists(bentham_img_dir):
        logger.warning("Bentham folders not found. Skipping local Bentham dataset.")
        return samples
        
    ns = {'p': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19'}
    xml_files = glob.glob(os.path.join(bentham_gt_dir, "*.xml"))
    
    logger.info(f"Parsing {len(xml_files)} Bentham layout XML pages...")
    for xml_path in xml_files:
        base_name = os.path.splitext(os.path.basename(xml_path))[0]
        img_path = os.path.join(bentham_img_dir, base_name + ".jpg")
        if not os.path.exists(img_path):
            continue
            
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            # Find all TextLine tags
            lines = root.findall(".//p:TextLine", ns)
            for line in lines:
                # Get unicode text
                unicode_elem = line.find(".//p:Unicode", ns)
                text = unicode_elem.text.strip() if unicode_elem is not None and unicode_elem.text else ""
                if not text:
                    continue
                    
                # Get points coordinates
                point_elems = line.findall(".//p:Point", ns)
                if not point_elems:
                    continue
                xs = [int(p.get("x")) for p in point_elems if p.get("x") is not None]
                ys = [int(p.get("y")) for p in point_elems if p.get("y") is not None]
                if not xs or not ys:
                    continue
                    
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                
                if (xmax - xmin) > 10 and (ymax - ymin) > 10:
                    samples.append({
                        "img_path": img_path,
                        "bbox": (xmin, ymin, xmax, ymax),
                        "text": text
                    })
        except Exception as e:
            logger.warning(f"Error parsing Bentham XML {xml_path}: {e}")
            
    logger.info(f"Successfully loaded {len(samples)} Bentham line crops from layout XMLs.")
    return samples

# Dataset class mapping
class HTRDataset(Dataset):
    def __init__(self, samples, processor, augment=False):
        self.samples = samples
        self.processor = processor
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Check if the sample is from Hugging Face or our local parsed list
        if "img_path" in item:
            # Local parsed sample with bounding box coordinates
            img_path = item["img_path"]
            bbox = item["bbox"]
            text = item["text"]
            
            image = Image.open(img_path).convert("RGB")
            # bbox is (xmin, ymin, xmax, ymax)
            image = image.crop(bbox)
        else:
            # Hugging Face sample (already cropped PIL image)
            image = item["image"].convert("RGB")
            text = item["text"]
            
        # Apply OpenCV pre-processing pipeline
        img_np = np.array(image)
        clean_img_np = preprocess_opencv_image(img_np)
        clean_pil = Image.fromarray(clean_img_np)
        
        if self.augment:
            augmented = train_transforms(image=np.array(clean_pil))
            clean_pil = Image.fromarray(augmented["image"])
            
        pixel_values = self.processor(clean_pil, return_tensors="pt").pixel_values.squeeze()
        
        # Tokenize label
        labels = self.processor.tokenizer(text, padding="max_length", max_length=64, truncation=True).input_ids
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def main():
    default_model = "fine_tuned_trocr" if os.path.exists("fine_tuned_trocr") else "microsoft/trocr-small-handwritten"
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR HTR pipeline.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples limit for debug runs.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, mps, cuda).")
    parser.add_argument("--model", type=str, default=default_model, help="Model name or path to resume/load.")
    default_initial_epoch = 1 if default_model == "fine_tuned_trocr" else 0
    parser.add_argument("--initial_epoch", type=int, default=default_initial_epoch, help="Epoch number to start training from.")
    args = parser.parse_args()

    # Hardware device configuration
    device = torch.device(args.device)
    logger.info(f"Target calculation device: {device}")

    # Load Model and Tokenizers
    logger.info(f"Loading processor and model from '{args.model}'...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model)
    img_processor = ViTImageProcessor.from_pretrained(args.model)
    processor = TrOCRProcessor(image_processor=img_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(args.model)
    model.to(device)

    # Set special token configs
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    # Load IAM from HF (Hugging Face)
    logger.info("Loading IAM dataset from Hugging Face...")
    try:
        dataset_iam = datasets.load_dataset("Teklia/IAM-line", trust_remote_code=True)
    except Exception as e:
        logger.error(f"Failed to load Teklia/IAM-line: {e}")
        sys.exit(1)
        
    iam_train = [{"image": x["image"], "text": x["text"]} for x in dataset_iam["train"]]
    iam_val = [{"image": x["image"], "text": x["text"]} for x in dataset_iam["validation"]]

    # Load Bentham locally
    logger.info("Loading Bentham layout XMLs...")
    bentham_samples = load_local_bentham()
    
    # Split Bentham into train and validation sets
    random.seed(42)
    random.shuffle(bentham_samples)
    split_idx = int(0.9 * len(bentham_samples))
    bentham_train = bentham_samples[:split_idx]
    bentham_val = bentham_samples[split_idx:]

    # Combine datasets
    train_samples = iam_train + bentham_train
    val_samples = iam_val + bentham_val

    # Debug samples capping
    if args.max_samples > 0:
        logger.info(f"Limiting dataset sizes to {args.max_samples} samples for quick validation/debug run...")
        train_samples = train_samples[:args.max_samples]
        val_samples = val_samples[:args.max_samples]

    logger.info(f"Final training set size: {len(train_samples)} lines")
    logger.info(f"Final validation set size: {len(val_samples)} lines")

    train_dataset = HTRDataset(train_samples, processor, augment=True)
    val_dataset = HTRDataset(val_samples, processor, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    writer = SummaryWriter(log_dir="runs/htr_experiment")

    # Mixed Precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    epochs_no_improve = 0
    patience = 10
    logger.info("=== HTR Fine-Tuning Started ===")

    for epoch in range(args.initial_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss

            if device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
                
            lr_scheduler.step()
            epoch_loss += loss.item()

            if batch_idx % 20 == 0:
                logger.info(f"Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation epoch
        model.eval()
        val_loss = 0.0
        references = []
        hypotheses = []

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = model(pixel_values=pixel_values, labels=labels)
                    val_loss += outputs.loss.item()

                # Generate transcription test
                generated_ids = model.generate(pixel_values)
                pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Decode references, filtering out padding tokens (-100)
                clean_labels = []
                for label_seq in labels:
                    filtered = [t for t in label_seq.tolist() if t != -100]
                    clean_labels.append(filtered)
                gt_texts = processor.batch_decode(clean_labels, skip_special_tokens=True)

                references.extend(gt_texts)
                hypotheses.extend(pred_texts)

        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate CER and WER metrics
        references = [r.strip() for r in references]
        hypotheses = [h.strip() for h in hypotheses]
        cer = jiwer.cer(references, hypotheses) if references else 0.0
        wer = jiwer.wer(references, hypotheses) if references else 0.0

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("Metrics/CER", cer, epoch)
        writer.add_scalar("Metrics/WER", wer, epoch)

        logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"CER: {cer*100:.2f}% | WER: {wer*100:.2f}%")

        # Save checkpoint if loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            output_dir = "fine_tuned_trocr"
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Validation loss improved. Saving best model checkpoint to '{output_dir}'...")
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs due to no validation loss improvement.")
                break

    writer.close()
    logger.info("✅ Training pipeline completed successfully. Model and processor stored.")

if __name__ == "__main__":
    main()
