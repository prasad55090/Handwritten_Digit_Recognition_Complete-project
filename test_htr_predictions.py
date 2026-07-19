import os
import time
import numpy as np
import torch
import ssl
import argparse

# Bypass SSL certificate verification on macOS Python
ssl._create_default_https_context = ssl._create_unverified_context
from PIL import Image
import datasets
import jiwer
from transformers import (
    XLMRobertaTokenizer,
    ViTImageProcessor,
    TrOCRProcessor,
    VisionEncoderDecoderModel
)
from train_htr import preprocess_opencv_image

is_fine_tuned = os.path.exists("fine_tuned_trocr") and os.path.exists("fine_tuned_trocr/processor_config.json")
MODEL_PATH = "fine_tuned_trocr" if is_fine_tuned else "microsoft/trocr-small-handwritten"

def main():
    parser = argparse.ArgumentParser(description="HTR Evaluation Suite.")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index of samples to evaluate.")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to evaluate.")
    args = parser.parse_args()

    print(f"=== HTR Evaluation Suite: Testing on {args.num_samples} Handwriting Images (Start Index: {args.start_idx}) ===")
    print(f"Loading HTR Model from: '{MODEL_PATH}'...")
    
    # Load model and processor
    device = torch.device("cpu")
    print(f"Executing evaluations on hardware device: {device}")
    
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
    img_processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    processor = TrOCRProcessor(image_processor=img_processor, tokenizer=tokenizer)
    
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    # Load 20 images from the validation split of the IAM database
    print("\nDownloading validation split samples from Hugging Face Hub...")
    try:
        dataset_iam = datasets.load_dataset("Teklia/IAM-line", split="validation", trust_remote_code=True)
    except Exception:
        # Fallback to public mirror if primary is down
        dataset_iam = datasets.load_dataset("gagan3012/project-e2e-iam-handwriting-dataset", split="test", trust_remote_code=True)
        
    start_idx = max(0, min(args.start_idx, len(dataset_iam) - 1))
    end_idx = min(start_idx + args.num_samples, len(dataset_iam))
    eval_count = end_idx - start_idx
    print(f"Successfully loaded validation set. Evaluating on {eval_count} samples (indices {start_idx} to {end_idx - 1})...\n")

    # Table Header
    print(f"{'Idx':<4} | {'Ground Truth Text':<35} | {'Predicted Transcription':<35} | {'Conf':<6} | {'CER':<6} | {'WER':<6}")
    print("-" * 110)

    total_cer = 0.0
    total_wer = 0.0
    total_conf = 0.0
    
    # Auto-detect column names
    img_col = "image" if "image" in dataset_iam.features else "img"
    txt_col = "text" if "text" in dataset_iam.features else "label"

    for idx in range(start_idx, end_idx):
        item = dataset_iam[idx]
        image_pil = item[img_col].convert("RGB")
        gt_text = (item[txt_col] or "").strip()

        # Apply production OpenCV preprocessing pipeline
        img_np = np.array(image_pil)
        clean_img_np = preprocess_opencv_image(img_np)
        clean_pil = Image.fromarray(clean_img_np)

        # Run model prediction
        start_time = time.time()
        pixel_values = processor(clean_pil, return_tensors="pt").pixel_values.to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, return_dict_in_generate=True, output_scores=True)
            
        transcription = processor.tokenizer.decode(generated_ids.sequences[0], skip_special_tokens=True).strip()
        
        # Calculate transcription confidence (mean probability of generated tokens)
        logits = generated_ids.scores
        probs = []
        for step_logit in logits:
            step_prob = torch.softmax(step_logit, dim=-1)
            max_prob = torch.max(step_prob).item()
            probs.append(max_prob)
        confidence = np.mean(probs) if probs else 1.0

        # Calculate metrics using jiwer
        cer = jiwer.cer(gt_text, transcription) if gt_text else 0.0
        wer = jiwer.wer(gt_text, transcription) if gt_text else 0.0
        
        total_cer += cer
        total_wer += wer
        total_conf += confidence

        # Truncate strings for formatting neatness
        gt_disp = gt_text[:32] + "..." if len(gt_text) > 35 else gt_text
        pred_disp = transcription[:32] + "..." if len(transcription) > 35 else transcription

        print(f"{idx+1:<4} | {gt_disp:<35} | {pred_disp:<35} | {confidence*100:4.1f}% | {cer*100:4.1f}% | {wer*100:4.1f}%")

    avg_cer = total_cer / eval_count if eval_count > 0 else 0.0
    avg_wer = total_wer / eval_count if eval_count > 0 else 0.0
    avg_conf = total_conf / eval_count if eval_count > 0 else 0.0

    print("-" * 110)
    print(f"AVERAGES: Confidence = {avg_conf*100:.2f}% | CER = {avg_cer*100:.2f}% | WER = {avg_wer*100:.2f}%")
    print(f"=== HTR Evaluation Complete ===")

if __name__ == "__main__":
    main()
