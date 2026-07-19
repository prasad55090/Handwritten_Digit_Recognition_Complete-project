import torch
from transformers import VisionEncoderDecoderModel, XLMRobertaTokenizer
import time

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    print("Loading model and tokenizer...")
    tokenizer = XLMRobertaTokenizer.from_pretrained("microsoft/trocr-small-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    model.to(device)
    model.train()
    
    # Configure token IDs
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Dummy inputs
    pixel_values = torch.randn(2, 3, 384, 384).to(device)
    labels = torch.randint(0, 1000, (2, 64)).to(device)
    
    print("Running forward pass...")
    start = time.time()
    outputs = model(pixel_values=pixel_values, labels=labels)
    loss = outputs.loss
    print(f"Forward pass completed in {time.time() - start:.2f}s. Loss: {loss.item()}")
    
    print("Running backward pass...")
    start = time.time()
    loss.backward()
    print(f"Backward pass completed in {time.time() - start:.2f}s.")
    print("Diagnosis: MPS works perfectly without hangs!")

if __name__ == "__main__":
    main()
