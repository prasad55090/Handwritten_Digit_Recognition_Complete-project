========================================================================
HANDWRITTEN TEXT RECOGNITION (HTR) MODEL FINE-TUNING GUIDE
========================================================================

This workspace contains a production-ready PyTorch training pipeline (`train_htr.py`) 
to fine-tune Microsoft TrOCR (Vision-Encoder-Decoder) on line-level handwriting datasets.

------------------------------------------------------------------------
1. Environment Setup
------------------------------------------------------------------------
Make sure all necessary machine learning, transformer, and evaluation packages 
are installed inside your virtual environment:

  $ .venv/bin/pip install -r requirements.txt

------------------------------------------------------------------------
2. Running the Fine-Tuning Script
------------------------------------------------------------------------
Run the main script `train_htr.py`. It will:
  - Download processed splits of the IAM Handwriting lines dataset and the CVL/Bentham dataset from Hugging Face.
  - Apply custom OpenCV image preprocessing (grayscale, bilateral noise filter, deskewing, illumination correction, CLAHE).
  - Apply Albumentations (random brightness, rotation, elastic distortion).
  - Execute training with mixed-precision, learning rate scheduler, early stopping, and automatic checkpoint saving.

Options:
  --epochs       Maximum epochs to train (default: 100).
  --batch_size   DataLoader batch size (default: 32).
  --lr           Optimizer learning rate (default: 5e-5).
  --patience     Early stopping patience (default: 8).
  --max_samples  Limit datasets for quick debugging runs.

Examples:

a) Launch a full GPU-accelerated training:
  $ .venv/bin/python train_htr.py --epochs 100 --batch_size 32

b) Run a quick sanity check with only 10 samples to test memory/code:
  $ .venv/bin/python train_htr.py --epochs 2 --max_samples 10

------------------------------------------------------------------------
3. Automatic Resumption
------------------------------------------------------------------------
If training stops (due to memory issues, power loss, or terminal termination), 
re-running the script will automatically find `htr_checkpoint.pt` and resume 
exactly from the last saved epoch, restoring model weights, optimizer, and learning rate.

------------------------------------------------------------------------
4. Monitoring with TensorBoard
------------------------------------------------------------------------
Training progress (Loss curves, CER, WER, Accuracy, Precision, Recall, F1-Score) 
are logged in real-time. Start TensorBoard to visualize these curves:

  $ .venv/bin/tensorboard --logdir runs/

Then, open http://localhost:6006/ in your web browser.

------------------------------------------------------------------------
5. Flask Application Integration
------------------------------------------------------------------------
The best weights are saved in a local folder called `fine_tuned_trocr/`.
The Flask backend (`app.py`) is updated to automatically detect this folder on startup:
  - If `fine_tuned_trocr/` exists: Flask loads your customized model.
  - If not: Flask falls back to `microsoft/trocr-small-handwritten` from HF Hub.

You do not need to modify any code or UI elements to deploy your new model.
Simply copy the folder `fine_tuned_trocr/` to the project directory and restart Flask.
