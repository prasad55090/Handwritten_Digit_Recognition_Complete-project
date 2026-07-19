import os
import glob
import datasets

def main():
    print("=== Counting Dataset Image Populations ===")
    
    # 1. CVL Dataset
    cvl_dir = "data/cvl/cvl-database-cropped-1-1"
    cvl_images = glob.glob(os.path.join(cvl_dir, "**", "*.tif"), recursive=True)
    print(f"CVL Handwriting Dataset local crops: {len(cvl_images)} images")

    # 2. Bentham Dataset
    bentham_dir = "data/bentham/BenthamDatasetR0-Images"
    bentham_images = glob.glob(os.path.join(bentham_dir, "**", "*.jpg"), recursive=True)
    print(f"Bentham Handwriting Dataset local page scans: {len(bentham_images)} images")

    # 3. IAM Handwriting Dataset (from cached HF Hub)
    print("Loading cached IAM dataset metadata from Hugging Face...")
    try:
        dataset_iam = datasets.load_dataset("Teklia/IAM-line", trust_remote_code=True)
        print(f"IAM Dataset (Hugging Face splits):")
        print(f"  - Train: {len(dataset_iam['train'])} lines")
        print(f"  - Validation: {len(dataset_iam['validation'])} lines")
        print(f"  - Test: {len(dataset_iam['test'])} lines")
        iam_total = len(dataset_iam['train']) + len(dataset_iam['validation']) + len(dataset_iam['test'])
    except Exception as e:
        print(f"Could not load IAM metadata: {e}")
        iam_total = 0

    total_images = len(cvl_images) + len(bentham_images) + iam_total
    print("-" * 50)
    print(f"TOTAL PREPARED HTR TRAINING SAMPLES: {total_images} images/lines")
    print("==========================================")

if __name__ == "__main__":
    main()
