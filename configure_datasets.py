import os
import sys
import time
import zipfile
import tarfile
import ssl
import requests
import datasets
import argparse

# Bypass SSL certificate verification on macOS Python
ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Datasets definition
DATASETS_CONFIG = {
    "CVL": {
        "url": "https://zenodo.org/records/1492267/files/cvl-database-cropped-1-1.zip?download=1",
        "zip_path": os.path.join(DATA_DIR, "cvl-database-cropped-1-1.zip"),
        "extract_path": os.path.join(DATA_DIR, "cvl"),
        "verify_folder": os.path.join(DATA_DIR, "cvl", "cvl-database-cropped-1-1"),
        "archive_type": "zip"
    },
    "Bentham_Images": {
        "url": "https://zenodo.org/records/44519/files/BenthamDatasetR0-Images.tbz?download=1",
        "zip_path": os.path.join(DATA_DIR, "BenthamDatasetR0-Images.tbz"),
        "extract_path": os.path.join(DATA_DIR, "bentham"),
        "verify_folder": os.path.join(DATA_DIR, "bentham", "BenthamDatasetR0-Images"),
        "archive_type": "tar"
    },
    "Bentham_GT": {
        "url": "https://zenodo.org/records/44519/files/BenthamDatasetR0-GT.tbz?download=1",
        "zip_path": os.path.join(DATA_DIR, "BenthamDatasetR0-GT.tbz"),
        "extract_path": os.path.join(DATA_DIR, "bentham"),
        "verify_folder": os.path.join(DATA_DIR, "bentham", "BenthamDatasetR0-GT"),
        "archive_type": "tar"
    }
}

def download_file_with_resume(url, target_path, max_retries=10):
    headers = {}
    file_mode = 'wb'
    existing_size = 0
    
    if os.path.exists(target_path):
        existing_size = os.path.getsize(target_path)
        try:
            # Send HEAD request to get total length
            head_res = requests.head(url, timeout=15, verify=False)
            total_size = int(head_res.headers.get('content-length', 0))
            if existing_size >= total_size and total_size > 0:
                print(f"File {os.path.basename(target_path)} is already fully downloaded ({existing_size / (1024*1024):.1f}MB).")
                return
            headers['Range'] = f"bytes={existing_size}-"
            file_mode = 'ab'
            print(f"Resuming download from byte {existing_size}...")
        except Exception:
            pass

    for attempt in range(max_retries):
        try:
            res = requests.get(url, headers=headers, stream=True, timeout=20, verify=False)
            if res.status_code not in (200, 206):
                raise ValueError(f"HTTP status code {res.status_code}")
                
            total_size = int(res.headers.get('content-length', 0)) + existing_size
            downloaded = existing_size
            
            with open(target_path, file_mode) as f:
                for chunk in res.iter_content(chunk_size=16384):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                        sys.stdout.write(f"\rProgress: {percent:.2f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)")
                        sys.stdout.flush()
            print(f"\nDownload of {os.path.basename(target_path)} finished successfully!")
            return
        except Exception as e:
            print(f"\nAttempt {attempt+1}/{max_retries} failed: {e}. Retrying in 5s...")
            time.sleep(5)
            if os.path.exists(target_path):
                existing_size = os.path.getsize(target_path)
                headers['Range'] = f"bytes={existing_size}-"
                file_mode = 'ab'
            else:
                existing_size = 0
                headers = {}
                file_mode = 'wb'
    raise IOError(f"Failed to download after {max_retries} attempts.")

def extract_archive(path, target_dir, archive_type):
    print(f"Extracting {os.path.basename(path)} into {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)
    if archive_type == "zip":
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
    elif archive_type == "tar":
        with tarfile.open(path, "r:bz2") as tar_ref:
            tar_ref.extractall(target_dir)
    print("Extraction finished!")
    # Delete archive file to save disk space
    os.remove(path)

def verify_datasets():
    print("\n=== Verifying Downloaded Datasets ===")
    all_valid = True
    for key, config in DATASETS_CONFIG.items():
        if os.path.exists(config["verify_folder"]):
            print(f"Dataset {key} verified successfully at: {config['verify_folder']}")
        else:
            print(f"Dataset {key} missing or invalid!")
            all_valid = False
            
    # Cache IAM dataset from Hugging Face
    print("\nPre-caching IAM line dataset from Hugging Face...")
    try:
        datasets.load_dataset("Teklia/IAM-line", trust_remote_code=True)
        print("IAM lines dataset successfully cached locally!")
    except Exception as e:
        print(f"Failed to cache IAM dataset: {e}")
        all_valid = False
        
    if all_valid:
        print("\nAll datasets (IAM, CVL, Bentham) configured and ready for training!")
    else:
        print("\nSome datasets failed validation. Please review the errors above.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Configure and download HTR datasets.")
    parser.add_argument("--download_public", action="store_true", help="Download Zenodo datasets automatically.")
    args = parser.parse_args()
    
    if args.download_public:
        print("=== Step 1: Downloading Public Datasets (CVL & Bentham) ===")
        for key, config in DATASETS_CONFIG.items():
            if not os.path.exists(config["verify_folder"]):
                print(f"\nProcessing download: {key}")
                download_file_with_resume(config["url"], config["zip_path"])
                extract_archive(config["zip_path"], config["extract_path"], config["archive_type"])
            else:
                print(f"\nDataset {key} already exists locally. Skipping download.")
                
    verify_datasets()

if __name__ == "__main__":
    main()
