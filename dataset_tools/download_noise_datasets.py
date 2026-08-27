"""
download_noise_datasets.py — Automatic Downloader for Noise and Speech Datasets
--------------------------------------------------------------------------------
Downloads LibriSpeech (speech) and ESC-50 (environmental noise) to 'noise_assets/'
without raw emojis to prevent Windows console CP949 crashes.
"""

import os
import sys
import zipfile
import tarfile
import requests
from tqdm import tqdm

# Target Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(os.path.dirname(BASE_DIR), "noise_assets")
ESC50_DIR = os.path.join(ASSETS_DIR, "esc50")
LIBRISPEECH_DIR = os.path.join(ASSETS_DIR, "librispeech")

# Download URLs
URLS = {
    "esc50": {
        "url": "https://github.com/karoldvl/ESC-50/archive/refs/heads/master.zip",
        "filename": "esc50.zip",
        "target_dir": ESC50_DIR
    },
    "librispeech": {
        "url": "http://www.openslr.org/resources/12/dev-clean.tar.gz",
        "filename": "librispeech.tar.gz",
        "target_dir": LIBRISPEECH_DIR
    }
}

def download_file(url, target_path):
    """Downloads a file showing a tqdm progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    with open(target_path, 'wb') as f, tqdm(
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        bar_format='{l_bar}{bar:40}{r_bar}'
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

def main():
    # Make sure noise assets root exists
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("  [INFO] Noise & Speech Dataset Downloader")
    print("="*60)
    print("  - Target assets folder: " + ASSETS_DIR)
    print("="*60 + "\n")
    
    for name, info in URLS.items():
        target_dir = info["target_dir"]
        
        # Skip if folder is already populated
        if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
            print(f"  [OK] {name.upper()} directory already populated. Skipping download.")
            continue
            
        temp_file = os.path.join(ASSETS_DIR, info["filename"])
        
        # Step 1: Download
        print(f"  [DOWN] Downloading {name.upper()}...")
        print(f"         From: {info['url']}")
        try:
            download_file(info["url"], temp_file)
            print(f"  [OK] Download finished: {info['filename']}")
        except Exception as e:
            print(f"  [FAIL] Failed to download {name}: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            sys.exit(1)
            
        # Step 2: Extract
        print(f"  [EXTRACT] Extracting {info['filename']}... Please wait.")
        try:
            os.makedirs(target_dir, exist_ok=True)
            if info["filename"].endswith(".zip"):
                with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                    # To keep it clean, extract all to a temporary subfolder and then restructure if needed
                    # For ESC-50, we want to extract the files into target_dir
                    zip_ref.extractall(ASSETS_DIR)
                
                # ESC-50 extracts to 'ESC-50-master'
                extracted_name = "ESC-50-master"
                old_path = os.path.join(ASSETS_DIR, extracted_name)
                if os.path.exists(old_path):
                    if os.path.exists(target_dir):
                        # Merge or rename
                        import shutil
                        shutil.rmtree(target_dir)
                    os.rename(old_path, target_dir)
                    
            elif info["filename"].endswith(".tar.gz"):
                with tarfile.open(temp_file, 'r:gz') as tar_ref:
                    tar_ref.extractall(ASSETS_DIR)
                
                # LibriSpeech extracts to 'LibriSpeech'
                extracted_name = "LibriSpeech"
                old_path = os.path.join(ASSETS_DIR, extracted_name)
                if os.path.exists(old_path):
                    if os.path.exists(target_dir):
                        import shutil
                        shutil.rmtree(target_dir)
                    os.rename(old_path, target_dir)
            
            print(f"  [OK] Extracted successfully to {target_dir}")
            
        except Exception as e:
            print(f"  [FAIL] Failed to extract {info['filename']}: {e}")
            sys.exit(1)
        finally:
            # Clean up temporary archive file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
    print("\n" + "="*60)
    print("  [OK] SUCCESS! All noise & speech datasets downloaded and extracted.")
    print("       Ready for dataset synthesis.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
