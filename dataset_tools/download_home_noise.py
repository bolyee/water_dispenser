"""
download_home_noise.py — Louder Domestic Noise Dataset Downloader (Zenodo DEMAND + GitHub Samples)
--------------------------------------------------------------------------------------------------
Downloads:
    1. Bustling Restaurant Babble (DEMAND PRESTAURANT) - Zenodo
    2. Loud Music (Drums/Jazz Trio) - GitHub
    3. TV Speech / Conversational Vocal - GitHub
    (Also supports DKITCHEN / DLIVING)

Usage:
    .\venv\Scripts\python.exe download_home_noise.py
"""

import os
import sys
import zipfile
import requests
from tqdm import tqdm

try:
    import soundfile as sf
    import numpy as np
except ImportError:
    print("[FAIL] 'soundfile' or 'numpy' library is not installed.")
    print("       Please run: .\\venv\\Scripts\\pip.exe install soundfile numpy")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOISE_DIR = os.path.join(os.path.dirname(BASE_DIR), "noise_assets", "home_noise")

# Direct download links for various noise categories
URLS = {
    # DEMAND Multichannel ZIPs (Zenodo)
    "DKITCHEN": {"type": "zip", "url": "https://zenodo.org/records/1227121/files/DKITCHEN_16k.zip?download=1"},
    "DLIVING": {"type": "zip", "url": "https://zenodo.org/records/1227121/files/DLIVING_16k.zip?download=1"},
    "PRESTAURANT": {"type": "zip", "url": "https://zenodo.org/records/1227121/files/PRESTAURANT_16k.zip?download=1"},
    
    # Loud Instrument Music (GitHub Raw)
    "MUSIC/drums.wav": {"type": "file", "url": "https://github.com/pdx-cs-sound/wavs/raw/master/submaster/drums.wav"},
    "MUSIC/trio.wav": {"type": "file", "url": "https://github.com/pdx-cs-sound/wavs/raw/master/submaster/trio.wav"},
    
    # TV Speech / Vocals (GitHub Raw, 16kHz)
    "SPEECH/m1.wav": {"type": "file", "url": "https://github.com/voxserv/audio_quality_testing_samples/raw/master/speech/orig/16k/m1.wav"},
    "SPEECH/f1.wav": {"type": "file", "url": "https://github.com/voxserv/audio_quality_testing_samples/raw/master/speech/orig/16k/f1.wav"},
}

def download_file(url, target_path):
    """Downloads a file displaying a tqdm progress bar."""
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

def extract_and_convert_to_mono(zip_path, folder_name):
    """Extracts zip archive and downmixes 16-channel WAV files to mono (channel 0)"""
    extract_temp_dir = os.path.join(NOISE_DIR, f"temp_{folder_name}")
    os.makedirs(extract_temp_dir, exist_ok=True)
    
    print(f"📦 Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_temp_dir)
        
    final_target_dir = os.path.join(NOISE_DIR, folder_name)
    os.makedirs(final_target_dir, exist_ok=True)
    
    search_dir = os.path.join(extract_temp_dir, folder_name)
    if not os.path.exists(search_dir):
        search_dir = extract_temp_dir
        
    wav_files = [f for f in os.listdir(search_dir) if f.endswith(".wav")]
    
    print(f"🔄 Converting 16-channel WAVs to mono (using Channel 1)...")
    for f_name in wav_files:
        src_path = os.path.join(search_dir, f_name)
        dst_path = os.path.join(final_target_dir, f_name)
        
        data, sr = sf.read(src_path)
        if len(data.shape) > 1 and data.shape[1] > 1:
            mono_data = data[:, 0]
        else:
            mono_data = data
            
        sf.write(dst_path, mono_data, sr)
        print(f"   -> Saved: {folder_name}/{f_name} (Sample Rate: {sr}Hz, Duration: {len(mono_data)/sr:.1f}s)")
        
    import shutil
    shutil.rmtree(extract_temp_dir)

def main():
    os.makedirs(NOISE_DIR, exist_ok=True)
    
    print("\n" + "="*60)
    print("  🏠 Louder Noise Dataset Downloader (Babble, Music, TV Speech)")
    print("="*60)
    print("  - Target Location: " + os.path.abspath(NOISE_DIR))
    print("="*60 + "\n")
    
    for name, info in URLS.items():
        # Handle folders for single files
        target_path = os.path.join(NOISE_DIR, name)
        target_folder = os.path.dirname(target_path) if info["type"] == "file" else target_path
        
        os.makedirs(target_folder, exist_ok=True)
        
        # Check if already populated
        if info["type"] == "zip":
            if os.path.exists(target_path) and len(os.listdir(target_path)) > 0:
                print(f"  [OK] Zip-based {name} directory already populated. Skipping.")
                continue
        elif info["type"] == "file":
            if os.path.exists(target_path):
                print(f"  [OK] File-based {name} already exists. Skipping.")
                continue
                
        # Action
        if info["type"] == "zip":
            zip_path = os.path.join(NOISE_DIR, f"{name}_16k.zip")
            print(f"📥 Downloading {name} noise archive from Zenodo...")
            try:
                download_file(info["url"], zip_path)
                print(f"  [OK] Download finished: {name}_16k.zip")
                extract_and_convert_to_mono(zip_path, name)
                print(f"  [OK] {name} folder processing complete.")
            except Exception as e:
                print(f"  [FAIL] Failed on {name}: {e}")
            finally:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
        else:
            # File download
            print(f"📥 Downloading sample file: {name}...")
            try:
                download_file(info["url"], target_path)
                print(f"  [OK] Downloaded: {name}")
            except Exception as e:
                print(f"  [FAIL] Failed to download file {name}: {e}")
                if os.path.exists(target_path):
                    os.remove(target_path)
                    
    print("\n" + "="*60)
    print("  [OK] SUCCESS! All loud noise and speech files are processed and ready.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
