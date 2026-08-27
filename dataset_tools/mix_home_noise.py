"""
mix_home_noise.py — Dataset Synthesis with Louder Background Noise (Babble, Music, Speech)
--------------------------------------------------------------------------------
Scans the clean pouring audios, mixes them with random segments of selected
household noise (babble, music, speech) at a target SNR, and saves to a new folder.

Usage:
    .\venv\Scripts\python.exe mix_home_noise.py --snr 5 --type babble
    .\venv\Scripts\python.exe mix_home_noise.py --snr 8 --type music
    .\venv\Scripts\python.exe mix_home_noise.py --snr 10 --type speech
"""

import os
import sys
import argparse
import random
import shutil
import numpy as np
import soundfile as sf
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "sound_of_water_dataset")
NOISY_DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "sound_of_water_dataset_noisy")
NOISE_DIR = os.path.join(os.path.dirname(BASE_DIR), "noise_assets", "home_noise")

TYPE_MAPPING = {
    "kitchen": "DKITCHEN",
    "living": "DLIVING",
    "babble": "PRESTAURANT",
    "music": "MUSIC",
    "speech": "SPEECH"
}

def get_all_wav_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".wav")]

def load_noise_source(noise_type):
    """Loads all wav files from selected noise directory and merges them."""
    noise_subfolder = TYPE_MAPPING[noise_type]
    folder_path = os.path.join(NOISE_DIR, noise_subfolder)
    
    if not os.path.exists(folder_path) or len(os.listdir(folder_path)) == 0:
        print(f"[FAIL] Noise source directory is empty or missing: {folder_path}")
        print("       Please run download_home_noise.py first!")
        sys.exit(1)
        
    wav_files = get_all_wav_files(folder_path)
    noise_audios = []
    
    print(f"📖 Loading noise sources from {noise_subfolder} ({len(wav_files)} files)...")
    for path in wav_files:
        data, sr = sf.read(path)
        noise_audios.append(data)
        
    # Concatenate all files into one long sequence
    return np.concatenate(noise_audios), sr

def mix_audio_at_snr(signal, noise_src, target_snr):
    """Mixes signal with random segment of noise at a target SNR."""
    sig_len = len(signal)
    
    # Select random segment of noise
    if len(noise_src) <= sig_len:
        noise_segment = np.tile(noise_src, int(np.ceil(sig_len / len(noise_src))))[:sig_len]
    else:
        start_idx = random.randint(0, len(noise_src) - sig_len - 1)
        noise_segment = noise_src[start_idx : start_idx + sig_len]
        
    # Calculate energy
    p_signal = np.mean(signal ** 2)
    p_noise_raw = np.mean(noise_segment ** 2)
    
    if p_signal == 0:
        return signal + noise_segment
        
    if p_noise_raw == 0:
        p_noise_raw = 1e-8
        
    # p_noise = p_signal / 10^(target_snr / 10)
    p_target_noise = p_signal / (10 ** (target_snr / 10.0))
    
    # Scale noise segment
    scale_factor = np.sqrt(p_target_noise / p_noise_raw)
    scaled_noise = noise_segment * scale_factor
    
    mixed = signal + scaled_noise
    
    # Prevent clipping: normalize if maximum exceeds 1.0
    max_val = np.max(np.abs(mixed))
    if max_val > 0.99:
        mixed = mixed / max_val * 0.95
        
    return mixed

def main():
    parser = argparse.ArgumentParser(description="Mix louder domestic noise with water pouring sounds.")
    parser.add_argument("--snr", type=float, default=10.0, help="Target Signal-to-Noise Ratio (dB). Default 10.0")
    parser.add_argument("--type", type=str, choices=list(TYPE_MAPPING.keys()), default="babble", 
                        help="Noise type: babble, music, speech, kitchen, living. Default: babble")
    args = parser.parse_args()
    
    clean_audio_dir = os.path.join(CLEAN_DATASET_DIR, "audios")
    if not os.path.exists(clean_audio_dir):
        print(f"[FAIL] Clean dataset not found at: {clean_audio_dir}")
        print("       Please download the dataset first.")
        sys.exit(1)
        
    # Load noise source
    noise_source, noise_sr = load_noise_source(args.type)
    
    # Setup Output Directories
    noisy_audio_dir = os.path.join(NOISY_DATASET_DIR, "audios")
    os.makedirs(noisy_audio_dir, exist_ok=True)
    
    # Copy metadata annotations and splits
    clean_annotations_dir = os.path.join(CLEAN_DATASET_DIR, "annotations")
    noisy_annotations_dir = os.path.join(NOISY_DATASET_DIR, "annotations")
    if os.path.exists(clean_annotations_dir):
        if os.path.exists(noisy_annotations_dir):
            shutil.rmtree(noisy_annotations_dir)
        shutil.copytree(clean_annotations_dir, noisy_annotations_dir)
        print("📁 Copied annotations directory to noisy dataset.")
        
    clean_splits_dir = os.path.join(CLEAN_DATASET_DIR, "splits")
    noisy_splits_dir = os.path.join(NOISY_DATASET_DIR, "splits")
    if os.path.exists(clean_splits_dir):
        if os.path.exists(noisy_splits_dir):
            shutil.rmtree(noisy_splits_dir)
        shutil.copytree(clean_splits_dir, noisy_splits_dir)
        print("📁 Copied splits directory to noisy dataset.")

    for f in ["README.md", ".gitattributes"]:
        src = os.path.join(CLEAN_DATASET_DIR, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(NOISY_DATASET_DIR, f))
            
    clean_files = get_all_wav_files(clean_audio_dir)
    print(f"\n⚡ Mixing {len(clean_files)} clean audio files with {args.type} noise at {args.snr}dB SNR...")
    
    for path in tqdm(clean_files, desc="Mixing progress"):
        sig_data, sig_sr = sf.read(path)
        
        # Verify sample rate matches
        if sig_sr != noise_sr:
            import librosa
            noise_source = librosa.resample(noise_source, orig_sr=noise_sr, target_sr=sig_sr)
            noise_sr = sig_sr
            
        mixed_data = mix_audio_at_snr(sig_data, noise_source, args.snr)
        
        file_name = os.path.basename(path)
        base_name, ext = os.path.splitext(file_name)
        noisy_file_name = f"{base_name}_noisy{ext}"
        out_path = os.path.join(noisy_audio_dir, noisy_file_name)
        sf.write(out_path, mixed_data, sig_sr)
        
    print("\n" + "="*60)
    print(f"[OK] Dataset synthesis complete!")
    print(f"     Noisy dataset saved to: {os.path.abspath(NOISY_DATASET_DIR)}")
    print(f"     Parameters used: SNR={args.snr}dB, Noise Type={args.type}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
