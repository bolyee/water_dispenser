"""
mix_yt_noise.py — Dataset Synthesis using YouTube Noises (With Metadata Recording)
--------------------------------------------------------------------------------
Scans the clean pouring audios, amplifies the water sound (Normalize),
mixes them with random segments of the downloaded YouTube noise files
at a target SNR, and records the mix mapping to 'mix_metadata.json'.

Usage:
    .\venv\Scripts\python.exe dataset_tools/mix_yt_noise.py --snr 3
"""

import os
import sys
import argparse
import random
import shutil
import json
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
CLEAN_DATASET_DIR = os.path.join(ROOT_DIR, "sound_of_water_dataset")
NOISY_DATASET_DIR = os.path.join(ROOT_DIR, "sound_of_water_dataset_noisy")
TARGET_SR = 16000  # 정수기 모델 표준 샘플 레이트 (16kHz)

def get_all_wav_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".wav")]

def get_youtube_noise_files():
    """Scans root directory for dataset_*.mp3 or dataset_*.wav files."""
    files = [f for f in os.listdir(ROOT_DIR) if f.startswith("dataset_") and (f.endswith(".mp3") or f.endswith(".wav"))]
    return [os.path.join(ROOT_DIR, f) for f in files]

def mix_audio_at_snr(signal, noise_segment, target_snr):
    """Mixes amplified signal with noise segment at a target SNR."""
    p_signal = np.mean(signal ** 2)
    p_noise_raw = np.mean(noise_segment ** 2)
    
    if p_signal == 0:
        return signal + noise_segment
        
    if p_noise_raw == 0:
        p_noise_raw = 1e-8
        
    # p_noise = p_signal / 10^(target_snr / 10)
    p_target_noise = p_signal / (10 ** (target_snr / 10.0))
    
    # Scale noise segment to match target SNR
    scale_factor = np.sqrt(p_target_noise / p_noise_raw)
    scaled_noise = noise_segment * scale_factor
    
    mixed = signal + scaled_noise
    
    # Prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 0.99:
        mixed = mixed / max_val * 0.95
        
    return mixed

def main():
    parser = argparse.ArgumentParser(description="Mix YouTube downloaded noise with water pouring sounds.")
    parser.add_argument("--snr", type=float, default=10.0, help="Target Signal-to-Noise Ratio (dB). Default 10.0")
    args = parser.parse_args()
    
    clean_audio_dir = os.path.join(CLEAN_DATASET_DIR, "audios")
    if not os.path.exists(clean_audio_dir):
        print(f"[FAIL] Clean dataset not found at: {clean_audio_dir}")
        print("       Please download the main dataset first.")
        sys.exit(1)
        
    # Scan for YouTube downloaded noises
    yt_noise_paths = get_youtube_noise_files()
    if not yt_noise_paths:
        print("[FAIL] No YouTube downloaded noise files found (e.g., dataset_01.mp3 ~ dataset_20.mp3).")
        print("       Please run download_yt.py first to extract the noises!")
        sys.exit(1)
        
    print(f"📖 Found {len(yt_noise_paths)} YouTube noise files.")
    
    # Load all YouTube noises into memory
    print("⏳ Loading and resampling YouTube noises to 16kHz mono...")
    noise_sources = {}
    for path in tqdm(yt_noise_paths, desc="Loading noises"):
        try:
            data, _ = librosa.load(path, sr=TARGET_SR, mono=True)
            noise_sources[os.path.basename(path)] = data
        except Exception as e:
            print(f"⚠️ Failed to load {os.path.basename(path)}: {e}")
            
    if not noise_sources:
        print("[FAIL] None of the YouTube noise files could be loaded.")
        sys.exit(1)
        
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
        
    clean_splits_dir = os.path.join(CLEAN_DATASET_DIR, "splits")
    noisy_splits_dir = os.path.join(NOISY_DATASET_DIR, "splits")
    if os.path.exists(clean_splits_dir):
        if os.path.exists(noisy_splits_dir):
            shutil.rmtree(noisy_splits_dir)
        shutil.copytree(clean_splits_dir, noisy_splits_dir)

    for f in ["README.md", ".gitattributes"]:
        src = os.path.join(CLEAN_DATASET_DIR, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(NOISY_DATASET_DIR, f))
            
    clean_files = get_all_wav_files(clean_audio_dir)
    print(f"\n⚡ Mixing {len(clean_files)} amplified clean audios with YouTube noise at {args.snr}dB SNR...")
    
    mix_metadata = {}
    
    for path in tqdm(clean_files, desc="Mixing progress"):
        sig_data, sig_sr = sf.read(path)
        
        # 16kHz 리샘플링 보장
        if sig_sr != TARGET_SR:
            sig_data = librosa.resample(sig_data, orig_sr=sig_sr, target_sr=TARGET_SR)
            sig_sr = TARGET_SR
            
        # 원본 물소리가 너무 작으므로 최대 피크를 0.8로 노멀라이즈(증폭)
        sig_max = np.max(np.abs(sig_data))
        if sig_max > 1e-6:
            sig_data = sig_data / sig_max * 0.8
            
        sig_len = len(sig_data)
        
        # 임의의 유튜브 소음 트랙 선택
        noise_name = random.choice(list(noise_sources.keys()))
        selected_noise = noise_sources[noise_name]
        
        if len(selected_noise) <= sig_len:
            noise_segment = np.tile(selected_noise, int(np.ceil(sig_len / len(selected_noise))))[:sig_len]
        else:
            start_idx = random.randint(0, len(selected_noise) - sig_len - 1)
            noise_segment = selected_noise[start_idx : start_idx + sig_len]
            
        # 합성
        mixed_data = mix_audio_at_snr(sig_data, noise_segment, args.snr)
        
        # 파일 저장
        file_name = os.path.basename(path)
        base_name, ext = os.path.splitext(file_name)
        noisy_file_name = f"{base_name}_noisy{ext}"
        out_path = os.path.join(noisy_audio_dir, noisy_file_name)
        sf.write(out_path, mixed_data, TARGET_SR)
        
        # 메타데이터 기록
        mix_metadata[noisy_file_name] = {
            "clean_file": file_name,
            "noise_file": noise_name,
            "snr_db": args.snr
        }
        
    # JSON 파일로 저장
    meta_path = os.path.join(NOISY_DATASET_DIR, "mix_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(mix_metadata, f, indent=4, ensure_ascii=False)
        
    print("\n" + "="*60)
    print(f"[OK] YouTube Noise Dataset synthesis complete!")
    print(f"     Noisy dataset saved to: {os.path.abspath(NOISY_DATASET_DIR)}")
    print(f"     Metadata saved to: {meta_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
