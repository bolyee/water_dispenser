import os
import sys
import random
import argparse
import shutil
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR = os.path.join(ROOT_DIR, "sound_of_water_dataset", "audios")
NOISE_FILE = os.path.join(ROOT_DIR, "tv_noise_2.mp3")
OUTPUT_DIR = os.path.join(ROOT_DIR, "unseen_test_dataset")

SR = 16000

def mix_audio_at_snr(signal, noise_segment, target_snr):
    p_signal = np.mean(signal ** 2)
    p_noise_raw = np.mean(noise_segment ** 2)
    if p_signal == 0:
        return signal + noise_segment
    if p_noise_raw == 0:
        p_noise_raw = 1e-8
    p_target_noise = p_signal / (10 ** (target_snr / 10.0))
    scale_factor = np.sqrt(p_target_noise / p_noise_raw)
    scaled_noise = noise_segment * scale_factor
    mixed = signal + scaled_noise
    max_val = np.max(np.abs(mixed))
    if max_val > 0.99:
        mixed = mixed / max_val * 0.95
    return mixed

def main():
    parser = argparse.ArgumentParser(description="Create unseen noise dataset with custom SNR")
    parser.add_argument("--snr", type=float, default=10.0, help="Target SNR in dB (e.g. 10.0, 5.0, 3.0)")
    args = parser.parse_args()
    
    if not os.path.exists(NOISE_FILE):
        print(f"❌ Error: {NOISE_FILE} not found!")
        sys.exit(1)
        
    # Clear output directory if it exists to overwrite completely
    if os.path.exists(OUTPUT_DIR):
        print(f"🧹 Clearing existing test dataset folder: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
        
    print(f"📖 Loading unseen noise source: {os.path.basename(NOISE_FILE)}...")
    noise, _ = librosa.load(NOISE_FILE, sr=SR, mono=True)
    print("✅ Unseen noise loaded successfully.")
    
    clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith(".wav")]
    if len(clean_files) < 10:
        print(f"❌ Error: Not enough clean files in {CLEAN_DIR}")
        sys.exit(1)
        
    random.seed(42)  # Fixed seed to extract consistent clean water pouring files
    selected_files = random.sample(clean_files, 10)
    print(f"🎯 Selected 10 clean files for mixing: {selected_files}")
    
    clean_out = os.path.join(OUTPUT_DIR, "clean")
    noisy_out = os.path.join(OUTPUT_DIR, "noisy")
    os.makedirs(clean_out, exist_ok=True)
    os.makedirs(noisy_out, exist_ok=True)
    
    for fname in tqdm(selected_files, desc=f"Mixing unseen noise (SNR {args.snr}dB)"):
        cpath = os.path.join(CLEAN_DIR, fname)
        cdata, csr = sf.read(cpath)
        if csr != SR:
            cdata = librosa.resample(cdata, orig_sr=csr, target_sr=SR)
        
        # Max peak normalize to 0.8
        cmax = np.max(np.abs(cdata))
        if cmax > 1e-6:
            cdata = cdata / cmax * 0.8
            
        clen = len(cdata)
        if len(noise) <= clen:
            noise_seg = np.tile(noise, int(np.ceil(clen / len(noise))))[:clen]
        else:
            start_idx = random.randint(0, len(noise) - clen - 1)
            noise_seg = noise[start_idx : start_idx + clen]
            
        ndata = mix_audio_at_snr(cdata, noise_seg, target_snr=args.snr)
        
        # Save clean and mixed noisy files under exact same filename for direct comparison
        sf.write(os.path.join(clean_out, fname), cdata, SR)
        sf.write(os.path.join(noisy_out, fname), ndata, SR)
        
    print(f"🎉 Unseen noise test dataset created successfully at {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
