import os
import sys
import random
import torch
import soundfile as sf
import numpy as np
import librosa

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from sound_of_water.audio_pitch.denoiser import AudioDenoisingWrapper

SR = 16000
CHUNK_LEN = 16000

def load_audio_local(path):
    data, sr = sf.read(path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    if sr != SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    
    # Peak amplitude norm to 0.8
    max_val = np.max(np.abs(data))
    if max_val > 1e-6:
        data = data / max_val * 0.8
    return data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")
    
    # Load denoiser
    denoiser = AudioDenoisingWrapper().to(device)
    ckpt_path = os.path.join(ROOT_DIR, "models", "denoiser_best.pth")
    if os.path.exists(ckpt_path):
        denoiser.unet.load_state_dict(torch.load(ckpt_path, map_location=device))
        denoiser.eval()
        print("✅ Loaded denoiser checkpoint.")
    else:
        print(f"❌ Error: checkpoint not found at {ckpt_path}")
        return
        
    clean_dir = os.path.join(ROOT_DIR, "sound_of_water_dataset", "audios")
    noisy_dir = os.path.join(ROOT_DIR, "sound_of_water_dataset_noisy", "audios")
    output_dir = os.path.join(ROOT_DIR, "demo_denoised")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find matching pairs
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith(".wav")])
    noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.endswith(".wav")])
    
    clean_map = {os.path.splitext(f)[0]: f for f in clean_files}
    noisy_map = {}
    for f in noisy_files:
        name, ext = os.path.splitext(f)
        if name.endswith("_noisy"):
            base = name[:-6]
        else:
            base = name
        noisy_map[base] = f
        
    common_bases = sorted(list(set(clean_map.keys()).intersection(noisy_map.keys())))
    
    # Select 3 random bases
    random.seed(42)  # For reproducibility
    selected_bases = random.sample(common_bases, 3)
    
    print(f"🎯 Selected files for comparison: {selected_bases}")
    
    for base in selected_bases:
        print(f"⏳ Processing {base}...")
        clean_path = os.path.join(clean_dir, clean_map[base])
        noisy_path = os.path.join(noisy_dir, noisy_map[base])
        
        clean_np = load_audio_local(clean_path)
        noisy_np = load_audio_local(noisy_path)
        
        # Prepare noisy chunks
        length = len(noisy_np)
        # Pad to multiple of CHUNK_LEN
        num_chunks = int(np.ceil(length / CHUNK_LEN))
        pad_len = num_chunks * CHUNK_LEN - length
        noisy_padded = np.pad(noisy_np, (0, pad_len))
        
        # Reshape to [1, T, 1, CHUNK_LEN]
        chunks = noisy_padded.reshape(1, num_chunks, 1, CHUNK_LEN)
        chunks_tensor = torch.tensor(chunks, dtype=torch.float32).to(device)
        
        # Run denoiser
        with torch.no_grad():
            denoised_chunks_tensor = denoiser(chunks_tensor)
            
        denoised_padded = denoised_chunks_tensor.cpu().numpy().reshape(-1)
        # Remove padding
        denoised_np = denoised_padded[:length]
        
        # Save files
        sf.write(os.path.join(output_dir, f"{base}_0_clean.wav"), clean_np, SR)
        sf.write(os.path.join(output_dir, f"{base}_1_noisy.wav"), noisy_np, SR)
        sf.write(os.path.join(output_dir, f"{base}_2_denoised.wav"), denoised_np, SR)
        
    print(f"🎉 Done! Saved outputs to {output_dir}")

if __name__ == "__main__":
    main()
