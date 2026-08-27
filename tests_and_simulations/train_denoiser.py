import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import librosa
from tqdm import tqdm

# Windows console encoding fix
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from sound_of_water.audio_pitch.denoiser import AudioDenoisingWrapper

# Constants
SR = 16000
CHUNK_LEN = 16000  # 1 second chunk for training

class AudioPairDataset(Dataset):
    """
    Dataset that pairs clean and noisy audio files.
    Returns random 1-second chunks from matching clean and noisy files.
    """
    def __init__(self, clean_dir, noisy_dir, chunk_len=CHUNK_LEN):
        self.clean_dir = os.path.join(clean_dir, "audios")
        self.noisy_dir = os.path.join(noisy_dir, "audios")
        self.chunk_len = chunk_len
        
        # List audio files
        clean_files = sorted([f for f in os.listdir(self.clean_dir) if f.endswith(".wav")])
        noisy_files = sorted([f for f in os.listdir(self.noisy_dir) if f.endswith(".wav")])
        
        clean_map = {os.path.splitext(f)[0]: f for f in clean_files}
        noisy_map = {}
        for f in noisy_files:
            name, ext = os.path.splitext(f)
            if name.endswith("_noisy"):
                base = name[:-6]
            else:
                base = name
            noisy_map[base] = f
            
        self.bases = sorted(list(set(clean_map.keys()).intersection(noisy_map.keys())))
        self.pairs = []
        for base in self.bases:
            clean_path = os.path.join(self.clean_dir, clean_map[base])
            noisy_path = os.path.join(self.noisy_dir, noisy_map[base])
            self.pairs.append((clean_path, noisy_path))
            
        print(f"📁 Dataset initialized with {len(self.pairs)} matching clean-noisy pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        clean_path, noisy_path = self.pairs[idx]
        
        try:
            # Load clean audio
            clean_data, sr_clean = sf.read(clean_path)
            if len(clean_data.shape) > 1:
                clean_data = np.mean(clean_data, axis=1)
            if sr_clean != SR:
                clean_data = librosa.resample(clean_data, orig_sr=sr_clean, target_sr=SR)
                
            # Load noisy audio
            noisy_data, sr_noisy = sf.read(noisy_path)
            if len(noisy_data.shape) > 1:
                noisy_data = np.mean(noisy_data, axis=1)
            if sr_noisy != SR:
                noisy_data = librosa.resample(noisy_data, orig_sr=sr_noisy, target_sr=SR)
        except Exception as e:
            # Fallback to zero arrays in case of reading error
            print(f"Error loading {clean_path}: {e}")
            clean_data = np.zeros(self.chunk_len, dtype=np.float32)
            noisy_data = np.zeros(self.chunk_len, dtype=np.float32)
            
        # Align lengths if they differ slightly
        min_len = min(len(clean_data), len(noisy_data))
        if min_len < self.chunk_len:
            # Zero pad if shorter than chunk length
            pad_len = self.chunk_len - min_len
            clean_data = np.pad(clean_data[:min_len], (0, pad_len))
            noisy_data = np.pad(noisy_data[:min_len], (0, pad_len))
            start_idx = 0
        else:
            clean_data = clean_data[:min_len]
            noisy_data = noisy_data[:min_len]
            # Select random 1-second chunk
            start_idx = random.randint(0, min_len - self.chunk_len)
            
        clean_chunk = clean_data[start_idx : start_idx + self.chunk_len]
        noisy_chunk = noisy_data[start_idx : start_idx + self.chunk_len]
        
        # Target peak amplitude norm (0.8) as in main evaluation code
        max_clean = np.max(np.abs(clean_chunk))
        if max_clean > 1e-6:
            clean_chunk = clean_chunk / max_clean * 0.8
            
        max_noisy = np.max(np.abs(noisy_chunk))
        if max_noisy > 1e-6:
            noisy_chunk = noisy_chunk / max_noisy * 0.8
            
        return {
            "clean": torch.tensor(clean_chunk, dtype=torch.float32),
            "noisy": torch.tensor(noisy_chunk, dtype=torch.float32)
        }

def train():
    import argparse
    parser = argparse.ArgumentParser(description="Train Lightweight 2D U-Net Denoising Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # Initialize directories
    clean_dir = os.path.join(ROOT_DIR, "sound_of_water_dataset")
    noisy_dir = os.path.join(ROOT_DIR, "sound_of_water_dataset_noisy")
    
    # Create dataset
    full_dataset = AudioPairDataset(clean_dir, noisy_dir)
    
    # Split into train/validation
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Initialize wrapper model (STFT + UNet + ISTFT)
    model = AudioDenoisingWrapper().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Multi-resolution Spectral loss helper
    def spectral_loss_fn(y_pred, y_true):
        # y_pred, y_true: [B, 1, 1, 16000] -> flattened to [B, 16000]
        yp_flat = y_pred.view(y_pred.shape[0], -1)
        yt_flat = y_true.view(y_true.shape[0], -1)
        
        # Compute STFT magnitude
        stft_p = torch.stft(yp_flat, n_fft=512, hop_length=160, win_length=512, window=model.window, return_complex=True)
        stft_t = torch.stft(yt_flat, n_fft=512, hop_length=160, win_length=512, window=model.window, return_complex=True)
        
        mag_p = torch.abs(stft_p)
        mag_t = torch.abs(stft_t)
        
        l1_mag = F.l1_loss(mag_p, mag_t)
        l1_log_mag = F.l1_loss(torch.log(mag_p + 1e-7), torch.log(mag_t + 1e-7))
        return l1_mag + l1_log_mag

    models_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(models_dir, "denoiser_best.pth")

    print("\n🚀 Starting Training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            noisy = batch["noisy"].to(device) # [B, 16000]
            clean = batch["clean"].to(device) # [B, 16000]
            
            # Format inputs to [B, T=1, C=1, NS=16000]
            noisy_in = noisy.unsqueeze(1).unsqueeze(1)
            clean_in = clean.unsqueeze(1).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Forward
            clean_pred = model(noisy_in) # [B, 1, 1, 16000]
            
            # Waveform Reconstruction Loss (MSE)
            loss_wave = F.mse_loss(clean_pred, clean_in)
            # Spectral Loss
            loss_spec = spectral_loss_fn(clean_pred, clean_in)
            
            # Combine losses
            loss = loss_wave * 100.0 + loss_spec
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Wave": f"{loss_wave.item():.5f}", "Spec": f"{loss_spec.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_wave = 0.0
        val_spec = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                noisy = batch["noisy"].to(device)
                clean = batch["clean"].to(device)
                noisy_in = noisy.unsqueeze(1).unsqueeze(1)
                clean_in = clean.unsqueeze(1).unsqueeze(1)
                
                clean_pred = model(noisy_in)
                loss_wave = F.mse_loss(clean_pred, clean_in)
                loss_spec = spectral_loss_fn(clean_pred, clean_in)
                loss = loss_wave * 100.0 + loss_spec
                
                val_loss += loss.item()
                val_wave += loss_wave.item()
                val_spec += loss_spec.item()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_wave = val_wave / len(val_loader)
        avg_val_spec = val_spec / len(val_loader)
        
        print(f"📊 Epoch {epoch} Summary - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} (Wave: {avg_val_wave:.5f}, Spec: {avg_val_spec:.4f})")
        
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save only the UNet weights to save space (Wrapper can wrap it back)
            torch.save(model.unet.state_dict(), best_ckpt_path)
            print(f"✨ Best model saved to {best_ckpt_path} with Val Loss: {avg_val_loss:.4f}")
            
    print("\n🎉 Training completed successfully.")

if __name__ == "__main__":
    train()
