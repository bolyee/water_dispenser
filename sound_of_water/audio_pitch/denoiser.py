import torch
import torch.nn as nn
import torch.nn.functional as F

class Lightweight2DUnet(nn.Module):
    """
    Spectrogram-based lightweight 2D U-Net for audio denoising.
    Predicts a multiplicative mask (0.0 to 1.0) applied to the input magnitude spectrogram.
    """
    def __init__(self):
        super().__init__()
        # Encoder
        # Input shape: [B, 1, F, T] (e.g., [B, 1, 257, Frames])
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU()
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Downsamp 2x

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Downsamp 2x

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU()
        )

        # Decoder
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1), # Concat channel (32+32)
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1), # Concat channel (16+16)
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU()
        )

        # Out (produces mask)
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # x: [B, 1, F, T]
        # Encoder
        x1 = self.enc1(x)
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        d2 = self.up2(b)
        
        # Adjust dimensions if odd-sized input caused size mismatch during pooling
        if d2.shape[2] != x2.shape[2] or d2.shape[3] != x2.shape[3]:
            d2 = F.pad(d2, [0, x2.shape[3] - d2.shape[3], 0, x2.shape[2] - d2.shape[2]])
            
        m2 = torch.cat([d2, x2], dim=1)
        dec2_out = self.dec2(m2)

        d1 = self.up1(dec2_out)
        if d1.shape[2] != x1.shape[2] or d1.shape[3] != x1.shape[3]:
            d1 = F.pad(d1, [0, x1.shape[3] - d1.shape[3], 0, x1.shape[2] - d1.shape[2]])
            
        m1 = torch.cat([d1, x1], dim=1)
        dec1_out = self.dec1(m1)

        mask = torch.sigmoid(self.out(dec1_out))
        return mask


class AudioDenoisingWrapper(nn.Module):
    """
    Wraps the spectrogram-based 2D U-Net.
    Accepts 1D Audio Waveforms [B, T, C, NS] and returns cleaned [B, T, C, NS].
    Processes STFT -> U-Net Masking -> ISTFT internally.
    """
    def __init__(self, n_fft=512, hop_length=160, win_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        self.unet = Lightweight2DUnet()
        
        # Register window tensor to device automatically
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, x):
        # Input shape: [B, T, C, NS]
        # We assume C = 1 (single channel) as per project requirements
        B, T, C, NS = x.shape
        assert C == 1, "Only single channel audio is supported"
        
        # Reshape to 2D batch for processing: (B*T) x NS
        x_flat = x.view(B * T, NS)
        
        # 1. STFT
        # output: [B*T, F, Frames] (Complex tensor)
        stft_res = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        
        magnitude = torch.abs(stft_res) # [B*T, F, Frames]
        phase = torch.angle(stft_res)     # [B*T, F, Frames]
        
        # Add channel dimension: [B*T, 1, F, Frames]
        mag_input = magnitude.unsqueeze(1)
        
        # 2. Denoise (Masking)
        mask = self.unet(mag_input) # [B*T, 1, F, Frames]
        clean_mag = mag_input * mask
        clean_mag = clean_mag.squeeze(1) # [B*T, F, Frames]
        
        # Reconstruct complex spectrogram
        clean_stft = torch.polar(clean_mag, phase)
        
        # 3. ISTFT
        # output: [B*T, NS]
        clean_waveform = torch.istft(
            clean_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            length=NS # Keep exactly the same length
        )
        
        # Reshape back to original dimensions [B, T, C, NS]
        return clean_waveform.view(B, T, C, NS)

if __name__ == "__main__":
    # Quick sanity check
    print("Testing AudioDenoisingWrapper...")
    wrapper = AudioDenoisingWrapper()
    # Mock data: Batch of 2, 3 clips, 1 channel, 16000 samples each
    x_test = torch.randn(2, 3, 1, 16000)
    print("Input shape:", x_test.shape)
    
    with torch.no_grad():
        out = wrapper(x_test)
    print("Output shape:", out.shape)
    assert x_test.shape == out.shape, "Shape mismatch!"
    print("Success! Dimensions match perfectly.")
