import sys
import os
import numpy as np
import librosa
sys.path.append('.')
from simulate_pitch_matching import build_calibration_templates

video_path = "media_assets/example_video.mp4"
ref_mels_db, ref_l_preds, full_audio_np, sr = build_calibration_templates(video_path)

start_idx = 0
end_idx = int(0.5 * sr)
live_chunk = full_audio_np[start_idx:end_idx]

print("Live chunk shape:", live_chunk.shape)
try:
    live_mels = librosa.feature.melspectrogram(y=live_chunk, sr=sr, n_fft=1024, hop_length=512, n_mels=64)
    live_mels_db = librosa.power_to_db(live_mels, ref=np.max)
    live_feature = np.mean(live_mels_db, axis=1)
    
    print("live_feature shape:", live_feature.shape)
    print("ref_mels_db.T shape:", ref_mels_db.T.shape)
    
    distances = np.linalg.norm(ref_mels_db.T - live_feature, axis=1)
    best_match_idx = np.argmin(distances)
    print("best_match_idx:", best_match_idx)
    current_pred = ref_l_preds[best_match_idx]
    print("current_pred:", current_pred)
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
