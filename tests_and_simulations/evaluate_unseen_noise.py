import os
import sys
import numpy as np
import librosa
import torch
import soundfile as sf
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from demo.util import load_model, get_model_output, visualise_args
import shared.utils as su
from sound_of_water.audio_pitch.denoiser import AudioDenoisingWrapper

# Configuration
TEST_DIR = os.path.join(ROOT_DIR, "unseen_test_dataset")
CLEAN_DIR = os.path.join(TEST_DIR, "clean")
NOISY_DIR = os.path.join(TEST_DIR, "noisy")
DENOISED_DIR = os.path.join(TEST_DIR, "denoised")

SR = 16000
CHUNK_LEN = 16000
FILL_RATIO = 0.55
MEL_WINDOW_S = 1.0
MEL_HOP_S = 0.25
N_MELS = 64
FMAX = 8000

def load_audio_tensor_local(path):
    data, sr = sf.read(path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    if sr != SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
        
    # Normalization peak at 0.8
    max_val = np.max(np.abs(data))
    if max_val > 1e-6:
        data = data / max_val * 0.8
        
    from transformers import Wav2Vec2FeatureExtractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    inputs = feature_extractor(data, sampling_rate=SR, return_tensors="pt", padding=False)
    return inputs.input_values.unsqueeze(0), data

def main():
    print("=" * 80)
    print(" 📊 Unseen Noise (TV Noise 2) Generalization Evaluation")
    print("=" * 80)
    
    if not os.path.exists(CLEAN_DIR) or not os.path.exists(NOISY_DIR):
        print("❌ Error: Test dataset directories not found. Please run create_unseen_test_dataset.py first.")
        sys.exit(1)
        
    os.makedirs(DENOISED_DIR, exist_ok=True)
    
    clean_files = sorted([f for f in os.listdir(CLEAN_DIR) if f.endswith(".wav")])
    if not clean_files:
        print("❌ Error: No files in test dataset.")
        sys.exit(1)
        
    # 1. Load AI model and Denoiser
    print("\n[AI 모델 및 디노이저 로드 중...]")
    model = load_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser = AudioDenoisingWrapper().to(device)
    ckpt_path = os.path.join(ROOT_DIR, "models", "denoiser_best.pth")
    if os.path.exists(ckpt_path):
        denoiser.unet.load_state_dict(torch.load(ckpt_path, map_location=device))
        denoiser.eval()
        print("✅ Loaded denoiser checkpoint.")
    else:
        print(f"❌ Warning: Denoiser weights not found at {ckpt_path}. Using random weights.")
    print("[로드 완료!]\n")
    
    # Results accumulators
    clean_maes = []
    noisy_maes = []
    denoised_maes = []
    
    clean_delays = []
    noisy_delays = []
    denoised_delays = []
    
    clean_stop_fail = 0
    noisy_stop_fail = 0
    denoised_stop_fail = 0
    
    print(f"{'파일명':<25} | {'Clean MAE':<9} | {'Noisy MAE':<9} | {'Denoised MAE':<12} | {'Clean Delay':<11} | {'Noisy Delay':<11} | {'Denoised Delay':<14}")
    print("-" * 115)
    
    for fname in tqdm(clean_files, desc="Evaluating"):
        clean_path = os.path.join(CLEAN_DIR, fname)
        noisy_path = os.path.join(NOISY_DIR, fname)
        denoised_path = os.path.join(DENOISED_DIR, fname)
        
        # Load clean and noisy arrays
        _, clean_np = load_audio_tensor_local(clean_path)
        _, noisy_np = load_audio_tensor_local(noisy_path)
        
        # 1. Denoise and save the denoised file
        length = len(noisy_np)
        num_chunks = int(np.ceil(length / CHUNK_LEN))
        pad_len = num_chunks * CHUNK_LEN - length
        noisy_padded = np.pad(noisy_np, (0, pad_len))
        
        chunks = noisy_padded.reshape(1, num_chunks, 1, CHUNK_LEN)
        chunks_tensor = torch.tensor(chunks, dtype=torch.float32).to(device)
        with torch.no_grad():
            denoised_chunks_tensor = denoiser(chunks_tensor)
        denoised_padded = denoised_chunks_tensor.cpu().numpy().reshape(-1)
        denoised_np = denoised_padded[:length]
        
        # Save denoised wave
        sf.write(denoised_path, denoised_np, SR)
        
        # Ground Truth Trajectory calculation (using Clean reference)
        clean_tensor, _ = load_audio_tensor_local(clean_path)
        with torch.no_grad():
            _, y_audio = get_model_output(clean_tensor, model)
            wavelengths = y_audio @ torch.linspace(
                0, visualise_args['w_max'], visualise_args['n_bins']
            ).to(y_audio.device)
            l_preds = su.physics.estimate_length_of_air_column(wavelengths).numpy()
            
        l_max = float(np.max(l_preds))
        threshold = l_max * (1.0 - FILL_RATIO)
        n_frames = len(l_preds)
        timestamps_eval = librosa.frames_to_time(
            np.arange(n_frames),
            sr=visualise_args['sr'],
            n_fft=visualise_args['n_fft'],
            hop_length=visualise_args['hop_length'],
        )
        
        t_stop_gt = None
        for idx, t_v in enumerate(timestamps_eval):
            if l_preds[idx] <= threshold:
                t_stop_gt = t_v
                break
                
        # Create matching calibration template using Clean
        win_samples = int(MEL_WINDOW_S * SR)
        hop_samples = int(MEL_HOP_S * SR)
        mel_windows_list = []
        lpred_per_window = []
        
        for start in range(0, len(clean_np) - win_samples + 1, hop_samples):
            chunk = clean_np[start : start + win_samples]
            mel = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=N_MELS, fmax=FMAX)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            feat = mel_db.mean(axis=1)
            t_c = (start + win_samples / 2) / SR
            idx_gt = int(np.argmin(np.abs(timestamps_eval - t_c)))
            mel_windows_list.append(feat)
            lpred_per_window.append(l_preds[idx_gt])
            
        mel_calib_arr = np.array(mel_windows_list, dtype=np.float32)
        norms = np.linalg.norm(mel_calib_arr, axis=1, keepdims=True) + 1e-8
        mel_calib_norm = mel_calib_arr / norms
        lpred_per_window = np.array(lpred_per_window, dtype=np.float32)
        
        # Real-time simulation settings
        t_step = 1.0
        total_len_s = len(clean_np) / SR
        
        errors_clean = []
        errors_noisy = []
        errors_denoised = []
        
        t_stop_clean = None
        t_stop_noisy = None
        t_stop_denoised = None
        
        # State variables
        accepted_clean = l_max
        consec_clean = 0
        is_stopped_clean = False
        
        accepted_noisy = l_max
        consec_noisy = 0
        is_stopped_noisy = False
        
        accepted_denoised = l_max
        consec_denoised = 0
        is_stopped_denoised = False
        
        water_start_time = None
        
        # Physical constraints
        MAX_CHANGE = 3.0
        CONFIRM_COUNT_REQUIRED = 2
        silence_threshold = 0.00075
        
        # 1-second interval loop
        for t in np.arange(MEL_WINDOW_S, total_len_s, t_step):
            end_idx = int(t * SR)
            idx_gt = int(np.argmin(np.abs(timestamps_eval - t)))
            gt_val = l_preds[idx_gt]
            
            # --- 1. Clean ---
            chunk_clean = clean_np[end_idx - win_samples : end_idx]
            chunk_clean_rms = float(np.sqrt(np.mean(chunk_clean ** 2)))
            
            if chunk_clean_rms >= silence_threshold and water_start_time is None:
                water_start_time = t
                
            if not is_stopped_clean:
                if chunk_clean_rms >= silence_threshold:
                    mel_clean = librosa.feature.melspectrogram(y=chunk_clean, sr=SR, n_mels=N_MELS, fmax=FMAX)
                    feat_clean = librosa.power_to_db(mel_clean, ref=np.max).mean(axis=1)
                    feat_clean_n = feat_clean / (np.linalg.norm(feat_clean) + 1e-8)
                    sims_clean = mel_calib_norm @ feat_clean_n
                    best_clean = int(np.argmax(sims_clean))
                    raw_pred_clean = float(lpred_per_window[best_clean])
                    
                    delta_clean = abs(raw_pred_clean - accepted_clean)
                    if delta_clean > MAX_CHANGE:
                        consec_clean = 0
                    else:
                        accepted_clean = raw_pred_clean
                        if accepted_clean <= threshold:
                            consec_clean += 1
                        else:
                            consec_clean = 0
                            
                    errors_clean.append(abs(gt_val - accepted_clean))
                    t_pour_clean = t - water_start_time if water_start_time is not None else 0.0
                    if consec_clean >= CONFIRM_COUNT_REQUIRED and t_pour_clean > 1.0:
                        is_stopped_clean = True
                        t_stop_clean = t
                else:
                    errors_clean.append(abs(gt_val - accepted_clean))
            else:
                errors_clean.append(abs(gt_val - accepted_clean))
                
            # --- 2. Noisy ---
            chunk_noisy = noisy_np[end_idx - win_samples : end_idx]
            chunk_noisy_rms = float(np.sqrt(np.mean(chunk_noisy ** 2)))
            
            if not is_stopped_noisy:
                if chunk_noisy_rms >= silence_threshold:
                    mel_noisy = librosa.feature.melspectrogram(y=chunk_noisy, sr=SR, n_mels=N_MELS, fmax=FMAX)
                    feat_noisy = librosa.power_to_db(mel_noisy, ref=np.max).mean(axis=1)
                    feat_noisy_n = feat_noisy / (np.linalg.norm(feat_noisy) + 1e-8)
                    sims_noisy = mel_calib_norm @ feat_noisy_n
                    best_noisy = int(np.argmax(sims_noisy))
                    raw_pred_noisy = float(lpred_per_window[best_noisy])
                    
                    delta_noisy = abs(raw_pred_noisy - accepted_noisy)
                    if delta_noisy > MAX_CHANGE:
                        consec_noisy = 0
                    else:
                        accepted_noisy = raw_pred_noisy
                        if accepted_noisy <= threshold:
                            consec_noisy += 1
                        else:
                            consec_noisy = 0
                            
                    errors_noisy.append(abs(gt_val - accepted_noisy))
                    t_pour_noisy = t - water_start_time if water_start_time is not None else 0.0
                    if consec_noisy >= CONFIRM_COUNT_REQUIRED and t_pour_noisy > 1.0:
                        is_stopped_noisy = True
                        t_stop_noisy = t
                else:
                    errors_noisy.append(abs(gt_val - accepted_noisy))
            else:
                errors_noisy.append(abs(gt_val - accepted_noisy))
                
            # --- 3. Denoised ---
            chunk_denoised = denoised_np[end_idx - win_samples : end_idx]
            chunk_denoised_rms = float(np.sqrt(np.mean(chunk_denoised ** 2)))
            
            if not is_stopped_denoised:
                if chunk_denoised_rms >= silence_threshold:
                    mel_denoised = librosa.feature.melspectrogram(y=chunk_denoised, sr=SR, n_mels=N_MELS, fmax=FMAX)
                    feat_denoised = librosa.power_to_db(mel_denoised, ref=np.max).mean(axis=1)
                    feat_denoised_n = feat_denoised / (np.linalg.norm(feat_denoised) + 1e-8)
                    sims_denoised = mel_calib_norm @ feat_denoised_n
                    best_denoised = int(np.argmax(sims_denoised))
                    raw_pred_denoised = float(lpred_per_window[best_denoised])
                    
                    delta_denoised = abs(raw_pred_denoised - accepted_denoised)
                    if delta_denoised > MAX_CHANGE:
                        consec_denoised = 0
                    else:
                        accepted_denoised = raw_pred_denoised
                        if accepted_denoised <= threshold:
                            consec_denoised += 1
                        else:
                            consec_denoised = 0
                            
                    errors_denoised.append(abs(gt_val - accepted_denoised))
                    t_pour_denoised = t - water_start_time if water_start_time is not None else 0.0
                    if consec_denoised >= CONFIRM_COUNT_REQUIRED and t_pour_denoised > 1.0:
                        is_stopped_denoised = True
                        t_stop_denoised = t
                else:
                    errors_denoised.append(abs(gt_val - accepted_denoised))
            else:
                errors_denoised.append(abs(gt_val - accepted_denoised))
                
        # Calculate file metrics
        mae_clean = np.mean(errors_clean) if errors_clean else 0.0
        mae_noisy = np.mean(errors_noisy) if errors_noisy else 0.0
        mae_denoised = np.mean(errors_denoised) if errors_denoised else 0.0
        
        clean_maes.append(mae_clean)
        noisy_maes.append(mae_noisy)
        denoised_maes.append(mae_denoised)
        
        delay_clean_str = "미정지"
        delay_noisy_str = "미정지"
        delay_denoised_str = "미정지"
        
        if t_stop_gt is not None:
            if t_stop_clean is not None:
                delay_clean = t_stop_clean - t_stop_gt
                clean_delays.append(delay_clean)
                delay_clean_str = f"{delay_clean:+.1f}s"
            else:
                clean_stop_fail += 1
                
            if t_stop_noisy is not None:
                delay_noisy = t_stop_noisy - t_stop_gt
                noisy_delays.append(delay_noisy)
                delay_noisy_str = f"{delay_noisy:+.1f}s"
            else:
                noisy_stop_fail += 1
                
            if t_stop_denoised is not None:
                delay_denoised = t_stop_denoised - t_stop_gt
                denoised_delays.append(delay_denoised)
                delay_denoised_str = f"{delay_denoised:+.1f}s"
            else:
                denoised_stop_fail += 1
        else:
            delay_clean_str = "N/A"
            delay_noisy_str = "N/A"
            delay_denoised_str = "N/A"
            
        print(f"{fname:<25} | {mae_clean:7.2f}cm | {mae_noisy:7.2f}cm | {mae_denoised:10.2f}cm | {delay_clean_str:<11} | {delay_noisy_str:<11} | {delay_denoised_str:<14}")

    # Summary
    avg_mae_clean = np.mean(clean_maes)
    avg_mae_noisy = np.mean(noisy_maes)
    avg_mae_denoised = np.mean(denoised_maes)
    
    avg_delay_clean = np.mean(clean_delays) if clean_delays else 0.0
    avg_delay_noisy = np.mean(noisy_delays) if noisy_delays else 0.0
    avg_delay_denoised = np.mean(denoised_delays) if denoised_delays else 0.0
    
    print("-" * 115)
    print("\n📊 [미학습 소음(TV Noise 2) 종합 정량 비교 리포트]")
    print(f"  - 평가 대상 파일 개수: {len(clean_files)}개")
    print()
    print("  1. 📏 수위 감지 평균 절대 오차 (MAE):")
    print(f"     * 소음 없음 (Clean):             {avg_mae_clean:.2f} cm")
    print(f"     * 미학습 소음 노출 (Noisy):      {avg_mae_noisy:.2f} cm")
    print(f"     * 디노이저 정제 후 (Denoised):   {avg_mae_denoised:.2f} cm")
    print(f"     * 👉 디노이저로 인한 에러 개선:  {avg_mae_noisy - avg_mae_denoised:+.2f} cm (Clean 대비 차이: {avg_mae_denoised - avg_mae_clean:+.2f} cm)")
    print()
    print("  2. 🛑 80% 수위 도달 시 자동 정지 평균 지연 시간 (Stop Latency):")
    print(f"     * 소음 없음 (Clean):             {avg_delay_clean:+.2f} 초 (정지 실패: {clean_stop_fail}건)")
    print(f"     * 미학습 소음 노출 (Noisy):      {avg_delay_noisy:+.2f} 초 (정지 실패: {noisy_stop_fail}건)")
    print(f"     * 디노이저 정제 후 (Denoised):   {avg_delay_denoised:+.2f} 초 (정지 실패: {denoised_stop_fail}건)")
    print(f"     * 👉 디노이저로 인한 지연 단축:  {avg_delay_noisy - avg_delay_denoised:+.2f} 초")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
