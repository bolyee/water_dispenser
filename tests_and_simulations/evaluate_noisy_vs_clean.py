"""
evaluate_noisy_vs_clean.py
─────────────────────────────────────────────
다중 음원 기반 클린 vs 노이즈 수위 매칭 정량 비교 평가 프로그램

1) 데이터셋에서 클린 및 노이즈 합성 오디오 파일 매칭 쌍 확보
2) 사용자 지정 개수만큼 무작위 오디오 선정
3) 각 오디오에 대해:
   - AI 모델로 Ground Truth(정답) 수위 궤적 계산
   - 클린 및 노이즈 오디오 각각에 대해 1초 슬라이딩 윈도우 Mel 매칭 시뮬레이션 수행
   - 평균 절대 오차(MAE) 및 80% 자동 정지 지연 시간 계산
4) 종합 요약 리포트(평균 오차 증가량, 정지 지연 비교) 출력
"""

import os
import sys

# Windows console encoding fix
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import random
import json
import numpy as np
import librosa
import torch
import soundfile as sf
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo.util import load_model, get_model_output, visualise_args
import shared.utils as su

# ──────────────────────────────────────────────
FILL_RATIO    = 0.55
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN_DIR     = os.path.join(ROOT_DIR, "sound_of_water_dataset")
NOISY_DIR     = os.path.join(ROOT_DIR, "sound_of_water_dataset_noisy")
SR            = 16000
MEL_WINDOW_S  = 1.0    # 실시간 윈도우 크기 (초)
MEL_HOP_S     = 0.25   # 캘리브레이션 윈도우 간격 (초)
N_MELS        = 64
FMAX          = 8000
# ──────────────────────────────────────────────

def get_audio_files(directory):
    audio_dir = os.path.join(directory, "audios")
    if not os.path.exists(audio_dir):
        return []
    return sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])

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
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate noisy vs clean audios")
    parser.add_argument("--use_denoiser", action="store_true", help="Apply 2D U-Net denoiser to noisy audio")
    parser.add_argument("--denoiser_ckpt", type=str, default=None, help="Path to denoiser weights (e.g., models/denoiser_best.pth)")
    parser.add_argument("--num_eval", type=int, default=0, help="Number of files to evaluate. If 0, prompts user.")
    args = parser.parse_args()

    print("=" * 80)
    print(" 📊 클린 vs 노이즈 수위 매칭 정량 비교 평가 프로그램")
    if args.use_denoiser:
        print(" 🛡️ [디노이징 전처리 활성화 (Light-weight 2D U-Net)]")
    print("=" * 80)

    clean_audios = get_audio_files(CLEAN_DIR)
    noisy_audios = get_audio_files(NOISY_DIR)

    if not clean_audios or not noisy_audios:
        print("[FAIL] 데이터셋 폴더를 찾을 수 없거나 오디오 파일이 없습니다.")
        sys.exit(1)

    # 1. 파일 매핑 매치
    clean_map = {os.path.splitext(f)[0]: f for f in clean_audios}
    noisy_map = {}
    for f in noisy_audios:
        name, ext = os.path.splitext(f)
        if name.endswith("_noisy"):
            base = name[:-6]
        else:
            base = name
        noisy_map[base] = f

    common_bases = sorted(list(set(clean_map.keys()).intersection(noisy_map.keys())))
    if not common_bases:
        print("[FAIL] 깨끗한 오디오와 소음 오디오 간 매칭 쌍이 없습니다.")
        sys.exit(1)

    total_pairs = len(common_bases)
    print(f"📖 총 {total_pairs}개의 물소리-소음 오디오 쌍이 감지되었습니다.")

    # 사용자 입력 개수 결정
    num_eval = args.num_eval
    if num_eval == 0:
        try:
            ans = input(f"평가할 파일 개수를 입력하세요 (1~{total_pairs}, 기본값: 20): ").strip()
            if ans:
                num_eval = int(ans)
                if num_eval < 1: num_eval = 1
                if num_eval > total_pairs: num_eval = total_pairs
            else:
                num_eval = 20
        except (ValueError, KeyboardInterrupt, Exception):
            num_eval = 20
    if num_eval > total_pairs:
        num_eval = total_pairs

    print(f"⚡ {num_eval}개의 무작위 오디오 파일 쌍을 분석하여 비교 평가를 시작합니다.")
    selected_bases = random.sample(common_bases, num_eval)

    # 2. AI 모델 로딩
    print("\n[AI 모델 로딩 중...]")
    model = load_model()
    print("[AI 모델 로딩 완료!]")

    # 2.1 디노이저 로딩
    denoiser_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.use_denoiser:
        print("[디노이징 모듈 로딩 중...]")
        from sound_of_water.audio_pitch.denoiser import AudioDenoisingWrapper
        denoiser_model = AudioDenoisingWrapper().to(device)
        
        # Load weights
        ckpt_path = args.denoiser_ckpt
        if ckpt_path is None:
            ckpt_path = os.path.join(ROOT_DIR, "models", "denoiser_best.pth")
            
        if os.path.exists(ckpt_path):
            print(f"[>>>] Loading denoiser weights from {ckpt_path}")
            denoiser_model.unet.load_state_dict(torch.load(ckpt_path, map_location=device))
            denoiser_model.eval()
            print("[디노이징 모듈 로딩 완료!]\n")
        else:
            print(f"[>>>] WARNING: Denoiser checkpoint not found at {ckpt_path}. Using random weights!\n")

    # 결과 누적 변수
    results = []
    clean_maes = []
    noisy_maes = []
    clean_delays = []
    noisy_delays = []
    
    clean_stop_fail_count = 0
    noisy_stop_fail_count = 0

    print(f"{'파일명':<30} | {'Clean MAE':<10} | {'Noisy MAE':<10} | {'Clean Delay':<11} | {'Noisy Delay':<11}")
    print("-" * 82)

    for base in tqdm(selected_bases, desc="Evaluating pairs"):
        clean_file = clean_map[base]
        noisy_file = noisy_map[base]
        
        clean_path = os.path.join(CLEAN_DIR, "audios", clean_file)
        noisy_path = os.path.join(NOISY_DIR, "audios", noisy_file)

        # 오디오 로드
        _, clean_np = load_audio_tensor_local(clean_path)
        _, noisy_np = load_audio_tensor_local(noisy_path)

        # AI Ground Truth 분석 (Clean 기준)
        clean_tensor, _ = load_audio_tensor_local(clean_path)
        with torch.no_grad():
            _, y_audio = get_model_output(clean_tensor, model)
            wavelengths = y_audio @ torch.linspace(
                0, visualise_args['w_max'], visualise_args['n_bins']
            ).to(y_audio.device)
            l_preds = su.physics.estimate_length_of_air_column(wavelengths).numpy()

        l_max = float(np.max(l_preds))
        threshold = l_max * (1.0 - FILL_RATIO)  # 남은공간 임계치
        n_frames = len(l_preds)
        timestamps_eval = librosa.frames_to_time(
            np.arange(n_frames),
            sr=visualise_args['sr'],
            n_fft=visualise_args['n_fft'],
            hop_length=visualise_args['hop_length'],
        )

        # Ground Truth 정지 시점
        t_stop_gt = None
        for idx, t_v in enumerate(timestamps_eval):
            if l_preds[idx] <= threshold:
                t_stop_gt = t_v
                break

        # 캘리브레이션 템플릿용 윈도우 생성 (Clean 기준)
        win_samples = int(MEL_WINDOW_S * SR)
        hop_samples = int(MEL_HOP_S    * SR)
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

        # 실시간 매칭 평가 루프 (realtime_esp32_mic.py와 완전히 동일한 파라미터 적용)
        t_step = 1.0  # INFERENCE_INTERVAL = 1.0s
        total_len_s = len(clean_np) / SR
        
        errors_clean = []
        errors_noisy = []
        
        t_stop_clean = None
        t_stop_noisy = None
        
        # 클린 매칭 상태 변수
        accepted_clean = l_max
        consec_clean = 0
        is_stopped_clean = False
        
        # 노이즈 매칭 상태 변수
        accepted_noisy = l_max
        consec_noisy = 0
        is_stopped_noisy = False
        
        # 물 흘러내림 시작 지점 (두 음원이 동기화되어 있으므로 클린 음원 기준 감지 시점을 공유)
        water_start_time = None
        
        # 물리 파라미터 설정
        MAX_CHANGE = 3.0  # cm
        CONFIRM_COUNT_REQUIRED = 2
        silence_threshold = 0.00075  # realtime_esp32_mic.py의 기본값 기준
        
        # 1초 간격 루프
        for t in np.arange(MEL_WINDOW_S, total_len_s, t_step):
            end_idx = int(t * SR)
            
            # Ground truth 참값 수위 (현재 시점 기준)
            idx_gt = int(np.argmin(np.abs(timestamps_eval - t)))
            gt_val = l_preds[idx_gt]

            # 1. Clean 매칭 시뮬레이션
            chunk_clean = clean_np[end_idx - win_samples : end_idx]
            chunk_clean_rms = float(np.sqrt(np.mean(chunk_clean ** 2)))
            
            # 클린 오디오 기준으로 최초 감지 지점을 공동 시작점으로 결정
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
                    
                    # 물리적 변화 제약 체크
                    delta_clean = abs(raw_pred_clean - accepted_clean)
                    if delta_clean > MAX_CHANGE:
                        # 오값 홀드 및 연속 카운터 리셋
                        consec_clean = 0
                    else:
                        accepted_clean = raw_pred_clean
                        if accepted_clean <= threshold:
                            consec_clean += 1
                        else:
                            consec_clean = 0
                            
                    # 최종 채택된 예측값을 에러 계산에 활용
                    errors_clean.append(abs(gt_val - accepted_clean))
                    
                    # 자동 정지 감지 (물 감지 이후 1.0초 경과 필터)
                    t_pour_clean = t - water_start_time if water_start_time is not None else 0.0
                    if consec_clean >= CONFIRM_COUNT_REQUIRED and t_pour_clean > 1.0:
                        is_stopped_clean = True
                        t_stop_clean = t
                else:
                    # 무음 대기 중에는 에러 평가에 accepted_clean(직전값) 유지
                    errors_clean.append(abs(gt_val - accepted_clean))
            else:
                # 이미 멈춘 상태에서는 accepted_clean(정지 시점의 최종값) 유지하며 에러 평가
                errors_clean.append(abs(gt_val - accepted_clean))

            # 2. Noisy 매칭 시뮬레이션
            chunk_noisy = noisy_np[end_idx - win_samples : end_idx]
            
            if args.use_denoiser and denoiser_model is not None:
                # Denoise chunk [1, 1, 1, win_samples]
                chunk_tensor = torch.tensor(chunk_noisy, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    clean_chunk_tensor = denoiser_model(chunk_tensor)
                chunk_noisy = clean_chunk_tensor.squeeze().cpu().numpy()
                
            chunk_noisy_rms = float(np.sqrt(np.mean(chunk_noisy ** 2)))
            
            if not is_stopped_noisy:
                # 노이즈 음원은 음량이 크므로 무음 대기 통과
                if chunk_noisy_rms >= silence_threshold:
                    mel_noisy = librosa.feature.melspectrogram(y=chunk_noisy, sr=SR, n_mels=N_MELS, fmax=FMAX)
                    feat_noisy = librosa.power_to_db(mel_noisy, ref=np.max).mean(axis=1)
                    feat_noisy_n = feat_noisy / (np.linalg.norm(feat_noisy) + 1e-8)
                    
                    sims_noisy = mel_calib_norm @ feat_noisy_n
                    best_noisy = int(np.argmax(sims_noisy))
                    raw_pred_noisy = float(lpred_per_window[best_noisy])
                    
                    # 물리적 변화 제약 체크
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
                    
                    # 자동 정지 감지 (물 감지 이후 1.0초 경과 필터)
                    t_pour_noisy = t - water_start_time if water_start_time is not None else 0.0
                    if consec_noisy >= CONFIRM_COUNT_REQUIRED and t_pour_noisy > 1.0:
                        is_stopped_noisy = True
                        t_stop_noisy = t
                else:
                    errors_noisy.append(abs(gt_val - accepted_noisy))
            else:
                errors_noisy.append(abs(gt_val - accepted_noisy))

        # 오차 평균
        mae_clean = np.mean(errors_clean) if errors_clean else 0.0
        mae_noisy = np.mean(errors_noisy) if errors_noisy else 0.0
        clean_maes.append(mae_clean)
        noisy_maes.append(mae_noisy)

        # 정지 지연 시간 계산 (오직 참값 도달 시점이 계산 가능한 경우만 지연 시간 누적)
        delay_clean_str = "미정지"
        delay_noisy_str = "미정지"

        if t_stop_gt is not None:
            if t_stop_clean is not None:
                delay_clean = t_stop_clean - t_stop_gt
                clean_delays.append(delay_clean)
                delay_clean_str = f"{delay_clean:+.1f}s"
            else:
                clean_stop_fail_count += 1
                
            if t_stop_noisy is not None:
                delay_noisy = t_stop_noisy - t_stop_gt
                noisy_delays.append(delay_noisy)
                delay_noisy_str = f"{delay_noisy:+.1f}s"
            else:
                noisy_stop_fail_count += 1
        else:
            # 컵이 덜 채워진 상태로 물이 멈췄거나 데이터가 짧아 도달하지 못한 경우
            delay_clean_str = "N/A"
            delay_noisy_str = "N/A"

        # 콘솔 테이블 한 줄 출력
        print(f"{clean_file:<30} | {mae_clean:9.2f}cm | {mae_noisy:9.2f}cm | {delay_clean_str:<11} | {delay_noisy_str:<11}")
        
        results.append({
            "file": clean_file,
            "mae_clean": mae_clean,
            "mae_noisy": mae_noisy,
            "delay_clean": delay_clean_str,
            "delay_noisy": delay_noisy_str
        })

    # 전체 요약 통계 계산
    avg_mae_clean = np.mean(clean_maes)
    avg_mae_noisy = np.mean(noisy_maes)
    diff_mae = avg_mae_noisy - avg_mae_clean

    avg_delay_clean = np.mean(clean_delays) if clean_delays else 0.0
    avg_delay_noisy = np.mean(noisy_delays) if noisy_delays else 0.0
    diff_delay = avg_delay_noisy - avg_delay_clean

    print("-" * 82)
    print("\n📊 [종합 정량 비교 리포트]")
    print(f"  - 평가 대상 파일 개수: {num_eval}개")
    print(f"  - 컵 자동 정지 임계 기준: {int(FILL_RATIO*100)}% 채워짐 기준")
    print()
    print(f"  1. 📏 수위 감지 평균 절대 오차 (MAE):")
    print(f"     * 소음 없음 (Clean): {avg_mae_clean:.2f} cm")
    print(f"     * 소음 있음 (Noisy): {avg_mae_noisy:.2f} cm")
    print(f"     * 👉 소음으로 인한 오차 증가량: {diff_mae:+.2f} cm")
    print()
    print(f"  2. 🛑 80% 수위 도달 시 자동 정지 평균 지연 시간 (Stop Latency):")
    print(f"     * 소음 없음 (Clean): {avg_delay_clean:+.2f} 초 (정지 실패: {clean_stop_fail_count}건)")
    print(f"     * 소음 있음 (Noisy): {avg_delay_noisy:+.2f} 초 (정지 실패: {noisy_stop_fail_count}건)")
    print(f"     * 👉 소음으로 인한 추가 지연: {diff_delay:+.2f} 초")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
