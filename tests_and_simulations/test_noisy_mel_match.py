"""
test_noisy_mel_match.py — Noisy Audio Level Estimation & Matching Simulator (With Metadata Info)
--------------------------------------------------------------------------------
1. Loads a noisy audio file (with TV noise) and its clean original version.
2. Computes the ground truth water level trajectory from the clean audio using AI model.
3. Simulates the real-time dispenser algorithm (1s sliding window Mel cosine similarity)
   on the NOISY audio.
4. Outputs the comparison log to verify if it can accurately track water levels
   under loud 3dB TV noise, showing the exact clean and noise tracks used.

Usage:
    .\venv\Scripts\python.exe tests_and_simulations/test_noisy_mel_match.py
"""

import os
import sys
import random
import json
import numpy as np
import soundfile as sf
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo.util import load_model, get_model_output, visualise_args
import shared.utils as su

CLEAN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sound_of_water_dataset")
NOISY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sound_of_water_dataset_noisy")
SR = 16000
WINDOW_S = 1.0
INFERENCE_INTERVAL = 0.5  # 0.5초 주기로 스캔 시뮬레이션
N_MELS = 64

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
    
    # Normalize input amplitude to match scaling in mix_yt_noise.py
    max_val = np.max(np.abs(data))
    if max_val > 1e-6:
        data = data / max_val * 0.8
        
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    inputs = feature_extractor(data, sampling_rate=SR, return_tensors="pt", padding=False)
    return inputs.input_values.unsqueeze(0), data

def main():
    print("=" * 60)
    print(" 🧠 TV 소음 합성 데이터셋 수위 감지 및 정지 정확도 검증 스크립트")
    print("=" * 60)
    
    clean_audios = get_audio_files(CLEAN_DIR)
    noisy_audios = get_audio_files(NOISY_DIR)
    
    if not clean_audios or not noisy_audios:
        print("[FAIL] 데이터셋 폴더를 찾을 수 없거나 오디오 파일이 없습니다.")
        sys.exit(1)
        
    # 공통으로 매칭되는 오디오 리스트 확보 (파일명에 _noisy 접미사가 붙어도 매칭 가능하도록 수정)
    clean_map = {os.path.splitext(f)[0]: f for f in clean_audios}
    noisy_map = {}
    for f in noisy_audios:
        name, ext = os.path.splitext(f)
        if name.endswith("_noisy"):
            base = name[:-6]  # "_noisy" 제외
        else:
            base = name
        noisy_map[base] = f
        
    common_bases = list(set(clean_map.keys()).intersection(noisy_map.keys()))
    if not common_bases:
        print("[FAIL] 깨끗한 오디오와 소음 합성 오디오 간에 매칭되는 파일이 없습니다.")
        sys.exit(1)
        
    # mix_metadata.json 로드
    metadata_path = os.path.join(NOISY_DIR, "mix_metadata.json")
    mix_info = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                mix_info = json.load(f)
        except Exception as e:
            print(f"⚠️ mix_metadata.json 읽기 실패: {e}")
            
    print(f"📖 총 {len(common_bases)}개의 평가용 오디오셋 중 랜덤 1종을 선택해 시뮬레이션합니다.")
    selected_base = random.choice(common_bases)
    
    clean_file = clean_map[selected_base]
    noisy_file = noisy_map[selected_base]
    
    clean_path = os.path.join(CLEAN_DIR, "audios", clean_file)
    noisy_path = os.path.join(NOISY_DIR, "audios", noisy_file)
    
    # 1. AI 모델 로드
    print("\n[AI 모델 로딩 중...]")
    model = load_model()
    print("[AI 모델 로딩 완료!]")
    
    # 2. 오디오 로드 및 특징 계산
    # 메타데이터로부터 정보 조회
    noise_source_name = "알 수 없음 (다시 mix_yt_noise.py를 실행하면 기록됩니다)"
    snr_value = "알 수 없음"
    
    # 합성된 파일명(noisy_file) 또는 원본 파일명(clean_file) 둘 다 매칭 확인
    if noisy_file in mix_info:
        noise_source_name = mix_info[noisy_file].get("noise_file", noise_source_name)
        snr_value = f"{mix_info[noisy_file].get('snr_db', snr_value)}dB"
    elif clean_file in mix_info:
        noise_source_name = mix_info[clean_file].get("noise_file", noise_source_name)
        snr_value = f"{mix_info[clean_file].get('snr_db', snr_value)}dB"
        
    print("\n" + "="*70)
    print(f"🎧 테스트 파일 매핑 정보")
    print(f"   - 실제 물소리 원본: {clean_file}")
    print(f"   - 합성된 노이즈 오디오: {noisy_file}")
    print(f"   - 합성된 TV 소음원: {noise_source_name}")
    print(f"   - 설정된 노이즈 강도: {snr_value} SNR")
    print("="*70)
    
    # 깨끗한 오디오 기반 -> Ground Truth(정답) 수위 계산
    clean_tensor, clean_np = load_audio_tensor_local(clean_path)
    # 소음 오디오 기반 -> 실시간 마이크 입력 대용으로 사용
    noisy_tensor, noisy_np = load_audio_tensor_local(noisy_path)
    
    with torch.no_grad():
        # 깨끗한 원본 수위 계산 (AI 정답 궤적 구축)
        _, y_clean = get_model_output(clean_tensor, model)
        w_clean = y_clean @ torch.linspace(0, visualise_args['w_max'], visualise_args['n_bins']).to(y_clean.device)
        l_preds_clean = su.physics.estimate_length_of_air_column(w_clean).numpy()
        
        # 캘리브레이션 템플릿용 Mel 윈도우 구축 (깨끗한 상태 기준)
        mel_windows_list = []
        lpred_per_window = []
        win_samples = int(WINDOW_S * SR)
        hop_samples = int(0.25 * SR) # 0.25초 단위 조밀 매칭
        
        n_frames = len(l_preds_clean)
        timestamps_eval = librosa.frames_to_time(
            np.arange(n_frames),
            sr=visualise_args['sr'],
            n_fft=visualise_args['n_fft'],
            hop_length=visualise_args['hop_length'],
        )
        
        for start in range(0, len(clean_np) - win_samples + 1, hop_samples):
            chunk = clean_np[start : start + win_samples]
            mel = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=N_MELS, fmax=8000)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_feat = mel_db.mean(axis=1)
            
            t_center = (start + win_samples / 2) / SR
            idx_lpred = int(np.argmin(np.abs(timestamps_eval - t_center)))
            
            mel_windows_list.append(mel_feat)
            lpred_per_window.append(l_preds_clean[idx_lpred])
            
        mel_calib_arr = np.array(mel_windows_list, dtype=np.float32)
        norms = np.linalg.norm(mel_calib_arr, axis=1, keepdims=True) + 1e-8
        mel_calib_norm = mel_calib_arr / norms
        lpred_per_window = np.array(lpred_per_window, dtype=np.float32)
        
    # 3. 시뮬레이션 파라미터 세팅
    l_max = float(np.max(l_preds_clean))
    threshold = l_max * 0.20  # 남은 공간 20% (즉 80% 채워짐 기준)
    
    print("\n" + "="*70)
    print(f" ⚙️ 시뮬레이션 설정")
    print(f"    - 컵 최대 높이: {l_max:.2f}cm")
    print(f"    - 목표 정지 임계치(남은공간): {threshold:.2f}cm 이하")
    print("="*70)
    print("  시간(초)  |  실제 수위(정답) | 소음 환경 예측 |    오차    | 코사인유사도")
    print("-" * 70)
    
    t_step = INFERENCE_INTERVAL
    total_len_s = len(noisy_np) / SR
    
    # 0.5초 간격으로 오디오를 흘려보내며 매칭 수행
    ai_stopped_t = None
    mel_stopped_t = None
    
    for idx, t in enumerate(np.arange(WINDOW_S, total_len_s, t_step)):
        start_idx = 0
        end_idx = int(t * SR)
        
        # 실시간처럼 들어오는 1초 윈도우 추출 (소음 섞인 신호에서)
        chunk = noisy_np[end_idx - win_samples : end_idx]
        
        # 1. 실제 정답 탐색 (t시점과 가장 가까운 프레임의 깨끗한 수위)
        idx_clean = int(np.argmin(np.abs(timestamps_eval - t)))
        gt_val = l_preds_clean[idx_clean]
        
        # 2. 소음 상태에서 Mel 스펙트로그램 특징 계산
        mel_live = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=N_MELS, fmax=8000)
        mel_feat = librosa.power_to_db(mel_live, ref=np.max).mean(axis=1)
        norm = np.linalg.norm(mel_feat) + 1e-8
        mel_feat_norm = mel_feat / norm
        
        # 3. 캘리브레이션 템플릿과 비교 매칭
        sims = mel_calib_norm @ mel_feat_norm
        best_idx = int(np.argmax(sims))
        pred_val = float(lpred_per_window[best_idx])
        sim_score = sims[best_idx]
        
        error = abs(gt_val - pred_val)
        err_icon = "✅" if error <= 1.5 else "⚠️"
        
        print(f"    {t:4.1f}s    |    {gt_val:6.2f}cm    |   {pred_val:6.2f}cm   |  {error:5.2f}cm {err_icon} |    {sim_score:.4f}")
        
        # 정지 시점 기록
        if gt_val <= threshold and ai_stopped_t is None:
            ai_stopped_t = t
        if pred_val <= threshold and mel_stopped_t is None:
            mel_stopped_t = t
            
    print("-" * 70)
    print("\n📊 [정지 성능 분석 보고서]")
    if ai_stopped_t is not None:
        print(f"  - 원본 기준 목표 도달 시점 (AI 정답): {ai_stopped_t:.1f}초")
    else:
        print("  - 원본 기준 목표 수위에 도달하지 못함")
        
    if mel_stopped_t is not None:
        print(f"  - 소음 속 감지 및 정지 시점 (실시간 Mel): {mel_stopped_t:.1f}초")
    else:
        print("  - 소음 속에서 컵이 찬 것을 감지하지 못함 (미정지 에러)")
        
    if ai_stopped_t is not None and mel_stopped_t is not None:
        diff = mel_stopped_t - ai_stopped_t
        if diff >= 0:
            print(f"  🟢 지연 시간: +{diff:.1f}초 (지연 범위 정상)")
        else:
            print(f"  🔴 조기 차단: {diff:.1f}초 (물이 덜 찼는데 미리 잠금 - 감지 오작동)")
            
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
