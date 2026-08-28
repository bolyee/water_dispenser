"""
realtime_mic.py  —  마이크 기반 실시간 정수기 수위 모니터
--------------------------------------------------------------
사용법:
    pip install sounddevice
    python realtime_mic.py

컵 학습 방법:
    1. 프로그램 실행
    2. "새 컵 학습" 선택
    3. 안내에 따라 빈 컵에 끝까지 물을 따르면서 마이크로 소리를 녹음
    4. 학습 결과가 자동 저장됨 (이후부터는 빠른 로드 가능)
"""

import os
import sys
import threading
import time

import numpy as np
import torch
import socket
import cv2
import soundfile as sf
import librosa
import requests
from transformers import Wav2Vec2FeatureExtractor

# 이 파일은 realtime/ 안에 있으므로 저장소 루트는 한 단계 위.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from demo.util import load_model, get_model_output, visualise_args
import shared.utils as su

# 모델 학습 시와 동일한 오디오 정규화기 (zero-mean, unit-variance)
_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

def preprocess_audio(audio_np: np.ndarray) -> torch.Tensor:
    """마이크 raw numpy 배열을 모델 입력 형식 [1,1,L] 텐서로 변환합니다."""
    inputs = _feature_extractor(
        audio_np, sampling_rate=SR, return_tensors="pt", padding=False
    )
    # input_values: [1, L] → unsqueeze → [1, 1, L]
    return inputs.input_values.unsqueeze(0)


FILL_RATIO       = 0.55   # 몇 % 채워지면 멈출지 (지연 보완을 위해 55%로 하향)
SR               = 16000  # 마이크 샘플링 레이트 (16kHz 고정)
INFERENCE_INTERVAL = 1.0  # AI 추론 주기 (초)
UDP_PORT         = 5005   # ESP32 오디오 수신 포트
ESP32_IP         = "20.30.88.125"  # ESP32 IP 주소 (지연 진단 및 HTTP 제어용)
# ============================================================

CACHE_DIR = os.path.join(ROOT_DIR, "calibration_cache")

# 공유 상태
shared = {
    "current_l_pred": None,
    "is_stopped": False,
    "running": True,
    "audio_buffer": np.array([], dtype=np.float32),
}
lock = threading.Lock()


# ─────────────────────────────────────────────
#  캐시 / 컵 선택 메뉴
# ─────────────────────────────────────────────

def list_caches():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return sorted([f for f in os.listdir(CACHE_DIR) if f.endswith(".npz")])


def select_or_create_cache(model):
    """학습된 컵 목록을 보여주고 선택하거나, 새 컵을 학습한다."""
    caches = list_caches()

    print("\n" + "="*58)
    print("  [컵 선택 메뉴]")
    for i, name in enumerate(caches):
        info = np.load(os.path.join(CACHE_DIR, name))
        l_max = float(info['l_max'])
        l_min = float(info['l_min'])
        print(f"  [{i+1}] {name.replace('_calibration.npz', '')}")
        print(f"       컵 높이: {l_max:.2f}cm | 꽉 찰 때: {l_min:.2f}cm")
    print(f"  [0] 새 컵 학습 (마이크로 녹음)")
    print("="*58)

    while True:
        try:
            ans = int(input(f"  번호 선택 (0~{len(caches)}): ").strip())
            if ans == 0:
                return calibrate_new_cup(model)
            elif 1 <= ans <= len(caches):
                selected_path = os.path.join(CACHE_DIR, caches[ans - 1])
                info = np.load(selected_path)
                l_max = float(info['l_max'])
                print(f"\n✅ 기존 컵 로드 완료! 전체 높이: {l_max:.2f}cm | 임계값: {l_max*(1-FILL_RATIO):.2f}cm\n")
                return l_max, selected_path
            else:
                print(f"  0~{len(caches)} 사이 숫자를 입력해 주세요.")
        except ValueError:
            print("  숫자를 입력해 주세요.")




# ─────────────────────────────────────────────
#  UDP 리스너 (백그라운드 스레드)
# ─────────────────────────────────────────────
def udp_listener():
    UDP_IP = "0.0.0.0"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.5)
    print(f"📡 UDP 오디오 수신 대기 중... (포트: {UDP_PORT})")
    
    while True:
        with lock:
            if not shared["running"]:
                break
        try:
            data, addr = sock.recvfrom(4096)
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            with lock:
                shared["audio_buffer"] = np.concatenate([shared["audio_buffer"], audio_chunk])
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[UDP 수신 오류] {e}")
            break
    sock.close()

# ─────────────────────────────────────────────
#  마이크 사전 검증
# ─────────────────────────────────────────────

def check_mic():
    print("\n🔍 ESP32 마이크 데이터 수신 및 주변 노이즈 상태를 확인합니다... (2초 대기)")
    with lock:
        shared["audio_buffer"] = np.array([], dtype=np.float32)
    time.sleep(2.0)
    
    with lock:
        audio = shared["audio_buffer"].copy()
    
    if len(audio) == 0:
        print("❌ ESP32에서 오디오 데이터가 오지 않습니다.")
        print("   -> ESP32 와이파이 연결 여부 및 PC IP 주소를 확인하세요.")
        return False, 0.0
    
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"   측정된 배경 노이즈 레벨(RMS): {rms:.5f}")
    if rms < 1e-5:
        print("❌ 데이터는 오고 있지만 소리가 너무 작습니다. I2S 핀 연결을 확인하세요.")
        return False, 0.0

    print("✅ ESP32 마이크 정상 작동 및 수신 완료!")
    return True, rms


def validate_recording(audio_np, noise_rms):
    """
    녹음된 오디오가 학습에 적합한지 판단합니다.
    """
    MIN_DURATION_S = 3.0   # 최소 녹음 길이
    MIN_RMS        = 5e-4  # 전체 최소 신호 레벨
    MIN_VAR_RATIO  = 0.10  # 앞/뒤 RMS 변화 비율 최솟값 (물이 차면 소리가 변해야 함)

    duration = len(audio_np) / SR
    if duration < MIN_DURATION_S:
        return False, f"녹음이 너무 짧습니다 ({duration:.1f}초). 최소 {MIN_DURATION_S}초 이상 녹음해 주세요."

    rms_total = float(np.sqrt(np.mean(audio_np ** 2)))
    if rms_total < MIN_RMS:
        return False, f"소리 신호가 너무 약합니다 (RMS={rms_total:.5f}). 마이크를 컵 가까이 대거나 볼륨을 높여 주세요."

    # 노이즈 체크 (물소리가 주변 소음보다 압도적으로 커야 함)
    if rms_total < noise_rms * 1.5:
        return False, f"물소리(RMS={rms_total:.5f})가 주변 환경 노이즈(RMS={noise_rms:.5f})에 완전히 묻혔습니다!\n   마이크 게인을 조절하시거나, 주변(선풍기/PC 소음 등)을 조용하게 만든 뒤 다시 시도해 주세요."

    # 앞 20% vs 뒤 20% 에너지 변화 (물이 차면 공명 주파수가 달라지므로 변화 있어야 함)
    n = len(audio_np)
    rms_start = float(np.sqrt(np.mean(audio_np[:n//5] ** 2)))
    rms_end   = float(np.sqrt(np.mean(audio_np[-n//5:] ** 2)))
    if rms_start < 1e-6 or rms_end < 1e-6:
        return False, "앞부분이나 뒷부분 소리가 거의 없습니다. 처음부터 끝까지 물을 천천히 따라 주세요."

    var_ratio = abs(rms_start - rms_end) / max(rms_start, rms_end)
    if var_ratio < MIN_VAR_RATIO:
        return False, (f"소리 변화가 너무 적습니다 (변화율={var_ratio:.1%}). "
                       "빈 컵에 물을 따르면서 소리가 변해야 합니다. 배경 소음만 녹음된 것은 아닌지 확인해 주세요.")

    return True, f"✅ 녹음 품질 양호! (길이={duration:.1f}s, RMS={rms_total:.5f}, 변화율={var_ratio:.1%})"


# ─────────────────────────────────────────────
#  새 컵 학습 (마이크 녹음 → AI 분석 → 캐시 저장)
# ─────────────────────────────────────────────

def calibrate_new_cup(model):
    """마이크로 새 컵에 물 따르는 소리를 녹음하고 AI로 분석해 캐시를 저장한다."""

    cup_name = input("\n  새 컵 이름을 입력하세요 (예: tall_glass, mug): ").strip().replace(" ", "_")
    if not cup_name:
        cup_name = "new_cup"

    print("\n" + "="*58)
    print("  [새 컵 학습 모드]")
    print("="*58)

    # ① 마이크 연결 사전 검사 및 주변 소음(노이즈) 측정
    noise_rms = 0.0
    while True:
        ok, measured_noise = check_mic()
        if ok:
            noise_rms = measured_noise
            break
        retry = input("  마이크 문제를 해결한 후 다시 시도하시겠습니까? [Y/n]: ").strip().lower()
        if retry in ('n', 'no'):
            sys.exit(1)

    print("\n  준비되면 Enter를 누르고, 빈 컵에 물을 끝까지 따르세요.")
    print("  물 따르기가 완전히 끝나면 다시 Enter를 누르세요.")
    input("  ▶ 준비됐으면 Enter ▶ ")

    # ESP32 녹음 시작
    print("🎙️  [녹음 시작] 지금 컵에 물을 따르세요...")
    with lock:
        shared["audio_buffer"] = np.array([], dtype=np.float32)
        
    input("  ⏹  물을 다 따랐으면 Enter ▶ ")
    
    with lock:
        audio_np = shared["audio_buffer"].copy()
        
    if len(audio_np) == 0:
        print("[ERROR] 수신된 오디오가 없습니다.")
        sys.exit(1)
    duration = len(audio_np) / SR
    print(f"  총 {duration:.1f}초 녹음 완료.")

    # ② 녹음 품질 검증 (노이즈 대비 물소리 크기 확인)
    ok, msg = validate_recording(audio_np, noise_rms)
    if not ok:
        print(f"\n⚠️  녹음 품질 문제: {msg}")
        retry = input("  다시 녹음하시겠습니까? [Y/n]: ").strip().lower()
        if retry not in ('n', 'no'):
            return calibrate_new_cup(model)  # 재귀 호출로 처음부터 다시
        else:
            print("학습을 취소합니다.")
            sys.exit(1)
    print(msg)

    # AI로 수위 분석
    # → 마이크 raw 오디오를 WAV 파일로 저장하고,
    #   simulate_stop.py와 완전히 동일한 load_audio_tensor() 파이프라인을 통과시킵니다.
    print("\n🧠 AI가 녹음된 소리를 분석합니다... (잠시만 기다려 주세요)")

    from demo.util import load_audio_tensor

    # 임시 WAV 파일로 저장
    tmp_wav = os.path.join(CACHE_DIR, f"_tmp_{cup_name}.wav")
    os.makedirs(CACHE_DIR, exist_ok=True)
    sf.write(tmp_wav, audio_np, SR)
    print(f"  임시 WAV 저장: {tmp_wav}")

    # simulate_stop.py와 동일한 전처리 파이프라인
    audio_tensor = load_audio_tensor(tmp_wav)

    with torch.no_grad():
        z_audio, y_audio = get_model_output(audio_tensor, model)
        wavelengths_tensor = y_audio @ torch.linspace(
            0, visualise_args['w_max'], visualise_args['n_bins']
        ).to(y_audio.device)
        l_preds = su.physics.estimate_length_of_air_column(wavelengths_tensor).numpy()

    wavelengths_np = wavelengths_tensor.cpu().numpy()

    # 임시 파일 삭제
    os.remove(tmp_wav)

    # 수위 범위 산출
    l_max = float(np.max(l_preds))
    l_min = float(np.mean(l_preds[-10:]))
    print(f"[정보] 컵 범위: {l_max:.2f}cm (빈) ~ {l_min:.2f}cm (꽉 참)")

    # ─────────────────────────────────────────────
    # Mel 스펙트로그램 윈도우 저장 (1초 간격 반영)
    # ─────────────────────────────────────────────
    MEL_WINDOW_S = 1.0
    MEL_HOP_S    = 1.0   # 1초 간격으로 큼직하게 (사용자 요청)
    N_MELS       = 64
    win_samples  = int(MEL_WINDOW_S * SR)
    hop_samples  = int(MEL_HOP_S    * SR)

    mel_windows_list   = []
    lpred_per_window   = []
    rms_per_window     = []  # ★ 음량 정보 추가

    n_frames_total = len(l_preds)
    timestamps_eval = librosa.frames_to_time(
        np.arange(n_frames_total),
        sr=visualise_args['sr'],
        n_fft=visualise_args['n_fft'],
        hop_length=visualise_args['hop_length'],
    )

    for start in range(0, len(audio_np) - win_samples + 1, hop_samples):
        chunk = audio_np[start : start + win_samples]
        # RMS 계산
        c_rms = float(np.sqrt(np.mean(chunk ** 2)))
        # Mel 계산
        mel   = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=N_MELS, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_feat = mel_db.mean(axis=1)

        t_center = (start + win_samples / 2) / SR
        idx_lpred = int(np.argmin(np.abs(timestamps_eval - t_center)))

        mel_windows_list.append(mel_feat)
        lpred_per_window.append(l_preds[idx_lpred])
        rms_per_window.append(c_rms)

    mel_windows_arr  = np.array(mel_windows_list, dtype=np.float32)
    lpred_win_arr    = np.array(lpred_per_window, dtype=np.float32)
    rms_win_arr      = np.array(rms_per_window, dtype=np.float32)

    print(f"[정보] Mel 윈도우({MEL_HOP_S}s 간격) 생성 완료: {len(mel_windows_arr)}개")
    print(f"      평균 음량(RMS): {rms_win_arr.mean():.5f}")

    # 코사인 유사도를 위한 사전 정규화
    norms = np.linalg.norm(mel_windows_arr, axis=1, keepdims=True) + 1e-8
    mel_windows_norm = mel_windows_arr / norms

    # 캐시 저장
    cache_path = os.path.join(CACHE_DIR, f"{cup_name}_calibration.npz")
    np.savez(
        cache_path,
        timestamps_eval  = timestamps_eval,
        l_pred           = l_preds,
        l_max            = l_max,
        l_min            = l_min,
        z_audio          = z_audio.numpy(),
        wavelengths      = wavelengths_np,
        mel_windows_norm = mel_windows_norm,
        lpred_per_window = lpred_win_arr,
        rms_per_window   = rms_win_arr,
        noise_rms        = noise_rms,          # ★ 배경 노이즈 레벨 저장
    )
    print(f"💾 학습 결과 저장 완료 (RMS 포함): {cache_path}\n")

    return l_max, cache_path






# ─────────────────────────────────────────────
#  실시간 Mel 스펙트로그램 매칭 스레드
# ─────────────────────────────────────────────

def ai_worker(model, l_max, threshold, cache_path):
    """
    캘리브레이션 때 저장한 1초 Mel 스펙트로그램과 실시간 마이크 오디오를 비교하여 수위를 추정합니다.
    - Mel 스펙트로그램 코사인 유사도로 가장 비슷한 시점의 수위를 예측
    - 물리적 제약 (1.5cm/초) + 연속 확인 (2회)으로 노이즈 오작동 방지
    """
    data = np.load(cache_path)

    if 'mel_windows_norm' not in data:
        print("[경고] 이 캐시는 Mel 윈도우가 없습니다. 컵을 다시 학습해 주세요.")
        with lock:
            shared["running"] = False
        return

    mel_calib_norm   = data['mel_windows_norm']   # [N_win, N_MELS]
    lpred_per_window = data['lpred_per_window']   # [N_win]
    N_MELS   = mel_calib_norm.shape[1]
    WINDOW_S = 1.0

    # 저장된 노이즈 레벨을 기반으로 동적 대기 임계치 계산 (기존 하드코딩 0.0003 개선)
    noise_rms = float(data['noise_rms']) if 'noise_rms' in data else 0.0005
    silence_threshold = max(3e-4, noise_rms * 1.5)

    print(f"[Mel Matcher] 윈도우 {len(mel_calib_norm)}개 로드 (N_MELS={N_MELS})")
    print(f"  수위 범위: {lpred_per_window.max():.1f}cm ~ {lpred_per_window.min():.1f}cm")
    print(f"  수위 임계치(목표 공간): {threshold:.2f}cm 이하")
    print(f"  소리 감지 임계치 (RMS): {silence_threshold:.5f} (이보다 소리가 커야 동작 시작)")
    print("  컵에 물을 부어 주세요!\n")

    # ESP32 서보모터 밸브 오픈 신호 전송
    try:
        print(f"📡 ESP32 SG90 서보모터 밸브 오픈 신호 전송 중... (http://{ESP32_IP}/open)")
        headers = {'Connection': 'close', 'User-Agent': 'Mozilla/5.0'}
        response = requests.get(f"http://{ESP32_IP}/open", headers=headers, timeout=3)
        print(f"✅ ESP32 밸브(서보모터) 오픈 성공! (응답 코드: {response.status_code})")
    except Exception as e:
        print(f"❌ 밸브 오픈 통신 지연/에러 발생 (에러 무시 후 분석 진행): {e}")

    consecutive_below = 0
    accepted_pred = l_max  # 빈 컵에서 시작
    MAX_CHANGE = 3.0  # cm — 1초당 허용되는 최대 수위 변화 (Mel 윈도우 간격 고려)
    CONFIRM_COUNT_REQUIRED = 2
    water_start_time = None

    while True:
        with lock:
            if not shared["running"]:
                break
            if shared["is_stopped"]:
                time.sleep(0.1)
                continue
            buf = shared["audio_buffer"].copy()

        t_elapsed = len(buf) / SR
        if t_elapsed < WINDOW_S:
            time.sleep(0.2)
            continue

        # 가장 최근 1초 추출
        chunk     = buf[-int(WINDOW_S * SR):]
        chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))

        if chunk_rms < silence_threshold:
            print(f"[Mel] t={t_elapsed:.1f}s | 소리 대기 중... (RMS={chunk_rms:.5f})")
            time.sleep(INFERENCE_INTERVAL)
            continue

        # 물 흘러내림(소리 감지) 시작 절대 시간 기록
        if water_start_time is None:
            water_start_time = t_elapsed
            print(f"💧 물 흘러내림 감지 시작! (기준 시간: {water_start_time:.1f}초)")

        # ① Mel 스펙트로그램 계산 및 코사인 유사도 비교
        mel_live = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=N_MELS, fmax=8000)
        mel_feat = librosa.power_to_db(mel_live, ref=np.max).mean(axis=1)

        norm          = np.linalg.norm(mel_feat) + 1e-8
        mel_feat_norm = mel_feat / norm
        sims          = mel_calib_norm @ mel_feat_norm

        best_idx = int(np.argmax(sims))
        raw_pred = float(lpred_per_window[best_idx])

        # ② 물리적 타당성 검사: 1초 만에 수위가 1.5cm 이상 변할 수 없음
        delta = abs(raw_pred - accepted_pred)
        if delta > MAX_CHANGE:
            # 변화가 너무 크면 노이즈 → 이전 값 유지
            print(f"[Mel] t={t_elapsed:.1f}s | sim={sims[best_idx]:.3f} | (Δ={delta:.1f}cm > {MAX_CHANGE}) → Hold ({accepted_pred:.2f}cm) | RMS={chunk_rms:.5f}")
            consecutive_below = 0
            with lock:
                shared["current_l_pred"] = accepted_pred
            time.sleep(INFERENCE_INTERVAL)
            continue

        # ③ 타당한 예측 → 채택
        accepted_pred = raw_pred

        # ④ 연속 확인 카운터 — 연속 2회 이상 임계치 이하여야 정지
        if accepted_pred <= threshold:
            consecutive_below += 1
        else:
            consecutive_below = 0

        below_status = f" [{consecutive_below}/{CONFIRM_COUNT_REQUIRED}]" if accepted_pred <= threshold else ""
        print(f"[Mel] t={t_elapsed:.1f}s | sim={sims[best_idx]:.3f} → Space: {accepted_pred:.2f}cm (thr: {threshold:.2f}cm) | RMS={chunk_rms:.5f}{below_status}")

        trigger_stop = False
        with lock:
            shared["current_l_pred"] = accepted_pred
            t_pour = t_elapsed - water_start_time if water_start_time is not None else 0.0
            if consecutive_below >= CONFIRM_COUNT_REQUIRED and t_pour > 1.0 and not shared["is_stopped"]:
                trigger_stop = True

        if trigger_stop:
            print(f"\n⚠️ 수위 임계치 도달! ({accepted_pred:.2f}cm ≤ {threshold:.2f}cm)")
            with lock:
                shared["is_stopped"] = True
            try:
                print(f"📡 ESP32 SG90 서보모터 정지 신호 전송 중... (http://{ESP32_IP}/stop)")
                headers = {'Connection': 'close', 'User-Agent': 'Mozilla/5.0'}
                response = requests.get(f"http://{ESP32_IP}/stop", headers=headers, timeout=5)
                if response.status_code == 200:
                    print(f"✅ ESP32 밸브(서보모터) 물리적 잠금 성공!")
                else:
                    print(f"❌ ESP32 응답 코드: {response.status_code}")
            except Exception as e:
                print(f"❌ 통신 에러: {e}")
                print("   → 수동으로 밸브를 잠가 주세요!")
            print("🚨 [AUTO STOP] 시스템 정지 완료.")

        time.sleep(INFERENCE_INTERVAL)


# ─────────────────────────────────────────────
#  OpenCV 디스플레이
# ─────────────────────────────────────────────

def display_loop(l_max, threshold):
    win_w, win_h = 600, 400

    while True:
        with lock:
            l_pred = shared["current_l_pred"]
            is_stopped = shared["is_stopped"]
            t_elapsed = len(shared["audio_buffer"]) / SR

        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

        cv2.putText(canvas, "Real-Time Mic Water Level Monitor", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(canvas, f"Recording: {t_elapsed:.1f} s", (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

        if l_pred is not None:
            cv2.putText(canvas, f"Empty Space: {l_pred:.2f} cm", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # 수위 바 (l_max 기준 채워진 비율)
            bar_x, bar_y, bar_w, bar_h = 30, 160, win_w - 60, 60
            fill_frac = min(max((l_max - l_pred) / l_max, 0.0), 1.0)
            filled_w = int(bar_w * fill_frac)
            bar_color = (0, 200, 255) if not is_stopped else (0, 0, 255)
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), bar_color, -1)
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 2)
            cv2.putText(canvas, f"{int(fill_frac*100)}% filled", (bar_x + 5, bar_y + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(canvas, f"Stop at {int(FILL_RATIO*100)}% ({threshold:.2f} cm)",
                        (bar_x, bar_y + bar_h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 100), 1)
        else:
            cv2.putText(canvas, "Listening... (waiting for audio buffer)", (30, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        if is_stopped:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (win_w, win_h), (0, 0, 255), -1)
            canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)
            cv2.putText(canvas, "AUTO STOP TRIGGERED!", (50, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)
            if l_pred is not None:
                cv2.putText(canvas, f"Water reached {l_pred:.2f} cm!", (80, 330),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.putText(canvas, "Press 'q' to quit", (win_w - 200, win_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        cv2.imshow("Mic Water Level Monitor", canvas)
        try:
            if cv2.waitKey(100) & 0xFF == ord('q'):
                with lock:
                    shared["running"] = False
                break
        except KeyboardInterrupt:
            with lock:
                shared["running"] = False
            break

    cv2.destroyAllWindows()


def measure_latency(ip):
    """ESP32의 /ping 라우트로 5회 테스트 요청을 보내 무선 통신 레이턴시를 구하고 품질을 평가합니다."""
    print("\n" + "="*58)
    print(f" 📡 ESP32 무선 통신 지연 시간(Latency) 진단 중... (대상: {ip})")
    print("="*58)
    
    url = f"http://{ip}/ping"
    headers = {'Connection': 'close', 'User-Agent': 'Mozilla/5.0'}
    latencies = []
    
    # 5회 핑 테스트
    for i in range(5):
        try:
            start_time = time.time()
            response = requests.get(url, headers=headers, timeout=2.0)
            elapsed = (time.time() - start_time) * 1000.0  # ms 단위
            if response.status_code == 200 and response.text.strip() == "pong":
                latencies.append(elapsed)
                print(f"   [{i+1}/5] 응답 수신: {elapsed:.1f} ms")
            else:
                print(f"   [{i+1}/5] 오류: HTTP 코드 {response.status_code} 수신")
        except requests.exceptions.RequestException as e:
            print(f"   [{i+1}/5] 연결 실패: {e}")
        time.sleep(0.1)  # 요청 간의 짧은 대기시간
        
    if not latencies:
        print("\n❌ [경고] ESP32와 HTTP 통신이 불가능합니다!")
        print("   - IP 주소가 올바른지, ESP32의 웹서버가 켜져 있는지 확인해 주세요.")
        print("="*58 + "\n")
        return None
        
    avg_rtt = sum(latencies) / len(latencies)
    max_rtt = max(latencies)
    
    # 지연 상태 판정
    if avg_rtt < 50.0:
        status_str = "🟢 Excellent (안전 - 연결 상태 우수)"
    elif avg_rtt < 150.0:
        status_str = "🟡 Good (양호 - 미세한 반응 지연 있을 수 있음)"
    else:
        status_str = "🔴 Danger (오버플로우 위험 - 통신 지연 심각)"
        
    print(f"\n📊 [레이턴시 진단 결과]")
    print(f"   - 평균 지연 시간: {avg_rtt:.1f} ms")
    print(f"   - 최대 지연 시간: {max_rtt:.1f} ms")
    print(f"   - 연결 신뢰성 등급: {status_str}")
    
    if avg_rtt >= 150.0:
        print("\n⚠️ [경고] 무선 상태가 불안정하여 정지(/stop) 명령이 늦게 도착할 수 있습니다.")
        print("   정수기 물이 넘치는 것을 막기 위해 안전을 확보하고 시연하세요.")
    print("="*58 + "\n")
    return avg_rtt


# ─────────────────────────────────────────────
#  메인
# ─────────────────────────────────────────────

def main():
    print("=" * 58)
    print("   🎙️  실시간 마이크 정수기 수위 모니터   ")
    print("=" * 58)

    # AI 모델은 항상 먼저 로드 (학습에도, 추론에도 필요)
    print("\n[AI 모델 로딩 중... 잠시만 기다려 주세요]")
    model = load_model()
    print("[AI 모델 로딩 완료!]")

    # UDP 리스너 시작 (마이크 검사 및 학습을 위해 가장 먼저 켜야 함)
    listener_thread = threading.Thread(target=udp_listener, daemon=True)
    listener_thread.start()
    print(f"🎙️  ESP32 실시간 리스닝 시작! (포트: {UDP_PORT})")
    time.sleep(1.0) # 버퍼가 쌓일 시간을 확보

    # 컵 선택 (기존 캐시 재사용 or 새 컵 마이크 학습)
    l_max, cache_path = select_or_create_cache(model)
    threshold = l_max * (1.0 - FILL_RATIO)

    # ESP32 네트워크 레이턴시(지연 시간) 사전 측정
    measure_latency(ESP32_IP)

    # 특징 매칭 백그라운드 스레드 (무거운 AI 재추론 없음)
    worker = threading.Thread(target=ai_worker, args=(model, l_max, threshold, cache_path), daemon=True)
    worker.start()

    # 디스플레이 (메인 스레드)
    display_loop(l_max, threshold)

    # 종료
    print("\n🛑 스트리밍 종료.")


if __name__ == "__main__":
    main()
