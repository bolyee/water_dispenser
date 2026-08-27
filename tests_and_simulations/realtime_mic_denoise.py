"""
realtime_mic_denoise.py  —  마이크 기반 실시간 정수기 수위 모니터 (노이즈 감소 알고리즘 통합 버전)
-----------------------------------------------------------------------------------------
사용법:
    pip install sounddevice noisereduce scipy
    python realtime_mic_denoise.py

특징:
    1. 시작 시 2초간 배경 잡음(Noise Floor)을 수집하여 실시간 스펙트럼 차감(Spectral Subtraction) 노이즈 가공 필터 생성.
    2. 수위 측정 학습(Calibration)과 실시간 추론 루프 전체에 노이즈 제거 필터를 적용하여 소음이 심한 환경에서도 높은 정확도 유지.
"""

import os
import sys
import threading
import time

import numpy as np
import torch
import sounddevice as sd
import cv2
import soundfile as sf
import librosa
import requests
from transformers import Wav2Vec2FeatureExtractor

# 외부 라이브러리 예외 처리
try:
    import noisereduce as nr
    DENOISE_AVAILABLE = True
except ImportError:
    DENOISE_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from demo.util import load_model, get_model_output, visualise_args
import shared.utils as su

# 모델 학습 시와 동일한 오디오 정규화기 (zero-mean, unit-variance)
_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

def preprocess_audio(audio_np: np.ndarray) -> torch.Tensor:
    """마이크 raw numpy 배열을 모델 입력 형식 [1,1,L] 텐서로 변환합니다."""
    inputs = _feature_extractor(
        audio_np, sampling_rate=SR, return_tensors="pt", padding=False
    )
    return inputs.input_values.unsqueeze(0)


# ============================================================
# ▼▼▼ 설정 값 ▼▼▼
# ============================================================
FILL_RATIO       = 0.55   # 몇 % 채워지면 멈출지 (지연 보완을 위해 55%로 하향)
SR               = 16000  # 마이크 샘플링 레이트 (16kHz 고정)
INFERENCE_INTERVAL = 1.0  # AI 추론 주기 (초)
# ============================================================

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_cache")

# 공유 상태
shared = {
    "current_l_pred": None,
    "is_stopped": False,
    "running": True,
    "audio_buffer": np.array([], dtype=np.float32),
}
lock = threading.Lock()




# ─────────────────────────────────────────────
#  배경 잡음 수집 (Noise Profile)
# ─────────────────────────────────────────────
def capture_noise_profile():
    """시작 시 2초간 주변의 고정된 배경 잡음(Noise Floor)을 녹음하여 분석합니다."""
    print("\n" + "="*58)
    print("  [🔇 1단계: 주변 배경 소음(Noise Floor) 분석]")
    print("="*58)
    print("🔍 주변 잡음을 추출하여 필터를 생성합니다. (2초간 아무 소리도 내지 마세요!)")
    
    test_buf = []
    def cb(indata, frames, t, status):
        test_buf.append(indata[:, 0].copy().astype(np.float32))

    try:
        stream = sd.InputStream(samplerate=SR, channels=1, callback=cb)
        stream.start()
        time.sleep(2.0)
        stream.stop()
        stream.close()
    except Exception as e:
        print("\n[오류] 마이크를 초기화할 수 없습니다. 마이크 연결을 확인하세요.")
        print(f"상세 에러: {e}")
        return None

    if not test_buf:
        return None
        
    noise_data = np.concatenate(test_buf)
    rms = float(np.sqrt(np.mean(noise_data ** 2)))
    print(f"[OK] 배경 소음 수집 성공! (기본 노이즈 레벨 RMS: {rms:.5f})")
    print("="*58)
    return noise_data


# ─────────────────────────────────────────────
#  캐시 / 컵 선택 메뉴
# ─────────────────────────────────────────────
def list_caches():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return sorted([f for f in os.listdir(CACHE_DIR) if f.endswith(".npz")])


def select_or_create_cache(model, noise_profile):
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
                return calibrate_new_cup(model, noise_profile)
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


def validate_recording(audio_np, noise_rms):
    """녹음된 오디오가 학습에 적합한지 판단합니다."""
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
        return False, f"물소리(RMS={rms_total:.5f})가 주변 환경 노이즈(RMS={noise_rms:.5f})에 완전히 묻혔습니다!"

    n = len(audio_np)
    rms_start = float(np.sqrt(np.mean(audio_np[:n//5] ** 2)))
    rms_end   = float(np.sqrt(np.mean(audio_np[-n//5:] ** 2)))
    if rms_start < 1e-6 or rms_end < 1e-6:
        return False, "앞부분이나 뒷부분 소리가 거의 없습니다. 처음부터 끝까지 물을 천천히 따라 주세요."

    var_ratio = abs(rms_start - rms_end) / max(rms_start, rms_end)
    if var_ratio < MIN_VAR_RATIO:
        return False, (f"소리 변화가 너무 적습니다 (변화율={var_ratio:.1%}). 소음만 녹음된 것은 아닌지 확인해 주세요.")

    return True, f"✅ 녹음 품질 양호! (길이={duration:.1f}s, RMS={rms_total:.5f}, 변화율={var_ratio:.1%})"


# ─────────────────────────────────────────────
#  새 컵 학습 (마이크 녹음 → 노이즈 제거 → AI 분석 → 캐시 저장)
# ─────────────────────────────────────────────
def calibrate_new_cup(model, noise_profile):
    """마이크로 새 컵에 물 따르는 소리를 녹음하고 노이즈 필터를 입힌 후 AI로 분석해 캐시를 저장한다."""

    cup_name = input("\n  새 컵 이름을 입력하세요 (예: tall_glass, mug): ").strip().replace(" ", "_")
    if not cup_name:
        cup_name = "new_cup"

    print("\n" + "="*58)
    print("  [새 컵 학습 모드 (노이즈 필터 가동)]")
    print("="*58)

    # 1. 주변 소음(배경) 측정
    noise_rms = 0.005
    if noise_profile is not None:
        noise_rms = float(np.sqrt(np.mean(noise_profile ** 2)))
    else:
        print("[경고] 소음 필터가 활성화되지 않았습니다. 기본 품질 검사만 적용됩니다.")

    print("\n  준비되면 Enter를 누르고, 빈 컵에 물을 끝까지 따르세요.")
    print("  물 따르기가 완전히 끝나면 다시 Enter를 누르세요.")
    input("  ▶ 준비됐으면 Enter ▶ ")

    # 마이크 녹음 시작
    print("🎙️  [녹음 시작] 지금 컵에 물을 따르세요...")
    rec_buffer = []

    def rec_callback(indata, frames, t, status):
        rec_buffer.append(indata[:, 0].copy().astype(np.float32))

    try:
        stream = sd.InputStream(samplerate=SR, channels=1, callback=rec_callback)
        stream.start()
        input("  ⏹  물을 다 따랐으면 Enter ▶ ")
        stream.stop()
        stream.close()
    except Exception as e:
        print(f"\n❌ [오류] 마이크 녹음 실패: {e}")
        sys.exit(1)

    if not rec_buffer:
        print("[ERROR] 녹음된 오디오가 없습니다.")
        sys.exit(1)

    audio_np = np.concatenate(rec_buffer)
    duration = len(audio_np) / SR
    print(f"  총 {duration:.1f}초 녹음 완료.")

    # 2. 녹음 품질 검증 (노이즈 대비 물소리 크기 확인)
    ok, msg = validate_recording(audio_np, noise_rms)
    if not ok:
        print(f"\n⚠️  녹음 품질 문제: {msg}")
        retry = input("  다시 녹음하시겠습니까? [Y/n]: ").strip().lower()
        if retry not in ('n', 'no'):
            return calibrate_new_cup(model, noise_profile)
        else:
            print("학습을 취소합니다.")
            sys.exit(1)
    print(msg)

    # 3. 노이즈 제거 알고리즘 전처리 적용
    if DENOISE_AVAILABLE and noise_profile is not None:
        print("\n⚡ 녹음 파일에 스펙트럼 차감 잡음 필터를 적용 중입니다...")
        # Spectral Subtraction Denoise (배경 잡음 제거)
        audio_np = nr.reduce_noise(y=audio_np, sr=SR, y_noise=noise_profile, prop_decrease=0.90)
        print("✅ 잡음 제거 전처리 완료!")

    # 4. AI로 수위 분석
    print("\n🧠 AI가 깨끗하게 정제된 소리를 분석합니다... (잠시만 기다려 주세요)")
    from demo.util import load_audio_tensor

    # 임시 WAV 파일로 저장
    tmp_wav = os.path.join(CACHE_DIR, f"_tmp_{cup_name}.wav")
    os.makedirs(CACHE_DIR, exist_ok=True)
    sf.write(tmp_wav, audio_np, SR)

    audio_tensor = load_audio_tensor(tmp_wav)

    with torch.no_grad():
        z_audio, y_audio = get_model_output(audio_tensor, model)
        wavelengths_tensor = y_audio @ torch.linspace(
            0, visualise_args['w_max'], visualise_args['n_bins']
        ).to(y_audio.device)
        l_preds = su.physics.estimate_length_of_air_column(wavelengths_tensor).numpy()

    wavelengths_np = wavelengths_tensor.cpu().numpy()
    os.remove(tmp_wav)

    # 수위 범위 산출
    l_max = float(np.max(l_preds))
    l_min = float(np.mean(l_preds[-10:]))
    print(f"[정보] 컵 수위 범위: {l_max:.2f}cm (빈) ~ {l_min:.2f}cm (꽉 참)")

    # 5. Mel 스펙트로그램 윈도우 추출 및 저장
    MEL_WINDOW_S = 1.0
    MEL_HOP_S    = 1.0
    N_MELS       = 64
    win_samples  = int(MEL_WINDOW_S * SR)
    hop_samples  = int(MEL_HOP_S    * SR)

    mel_windows_list   = []
    lpred_per_window   = []
    rms_per_window     = []

    n_frames_total = len(l_preds)
    timestamps_eval = librosa.frames_to_time(
        np.arange(n_frames_total),
        sr=visualise_args['sr'],
        n_fft=visualise_args['n_fft'],
        hop_length=visualise_args['hop_length'],
    )

    for start in range(0, len(audio_np) - win_samples + 1, hop_samples):
        chunk = audio_np[start : start + win_samples]
        c_rms = float(np.sqrt(np.mean(chunk ** 2)))
        
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
        noise_rms        = noise_rms,
    )
    print(f"💾 학습 결과 및 노이즈 필터링 템플릿 저장 완료: {cache_path}\n")

    return l_max, cache_path


# ─────────────────────────────────────────────
#  마이크 콜백 (실시간 녹음)
# ─────────────────────────────────────────────
def mic_callback(indata, frames, time_info, status):
    audio_chunk = indata[:, 0].astype(np.float32)
    with lock:
        shared["audio_buffer"] = np.concatenate([shared["audio_buffer"], audio_chunk])


# ─────────────────────────────────────────────
#  실시간 Mel 스펙트로그램 매칭 스레드 (노이즈 감소 적용)
# ─────────────────────────────────────────────
def ai_worker(model, l_max, threshold, cache_path, noise_profile):
    """캘리브레이션 때 저장한 Mel 템플릿과 노이즈가 제거된 실시간 마이크 오디오를 비교하여 수위를 추정합니다."""
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

    print(f"[Mel Matcher] 템플릿 {len(mel_calib_norm)}개 로드 완료 (N_MELS={N_MELS})")
    print(f"  수위 감지 범위: {lpred_per_window.max():.1f}cm ~ {lpred_per_window.min():.1f}cm")
    print(f"  목표 수위 임계치: {threshold:.2f}cm 이하 도달 시 자동 정지")
    print("  준비 완료! 컵에 물을 부어 주세요!\n")

    # ESP32 서보모터 밸브 오픈 신호 전송
    try:
        print("📡 ESP32 SG90 서보모터 밸브 오픈 신호 전송 중... (http://192.168.0.250/open)")
        headers = {'Connection': 'close', 'User-Agent': 'Mozilla/5.0'}
        response = requests.get("http://192.168.0.250/open", headers=headers, timeout=3)
        print(f"✅ ESP32 밸브(서보모터) 오픈 성공! (응답 코드: {response.status_code})")
    except Exception as e:
        print(f"❌ 밸브 오픈 통신 지연/에러 발생 (에러 무시 후 분석 진행): {e}")

    consecutive_below = 0
    accepted_pred = l_max  # 빈 컵에서 시작
    MAX_CHANGE = 3.0       # 1초당 허용되는 최대 수위 변화 (cm)
    CONFIRM_COUNT_REQUIRED = 2

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

        if chunk_rms < 3e-4:
            print(f"[Mel] t={t_elapsed:.1f}s | 소리 대기 중... (RMS={chunk_rms:.5f})")
            time.sleep(INFERENCE_INTERVAL)
            continue

        # [핵심] 실시간 오디오 노이즈 감쇄 필터링
        if DENOISE_AVAILABLE and noise_profile is not None:
            # Spectral Subtraction (배경 잡음 제거)
            chunk = nr.reduce_noise(y=chunk, sr=SR, y_noise=noise_profile, prop_decrease=0.90)

        # 1. Mel 스펙트로그램 계산 및 코사인 유사도 비교
        mel_live = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=N_MELS, fmax=8000)
        mel_feat = librosa.power_to_db(mel_live, ref=np.max).mean(axis=1)

        norm          = np.linalg.norm(mel_feat) + 1e-8
        mel_feat_norm = mel_feat / norm
        sims          = mel_calib_norm @ mel_feat_norm

        best_idx = int(np.argmax(sims))
        raw_pred = float(lpred_per_window[best_idx])

        # 2. 물리적 타당성 검사 (1초 내 급격한 수위 변동 노이즈 2차 차단)
        delta = abs(raw_pred - accepted_pred)
        if delta > MAX_CHANGE:
            print(f"[Filter] t={t_elapsed:.1f}s | sim={sims[best_idx]:.3f} | 급격한 수위 노이즈 감지 (Δ={delta:.1f}cm > {MAX_CHANGE}) -> 이전 수위 유지 ({accepted_pred:.2f}cm)")
            consecutive_below = 0
            with lock:
                shared["current_l_pred"] = accepted_pred
            time.sleep(INFERENCE_INTERVAL)
            continue

        accepted_pred = raw_pred

        # 3. 연속 수위 임계치 도달 확인
        if accepted_pred <= threshold:
            consecutive_below += 1
        else:
            consecutive_below = 0

        below_status = f" [{consecutive_below}/{CONFIRM_COUNT_REQUIRED}]" if accepted_pred <= threshold else ""
        print(f"[Mel] t={t_elapsed:.1f}s | sim={sims[best_idx]:.3f} → 남은공간: {accepted_pred:.2f}cm (목표: {threshold:.2f}cm) | RMS={chunk_rms:.5f}{below_status}")

        trigger_stop = False
        with lock:
            shared["current_l_pred"] = accepted_pred
            if consecutive_below >= CONFIRM_COUNT_REQUIRED and t_elapsed > 5.0 and not shared["is_stopped"]:
                trigger_stop = True

        if trigger_stop:
            print(f"\n⚠️ 수위 임계치 도달! 정수기 자동 정지 신호를 쏩니다. ({accepted_pred:.2f}cm ≤ {threshold:.2f}cm)")
            with lock:
                shared["is_stopped"] = True
            try:
                print("📡 ESP32 SG90 서보모터 정지 신호 전송 중... (http://192.168.0.250/stop)")
                headers = {'Connection': 'close', 'User-Agent': 'Mozilla/5.0'}
                response = requests.get("http://192.168.0.250/stop", headers=headers, timeout=5)
                if response.status_code == 200:
                    print(f"✅ ESP32 밸브(서보모터) 물리적 잠금 성공!")
                else:
                    print(f"❌ ESP32 응답 코드: {response.status_code}")
            except Exception as e:
                print(f"❌ 통신 에러: {e} -> 수동으로 밸브를 잠가 주세요!")
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
            is_running = shared["running"]

        if not is_running:
            break

        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

        # Title and Stats
        cv2.putText(canvas, "Real-Time Mic Water Level Monitor", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(canvas, "Status: Denoise Filter Active [Bandpass + Spec Subtraction]", (30, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)
        cv2.putText(canvas, f"Recording: {t_elapsed:.1f} s", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

        if l_pred is not None:
            cv2.putText(canvas, f"Empty Space: {l_pred:.2f} cm", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # 수위 바 (l_max 기준 채워진 비율)
            bar_x, bar_y, bar_w, bar_h = 30, 170, win_w - 60, 60
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
            cv2.putText(canvas, "Listening... (waiting for audio buffer)", (30, 160),
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

        cv2.imshow("Mic Water Level Monitor [Denoised]", canvas)
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


# ─────────────────────────────────────────────
#  메인
# ─────────────────────────────────────────────
def main():
    print("=" * 58)
    print("   🎙️  실시간 마이크 수위 모니터 [노이즈 제거 버전]   ")
    print("=" * 58)

    # 노이즈 라이브러리 검사
    if not DENOISE_AVAILABLE:
        print("\n[알림] 'noisereduce' 또는 'scipy' 라이브러리가 존재하지 않습니다.")
        print("       노이즈 제거 엔진이 자동으로 꺼집니다.")
        print("       활성화하려면 다음을 실행하세요: pip install noisereduce scipy\n")
        time.sleep(1)

    # 1. AI 모델 로딩
    print("[AI 모델 로딩 중... 잠시만 기다려 주세요]")
    model = load_model()
    print("[AI 모델 로딩 완료!]")

    # 2. 배경 소음 측정 (노이즈 감소용 프로필 수집)
    noise_profile = None
    if DENOISE_AVAILABLE:
        noise_profile = capture_noise_profile()

    # 3. 컵 선택 (기존 캐시 재사용 or 새 컵 마이크 학습)
    l_max, cache_path = select_or_create_cache(model, noise_profile)
    threshold = l_max * (1.0 - FILL_RATIO)

    # 4. 실시간 마이크 스트림 시작
    try:
        stream = sd.InputStream(samplerate=SR, channels=1, callback=mic_callback)
        stream.start()
        print(f"🎙️  실시간 리스닝 시작! (샘플레이트: {SR}Hz)")
    except Exception as e:
        print("\n❌ [치명적 오류] 마이크에 접근할 수 없습니다.")
        print(f"   상세 에러: {e}")
        return

    # 5. 특징 매칭 백그라운드 스레드 가동
    worker = threading.Thread(target=ai_worker, args=(model, l_max, threshold, cache_path, noise_profile), daemon=True)
    worker.start()

    # 6. 디스플레이 (메인 스레드)
    display_loop(l_max, threshold)

    # 7. 종료 프로세스
    stream.stop()
    stream.close()
    print("\n🛑 마이크 스트리밍 종료.")


if __name__ == "__main__":
    main()
