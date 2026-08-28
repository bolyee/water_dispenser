"""
realtime_noesp_camera.py  —  노트북 마이크 + 카메라 자동 컵 인식 + ESP32 연동 시각장애인용 정수기 수위 모니터
--------------------------------------------------------------------------------------------------
사용법:
    pip install sounddevice opencv-python
    python realtime_noesp_camera.py

특징:
    1. 시각장애인 편의성 향상: 화면에서 컵을 수동 선택할 필요 없이, 카메라에 컵을 보여주면 색상(HSV 히스토그램)을 비교해 자동 인식
    2. 모든 주요 안내 및 차단 상태를 음성(Mac say TTS)으로 안내
    3. 수위 임계치 도달 시 밸브 자동 차단 및 안내 음성("물이 다 찼습니다. 멈추세요.") 출력
"""

import os
import sys
import threading
import time
import subprocess

import numpy as np
import torch
import sounddevice as sd
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
    return inputs.input_values.unsqueeze(0)


# ============================================================
# ▼▼▼ 설정 값 ▼▼▼
# ============================================================
FILL_RATIO       = 0.55   # 몇 % 채워지면 멈출지 (지연 보완을 위해 55%로 하향)
SR               = 16000  # 마이크 샘플링 레이트 (16kHz 고정)
INFERENCE_INTERVAL = 1.0  # AI 추론 주기 (초)
ESP32_IP         = "20.30.88.125"  # ESP32 IP 주소 (서보 모터 제어용)
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
#  컴퓨터 비전: 전경 분리 및 HSV 색상 히스토그램 계산
# ─────────────────────────────────────────────
def extract_cup_foreground_grabcut(image, roi_rect):
    """
    GrabCut 알고리즘을 사용해 이미지에서 배경을 완전히 분리하고 컵(전경) 영역만 마스크로 추출합니다.
    (새 컵을 등록할 때 딱 한 번 실행되므로 연산이 정확하고 컵 지문을 깔끔하게 얻습니다.)
    """
    x, y, w, h = roi_rect
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (x, y, w, h)
    
    try:
        # 3회 반복으로 GrabCut 수행하여 정밀한 컵 윤곽 마스크 추출
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        
        # 가이드 박스 외부 영역은 안전하게 0으로 자름
        out_mask = np.zeros_like(fg_mask)
        out_mask[y:y+h, x:x+w] = fg_mask[y:y+h, x:x+w]
        return out_mask
    except Exception as e:
        print(f"⚠️ GrabCut 실패, 기본 ROI 영역 사용: {e}")
        out_mask = np.zeros(image.shape[:2], np.uint8)
        out_mask[y:y+h, x:x+w] = 255
        return out_mask


def get_lightweight_foreground_mask(image, roi_rect):
    """
    실시간 대조용 가볍고 빠른 배경 제거 필터입니다.
    어두운 그림자(V<40)와 너무 밝거나 하얀 배경(S<20, V>180) 같은 무채색 영역을 무시하여 배경 혼입을 막습니다.
    """
    x, y, w, h = roi_rect
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_roi = hsv[y:y+h, x:x+w]
    
    # H: 0~180, S(채도): 20~255, V(밝기): 40~255 범위 필터링
    lower_bg = np.array([0, 20, 40])
    upper_bg = np.array([180, 255, 245])
    
    roi_mask = cv2.inRange(hsv_roi, lower_bg, upper_bg)
    
    # 잔노이즈 제거 (Opening 연산)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)
    
    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    full_mask[y:y+h, x:x+w] = roi_mask
    return full_mask


def calculate_hsv_histogram(image, mask=None, roi_rect=None):
    """
    이미지(BGR)에서 마스크 영역 혹은 지정된 ROI 영역에 대해 HSV 색상 히스토그램을 추출합니다.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if mask is not None:
        # 마스크된 컵 영역만으로 정밀하게 히스토그램을 구합니다.
        hist = cv2.calcHist([hsv], [0, 1], mask, [50, 60], [0, 180, 0, 256])
    elif roi_rect is not None:
        x, y, w, h = roi_rect
        hsv_roi = hsv[y:y+h, x:x+w]
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [50, 60], [0, 180, 0, 256])
    else:
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


# ─────────────────────────────────────────────
#  컵 이미지 캡처 (새 컵 등록 시 사용)
# ─────────────────────────────────────────────
def capture_cup_image_and_hist():
    """카메라를 열어 컵을 촬영하고 컵의 HSV 히스토그램 특징 벡터를 반환합니다."""
    cam_idx = find_iphone_camera_index()
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다. 카메라 연결 상태를 확인해 주세요.")
        try:
            subprocess.Popen(["say", "카메라를 켤 수 없습니다."])
        except:
            pass
        return None

    try:
        subprocess.Popen(["say", "새로운 컵의 모습을 등록합니다. 컵을 화면 중앙 주황색 박스 안에 맞추고 스페이스바를 눌러 촬영하세요."])
    except:
        pass

    print("\n[📸 카메라 촬영] 컵을 주황색 가이드라인에 맞춘 후 'Space' 키를 눌러 촬영하세요. ('q'는 취소)")
    hist_result = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        # 화면 중앙의 40% x 50% 영역 가이드 박스 정의
        box_w, box_h = int(w * 0.4), int(h * 0.5)
        box_x, box_y = (w - box_w) // 2, (h - box_h) // 2

        display_frame = frame.copy()
        # 가이드 박스 그리기
        cv2.rectangle(display_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 165, 255), 2)
        cv2.putText(display_frame, "Align cup inside box & Press SPACE", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(display_frame, "Press 'q' to quit registration", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        cv2.imshow("Register Cup Image", display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):  # 스페이스 바 누르면 캡처 완료
            print("⏳ 컵 전경(GrabCut) 분리 분석 중... (잠시만 기다려 주세요)")
            try:
                subprocess.Popen(["say", "컵 이미지를 분석 중입니다."])
            except:
                pass
            fg_mask = extract_cup_foreground_grabcut(frame, (box_x, box_y, box_w, box_h))
            hist_result = calculate_hsv_histogram(frame, mask=fg_mask)
            try:
                subprocess.Popen(["say", "컵 촬영과 분석이 완료되었습니다."])
            except:
                pass
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return hist_result


# ─────────────────────────────────────────────
#  카메라 기반 컵 자동 감지 및 로드
# ─────────────────────────────────────────────
def list_caches():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return sorted([f for f in os.listdir(CACHE_DIR) if f.endswith(".npz")])


def auto_detect_cup_by_camera(allow_calibration=True):
    """
    저장된 캘리브레이션 컵 정보들을 불러와, 카메라 영상과 실시간으로 대조 및 자동 분류합니다.
    """
    caches = list_caches()
    valid_templates = []

    # 기존 학습된 데이터 중 카메라 컵 히스토그램이 포함된 것 필터링
    for name in caches:
        path = os.path.join(CACHE_DIR, name)
        info = np.load(path)
        if 'cup_hist' in info:
            valid_templates.append({
                'name': name.replace('_calibration.npz', '').replace('_', ' '),
                'hist': info['cup_hist'],
                'l_max': float(info['l_max']),
                'path': path
            })

    if not valid_templates:
        print("\n⚠️ 카메라 자동 인식용으로 등록된 컵 데이터가 존재하지 않습니다.")
        try:
            subprocess.Popen(["say", "등록된 컵이 없습니다. 스페이스 바를 눌러 새로운 컵을 등록해 주세요."])
        except:
            pass
        
        # fallback 처리: 스페이스 입력 받으면 새 컵 학습으로
        cam_idx = find_iphone_camera_index()
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.putText(frame, "No cups registered. Press SPACE to start calibration.", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("No Cup Registered", frame)
                key = cv2.waitKey(50) & 0xFF
                if key == ord(' '):
                    cap.release()
                    cv2.destroyAllWindows()
                    return None, None
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
        return None, None

    # 카메라 오픈
    cam_idx = find_iphone_camera_index()
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return None, None

    try:
        subprocess.Popen(["say", "카메라에 사용할 컵을 비춰 주세요."])
    except:
        pass

    print("\n🔍 [컵 자동 감지] 카메라 창의 박스 영역에 컵을 비춰 주세요.")
    if allow_calibration:
        print("   -> 새 컵 학습을 원하시면 창을 활성화한 상태에서 [Space] 키를 누르세요. ('q'는 종료)")
    else:
        print("   (카메라 창에서 'q'를 누르면 종료됩니다.)")

    consecutive_match = 0
    last_best_name = None
    match_threshold = 0.70   # 색상 유사도 점수 기준값
    required_frames = 15     # 연속 프레임 일치도 수치 (약 0.5초 연속 동일 판단)
    
    detected_cup = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        box_w, box_h = int(w * 0.4), int(h * 0.5)
        box_x, box_y = (w - box_w) // 2, (h - box_h) // 2

        display_frame = frame.copy()
        cv2.rectangle(display_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 0, 0), 2)
        cv2.putText(display_frame, "Place your cup here", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        if allow_calibration:
            cv2.putText(display_frame, "Press SPACE to add a new cup", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 실시간 초경량 배경 제거 마스크 생성 후 히스토그램 추출
        light_mask = get_lightweight_foreground_mask(frame, (box_x, box_y, box_w, box_h))
        curr_hist = calculate_hsv_histogram(frame, mask=light_mask)

        # 모든 등록된 컵과의 색상 유사도 점수 계산
        scores = []
        for temp in valid_templates:
            score = cv2.compareHist(temp['hist'], curr_hist, cv2.HISTCMP_CORREL)
            scores.append((temp, score))

        # 점수 기준 내림차순 정렬
        scores.sort(key=lambda x: x[1], reverse=True)

        # 화면에 모든 등록된 컵의 매칭률 표시
        y_pos = 110
        for i, (temp, score) in enumerate(scores):
            match_name = temp['name']
            
            if i == 0 and score >= match_threshold:
                color = (0, 255, 0)      # 통과 (초록색)
                label = f"-> {match_name}: {score:.2f}"
            elif i == 0:
                color = (0, 165, 255)    # 1위지만 미달 (주황색)
                label = f"-> {match_name}: {score:.2f}"
            else:
                color = (200, 200, 200)  # 다른 후보들 (회색)
                label = f"   {match_name}: {score:.2f}"
                
            cv2.putText(display_frame, label, (30, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 25

        if scores:
            best_match, best_score = scores[0]
            if best_score >= match_threshold:
                match_name = best_match['name']
                if match_name == last_best_name:
                    consecutive_match += 1
                else:
                    consecutive_match = 0
                    last_best_name = match_name
            else:
                consecutive_match = 0
                last_best_name = None
        else:
            consecutive_match = 0
            last_best_name = None

        # 컵 인식 확정
        if consecutive_match >= required_frames:
            detected_cup = best_match
            cv2.rectangle(display_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 4)
            cv2.putText(display_frame, "CONFIRMED!", (box_x + 10, box_y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow("Auto Cup Detection", display_frame)
            cv2.waitKey(1000)  # 확인을 위한 1초 대기
            break

        cv2.imshow("Auto Cup Detection", display_frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' ') and allow_calibration:  # 스페이스 바 누르면 자동 감지를 건너뛰고 새 컵 학습으로 전환
            detected_cup = None
            break
        elif key == ord('q'):
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()

    if detected_cup is not None:
        try:
            # 컵 인식 완료 음성 피드백
            subprocess.Popen(["say", f"{detected_cup['name']} 컵이 확인되었습니다. 물을 따르기 시작하세요."])
        except:
            pass
        return detected_cup['l_max'], detected_cup['path']
    else:
        return None, None


# ─────────────────────────────────────────────
#  글로벌로 선택된 카메라 및 마이크 인덱스
# ─────────────────────────────────────────────
CHOSEN_CAMERA_IDX = 0
CHOSEN_MIC_IDX    = None

def find_iphone_camera_index():
    return CHOSEN_CAMERA_IDX

def find_iphone_mic_index():
    global CHOSEN_MIC_IDX
    if CHOSEN_MIC_IDX is None:
        CHOSEN_MIC_IDX = select_mic_index()
    return CHOSEN_MIC_IDX

def select_camera_index():
    print("\n📷 [비디오 장치] 사용 가능한 카메라 검색 중...")
    available_cams = []
    # 0부터 4까지 카메라를 열어 확인
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cams.append(i)
            cap.release()
    
    if not available_cams:
        print("⚠️ 감지된 카메라가 없습니다. 기본값 0번을 사용합니다.")
        return 0
        
    print("\n사용할 카메라 번호를 선택해 주세요:")
    for idx in available_cams:
        print(f"  [{idx}]: 카메라 {idx}")
        
    try:
        sel = input(f"  선택 (기본값 {available_cams[0]}): ").strip()
        if not sel:
            return available_cams[0]
        return int(sel)
    except:
        return available_cams[0]

def select_mic_index():
    print("\n🎙️ [오디오 장치] 연결된 마이크 검색 중...")
    try:
        devices = sd.query_devices()
        input_devices = []
        default_device_idx = sd.default.device[0] # default input device index
        
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_devices.append((idx, dev['name']))
        
        if not input_devices:
            print("⚠️ 감지된 마이크가 없습니다.")
            return None
            
        print("\n사용할 마이크 번호를 선택해 주세요:")
        for idx, name in input_devices:
            is_default = " (기본값)" if idx == default_device_idx else ""
            is_iphone = " [📱 iPhone 마이크]" if "iphone" in name.lower() else ""
            print(f"  [{idx}]: {name}{is_default}{is_iphone}")
            
        try:
            sel = input(f"  선택 (기본값 {default_device_idx}): ").strip()
            if not sel:
                return default_device_idx
            return int(sel)
        except:
            return default_device_idx
    except Exception as e:
        print(f"⚠️ 마이크 검색 오류: {e}")
        return None

global_stream = None

def ensure_mic_stream_started():
    global global_stream
    if global_stream is not None:
        return
        
    mic_idx = find_iphone_mic_index()
    if mic_idx is not None:
        print(f"\n📱 [오디오 센서] 선택된 마이크 (기기 번호: {mic_idx})를 연결합니다.")
    else:
        print("\n💻 [오디오 센서] 기본 마이크를 연결합니다.")

    try:
        global_stream = sd.InputStream(samplerate=SR, channels=1, callback=mic_callback, device=mic_idx)
        global_stream.start()
        print(f"🎙️  글로벌 오디오 스트림 활성화 완료. (샘플레이트: {SR}Hz)")
    except Exception as e:
        if mic_idx is not None:
            print(f"⚠️ [오류] 선택된 마이크 구동 실패 ({e}). 노트북 내장 마이크로 대체합니다.")
            try:
                global_stream = sd.InputStream(samplerate=SR, channels=1, callback=mic_callback, device=None)
                global_stream.start()
                print(f"🎙️  글로벌 오디오 스트림 활성화 완료 (노트북 마이크). (샘플레이트: {SR}Hz)")
            except Exception as e2:
                print(f"\n❌ [치명적 오류] 마이크를 초기화할 수 없습니다: {e2}")
                sys.exit(1)
        else:
            print(f"\n❌ [치명적 오류] 마이크를 초기화할 수 없습니다: {e}")
            sys.exit(1)


# ─────────────────────────────────────────────
#  마이크 사전 검증
# ─────────────────────────────────────────────
def check_mic():
    """상시 활성화된 오디오 스트림 버퍼를 분석하여 2초간 주변 노이즈 상태(Noise Floor)를 확인합니다."""
    print("\n🔍 오디오 수신 및 주변 노이즈 상태를 확인합니다... (2초간 아무 소리도 내지 마세요!)")
    
    # 2초 측정을 위해 기존 버퍼를 비워 줍니다.
    with lock:
        shared["audio_buffer"] = np.array([], dtype=np.float32)
        
    time.sleep(2.0)
    
    with lock:
        audio = shared["audio_buffer"].copy()
        
    if len(audio) == 0:
        print("❌ 마이크에서 들어오는 신호가 없습니다. (연동 여부 및 권한을 확인하세요)")
        return False, 0.0
        
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"   측정된 배경 노이즈 레벨(RMS): {rms:.5f}")
    if rms < 1e-5:
        print("❌ 마이크가 잡히지만 무음 상태입니다. (음소거 해제를 확인하세요)")
        return False, 0.0
        
    print("✅ 마이크 정상 작동 및 노이즈 측정 완료!")
    return True, rms


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

    if rms_total < noise_rms * 1.5:
        return False, f"물소리(RMS={rms_total:.5f})가 주변 환경 노이즈(RMS={noise_rms:.5f})에 완전히 묻혔습니다!"

    n = len(audio_np)
    rms_start = float(np.sqrt(np.mean(audio_np[:n//5] ** 2)))
    rms_end   = float(np.sqrt(np.mean(audio_np[-n//5:] ** 2)))
    if rms_start < 1e-6 or rms_end < 1e-6:
        return False, "앞부분이나 뒷부분 소리가 거의 없습니다. 처음부터 끝까지 물을 천천히 따라 주세요."

    var_ratio = abs(rms_start - rms_end) / max(rms_start, rms_end)
    if var_ratio < MIN_VAR_RATIO:
        return False, f"소리 변화가 너무 적습니다 (변화율={var_ratio:.1%}). 물소리 주파수 변화가 식별되지 않습니다."

    return True, f"✅ 녹음 품질 양호! (길이={duration:.1f}s, RMS={rms_total:.5f})"


# ─────────────────────────────────────────────
#  새 컵 학습 (카메라 캡처 + 마이크 녹음 → AI 분석 → 캐시 저장)
# ─────────────────────────────────────────────
def calibrate_new_cup(model):
    """카메라로 컵을 촬영하고 물 따르는 소리를 녹음하여 캐시를 생성합니다."""
    
    # ① 카메라 촬영을 통해 컵의 고유 색상 히스토그램 등록
    cup_hist = capture_cup_image_and_hist()
    if cup_hist is None:
        print("❌ 컵 이미지 캡처가 취소되었습니다. 학습을 중단합니다.")
        sys.exit(0)

    try:
        subprocess.Popen(["say", "등록할 컵의 이름을 콘솔창에 입력해 주세요."])
    except:
        pass

    cup_name = input("\n  새 컵 이름을 입력하세요 (예: tall_glass, mug): ").strip().replace(" ", "_")
    if not cup_name:
        cup_name = "new_cup"

    print("\n" + "="*58)
    print("  [새 컵 학습 모드 - 오디오 녹음 단계]")
    print("="*58)

    # ② 마이크 연결 사전 검사 및 주변 소음(노이즈) 측정
    ensure_mic_stream_started()
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
    
    try:
        subprocess.Popen(["say", "준비되면 엔터를 누르고 빈 컵에 물을 끝까지 따라 주세요. 다 차면 다시 엔터를 누르세요."])
    except:
        pass

    input("  ▶ 준비됐으면 Enter ▶ ")

    # 마이크 녹음 시작 (글로벌 단일 스트림 버퍼 슬라이싱 활용)
    with lock:
        start_idx = len(shared["audio_buffer"])
        
    print("🎙️  [녹음 시작] 지금 컵에 물을 따르세요...")
    
    input("  ⏹  물을 다 따랐으면 Enter ▶ ")
    
    with lock:
        end_idx = len(shared["audio_buffer"])
        audio_np = shared["audio_buffer"][start_idx:end_idx].copy()

    if len(audio_np) == 0:
        print("[ERROR] 녹음된 오디오가 없습니다.")
        sys.exit(1)

    duration = len(audio_np) / SR
    print(f"  총 {duration:.1f}초 녹음 완료.")

    # ③ 녹음 품질 검증
    ok, msg = validate_recording(audio_np, noise_rms)
    if not ok:
        print(f"\n⚠️  녹음 품질 문제: {msg}")
        retry = input("  다시 녹음하시겠습니까? [Y/n]: ").strip().lower()
        if retry not in ('n', 'no'):
            return calibrate_new_cup(model)
        else:
            print("학습을 취소합니다.")
            sys.exit(1)
    print(msg)

    # ④ AI로 수위 분석
    print("\n🧠 AI가 녹음된 소리를 분석합니다... (잠시만 기다려 주세요)")
    from demo.util import load_audio_tensor

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

    l_max = float(np.max(l_preds))
    l_min = float(np.mean(l_preds[-10:]))
    print(f"[정보] 컵 범위: {l_max:.2f}cm (빈) ~ {l_min:.2f}cm (꽉 참)")

    # ⑤ Mel 스펙트로그램 윈도우 추출 및 저장
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
        cup_hist         = cup_hist,          # ★ 컵 이미지 분석용 히스토그램 추가
    )
    print(f"💾 학습 결과 저장 완료 (카메라 데이터 포함): {cache_path}\n")
    try:
        subprocess.Popen(["say", "새 컵이 성공적으로 등록되었습니다."])
    except:
        pass

    return l_max, cache_path


# ─────────────────────────────────────────────
#  마이크 콜백 (실시간 녹음)
# ─────────────────────────────────────────────
def mic_callback(indata, frames, time_info, status):
    audio_chunk = indata[:, 0].astype(np.float32)
    with lock:
        shared["audio_buffer"] = np.concatenate([shared["audio_buffer"], audio_chunk])


# ─────────────────────────────────────────────
#  실시간 Mel 스펙트로그램 매칭 스레드
# ─────────────────────────────────────────────
def ai_worker(model, l_max, threshold, cache_path):
    data = np.load(cache_path)

    if 'mel_windows_norm' not in data:
        print("[경고] 이 캐시는 Mel 윈도우가 없습니다. 컵을 다시 학습해 주세요.")
        with lock:
            shared["running"] = False
        return

    mel_calib_norm   = data['mel_windows_norm']
    lpred_per_window = data['lpred_per_window']
    N_MELS   = mel_calib_norm.shape[1]
    WINDOW_S = 1.0

    # ESP32 서보모터 밸브 오픈 신호 전송
    try:
        print(f"📡 ESP32 SG90 서보모터 밸브 오픈 신호 전송 중... (http://{ESP32_IP}/open)")
        headers = {'Connection': 'close', 'User-Agent': 'Mozilla/5.0'}
        response = requests.get(f"http://{ESP32_IP}/open", headers=headers, timeout=3)
        print(f"✅ ESP32 밸브 오픈 성공!")
    except Exception as e:
        print(f"❌ 밸브 오픈 실패 (분석 진행): {e}")

    consecutive_below = 0
    accepted_pred = l_max
    MAX_CHANGE = 3.0
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

        chunk     = buf[-int(WINDOW_S * SR):]
        chunk_rms = float(np.sqrt(np.mean(chunk ** 2)))

        if chunk_rms < 3e-4:
            time.sleep(INFERENCE_INTERVAL)
            continue

        if water_start_time is None:
            water_start_time = t_elapsed
            print(f"💧 물 흐러내림 감지 시작! (기준 시간: {water_start_time:.1f}초)")

        mel_live = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=N_MELS, fmax=8000)
        mel_feat = librosa.power_to_db(mel_live, ref=np.max).mean(axis=1)

        norm          = np.linalg.norm(mel_feat) + 1e-8
        mel_feat_norm = mel_feat / norm
        sims          = mel_calib_norm @ mel_feat_norm

        best_idx = int(np.argmax(sims))
        raw_pred = float(lpred_per_window[best_idx])

        delta = abs(raw_pred - accepted_pred)
        if delta > MAX_CHANGE:
            consecutive_below = 0
            with lock:
                shared["current_l_pred"] = accepted_pred
            time.sleep(INFERENCE_INTERVAL)
            continue

        accepted_pred = raw_pred

        if accepted_pred <= threshold:
            consecutive_below += 1
        else:
            consecutive_below = 0

        below_status = f" [{consecutive_below}/{CONFIRM_COUNT_REQUIRED}]" if accepted_pred <= threshold else ""
        print(f"[Mel] Space: {accepted_pred:.2f}cm (thr: {threshold:.2f}cm) | RMS={chunk_rms:.5f}{below_status}")

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

            # 🔊 음성 경고 (Mac 'say' 명령어 사용 - 비동기 실행)
            try:
                subprocess.Popen(["say", "물이 다 찼습니다. 멈추세요."])
            except Exception as e:
                print(f"🔊 음성 경고 출력 실패: {e}")

            # 밸브 닫기 API 호출
            try:
                print(f"📡 ESP32 SG90 서보모터 정지 신호 전송 중... (http://{ESP32_IP}/stop)")
                headers = {'Connection': 'close', 'User-Agent': 'Mozilla/5.0'}
                response = requests.get(f"http://{ESP32_IP}/stop", headers=headers, timeout=5)
                if response.status_code == 200:
                    print(f"✅ ESP32 밸브 물리적 잠금 성공!")
                else:
                    print(f"❌ ESP32 응답 코드: {response.status_code}")
            except Exception as e:
                print(f"❌ 통신 에러: {e} -> 즉시 수동으로 잠가 주세요!")
            print("🚨 [AUTO STOP] 시스템 정지 완료.")

        time.sleep(INFERENCE_INTERVAL)


# ─────────────────────────────────────────────
#  OpenCV 디스플레이 루프
# ─────────────────────────────────────────────
def display_loop(l_max, threshold):
    win_w, win_h = 600, 400

    while True:
        with lock:
            l_pred = shared["current_l_pred"]
            is_stopped = shared["is_stopped"]
            t_elapsed = len(shared["audio_buffer"]) / SR

        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

        cv2.putText(canvas, "Water Level Monitor (Camera Activated)", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(canvas, f"Recording: {t_elapsed:.1f} s", (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        if l_pred is not None:
            cv2.putText(canvas, f"Empty Space: {l_pred:.2f} cm", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

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
            cv2.putText(canvas, "Listening... (waiting for water sound)", (30, 150),
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

        cv2.imshow("Water Level Monitor", canvas)
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
    print("   🎙️📸 시각장애인 정수기 자동 컵 인식 시스템 (노스 ESP 마이크)   ")
    print("=" * 58)

    # 1. 카메라 선택 진행
    global CHOSEN_CAMERA_IDX
    CHOSEN_CAMERA_IDX = select_camera_index()

    # 2. 작업 모드 선택
    print("\n진행할 작업을 선택해 주세요:")
    print("  [1]: 기존 등록된 컵 인식 (카메라 매칭)")
    print("  [2]: 새 컵 등록 및 학습 (새로 촬영 + 물 소리 녹음)")
    
    work_mode = "1"
    try:
        work_mode = input("  선택 (기본값 1): ").strip()
        if work_mode not in ("1", "2"):
            work_mode = "1"
    except KeyboardInterrupt:
        return

    print("\n[AI 모델 로딩 중...]")
    model = load_model()
    print("[AI 모델 로드 성공]")

    l_max = None
    cache_path = None

    if work_mode == "1":
        # 기존 등록된 컵 인식 진행 (스페이스바 등록 비활성화)
        l_max, cache_path = auto_detect_cup_by_camera(allow_calibration=False)
        
        # 만약 인식 실패했거나 스페이스바로 넘어왔다면 새 컵 등록 실행
        if l_max is None or cache_path is None:
            print("\n💡 등록된 컵 인식에 실패했거나 건너뛰었습니다. 새 컵 등록을 진행합니다.")
            l_max, cache_path = calibrate_new_cup(model)
    else:
        # 새 컵 등록 바로 실행
        l_max, cache_path = calibrate_new_cup(model)

    # 3. 컵 선택 및 캘리브레이션 완료 후, 실시간 모니터링 시작 직전에 마이크 스트림 활성화
    ensure_mic_stream_started()

    threshold = l_max * (1.0 - FILL_RATIO)

    # 4. 실시간 분석 시작 직전에 누적 버퍼를 비워 줍니다.
    with lock:
        shared["audio_buffer"] = np.array([], dtype=np.float32)

    # 실시간 감지 분석 백그라운드 구동
    worker = threading.Thread(target=ai_worker, args=(model, l_max, threshold, cache_path), daemon=True)
    worker.start()

    # GUI 디스플레이 구동 (메인 스레드)
    display_loop(l_max, threshold)

    # 5. 종료 클린업
    global global_stream
    if global_stream is not None:
        global_stream.stop()
        global_stream.close()
    print("\n🛑 정수기 모니터링 세션이 종료되었습니다.")


if __name__ == "__main__":
    main()
