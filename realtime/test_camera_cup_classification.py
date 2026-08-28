#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_camera_cup_classification.py
------------------------------------------------------------------
카메라 기반 컵 자동 인식 및 분류 성능 테스트 스크립트.
(오디오 입력 및 AI 모델 연동 배제, ResNet-18 임베딩 기반 머신러닝 매칭 프로그램)

사용법:
    python test_camera_cup_classification.py

단축키:
    [q]       : 종료
    [Space]   : 감지 일시정지 / 재개
    [c] / [n] : 새로운 컵 등록 (카메라 촬영)
    [+] / [=] : 머신러닝 매칭 임계값 증가 (0.02 단위)
    [-]       : 머신러닝 매칭 임계값 감소 (0.02 단위)
"""

import os
import sys
import time
import subprocess
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# 프로젝트 루트 경로 설정
# 이 파일은 realtime/ 안에 있으므로 저장소 루트는 한 단계 위.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT_DIR, "calibration_cache")

# 기본 설정값
DEFAULT_ML_THRESHOLD = 0.80     # ResNet-18 코사인 유사도 기본 임계값
DEFAULT_HIST_THRESHOLD = 0.70   # HSV 히스토그램 기본 임계값 (하위 호환용)
REQUIRED_FRAMES = 15            # 연속 프레임 일치 횟수 (약 0.5초 연속 유지)

# PyTorch ImageNet 표준 정규화 전처리 설정
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_resnet_embedding(model, crop_bgr):
    """
    크롭된 컵 이미지로부터 ResNet-18 특징 벡터(Embedding)를 추출하고 L2 정규화하여 반환합니다.
    """
    # OpenCV BGR -> PyTorch RGB 변환
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img_t = preprocess(crop_rgb).unsqueeze(0)  # [1, 3, 224, 224] Tensor
    
    with torch.no_grad():
        embedding = model(img_t).squeeze(0).cpu().numpy()  # 512차원 벡터 추출
        
    # L2 정규화 적용 (코사인 유사도 계산이 용이하도록 처리)
    norm = np.linalg.norm(embedding)
    if norm > 1e-8:
        embedding = embedding / norm
    return embedding


def get_lightweight_foreground_mask(image, roi_rect):
    """
    [하위 호환용] 실시간 대조용 가볍고 빠른 배경 제거 필터 (HSV 히스토그램 매칭용).
    """
    x, y, w, h = roi_rect
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_roi = hsv[y:y+h, x:x+w]
    
    lower_bg = np.array([0, 20, 40])
    upper_bg = np.array([180, 255, 245])
    
    roi_mask = cv2.inRange(hsv_roi, lower_bg, upper_bg)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)
    
    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    full_mask[y:y+h, x:x+w] = roi_mask
    return full_mask


def calculate_hsv_histogram(image, mask=None, roi_rect=None):
    """
    [하위 호환용] 이미지에서 HSV 색상 히스토그램을 추출합니다.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if mask is not None:
        hist = cv2.calcHist([hsv], [0, 1], mask, [50, 60], [0, 180, 0, 256])
    elif roi_rect is not None:
        x, y, w, h = roi_rect
        hsv_roi = hsv[y:y+h, x:x+w]
        hist = cv2.calcHist([hsv_roi], [0, 1], None, [50, 60], [0, 180, 0, 256])
    else:
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def extract_cup_foreground_grabcut(image, roi_rect):
    """
    GrabCut 알고리즘을 사용해 배경을 지우고 컵 영역만 분리해 냅니다 (하위 호환 히스토그램용).
    """
    x, y, w, h = roi_rect
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (x, y, w, h)
    
    try:
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
        out_mask = np.zeros_like(fg_mask)
        out_mask[y:y+h, x:x+w] = fg_mask[y:y+h, x:x+w]
        return out_mask
    except Exception as e:
        print(f"⚠️ GrabCut 실패, 기본 ROI 영역 사용: {e}")
        out_mask = np.zeros(image.shape[:2], np.uint8)
        out_mask[y:y+h, x:x+w] = 255
        return out_mask


def list_caches():
    os.makedirs(CACHE_DIR, exist_ok=True)
    return sorted([f for f in os.listdir(CACHE_DIR) if f.endswith(".npz")])


def load_cup_templates():
    """저장된 캘리브레이션 컵 정보들을 불러옵니다."""
    caches = list_caches()
    templates = []
    
    for name in caches:
        path = os.path.join(CACHE_DIR, name)
        try:
            info = np.load(path)
            template_data = {
                'name': name.replace('_calibration.npz', '').replace('_', ' '),
                'l_max': float(info['l_max']) if 'l_max' in info else 15.0,
                'path': path
            }
            # 새로운 ResNet 머신러닝 임베딩 확인
            if 'cup_embedding' in info:
                template_data['embedding'] = info['cup_embedding']
            
            # 기존 하위 호환 히스토그램 확인
            if 'cup_hist' in info:
                template_data['hist'] = info['cup_hist']
                
            if 'embedding' in template_data or 'hist' in template_data:
                templates.append(template_data)
        except Exception as e:
            print(f"⚠️ {name} 파일을 로드하는 데 실패했습니다: {e}")
            
    return templates


def select_camera():
    """사용 가능한 카메라 리스트를 조회하고 선택을 입력받습니다."""
    print("\n📷 사용 가능한 카메라 검색 중...")
    available_cams = []
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


def capture_cup_image_and_features(cam_idx, model):
    """카메라를 열어 새 컵을 촬영하고 ResNet 임베딩 및 HSV 히스토그램을 추출합니다."""
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return None, None

    # 워밍업
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            break
        time.sleep(0.1)

    print("\n[📸 카메라 촬영] 컵을 주황색 가이드라인에 맞춘 후 'Space' 키를 눌러 촬영하세요. ('q'는 취소)")
    embedding_result = None
    hist_result = None
    consecutive_failures = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 30:
                print("⚠️ 촬영 중 카메라 연결이 손실되었습니다.")
                break
            time.sleep(0.03)
            continue
        consecutive_failures = 0

        h, w, _ = frame.shape
        box_w, box_h = int(w * 0.4), int(h * 0.5)
        box_x, box_y = (w - box_w) // 2, (h - box_h) // 2

        display_frame = frame.copy()
        cv2.rectangle(display_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 165, 255), 2)
        cv2.putText(display_frame, "Align cup inside box & Press SPACE", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(display_frame, "Press 'q' to cancel registration", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        cv2.imshow("Register Cup Image", display_frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '):
            print("⏳ 컵 특징 벡터(Embedding) 분석 중... (잠시만 기다려 주세요)")
            crop_bgr = frame[box_y:box_y+box_h, box_x:box_x+box_w]
            embedding_result = extract_resnet_embedding(model, crop_bgr)
            
            # 하위 호환용 HSV 히스토그램 추출
            fg_mask = extract_cup_foreground_grabcut(frame, (box_x, box_y, box_w, box_h))
            hist_result = calculate_hsv_histogram(frame, mask=fg_mask)
            break
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyWindow("Register Cup Image")
    return embedding_result, hist_result


def speak(text):
    """Mac OS 'say' 명령어로 음성 피드백을 제공합니다."""
    try:
        subprocess.Popen(["say", text])
    except:
        pass


def register_new_cup(cam_idx, model):
    """사용자가 카메라를 통해 직접 컵 비전 임베딩을 등록하도록 유도하고 저장합니다."""
    print("\n" + "=" * 60)
    print("                [ 머신러닝 기반 컵 비전 등록 및 학습 ]")
    print("=" * 60)
    
    speak("새로운 컵 비전 등록을 시작합니다.")
    
    # 1. 컵 이름 입력 받기
    speak("등록할 컵의 이름을 터미널창에 입력하세요.")
    cup_name = input("  새 컵의 이름을 입력하세요 (예: green_mug, tall_glass): ").strip().replace(" ", "_")
    if not cup_name:
        print("❌ 이름이 입력되지 않아 등록을 취소합니다.")
        return load_cup_templates()
        
    # 2. 카메라로 촬영하여 ResNet 특징 임베딩 취득
    print("\n  -> 카메라 창에서 스페이스 바를 눌러 컵을 촬영하세요.")
    speak("카메라 창에 컵을 맞추고 스페이스 바를 눌러 촬영하세요.")
    cup_emb, cup_hist = capture_cup_image_and_features(cam_idx, model)
    
    if cup_emb is None:
        print("❌ 컵 촬영이 취소되었습니다.")
        speak("컵 추가가 취소되었습니다.")
        return load_cup_templates()
        
    # 3. 캘리브레이션 캐시 NPZ 파일 저장
    cache_path = os.path.join(CACHE_DIR, f"{cup_name}_calibration.npz")
    np.savez(
        cache_path,
        cup_embedding=cup_emb,
        cup_hist=cup_hist,  # 하위 호환용
        l_max=15.0,         # 수위 측정용 기본 높이값
    )
    print(f"💾 컵 임베딩 등록 완료 (ResNet-18 적용): {cache_path}")
    speak("새로운 컵 비전 등록이 완료되었습니다.")
    
    return load_cup_templates()


def main():
    print("=" * 60)
    print("   🔍 컵 분류 알고리즘 (ResNet-18 Embedding & Cosine Similarity) 테스트   ")
    print("=" * 60)
    
    # 1. 모델 로딩
    print("🧠 ResNet-18 머신러닝 특징 추출기 로드 중 (약 5초 소요)...")
    try:
        # TIMM 대신 torchvision 기본을 활용하여 속도 및 안정성 보장
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        # 마지막 전결합층(fc) 제거하여 512차원 특징 맵(임베딩) 획득
        model.fc = torch.nn.Identity()
        model.eval()
        print("✅ ResNet-18 모델 빌드 완료!")
    except Exception as e:
        print(f"❌ ResNet-18 모델 로드 실패: {e}")
        sys.exit(1)

    # 2. 템플릿 로드
    templates = load_cup_templates()
    if not templates:
        print("⚠️ 등록된 컵 데이터가 존재하지 않습니다. 먼저 컵을 추가해 주세요.")
        
    # 3. 카메라 선택
    cam_idx = select_camera()
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"❌ 카메라 {cam_idx}를 열 수 없습니다.")
        sys.exit(1)
        
    print("⏳ 카메라 연결 초기화 및 워밍업 중...")
    has_frame = False
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            has_frame = True
            break
        time.sleep(0.1)
        
    if not has_frame:
        print("\n⚠️ [경고] 첫 프레임을 읽어오지 못했습니다. 카메라 권한 및 장치 사용 여부를 확인하세요.\n")
        
    # 창 설정
    cv2.namedWindow("Cup Classification Test", cv2.WINDOW_NORMAL)
    
    ml_threshold = DEFAULT_ML_THRESHOLD
    consecutive_match = 0
    last_best_name = None
    confirmed_cup = None
    confirmed_time = 0
    paused = False
    consecutive_failures = 0
    
    speak("머신러닝 기반 컵 식별 테스트를 시작합니다. 컵을 화면 중앙 파란색 박스에 비춰 주세요.")
    
    print("\n🚀 테스트 시작!")
    print("  - [q] 누르면 종료됩니다.")
    print("  - [Space] 누르면 감지가 일시정지됩니다.")
    print("  - [c] 또는 [n] 누르면 새로운 컵을 바로 등록합니다.")
    print("  - [+ / -] 누르면 머신러닝 매칭 임계값(Threshold)이 변경됩니다.")
    print("-" * 60)

    while True:
        # 템플릿이 없는 경우 등록 대기 화면 표시
        if not templates:
            ret, frame = cap.read()
            if ret:
                h, w, _ = frame.shape
                cv2.putText(frame, "No cups registered. Press 'c' to register a new cup.", (30, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("Cup Classification Test", frame)
            key = cv2.waitKey(100) & 0xFF
            if key in (ord('c'), ord('n')):
                cap.release()
                cv2.destroyAllWindows()
                
                templates = register_new_cup(cam_idx, model)
                
                cap = cv2.VideoCapture(cam_idx)
                cv2.namedWindow("Cup Classification Test", cv2.WINDOW_NORMAL)
                paused = False
                confirmed_cup = None
            elif key == ord('q'):
                break
            continue

        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 30:
                print("⚠️ [오류] 카메라 프레임을 연속적으로 읽을 수 없습니다.")
                speak("카메라 연결을 잃었습니다.")
                break
            time.sleep(0.03)
            continue
        consecutive_failures = 0

        h, w, _ = frame.shape
        box_w, box_h = int(w * 0.4), int(h * 0.5)
        box_x, box_y = (w - box_w) // 2, (h - box_h) // 2
        
        display_frame = frame.copy()
        box_color = (255, 0, 0)
        best_score = 0.0
        
        if not paused:
            # 컵 영역 크롭
            crop_bgr = frame[box_y:box_y+box_h, box_x:box_x+box_w]
            
            # 검은색 화면 및 특징 없는 배경 감지 (어둡거나 평균 밝기 편차가 너무 작은 경우)
            gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            mean_val = np.mean(gray_crop)
            std_dev = np.std(gray_crop)
            
            # 평균 밝기가 15.0 미만이거나(검은 화면) 표준편차가 8.0 미만이면(무채색/단색) 빈 화면으로 판단
            is_empty_or_black = (mean_val < 15.0 or std_dev < 8.0)
            
            scores = []
            
            if is_empty_or_black:
                # 카메라가 가려졌거나 어두운 상태이면 매칭 연산을 건너뛰고 점수를 0.0으로 초기화
                for temp in templates:
                    if 'embedding' in temp:
                        scores.append((temp, 0.0, False, 'ML'))
                    elif 'hist' in temp:
                        scores.append((temp, 0.0, False, 'Hist'))
                # 화면에 시각적 알림 표시
                cv2.putText(display_frame, "No Object / Camera Covered", (box_x, box_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
                consecutive_match = max(0, consecutive_match - 2)
            else:
                # 매칭에 필요한 특징 계산 조건 분기 (중복 연산 방지)
                needs_ml = any('embedding' in temp for temp in templates)
                needs_hist = any('hist' in temp for temp in templates if 'embedding' not in temp)
                
                curr_embed = None
                curr_hist = None
                
                if needs_ml:
                    curr_embed = extract_resnet_embedding(model, crop_bgr)
                if needs_hist:
                    light_mask = get_lightweight_foreground_mask(frame, (box_x, box_y, box_w, box_h))
                    curr_hist = calculate_hsv_histogram(frame, mask=light_mask)
                
                # 모든 템플릿과 매칭 유사도 계산
                for temp in templates:
                    if 'embedding' in temp:
                        # L2 정규화 상태이므로 단순 내적 = 코사인 유사도
                        score = np.dot(temp['embedding'], curr_embed)
                        score = max(-1.0, min(1.0, float(score)))  # 클램핑
                        is_ok = (score >= ml_threshold)
                        scores.append((temp, score, is_ok, 'ML'))
                    elif 'hist' in temp:
                        # 하위 호환용 HSV 히스토그램 유사도
                        score = cv2.compareHist(temp['hist'], curr_hist, cv2.HISTCMP_CORREL)
                        score = max(0.0, score)
                        is_ok = (score >= DEFAULT_HIST_THRESHOLD)
                        scores.append((temp, score, is_ok, 'Hist'))
            
            # 유사도 내림차순 정렬
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # 매칭 검증 로직
            if scores:
                best_match, best_score, best_ok, best_type = scores[0]
                if best_ok:
                    match_name = best_match['name']
                    if match_name == last_best_name:
                        consecutive_match += 1
                    else:
                        consecutive_match = 0
                        last_best_name = match_name
                else:
                    consecutive_match = max(0, consecutive_match - 1)
            else:
                consecutive_match = 0
                
            # 인식 확정 판정
            if consecutive_match >= REQUIRED_FRAMES:
                if confirmed_cup != last_best_name:
                    confirmed_cup = last_best_name
                    confirmed_time = time.time()
                    print(f"🎯 [인식 확정] {confirmed_cup} (방식: {best_type}, 점수: {best_score:.3f})")
                    speak(f"{confirmed_cup} 컵이 확인되었습니다.")
            
            if confirmed_cup is not None:
                box_color = (0, 255, 0)
                cv2.rectangle(display_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), box_color, 4)
                cv2.putText(display_frame, "CONFIRMED!", (box_x + 10, box_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(display_frame, confirmed_cup, (box_x + 10, box_y + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                cv2.rectangle(display_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), box_color, 2)
                
            # 매칭 스코어 판넬 텍스트 그리기
            y_pos = 120
            cv2.putText(display_frame, f"ML Threshold: {ml_threshold:.2f}", (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, "Match Scores:", (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            
            for i, (temp, score, is_ok, m_type) in enumerate(scores):
                t_name = temp['name']
                if is_ok:
                    color = (0, 255, 0)        # 매칭 성공
                    indicator = " [OK]"
                elif i == 0:
                    color = (0, 165, 255)      # 1순위이나 미달
                    indicator = " [LOW]"
                else:
                    color = (180, 180, 180)    # 차순위 후보들
                    indicator = ""
                    
                label = f"{t_name} ({m_type}): {score:.3f}{indicator}"
                cv2.putText(display_frame, label, (30, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
                
                # 진행 바 시각화 (유사도 수준 표시)
                bar_len = int(max(0.0, score) * 150)
                if bar_len > 0:
                    cv2.rectangle(display_frame, (30, y_pos + 8), (30 + bar_len, y_pos + 12), color, -1)
                
                y_pos += 40
                
            # 연속 인식 카운터 시각화
            if consecutive_match > 0 and confirmed_cup is None:
                progress = min(1.0, consecutive_match / REQUIRED_FRAMES)
                w_bar = int(box_w * progress)
                cv2.rectangle(display_frame, (box_x, box_y + box_h + 10), 
                              (box_x + w_bar, box_y + box_h + 18), (0, 255, 255), -1)
                cv2.putText(display_frame, f"Matching... {int(progress*100)}%", 
                            (box_x, box_y + box_h + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.rectangle(display_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 255), 2)
            cv2.putText(display_frame, "PAUSED", (box_x + int(box_w*0.3), box_y + int(box_h*0.5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(display_frame, "Press SPACE to resume", (30, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # 사용 방법 표시 (하단 바)
        cv2.putText(display_frame, "Keys: [Space] Pause | [c] Register Cup | [+ / -] Threshold | [q] Exit", (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.imshow("Cup Classification Test", display_frame)
        
        # 키 입력 이벤트 핸들링 (30ms 대기)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("🛑 사용자에 의해 테스트가 종료되었습니다.")
            break
        elif key == ord(' '):
            paused = not paused
            confirmed_cup = None
            consecutive_match = 0
            last_best_name = None
            print("⏸️ 감지 모드 일시 정지" if paused else "▶️ 감지 모드 재개")
        elif key in (ord('c'), ord('n')):
            # 컵 등록 루프 진입을 위해 카메라 및 윈도우 일시 해제
            cap.release()
            cv2.destroyAllWindows()
            
            templates = register_new_cup(cam_idx, model)
            
            # 테스트 복구
            cap = cv2.VideoCapture(cam_idx)
            cv2.namedWindow("Cup Classification Test", cv2.WINDOW_NORMAL)
            paused = False
            confirmed_cup = None
            consecutive_match = 0
            last_best_name = None
            
        elif key in (ord('+'), ord('=')):
            ml_threshold = min(1.0, ml_threshold + 0.02)
            print(f"📈 머신러닝 매칭 임계값 증가 -> {ml_threshold:.2f}")
        elif key == ord('-'):
            ml_threshold = max(0.0, ml_threshold - 0.02)
            print(f"📉 머신러닝 매칭 임계값 감소 -> {ml_threshold:.2f}")
            
        if confirmed_cup is not None and not paused:
            # 코사인 유사도 마진 완화 적용 (0.05 편차 대응)
            if best_score < ml_threshold - 0.05:
                if time.time() - confirmed_time > 2.0:
                    print("🔄 확정 상태 리셋 (컵 감지 해제)")
                    confirmed_cup = None
                    consecutive_match = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
