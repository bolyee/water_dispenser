import os
import sys
import torch
import numpy as np
import cv2
import librosa

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.util import load_model, load_audio_tensor, get_model_output, visualise_args
import shared.utils as su

def get_cache_path(video_path):
    """영상 파일명 기반의 캐시 파일 경로를 반환합니다."""
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration_cache")
    os.makedirs(cache_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(cache_dir, f"{base_name}_calibration.npz")


def load_predictions(video_path):
    cache_path = get_cache_path(video_path)

    # --- 저장된 캐시가 있으면 재사용 여부 질문 ---
    if os.path.exists(cache_path):
        cache_info = np.load(cache_path)
        l_max_cached = float(cache_info['l_max'])
        l_min_cached = float(cache_info['l_min'])
        print("\n" + "="*58)
        print("  이전에 학습된 컵 데이터를 발견했습니다!")
        print(f"  컵 범위: {l_max_cached:.2f}cm (빈) ~ {l_min_cached:.2f}cm (다 참)")
        print("="*58)
        ans = input("  이전에 학습한 컵과 동일한 컵입니까? [Y/n]: ").strip().lower()
        if ans in ('', 'y', 'yes', 'ㅛ'):
            print("✅ 기존 학습 데이터를 불러옵니다. AI 재학습을 건너뜁니다!\n")
            return (
                cache_info['timestamps_eval'],
                cache_info['l_pred'],
                l_max_cached,
                l_min_cached,
            )
        else:
            print("🔄 새로운 컵으로 인식합니다. AI 재학습을 시작합니다...\n")

    # --- 새로 학습 ---
    print("1) 모델 로딩 및 소리 추출을 시작합니다... (잠시만 기다려주세요)")
    model = load_model()
    audio = load_audio_tensor(video_path)

    with torch.no_grad():
        print("2) AI가 영상을 통째로 스캔하며 초 단위 수위를 미리 분석 중입니다...")
        z_audio, y_audio = get_model_output(audio, model)

        wavelengths = y_audio @ torch.linspace(
            0, visualise_args['w_max'], visualise_args['n_bins']
        ).to(y_audio.device)

        l_pred = su.physics.estimate_length_of_air_column(wavelengths).numpy()

    n_frames = len(y_audio)
    timestamps_eval = librosa.frames_to_time(
        np.arange(n_frames),
        sr=visualise_args['sr'],
        n_fft=visualise_args['n_fft'],
        hop_length=visualise_args['hop_length'],
    )

    # 전체 컵 범위 계산
    l_max = np.mean(l_pred[:10])
    l_min = np.mean(l_pred[-10:])
    print(f"[정보] 컵 전체 범위: {l_max:.2f}cm (빈 상태) ~ {l_min:.2f}cm (꽉 찬 상태)")

    # --- 캐시 저장 ---
    np.savez(cache_path, timestamps_eval=timestamps_eval, l_pred=l_pred, l_max=l_max, l_min=l_min)
    print(f"💾 학습 결과를 저장했습니다: {cache_path}\n")

    return timestamps_eval, l_pred, l_max, l_min

def run_simulation(video_path, timestamps_eval, l_pred, l_max, l_min, fill_ratio=0.80):
    # 컵 전체 높이(l_max) 기준으로 fill_ratio만큼 채워졌을 때 정지
    # 예: l_max=9.16cm, fill_ratio=0.80 → 정지 시점 = 9.16 × (1-0.80) = 1.83cm
    threshold = l_max * (1.0 - fill_ratio)
    print(f"3) 준비 완료! 영상을 재생합니다.")
    print(f"   → 컵 전체 높이: {l_max:.2f}cm | 정지 임계값: {threshold:.2f}cm (컵의 {int(fill_ratio*100)}% 채워짐 기준)")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] 비디오 파일을 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0  # Fallback
    frame_delay = int(1000 / fps)
    
    stopped = False
    last_frame = None
    current_l_pred = 99.9
    t_video = 0.0
    
    while True:
        if not stopped:
            ret, frame = cap.read()
            if not ret:
                print("--- 영상이 끝까지 재생되었습니다. ---")
                break
                
            frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
            t_video = frame_idx / fps
            
            # 비디오 현재 시간과 가장 일치하는 AI 예측값 탐색
            idx = (np.abs(timestamps_eval - t_video)).argmin()
            current_l_pred = l_pred[idx]
            
            # 3.0cm 이하 달성 시 정지 트리거
            if current_l_pred <= threshold and t_video > 1.5:
                stopped = True
            
            frame = cv2.resize(frame, (800, 600))
            last_frame = frame.copy()
        else:
            # 정지 상태에서는 마지막 프레임을 계속 재사용
            frame = last_frame.copy()
        
        # --- 수위 UI (항상 표시) ---
        text_color = (0, 255, 0)
        cv2.rectangle(frame, (10, 10), (450, 120), (0,0,0), -1)
        cv2.putText(frame, f"Time: {t_video:.1f} s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Empty Space: {current_l_pred:.2f} cm", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        
        # --- 멈춤 시 경고 UI (정지 후 매 프레임 다시 그리기) ---
        if stopped:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (800, 600), (0,0,255), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            
            cv2.putText(frame, "AUTO STOP TRIGGERED!", (50, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)
            cv2.putText(frame, f"Water level reached {current_l_pred:.2f} cm!", (60, 320), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Auto-Stopping Purifier Simulator", frame)
        
        # 멈췄을 땐 0(무한 대기), 재생 중일 땐 프레임 지연(ms)
        wait_time = 0 if stopped else frame_delay
        key = cv2.waitKey(wait_time) & 0xFF
        
        # 'q' 누르면 언제든 종료
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v_path = os.path.join(ROOT_DIR, "media_assets/goWgiQQMugA_2.5_9.0.mp4")
    print("==================================================")
    print("   자동 멈춤 정수기 OpenCV 시뮬레이터 구동 시작   ")
    print("==================================================")
    t_eval, l_preds, l_max, l_min = load_predictions(v_path)
    run_simulation(v_path, t_eval, l_preds, l_max, l_min, fill_ratio=0.80)

