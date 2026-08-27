import os
import sys
import numpy as np
import cv2
import threading
import time
import librosa
import torch

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.util import load_model, load_audio_tensor, get_model_output, visualise_args
import shared.utils as su

# Shared state between Main(Video) and Worker(Audio) threads
shared_state = {
    "t_video": 0.0,
    "current_l_pred": 99.9, 
    "ai_running": True,     
    "is_stopped": False     
}
lock = threading.Lock()

def build_calibration_templates(video_path):
    print("\n=======================================================")
    print("[사전 캘리브레이션 단계 시작] - 컵을 영점 조절합니다.")
    print("=======================================================")
    print("1/3. 360MB 정수기 AI 모델을 불러옵니다... (최초 1회만 실행)")
    model = load_model()
    
    print("2/3. 특정 컵의 전체 물소리(Reference)를 분석하여 정답지를 만듭니다...")
    full_audio = load_audio_tensor(video_path)
    sr = 16000
    
    with torch.no_grad():
        z_audio, y_audio = get_model_output(full_audio, model)
        wavelengths = y_audio @ torch.linspace(
            0, visualise_args['w_max'], visualise_args['n_bins']
        ).to(y_audio.device)
        # Ground truth water levels
        l_preds = su.physics.estimate_length_of_air_column(wavelengths).numpy()
    
    # 오디오 신호 매칭(템플릿) 추출을 위한 작업
    print("3/3. 정답지의 소리 주파수(Mel Spectrogram)를 초경량 템플릿으로 변환 중입니다...")
    audio_np = full_audio.squeeze().numpy()
    
    # Calculate Melspectrogram (소리 지문)
    n_fft = 1024
    hop_length = 512
    mels = librosa.feature.melspectrogram(y=audio_np, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=64)
    mels_db = librosa.power_to_db(mels, ref=np.max)
    
    # 시간축 동기화 코딩 조절 (AI의 예측 시간대축을 주파수 시간대축으로 변환 및 삽입)
    time_mels = librosa.frames_to_time(np.arange(mels_db.shape[1]), sr=sr, hop_length=hop_length)
    n_frames_model = len(y_audio)
    time_model = librosa.frames_to_time(
        np.arange(n_frames_model),
        sr=visualise_args['sr'],
        n_fft=visualise_args['n_fft'],
        hop_length=visualise_args['hop_length'],
    )
    
    aligned_l_preds = np.interp(time_mels, time_model, l_preds)
    
    print("✅ [사전 학습 및 영점 조절 완벽히 성공!] 무거운 AI 메모리는 이제 컴퓨터에서 완전히 삭제됩니다.")
    del model 
    
    return mels_db, aligned_l_preds, audio_np, sr

def audio_matching_worker(ref_mels_db, ref_l_preds, full_audio_np, sr, threshold=3.0):
    print("\n[경량 매칭 엔진] 무거운 AI 대신, 라이브 스트리밍 리스닝을 시작합니다... (CPU 거의 안 씀)")
    
    while True:
        with lock:
            if not shared_state["ai_running"]:
                break
            t = shared_state["t_video"]
            is_stopped = shared_state["is_stopped"]
            
        if is_stopped:
            time.sleep(0.01)
            continue
            
        # [실시간 흉내내기]: 현재 비디오 재생 시점 기준 가장 최근 0.5초의 소리 조각 캡쳐 (마이크 동작 원리)
        chunk_sec = 0.5
        start_t = max(0.0, t - chunk_sec)
        end_t = t
        
        start_idx = int(start_t * sr)
        end_idx = int(end_t * sr)
        
        if end_idx - start_idx < sr * 0.1:
            time.sleep(0.01)
            continue
            
        live_chunk = full_audio_np[start_idx:end_idx]
        
        try:
            # 1. 방금 들린 0.5초 소리의 특징값(Mel DB) 초고속 추출 (0.001초 미만 소요)
            live_mels = librosa.feature.melspectrogram(y=live_chunk, sr=sr, n_fft=1024, hop_length=512, n_mels=64)
            live_mels_db = librosa.power_to_db(live_mels, ref=np.max)
            # 평균 특징점 1줄(Array)로 변환
            live_feature = np.mean(live_mels_db, axis=1)
            
            # 2. MATCHING 핵심 매커니즘: 
            # 과거 학습해둔 사전에 수많은 시간대 배열 중 현재 소리와 유클리드 거리가 가장 가까운(최소 차이) 놈을 찾습니다.
            distances = np.linalg.norm(ref_mels_db.T - live_feature, axis=1)
            best_match_idx = np.argmin(distances)
            
            # 3. 그 찾아낸 오리지널 시간대에 해당하는 수위(cm) 정답을 꺼내옵니다.
            current_pred = ref_l_preds[best_match_idx]
        except Exception as e:
            print(f"[ERROR in Matching Thread] {e}")
            time.sleep(0.01)
            continue
        
        with lock:
            shared_state["current_l_pred"] = current_pred
            # 목표 수위(3.0cm) 이하 달성 시 영상 강제 정지 명령 (초기 1초간 노이즈 무시)
            if current_pred <= threshold and t > 1.0:
                shared_state["is_stopped"] = True

        time.sleep(0.03) # 매칭이 너무 가벼워서 그냥 팍팍 돌려도 컴퓨터에 무리가 없습니다.

def run_simulation(video_path, threshold=3.0):
    # 1. 사전 캘리브레이션
    ref_mels_db, ref_l_preds, full_audio_np, sr = build_calibration_templates(video_path)
    
    print(f"\n[Main] 🚀 초경량 주파수 매칭 시뮬레이터 비디오 재생 시작 (목표 잔여수위: {threshold}cm)")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] 비디오 파일을 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0  
    frame_delay = int(1000 / fps)
    
    ai_thread = threading.Thread(target=audio_matching_worker, args=(ref_mels_db, ref_l_preds, full_audio_np, sr, threshold))
    ai_thread.start()
    
    try:
        while True:
            with lock:
                is_stopped = shared_state["is_stopped"]
                current_l_pred = shared_state["current_l_pred"]
                
            if not is_stopped:
                ret, frame = cap.read()
                if not ret:
                    print("--- 영상이 끝까지 재생되었습니다. ---")
                    break
                    
                frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
                with lock:
                    shared_state["t_video"] = frame_idx / fps

            frame = cv2.resize(frame, (800, 600))
            
            text_color = (0, 255, 0)
            cv2.rectangle(frame, (10, 10), (600, 150), (0,0,0), -1) 
            
            with lock:
                t_video = shared_state["t_video"]
                
            cv2.putText(frame, f"Live Time: {t_video:.1f} s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            val_text = f"{current_l_pred:.2f} cm" if current_l_pred != 99.9 else "--"
            cv2.putText(frame, f"Live Space (Match): {val_text}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            
            if is_stopped:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (800, 600), (0,0,255), -1)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                
                # 시각적으로 변경됨을 알리기 위해 텍스트 바꿈
                cv2.putText(frame, "LIVE PITCH MATCH-STOP TRIGGERED!", (30, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
                cv2.putText(frame, f"Water level hit {current_l_pred:.2f} cm!", (60, 320), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.imshow("Super-Fast Pitch Matching Simulator", frame)
            
            wait_time = 0 if is_stopped else frame_delay
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q'):
                break

    finally:
        with lock:
            shared_state["ai_running"] = False
        ai_thread.join()
        cap.release()
        cv2.destroyAllWindows()
        print("[Main] 🛑 Shutdown success.")

if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v_path = os.path.join(ROOT_DIR, "media_assets/example_video.mp4")
    run_simulation(v_path, threshold=3.0)
