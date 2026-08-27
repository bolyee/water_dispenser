import os
import sys
import torch
import numpy as np
import cv2
import threading
import time

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.util import load_model, load_audio_tensor, get_model_output, visualise_args
import shared.utils as su

# Shared state between Main(Video) and Worker(AI) threads
shared_state = {
    "t_video": 0.0,
    "current_l_pred": 99.9, # default safe value (Waiting)
    "ai_running": True,     # flag to cleanly exit thread
    "is_stopped": False     # flag to notify video player to pause
}
lock = threading.Lock()

def ai_worker_thread(video_path, threshold=3.0):
    print("[AI Thread] Loading 360MB Model into memory...")
    model = load_model()
    
    print("[AI Thread] Loading audio buffer to simulate a Live Microphone...")
    full_audio = load_audio_tensor(video_path)
    sr = visualise_args['sr']
    total_samples = full_audio.shape[1]
    
    last_inference_t = -1.0
    print("[AI Thread] ⚡ AI Engine Ready! Now listening live ⚡")
    
    while True:
        with lock:
            if not shared_state["ai_running"]:
                break
            t = shared_state["t_video"]
            is_stopped = shared_state["is_stopped"]
            
        # Optimization: Don't run inference if video is paused/stopped or time hasn't moved yet
        if t == last_inference_t or is_stopped:
            time.sleep(0.01)
            continue
            
        # AI 모델의 맥락(Context) 파악을 위해 0초부터 현재 t초까지 누적된 오디오 역사를 통째로 줍니다.
        # 영상이 진행될수록 분석하는 배열의 길이가 점점 길어집니다.
        start_t = 0.0
        end_t = t
        
        start_idx = 0
        end_idx = int(end_t * sr)
        
        # 초반 1초 정도는 소리가 누적되길 기다림
        if end_idx < sr * 1.0:
            time.sleep(0.01)
            continue
            
        # Ensure 3D audio chunk correctly sliced [1, 1, L]
        audio_chunk = full_audio[:, :, start_idx:end_idx]
        
        try:
            # TRUE LIVE INFERENCE 
            with torch.no_grad():
                z_audio, y_audio = get_model_output(audio_chunk, model)
                
                wavelengths = y_audio @ torch.linspace(
                    0, visualise_args['w_max'], visualise_args['n_bins']
                ).to(y_audio.device)
                
                l_preds = su.physics.estimate_length_of_air_column(wavelengths).numpy()
                
                # The latest prediction is the LAST frame of this audio chunk!
                current_pred = l_preds[-1]
                
        except Exception as e:
            # Handle potential edge cases (e.g. chunk too small for model layers) gracefully
            time.sleep(0.01)
            continue
            
        with lock:
            shared_state["current_l_pred"] = current_pred
            # Trigger auto stop if threshold reached (ignore initial > 1.5s noises)
            if current_pred <= threshold and t > 1.5:
                shared_state["is_stopped"] = True

        last_inference_t = t
        
        # CPU Throttle: Sleep for 50ms implies AI runs at ~20 Hz maximum.
        # This prevents 100% CPU lockup and keeps the video playing smoothly.
        time.sleep(0.05) 

def run_simulation(video_path, threshold=3.0):
    print(f"[Main] 🚀 Starting LIVE Streaming AI Simulator (Threshold: {threshold}cm)")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] 비디오 파일을 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0  
    frame_delay = int(1000 / fps)
    
    # Start the "Live Microphone AI Engine" in the background
    ai_thread = threading.Thread(target=ai_worker_thread, args=(video_path, threshold))
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
                # Share the live timestamp with the AI thread
                with lock:
                    shared_state["t_video"] = frame_idx / fps

            frame = cv2.resize(frame, (800, 600))
            
            # --- 평상시 UI 그리기 ---
            text_color = (0, 255, 0) # 초록색
            cv2.rectangle(frame, (10, 10), (450, 120), (0,0,0), -1) 
            
            with lock:
                t_video = shared_state["t_video"]
                
            cv2.putText(frame, f"Live Time: {t_video:.1f} s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show "--" until AI warms up and sends the first prediction
            val_text = f"{current_l_pred:.2f} cm" if current_l_pred != 99.9 else "--"
            cv2.putText(frame, f"Live Space: {val_text}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            
            # --- 멈춤 시 UI 그리기 ---
            if is_stopped:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (800, 600), (0,0,255), -1)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                
                cv2.putText(frame, "LIVE AUTO STOP TRIGGERED!", (50, 250), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 5)
                cv2.putText(frame, f"Water level reached {current_l_pred:.2f} cm!", (60, 320), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.imshow("Real-Time Streaming Simulator", frame)
            
            wait_time = 0 if is_stopped else frame_delay
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q'):
                break

    finally:
        # Securely shut down the AI thread
        with lock:
            shared_state["ai_running"] = False
            
        print("[Main] Shutting down AI worker...")
        ai_thread.join()
        
        cap.release()
        cv2.destroyAllWindows()
        print("[Main] 🛑 Shutdown success.")

if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v_path = os.path.join(ROOT_DIR, "media_assets/example_video.mp4")
    
    print("==================================================")
    print("   마이크 스트리밍 모사 정수기 라이브 뷰어 시작   ")
    print("==================================================")
    
    run_simulation(v_path, threshold=3.0)
