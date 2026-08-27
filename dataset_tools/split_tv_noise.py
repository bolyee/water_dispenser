"""
split_tv_noise.py — Local Audio Splitting Utility for tv_noise
--------------------------------------------------------------------------------
Detects 'tv_noise.*' in the project directory, loads its audio track using torchaudio,
resamples it to 16kHz mono, and slices it into 20 segments (2-min length, 5-min intervals)
as dataset_*.mp3.

Usage:
    .\venv\Scripts\python.exe split_tv_noise.py
"""

import os
import sys

try:
    import torch
    import torchaudio
except ImportError:
    print("[FAIL] 'torchaudio' is not installed in the virtual environment.")
    print("       Please activate your environment or install it.")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
total_count = 20  # 다운로드할 파일 총 개수
interval_minutes = 5  # 간격 (5분)
duration_minutes = 2  # 추출할 길이 (2분)
TARGET_SR = 16000  # 모델 표준 샘플 레이트 (16kHz)

def find_tv_noise_file():
    """Finds a file starting with 'tv_noise' in the root directory."""
    candidates = [f for f in os.listdir(ROOT_DIR) if f.startswith("tv_noise") and os.path.isfile(os.path.join(ROOT_DIR, f))]
    if not candidates:
        return None
    # Prioritize non-temporary files
    return os.path.join(ROOT_DIR, candidates[0])

def main():
    print("\n" + "="*60)
    print(" ✂️ 로컬 tv_noise 오디오 쪼개기 및 MP3 변환기")
    print("="*60)
    
    file_path = find_tv_noise_file()
    if not file_path:
        print("[FAIL] 폴더 내에서 'tv_noise' 파일(예: tv_noise.mp4, tv_noise.mp3 등)을 찾을 수 없습니다.")
        print("       수동으로 다운로드한 영상/음향 파일의 이름을 'tv_noise'로 수정해 넣어주세요.")
        sys.exit(1)
        
    file_name = os.path.basename(file_path)
    print(f"✅ 대상 파일 감지 성공: {file_name}")
    print("⏳ 오디오 정보를 분석하고 있습니다... (대용량 파일의 경우 10~30초 소요)")

    try:
        # torchaudio will automatically decode audio from video formats like mp4 if ffmpeg is available
        waveform, sr = torchaudio.load(file_path)
        
        # Mono channel downmix
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample to 16kHz
        if sr != TARGET_SR:
            print(f"🔄 샘플 레이트 변환: {sr}Hz -> {TARGET_SR}Hz")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
            waveform = resampler(waveform)
            sr = TARGET_SR
            
        total_duration_sec = waveform.shape[1] / sr
        print(f"✅ 오디오 분석 완료! (총 길이: {total_duration_sec/60:.1f}분 / {total_duration_sec:.1f}초)")
        
        print(f"⚡ {total_count}개의 구간 분할을 시작합니다...")
        for i in range(total_count):
            start_sec = i * interval_minutes * 60
            end_sec = start_sec + duration_minutes * 60
            
            start_idx = int(start_sec * sr)
            end_idx = int(end_sec * sr)
            
            # Stop if the segment starts beyond the audio track duration
            if start_idx >= waveform.shape[1]:
                print(f"⚠️ 구간 {i+1} ({start_sec/60:.1f}분)은 원본 음원의 범위를 벗어납니다. 슬라이싱을 종료합니다.")
                break
                
            end_idx = min(end_idx, waveform.shape[1])
            chunk = waveform[:, start_idx:end_idx]
            
            out_file_name = f"dataset_{str(i+1).zfill(2)}.mp3"
            out_file_path = os.path.join(ROOT_DIR, out_file_name)
            
            # Save segment as mp3
            torchaudio.save(out_file_path, chunk, sr, format="mp3")
            print(f"💾 [구간 {i+1}/{total_count}] 추출 성공: {out_file_name} ({start_sec/60:.1f}분 ~ {end_sec/60:.1f}분)")
            
        print("\n✅ 모든 데이터셋 오디오 추출이 정상 완료되었습니다!")
        
    except Exception as e:
        print(f"\n❌ 오디오 처리 중 에러 발생: {e}")
        print("💡 팁: 만약 인코딩 에러가 난다면, 사용자의 시스템 환경에 FFmpeg이 설치되어 있어야 mp4 영상 등에서 오디오를 추출할 수 있습니다.")

if __name__ == "__main__":
    main()
