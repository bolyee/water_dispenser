import os
import sys
import socket
import time
import numpy as np
import soundfile as sf

try:
    import noisereduce as nr
    DENOISE_AVAILABLE = True
except ImportError:
    DENOISE_AVAILABLE = False

UDP_PORT = 5005
SAMPLE_RATE = 16000


def main():
    if not DENOISE_AVAILABLE:
        print("\n[오류] 노이즈 제거 라이브러리(noisereduce, scipy)가 설치되어 있지 않습니다.")
        print("       가상환경을 활성화하고 다음을 실행해 주세요: pip install noisereduce scipy")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    original_path = os.path.join(script_dir, "water_pour_original.wav")
    denoised_path = os.path.join(script_dir, "water_pour_denoised.wav")

    print("="*60)
    print(" 🎙️  물 따르기 소리 녹음 및 노이즈 제거 변환기")
    print("="*60)
    print("  1. ESP32 전원을 켜고 이 PC의 IP로 스트리밍 중인지 확인하세요.")
    print("  2. 이 프로그램은 녹음 완료 후 [원본] 및 [노이즈 제거] 파일 2개를 생성합니다.")
    print("  3. 오디오 재생은 하지 않으므로 파일 생성 후 컴퓨터에서 직접 더블클릭해 들어보시면 됩니다.")
    print("="*60 + "\n")

    # 1. 소켓 준비
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", UDP_PORT))
        sock.settimeout(3.0)
    except Exception as e:
        print(f"[FAIL] 포트 {UDP_PORT} 바인딩 에러: {e}")
        print("       다른 프로그램이 5005 포트를 쓰고 있는지 확인하세요.")
        return

    # 2. ESP32 연결 대기
    print("ESP32 마이크 신호 대기 중...")
    try:
        data, addr = sock.recvfrom(4096)
        print(f"[OK] 연결 확인! 송신지 IP: {addr[0]}")
    except socket.timeout:
        print("\n[FAIL] 타임아웃: 3초 동안 ESP32로부터 들어온 패킷이 없습니다.")
        print("       ESP32의 WiFi 상태 및 PC IP 설정을 확인해 주세요.")
        sock.close()
        return

    # 3. 녹음 설정 안내
    print("\n" + "="*50)
    print("  💡 녹음 필수 수칙:")
    print("  - 시작 직후 [처음 2초 동안]은 아무 소리도 내지 마세요 (주변 소음 프로필 수집)")
    print("  - [2초 후]부터 물을 컵에 졸졸졸 따라 주세요.")
    print("="*50)
    
    input("\n▶ 녹음을 시작하려면 Enter를 누르세요... (기본 10초간 녹음) ")
    
    # 4. 녹음 시작
    print("\n🎙️ [녹음 시작] 처음 2초는 조용히 -> 그 후 물을 부어 주세요...")
    audio_frames = []
    start_time = time.time()
    sock.settimeout(1.0)
    
    RECORD_SECONDS = 10
    
    while time.time() - start_time < RECORD_SECONDS:
        try:
            data, addr = sock.recvfrom(4096)
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_frames.append(chunk)
        except socket.timeout:
            print("[WARN] 패킷 드롭 발생")
            
    sock.close()
    print("\n⏹️ [녹음 완료] 오디오 데이터를 가공하는 중입니다...")

    if len(audio_frames) == 0:
        print("[FAIL] 수집된 데이터가 없습니다.")
        return

    # 오디오 병합
    audio_data = np.concatenate(audio_frames)

    # 5. 원본 파일 저장
    sf.write(original_path, audio_data, SAMPLE_RATE)
    print(f"💾 1. 원본 파일 저장 완료: {original_path}")

    # 6. 노이즈 제거 프로세스
    # 처음 2초를 노이즈 구간으로 사용
    noise_samples = int(2.0 * SAMPLE_RATE)
    if len(audio_data) < noise_samples:
        noise_samples = len(audio_data)
        
    noise_profile = audio_data[:noise_samples]
    original_noise_rms = np.sqrt(np.mean(noise_profile ** 2))

    # 스펙트럼 차감 적용 (2초 간 측정된 배경 잡음만 제거)
    audio_denoised = nr.reduce_noise(y=audio_data, sr=SAMPLE_RATE, y_noise=noise_profile, prop_decrease=0.90)

    # 7. 제거된 소음 크기 확인 및 저장
    denoised_noise_profile = audio_denoised[:noise_samples]
    denoised_noise_rms = np.sqrt(np.mean(denoised_noise_profile ** 2))

    sf.write(denoised_path, audio_denoised, SAMPLE_RATE)
    print(f"💾 2. 노이즈 제거 파일 저장 완료: {denoised_path}")

    if denoised_noise_rms > 0 and original_noise_rms > 0:
        db_reduction = 20 * np.log10(original_noise_rms / denoised_noise_rms)
        print(f"\n📊 [분석 결과] 배경 소음 레벨이 약 {db_reduction:.2f} dB 감소했습니다!")
    
    print("\n이제 폴더에서 두 파일(original, denoised)을 더블클릭해서 비교 청취해 보세요!")

if __name__ == "__main__":
    main()
