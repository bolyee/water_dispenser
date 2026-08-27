import os
import sys
import numpy as np
import soundfile as sf

# scipy & noisereduce 임포트 가능 여부 확인
try:
    import noisereduce as nr
    import sounddevice as sd
    DENOISE_AVAILABLE = True
except ImportError:
    DENOISE_AVAILABLE = False


def main():
    if not DENOISE_AVAILABLE:
        print("\n[오류] 노이즈 제거 라이브러리(noisereduce, scipy, sounddevice)가 설치되어 있지 않습니다.")
        print("       가상환경을 활성화하고 다음을 실행해 주세요: pip install noisereduce scipy sounddevice")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "esp32_record_test.wav")
    output_path = os.path.join(script_dir, "esp32_record_test_denoised.wav")

    if not os.path.exists(input_path):
        print(f"\n[오류] 원본 테스트 파일이 없습니다: {input_path}")
        print("       먼저 ESP32 녹음기(tests_and_simulations/record_esp32_audio.py)를 실행해")
        print("       'esp32_record_test.wav' 파일을 생성해 주세요.")
        return

    # 오디오 로드
    data, sr = sf.read(input_path)
    print("\n" + "="*60)
    # 2채널인 경우 1채널로 변환
    if len(data.shape) > 1:
        data = data[:, 0]
    
    print(f"🎵 오디오 로드 완료: {os.path.basename(input_path)}")
    print(f"   - 샘플레이트: {sr}Hz")
    print(f"   - 총 길이: {len(data)/sr:.2f}초 ({len(data)} 샘플)")
    print("="*60)

    # 1. 노이즈 프로필 추출 (보통 처음 1.5초는 조용한 대기 상태로 가정)
    noise_duration = min(1.5, len(data)/sr)
    noise_samples = int(noise_duration * sr)
    noise_profile = data[:noise_samples]
    
    original_noise_rms = np.sqrt(np.mean(noise_profile ** 2))
    print(f"🔇 [원본] 배경 소음 레벨 (RMS): {original_noise_rms:.6f}")

    # 2. 노이즈 감쇄 필터 적용
    # 스펙트럼 차감 적용 (배경 잡음 제거)
    data_denoised = nr.reduce_noise(y=data, sr=sr, y_noise=noise_profile, prop_decrease=0.90)

    # 3. 감쇄 후 소음 측정
    denoised_noise_profile = data_denoised[:noise_samples]
    denoised_noise_rms = np.sqrt(np.mean(denoised_noise_profile ** 2))
    print(f"🔇 [정제] 배경 소음 레벨 (RMS): {denoised_noise_rms:.6f}")

    # 4. 수치적 개선도 계산 (데시벨, dB)
    if denoised_noise_rms > 0:
        db_reduction = 20 * np.log10(original_noise_rms / denoised_noise_rms)
        print(f"\n📊 [수치적 개선도] 노이즈 레벨이 약 {db_reduction:.2f} dB 감소했습니다!")
        print(f"   (노이즈 크기가 원래의 약 {denoised_noise_rms / original_noise_rms * 100:.1f}% 수준으로 줄어듦)")
    else:
        print("\n📊 노이즈가 완전히 제거되었습니다.")

    # 파일 저장
    sf.write(output_path, data_denoised, sr)
    print(f"💾 필터링된 파일 저장 완료: {output_path}")
    print("="*60)

    # 5. 귀로 직접 들어보기 비교 (재생)
    print("\n👂 귀로 들어보기 비교 (헤드폰/스피커를 준비해 주세요)")
    try:
        input("▶ [1/2] 원본 소리 듣기 (Enter를 누르면 재생 시작) ")
        sd.play(data, sr)
        sd.wait()

        input("▶ [2/2] 노이즈 제거된 소리 듣기 (Enter를 누르면 재생 시작) ")
        sd.play(data_denoised, sr)
        sd.wait()
        print("\n✅ 재생이 완료되었습니다.")
    except Exception as e:
        print(f"\n[알림] 오디오 장치 재생 중 오류 발생: {e}")
        print("       (WAV 파일이 저장되었으니 직접 더블클릭해서 들으셔도 됩니다.)")

if __name__ == "__main__":
    main()
