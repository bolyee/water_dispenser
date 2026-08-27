import os
import sys

# 이 스크립트가 실행되는 위치의 부모 폴더(SoundOfWater 폴더)를 Python 경로(path)에 추가해서 import 에러를 방지합니다.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

# 모델 및 오디오 파이프라인 관련 모듈들 (레포지토리 내부 파일들)
from demo.util import load_model, load_audio_tensor, get_model_output, visualise_args
import shared.utils as su

def main():
    print("="*60)
    print("SoundOfWater 단독 테스트 구동 시작")
    print("="*60)

    # 1) 테스트용 데모 영상 파일 경로 확인
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = os.path.join(ROOT_DIR, "media_assets/example_video.mp4")
    if not os.path.exists(video_path):
        print(f"[X] 샘플 비디오를 찾을 수 없습니다: {video_path}")
        return
    print(f"[O] 샘플 오디오 인지 성공: {video_path}")

    # 2) 모델 로딩 (demo/util.py 경로를 ./models 로 수정해두었습니다.)
    print("\n모델을 불러오고 있습니다...")
    try:
        model = load_model()
    except Exception as e:
        print(f"\n[X] 모델 로딩 중 에러 발생: {e}")
        print("    -> 모델 파일(real_finetuned_visual_cosupervision.pth)이 ./models/ 에 잘 다운로드 되었는지 확인해주세요!")
        return
    print("[O] 모델 로딩 완료!")

    # 3) 오디오 전처리 (텐서 변환)
    print("\n오디오를 처리하는 중입니다 (WAV 변환 및 모노/16kHz 리샘플링)...")
    try:
        # load_audio_tensor 함수 내부에서 데코드(decord)/토치오디오(torchaudio) 등을 통해 처리
        audio = load_audio_tensor(video_path)
    except Exception as e:
        print(f"[X] 오디오 처리 에러: {e}")
        return

    # 4) 모델 추론 (Inference)
    print("\n모델을 통해 남은 수위(공간)를 예측합니다...")
    with torch.no_grad():
        z_audio, y_audio = get_model_output(audio, model)

        # y_audio는 시간 프레임별 파장(Wavelength) 예측값 분포입니다.
        # 파장으로 변환 (해당 방식은 demo/util.py show_output 함수에서 발췌)
        wavelengths = y_audio @ torch.linspace(
            0, visualise_args['w_max'], visualise_args['n_bins']
        ).to(y_audio.device)

        # 파장 값을 바탕으로 물리적 특성 계산
        l_pred = su.physics.estimate_length_of_air_column(wavelengths)
        
        # 마지막 10프레임 정도의 평균으로 수위를 파악할 수 있습니다 (물이 차오를수록 l_pred 값이 작아짐)
        final_l_pred = l_pred[-10:].mean().item()
        
    print("="*60)
    print(f"[O] 예측된 남은 빈 공간(공기기둥) 길이(가장 최근): {final_l_pred:.2f} cm")
    
    # 5) 자동 정지 시그널 테스트
    THRESHOLD = 3.0 # 정지 임계값 (3.0 cm)
    if final_l_pred <= THRESHOLD:
        print(f"🚨 [자동 멈춤 신호 발생] 남은 빈 공간이 {THRESHOLD}cm 이하입니다. 작동을 중지합니다!")
    else:
        print("💡 물이 아직 가득 차지 않았습니다. 안전합니다.")
    print("="*60)

if __name__ == "__main__":
    main()
