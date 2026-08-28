# 📊 SoundOfWater 프로젝트 및 최근 작업 요약 (PROJECT SUMMARY)

이 문서는 프로젝트의 전체 구조와 최근 진행된 핵심 작업 내용을 요약한 것으로, 다른 대화나 세션에서 이어서 작업할 때 참고할 수 있도록 작성되었습니다.

---

## 1. 📁 레포지토리 구조 및 주요 파일 설명

### 펌웨어 및 실시간 실행 파일
* **[esp32_servo_i2s_mic.ino](../firmware/esp32_servo_i2s_mic/esp32_servo_i2s_mic.ino)**
  * **역할**: ESP32 보드용 아두이노 스케치. INMP441 마이크로부터 I2S 인터페이스로 오디오를 입력받아 16kHz 16bit PCM으로 변환 후 PC로 **UDP 전송**.
  * **제어 인터페이스**: HTTP 웹 서버를 열어 `/open` 및 `/stop` 요청을 수신하고 SG90 서보모터를 회전시켜 물 밸브를 제어함.
* **[realtime_esp32_mic.py](../realtime/realtime_esp32_mic.py)**
  * **역할**: PC에서 실행되는 실시간 수위 감지 및 제어 클라이언트.
  * **동작**: ESP32가 송신하는 UDP 오디오 스트림을 수집하여 Mel-spectrogram을 생성하고, AI 모델 템플릿과의 매칭(Mel Matching)을 통해 현재 수위를 예측하며, 만약 55% 수위에 도달하면 ESP32로 HTTP `/stop` 신호를 자동 전송함. (OpenCV를 통해 수위 및 매칭 정보 실시간 시각화)
* **[realtime_mic.py](../realtime/realtime_mic.py)**
  * **역할**: PC의 내장 마이크를 활용하여 단독으로 수위 감지를 테스트할 수 있는 실시간 프로그램 (동작 메커니즘은 `realtime_esp32_mic.py`와 동일).

### 테스트 및 평가 (Tests & Simulations)
* **[evaluate_noisy_vs_clean.py](../tests_and_simulations/evaluate_noisy_vs_clean.py)**
  * **역할**: 깨끗한 오디오 데이터셋(`sound_of_water_dataset`)과 노이즈가 합성된 데이터셋(`sound_of_water_dataset_noisy`)을 일대일로 매핑하여 수위 매칭 오차(MAE)와 80% 정지 지연 시간(Stop Latency)을 정량 비교 평가하는 프로그램.
* **`check_durations_trace.py` (임시 디버그 스크립트)**
  * **역할**: 데이터셋 음원들의 80% 도달 시점 및 총 길이를 출력하고, 실시간 알고리즘의 상태 변수(RMS, 예측값, 채택값, 연속 확인 횟수, 가드 타임)를 초 단위로 트레이싱하여 정지 실패 원인을 수학적으로 규명함.

### AI 모델 및 데이터셋 관련
* **`models/`**: 학습 및 미세조정(Fine-tuned)된 Wav2Vec 2.0 기반 공기주 길이 예측 모델(`WavelengthWithTime`) 체크포인트 파일 보관.
* **`sound_of_water_dataset/`**: 원본 깨끗한 물소리 WAV 파일 및 메타데이터.
* **`sound_of_water_dataset_noisy/`**: TV 소음(약 +3dB)이 합성된 노이즈 데이터셋.

---

## 2. ⚙️ 핵심 알고리즘 파라미터 (동기화 완료)

시뮬레이션(`evaluate_noisy_vs_clean.py`)과 실제 구동 스크립트(`realtime_*.py`) 간의 물리적 및 알고리즘적 파라미터가 완벽하게 일치되도록 조정되어 있습니다:
* **추론 주기 (`INFERENCE_INTERVAL`)**: `1.0초`
* **수위 상한 임계치 (`FILL_RATIO`)**: `0.55` (실제 정수기와 동일하게 55% 채워짐 기준으로 자동 정지하도록 동기화 완료)
* **물리적 변화 제한 (`MAX_CHANGE`)**: `3.0 cm` (노이즈 방지를 위해 1초간 수위 변화폭을 3cm 이내로 제한, 초과 시 직전 값 홀드)
* **연속 확인 횟수 (`CONFIRM_COUNT_REQUIRED`)**: `2회` (연속으로 2번 수위 임계치 이하로 관측되어야 최종 정지 판정)
* **최초 물 감지 임계치 (`silence_threshold`)**: RMS `0.00075`
* **물 흐름 시작 안전 가드 (`Guard Time`)**: **`1.0초`** (최초 감지 후 1초 초과 경과해야 정지 트리거 작동 가능)

---

## 3. 🛠️ 이번 세션 핵심 성과 및 변경 사항

1. **미정지(No Stop) 현상 수학적 규명 및 해결**
   * **기존 문제**: 길이가 짧은(5~8초 내외) 오디오 파일에서 정지 감지가 작동하지 않는 `미정지` 현상이 빈번히 발생함.
   * **원인**: 기존 물 흐름 안전 가드 시간(`2.0초`)과 연속 확인 횟수(`2회`, 1.0초 간격) 때문에 물이 흐르기 시작한 시점부터 최소 3~4초 이상의 물리적인 지속 시간이 필요했으나, 짧은 음원은 80% 도달 이후 오디오가 조기 종료되어 조건을 만족하지 못함.
   * **해결**: 안전 가드 시간을 기존 **`2.0초`에서 `1.0초`로 단축**하여 짧은 물소리 음원도 정상적으로 정지 트리거가 작동하도록 조치함.
2. **안전 가드 시간 일제 동기화**
   * 수정된 `1.0초` 안전 가드 조건을 실제 운용 프로그램(`realtime_esp32_mic.py`, `realtime_mic.py`)과 시뮬레이터(`evaluate_noisy_vs_clean.py`) 모두에 반영 완료.
3. **콘솔 한글 인코딩 에러(UnicodeEncodeError) 조치**
   * Windows 환경의 PowerShell이나 CMD 콘솔에서 이모지나 특수 문자가 출력될 때 발생하는 `UnicodeEncodeError`를 방지하기 위해, `evaluate_noisy_vs_clean.py` 시작 시점에 `sys.stdout` 인코딩을 UTF-8로 자동 변경하는 동적 코드 반영.
4. **평가 파일 개수 예외 처리 버그 수정**
   * 사용자가 전체 데이터셋 크기(10개)보다 큰 평가 수치(기본값 20 등)를 입력하거나 빈 값을 주어 EOF 에러가 발생해도 프로그램이 튕기지 않고 데이터셋 한도 내로 자동 조정(`capping`)하여 안정적으로 작동하도록 수정.
