# 🚰 Water Dispenser — 물소리 기반 실시간 수위 감지 및 자동 차단 시스템

물 따르는 **소리만으로** 컵의 수위를 실시간 추정하고, 목표 수위에 도달하면
ESP32 서보모터로 밸브를 자동 차단하는 시스템입니다.
카메라·수위센서 없이 마이크 하나로 동작하며, 시각장애인 사용자를 위한
카메라 기반 컵 자동 인식 + 음성 안내 모드를 포함합니다.

```
[물소리] ──► [2D U-Net 디노이저] ──► [Wav2Vec2 공기주 길이 추정] ──► [수위 %]
                                                                      │
                                        [ESP32 서보 밸브] ◄── HTTP /stop
```

---

## 📌 원본 프로젝트 (출처)

이 저장소는 아래 연구의 공개 코드를 **클론하여 확장한 것**입니다.
`sound_of_water/`, `shared/`, `demo/`, `playground.ipynb`,
사전학습 체크포인트는 모두 원저자의 결과물입니다.

> **The Sound of Water: Inferring Physical Properties from Pouring Liquids**
> Piyush Bagad, Makarand Tapaswi, Cees G. M. Snoek, Andrew Zisserman
> [프로젝트 페이지](https://bpiyush.github.io/pouring-water-website/) ·
> [arXiv](https://arxiv.org/abs/2411.11222) ·
> [원본 저장소](https://github.com/bpiyush/SoundOfWater) ·
> [HuggingFace 모델](https://huggingface.co/bpiyush/sound-of-water-models)

원본 라이선스는 [LICENSE](./LICENSE)를 따릅니다. 인용은 이 문서 맨 아래를 참고하세요.

---

## 🆕 이 저장소에서 추가한 것

원본은 오프라인 오디오 파일을 분석하는 연구 코드입니다.
여기에 **실시간 스트리밍 추론 + 하드웨어 제어 + 소음 대응**을 붙였습니다.

1. **실시간 추론 파이프라인** — 1초 버퍼 단위 스트리밍 처리, 컵별 공명 템플릿
   캘리브레이션, Mel 매칭 기반 수위 추정
2. **ESP32 하드웨어 연동** — I2S 마이크 UDP 스트리밍, HTTP 서보 밸브 제어
3. **2D U-Net 디노이저** — 생활 소음 환경에서의 오인식/오버플로우 해결
   (직접 학습, `models/denoiser_best.pth`)
4. **안전 제어 로직** — RMS 침묵 게이트, 물리적 변화 제한, 연속 확인, 안전 가드
5. **카메라 기반 컵 자동 인식** — ResNet-18 임베딩 매칭 + TTS 음성 안내
6. **시뮬레이션 도구** — 녹음 파일로 정지 로직을 재현하고 상태 변수를 추적

---

## 🗂️ 폴더 구조

```
water_dispenser/
├── firmware/                  ESP32 아두이노 스케치
├── realtime/                  실시간 실행 스크립트 (메인)
├── tests_and_simulations/     평가·시뮬레이션·단위 테스트
├── calibration_cache/         컵별 공명 템플릿 (.npz)
├── scripts/                   Windows 배치 스크립트
│
└── 원본 SoundOfWater 코드 (수정하지 않음)
    ├── sound_of_water/        모델 정의 (+ 직접 추가한 denoiser.py)
    ├── demo/                  Gradio 데모 및 모델 로딩 유틸
    ├── shared/                공용 유틸리티
    ├── models/                체크포인트
    ├── media_assets/          데모용 예제 영상
    └── playground.ipynb       원본 분석 노트북
```

모든 스크립트는 **저장소 루트 기준**으로 경로를 계산하므로, 어느 위치에서
실행해도 `calibration_cache/`와 `models/`를 정상적으로 찾습니다.

---

## 📁 직접 작성한 파일 설명

### 펌웨어 — `firmware/`

아두이노 IDE 규칙상 스케치는 파일명과 같은 이름의 폴더 안에 있어야 합니다.

| 파일 | 설명 |
| --- | --- |
| `firmware/esp32_servo_i2s_mic/` | **메인 펌웨어.** INMP441 I2S 마이크를 32bit로 읽어 16kHz 16bit PCM으로 변환 후 PC로 UDP(5005) 전송. 동시에 HTTP 서버를 열어 `/open`, `/stop` 요청으로 SG90 서보 밸브 제어. |
| `firmware/esp32_servo/` | 서보 제어만 하는 최소 버전. 배선·서보 각도 확인용. |

### 실시간 실행 스크립트 — `realtime/`

**7개 모두 ESP32 서보 밸브를 HTTP로 제어합니다.** 차이는 오디오를 어디서
받는지, U-Net 디노이저를 쓰는지, 카메라 인식이 붙는지입니다.

| 파일 | 오디오 입력 | U-Net 디노이저 | 카메라 |
| --- | --- | --- | --- |
| `realtime_esp32_mic.py` | ESP32 UDP (5005) | ✗ | ✗ |
| `realtime_esp32_mic_unet.py` | ESP32 UDP (5005) | ✓ | ✗ |
| `realtime_noesp_mic.py` | 로컬 마이크 | ✗ | ✗ |
| `realtime_noesp_mic_unet.py` | 로컬 마이크 | ✓ | ✗ |
| `realtime_noesp_camera.py` | 로컬 마이크 | ✗ | ✓ |
| `realtime_mic.py` | 로컬 마이크 | ✗ | ✗ |
| `realtime_mic_unet.py` | 로컬 마이크 | ✓ | ✗ |

- `noesp_` 접두사 = **오디오는 노트북 마이크로 받고, 밸브 제어만 ESP32로** 보내는 구성.
  ESP32 마이크 음질이 나쁘거나 UDP가 방화벽에 막힐 때 사용합니다.
- `realtime_mic.py` / `realtime_mic_unet.py`는 `noesp_` 버전의 초기 형태입니다.
  기능은 같지만 ESP32 주소가 파일 중간에 `http://192.168.0.250`으로 하드코딩되어
  있습니다. 새로 쓴다면 `noesp_` 쪽을 쓰세요.
- `realtime_noesp_camera.py`는 화면에서 컵을 마우스로 고를 필요 없이 카메라에
  컵을 비추면 HSV 히스토그램으로 자동 인식하고, 모든 안내를 macOS `say` TTS로
  음성 출력합니다. 시각장애인 사용을 상정한 버전입니다.

> ⚠️ ESP32 IP가 스크립트 상단에 `ESP32_IP = "20.30.88.125"`로 하드코딩되어
> 있습니다. 실행 전에 본인 ESP32의 주소로 바꿔야 합니다
> (`realtime_mic*.py`는 파일 중간의 URL 문자열을 직접 수정).

### 디노이저

| 파일 | 설명 |
| --- | --- |
| `sound_of_water/audio_pitch/denoiser.py` | 경량 2D U-Net + `AudioDenoisingWrapper`. 1초 스펙트로그램 magnitude에 곱할 마스크(0.0~1.0)를 예측하고, 원본 위상을 재사용해 파형으로 복원. |
| `models/denoiser_best.pth` | 학습 완료 가중치 (495KB). **HuggingFace에 없는 자체 학습 결과물**입니다. |

### 카메라 컵 인식

| 파일 | 설명 |
| --- | --- |
| `realtime/test_camera_cup_classification.py` | 오디오/AI 모델을 배제하고 카메라 인식만 검증하는 테스트 프로그램. ResNet-18 임베딩 코사인 유사도로 등록된 컵을 분류. 실행 중 `[c]` 컵 등록, `[+]/[-]` 임계값 조정, `[Space]` 일시정지. |

### 시뮬레이션 · 테스트 — `tests_and_simulations/`

| 파일 | 설명 |
| --- | --- |
| `compare_denoise.py` | DSP(스펙트럼 차감) vs 2D U-Net 디노이즈 결과 비교. |
| `realtime_mic_denoise.py`, `realtime_esp32_mic_denoise.py` | DSP 스펙트럼 차감 방식을 쓰던 초기 실시간 버전. U-Net 채택 전의 비교 대상. |
| `record_and_denoise.py` | 마이크로 녹음한 뒤 디노이저를 통과시켜 전후 WAV를 저장. |
| `simulate_realtime_stream.py` | WAV 파일을 실시간 스트림처럼 1초씩 흘려보내 정지 로직 재현. |
| `simulate_mel_match.py`, `simulate_pitch_matching.py`, `simulate_stop.py` | Mel 매칭 / 피치 매칭 / 정지 트리거 각각의 단위 시뮬레이션. |
| `record_esp32_audio.py` | ESP32 UDP 수신 확인 및 WAV 녹음. **배선 후 첫 점검용.** |
| `test_esp32_servo.py`, `test_mic_input.py`, `test_inference.py` | 서보 / 마이크 입력 / 모델 추론 개별 동작 확인. |
| `debug_chunk.py`, `debug_pitch.py` | 청크 단위 상태 변수(RMS, 예측값, 채택값, 연속 확인 횟수, 가드 타임) 트레이싱. |

### 기타

| 파일 | 설명 |
| --- | --- |
| `Dockerfile`, `docker-compose.yml`, `.dockerignore` | CUDA 12.1 기반 실행 환경. |
| `requirements.txt` | 실행에 필요한 파이썬 패키지 고정 버전. |
| `scripts/setup.bat`, `scripts/run.bat` | Windows용 venv 생성 / 실행 배치 스크립트. 어디서 실행하든 저장소 루트로 이동한 뒤 동작합니다. |
| `calibration_cache/*.npz` | 컵별 공명 템플릿 캐시. 한 번 학습하면 다음부터 즉시 로드. |

---

## 🔌 하드웨어 배선

#### INMP441 I2S 마이크
| 핀 | ESP32 |
| --- | --- |
| VDD | **3.3V** (5V 금지) |
| GND | GND |
| L/R | GND (Left 채널 선택) |
| SD | GPIO 2 (D2) |
| WS | GPIO 15 (D15) |
| SCK | GPIO 4 (D4) |

#### SG90 서보모터
| 핀 | ESP32 |
| --- | --- |
| Signal (주황/흰) | GPIO 13 (D13) |
| VCC (빨강) | 5V 또는 VIN |
| GND (갈색/검정) | GND (마이크 GND와 공통) |

---

## 🚀 실행 방법

### 1. 환경 준비

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Docker를 쓴다면 (CUDA 12.1 기반, Gradio 데모용):

```bash
docker compose up demo          # http://localhost:7860
docker compose run --rm shell   # 실시간 스크립트용 셸
```

모델 가중치는 이미지에 포함하지 않고 `./models`를 마운트합니다.
`realtime/` 스크립트는 마이크·카메라·시리얼 접근이 필요해서 컨테이너에서
바로 돌지 않습니다 (Docker Desktop은 macOS/Windows에서 오디오 입력 자체를
전달하지 못함). 해당 device 옵션은 `docker-compose.yml`에 주석으로 있습니다.

### 2. 모델 가중치 내려받기

메인 체크포인트(약 360MB)는 저장소에 포함되어 있지 않습니다.

```bash
huggingface-cli download bpiyush/sound-of-water-models --local-dir ./models
```

`models/denoiser_best.pth`(디노이저)는 저장소에 함께 들어 있습니다.

### 3. ESP32 펌웨어 업로드

1. Arduino IDE로 [`firmware/esp32_servo_i2s_mic/`](./firmware/esp32_servo_i2s_mic/)를 엽니다.
2. 상단 설정값을 본인 환경에 맞게 수정합니다.
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   const char* pc_ip = "192.168.0.206";  // 파이썬을 실행할 PC의 로컬 IP
   ```
3. 업로드 후 시리얼 모니터를 **115200 baud**로 엽니다.
   조용할 때 `Mic Volume`이 30~100 사이로 나오고, 마이크에 바람을 불면
   값이 크게 흔들리면 정상입니다.

### 4. 연결 확인

```bash
python tests_and_simulations/record_esp32_audio.py   # UDP 수신 확인 + WAV 녹음
python tests_and_simulations/test_esp32_servo.py     # 서보 개폐 확인
```

### 5. 실시간 실행

```bash
python realtime/realtime_esp32_mic_unet.py    # 권장: ESP32 마이크 + 디노이저
python realtime/realtime_noesp_mic_unet.py    # 노트북 마이크 + ESP32 밸브 제어
python realtime/realtime_noesp_camera.py      # 카메라 컵 자동 인식 + 음성 안내
```

실행 흐름:

1. **새 컵 학습** — 메뉴에서 `0`을 선택하고 이름을 입력한 뒤, 빈 컵에 가득 찰
   때까지 물을 따릅니다. 공명 템플릿이 `calibration_cache/`에 저장됩니다.
2. **저장된 컵 선택** — 다음 실행부터는 목록에서 컵을 고르면 즉시 시작합니다.
3. **물 따르기** — 밸브가 열리고(`/open`) 수위를 추적하다가, 목표 임계치에
   도달하면 정지 신호(`/stop`)를 보냅니다.

---

## ⚙️ 핵심 제어 파라미터

`realtime_*.py`와 시뮬레이터가 동일한 값을 공유합니다.

| 파라미터 | 값 | 의미 |
| --- | --- | --- |
| `INFERENCE_INTERVAL` | 1.0 초 | 추론 주기 |
| `FILL_RATIO` | 0.55 | 정지 트리거 임계치. 통신 RTT·모터 구동 시간·배관 잔류 유량을 감안해 **물리적 80~85% 시점에 맞도록 선제 차단**하는 값입니다. |
| `MAX_CHANGE` | 3.0 cm | 1초당 허용 수위 변화폭. 공기주 높이는 1초에 급변할 수 없으므로 초과 시 오인식으로 보고 직전 값을 홀드. |
| `CONFIRM_COUNT_REQUIRED` | 2 회 | 임계치를 연속 2회 만족해야 정지 확정 |
| `silence_threshold` | RMS 0.00075 | 침묵 게이트. `max(0.0003, noise_rms * 1.5)`로 동적 계산되며 이하 입력은 AI 연산을 우회 |
| Guard Time | 1.0 초 | 최초 물 감지 후 이 시간이 지나야 정지 트리거 작동 |

> 가드 타임은 원래 2.0초였습니다. 5~8초짜리 짧은 음원에서 "물 감지 → 가드
> 2초 → 연속 확인 2회(2초)"로 최소 4초가 필요해 오디오가 먼저 끝나버리는
> **미정지 현상**이 발생했고, 1.0초로 줄여 해결했습니다.

---

## 📊 디노이징 성능 비교

개발 당시 20건 평가 결과입니다. 평가에 쓰인 소음 합성 데이터셋과 그 생성
도구는 저장소에서 제외했으므로, 아래는 재현 가능한 벤치마크가 아니라 채택
근거 기록입니다.

| 시나리오 | 수위 MAE | 80% 정지 지연 | 자동 정지 실패 |
| --- | --- | --- | --- |
| Clean (무소음) | 0.85 cm | +0.3 초 | 0 건 |
| Noisy (필터 없음) | 4.20 cm | +2.8 초 | **6 건 (오버플로우)** |
| DSP Denoised (스펙트럼 차감) | 2.10 cm | +1.5 초 | 2 건 |
| **AI Denoised (2D U-Net)** | **1.25 cm** | **+0.8 초** | **0 건** |

초기에는 연산량이 적은 DSP 스펙트럼 차감법(`prop_decrease=0.90`, 기동 시 2초간
노이즈 플로어 수집)을 먼저 시도했으나, 정지 실패가 남아 2D U-Net으로 전환했습니다.
U-Net은 1초 오디오 처리에 약 **5ms**만 소요되어 실시간성에 영향을 주지 않고,
터미널에 `[Denoise Latency: X.Xms]`로 실측값이 계속 출력됩니다.

---

## 📜 인용

원본 연구를 사용하는 경우 아래를 인용해 주세요.

```bibtex
@article{sound_of_water_bagad,
  title={The {S}ound of {W}ater: {I}nferring {P}hysical {P}roperties from {P}ouring {L}iquids},
  author={Bagad, Piyush and Tapaswi, Makarand and Snoek, Cees G. M. and Zisserman, Andrew},
  journal={arXiv},
  year={2024}
}
```

## 🙏 감사

수위 추정 모델과 데이터셋 전부는 원저자
[Piyush Bagad 외](https://github.com/bpiyush/SoundOfWater)의 연구 결과입니다.
이 저장소는 그 위에 실시간 처리, 하드웨어 제어, 소음 대응을 얹은 것입니다.
