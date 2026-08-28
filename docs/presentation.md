---
marp: true
theme: default
paginate: true
backgroundColor: #ffffff
color: #1a1a2e
style: |
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');

  * { box-sizing: border-box; }

  section {
    font-family: 'Noto Sans KR', 'Segoe UI', sans-serif;
    background: #ffffff;
    color: #1a1a2e;
    padding: 52px 64px;
    font-size: 18px;
    line-height: 1.7;
  }

  /* ── 타이틀 슬라이드 ── */
  section.title {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 64px 80px;
    background: #ffffff;
    border-left: 8px solid #2563eb;
  }
  section.title h1 {
    font-size: 2em;
    font-weight: 700;
    color: #1a1a2e;
    border: none;
    margin-bottom: 8px;
    line-height: 1.3;
  }
  section.title h2 {
    font-size: 1.15em;
    font-weight: 400;
    color: #4b5563;
    border: none;
    margin-bottom: 32px;
  }
  section.title p {
    color: #6b7280;
    font-size: 0.9em;
  }

  /* ── 일반 슬라이드 ── */
  h1 {
    font-size: 1.5em;
    font-weight: 700;
    color: #1e3a8a;
    border-bottom: 2px solid #dbeafe;
    padding-bottom: 10px;
    margin-bottom: 24px;
  }
  h2 {
    font-size: 1.1em;
    font-weight: 600;
    color: #2563eb;
    margin-top: 20px;
    margin-bottom: 8px;
  }
  h3 {
    font-size: 0.95em;
    font-weight: 600;
    color: #374151;
  }

  /* ── 강조 ── */
  strong { color: #1e3a8a; }
  em { color: #2563eb; font-style: normal; font-weight: 500; }

  /* ── 코드 ── */
  code {
    background: #f0f4ff;
    color: #1d4ed8;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.88em;
    font-family: 'Consolas', 'D2Coding', monospace;
  }
  pre {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #2563eb;
    border-radius: 6px;
    padding: 16px 20px;
    font-size: 0.82em;
    line-height: 1.6;
  }
  pre code {
    background: none;
    color: #1e293b;
    padding: 0;
  }

  /* ── 표 ── */
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9em;
    margin-top: 12px;
  }
  th {
    background: #1e3a8a;
    color: #ffffff;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
  }
  td {
    padding: 9px 14px;
    border-bottom: 1px solid #e5e7eb;
    color: #374151;
  }
  tr:nth-child(even) td { background: #f8fafc; }

  /* ── 리스트 ── */
  ul { padding-left: 20px; }
  ul li { margin-bottom: 6px; }
  ul li::marker { color: #2563eb; }

  /* ── 페이지 번호 ── */
  section::after {
    color: #9ca3af;
    font-size: 0.78em;
  }

  /* ── 박스 ── */
  blockquote {
    border-left: 4px solid #93c5fd;
    background: #eff6ff;
    margin: 16px 0;
    padding: 12px 20px;
    border-radius: 0 6px 6px 0;
    color: #1e40af;
    font-size: 0.92em;
  }

  /* ── 섹션 구분자 ── */
  section.section-divider {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: #1e3a8a;
    color: #ffffff;
    text-align: center;
  }
  section.section-divider h1 {
    color: #ffffff;
    border-color: #3b82f6;
    font-size: 1.8em;
  }
  section.section-divider p {
    color: #bfdbfe;
    font-size: 1em;
  }

  /* ── 2단 레이아웃 ── */
  .cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px;
    margin-top: 16px;
  }
  .box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 20px;
  }
  .box-blue {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 16px 20px;
  }
  .success { color: #16a34a; font-weight: 600; }
  .warn    { color: #d97706; font-weight: 600; }
  .error   { color: #dc2626; font-weight: 600; }
---

<!-- _class: title -->

# 물소리로 수위를 측정하는 AI 시스템
## 시각 장애인을 위한 베리어 프리(Barrier-Free) 정수기 및 실시간 제어 구현

<br>

기반 논문: *"The Sound of Water: Inferring Physical Properties from Pouring Liquids"*
Bagad et al., Oxford / Amsterdam, 2024

---

# 프로젝트 개요

**핵심 아이디어:** 추가 센서 없이 *물소리만으로* 수위를 실시간 추론하고, 목표 수위 도달 시 서보모터 밸브를 자동 차단하여 시각 장애인의 안전한 물 따르기를 지원

<br>

```
[INMP441 마이크]
      │ I2S (32bit)
   [ESP32]  ─── UDP WiFi ───>  [PC / Python]
      │                              │
      │                       wav2vec2 AI 모델
      │                       수위 비율 추론
      │                              │
      └──────── HTTP /stop ──────────┘
   [서보모터 밸브 차단]
```

<br>

> 물기둥 높이 변화 → 공기 기둥 공명 주파수 변화 → 피치로 수위 역산

---

# 작업 전체 흐름

| 단계 | 내용 | 결과 |
|------|------|------|
| 0 | 기획 배경: 시각 장애인을 위한 베리어 프리 정수기 | 💡 기획 |
| 1 | 데이터셋 다운로드 (1.4 GB) | ✅ 완료 |
| 2 | 하드웨어 세부 통합 배선 (마이크/모터) | ✅ 완료 |
| 3 | 8kHz 수음 대역폭 제약 분석 | ✅ 완료 |
| 4 | 동적 RMS 침묵 게이트 설계 | ✅ 완료 |
| 5 | 유량 지연 보상 (FILL_RATIO 0.55) | ✅ 완료 |
| 6 | 실전 동작 성공 및 시연 영상 | 🎬 완료 |
| 7 | 무선 통신 지연(RTT) 사전 진단 시스템 | 📡 완료 |

---

<!-- _class: section-divider -->

# 0단계
## 기획 배경: 시각 장애인을 위한 베리어 프리(Barrier-Free) 정수기

---

# 0단계: 기획 배경 및 필요성

## 시각 장애인의 일상 속 보이지 않는 위험
* **컵 넘침 사고**: 시각 장애인이 수위를 눈으로 확인할 수 없어 컵이 넘치거나 뜨거운 물에 화상을 입는 안전사고가 빈번히 발생합니다.
* **기존 솔루션의 한계**: 
  - 컵 내부에 센서를 장착하는 경보기 등은 세척의 번거로움과 접촉에 따른 위생 문제가 존재합니다.
  - 전용 컵이 필요하거나 센서를 매번 직접 세팅해야 하는 번거로움이 있습니다.

## 비접촉식 AI 수위 제어 솔루션 (Barrier-Free)
* 어떠한 컵이나 용기에도 제한받지 않고, 오직 **물 따르는 소리(공기 기둥 공명음)**만을 분석하여 컵 수위를 실시간 예측합니다.
* 시각 장애인이 안심하고 안전하게 물을 따를 수 있도록 돕는 **베리어 프리(Barrier-Free) 자동 밸브 제어 정수 시스템**을 제안합니다.

---

<!-- _class: section-divider -->

# 1단계
## 데이터셋 확보

---

# 1단계: 데이터셋 확보

## 문제
Hugging Face 익명 IP 접근 → **Rate Limit** 차단으로 다운로드 실패

## 해결
액세스 토큰 인증으로 제한 해제

```python
from huggingface_hub import login, snapshot_download

login(token="hf_...")
snapshot_download(
    repo_id="bpiyush/sound-of-water",
    repo_type="dataset",
    local_dir="sound_of_water_dataset"
)
```

## 결과
**805개 영상, 1.4 GB** — 물 따르기 영상 + 물리적 속성 어노테이션 100% 수신 완료

---

<!-- _class: section-divider -->

# 2단계
## 하드웨어 세부 통합 배선

---

# 2단계: 하드웨어 세부 배선 및 핀 매핑

ESP32와 INMP441 I2S 마이크 및 SG90 서보모터의 물리적 핀 연결

### 1. INMP441 I2S 디지털 마이크 배선
| INMP441 핀 | ESP32 연결 핀 | 기능 | 설명 |
|:---:|:---:|:---:|---|
| **VDD** | 3.3V | 전원 공급 | 마이크 동작 전원 |
| **GND** | GND | 접지 | 공통 접지(Common Ground) |
| **L/R** | GND | 채널 선택 | GND 접지로 Left(좌) 채널 고정 |
| **SCK** | **GPIO 4** | I2S 비트 클럭 (BCLK) | 오디오 샘플 전송용 클럭 |
| **WS** | **GPIO 15** | I2S 워드 선택 (LRCK) | 좌/우 채널 구분 동기화 |
| **SD** | **GPIO 2** | I2S 직렬 데이터 (DOUT) | 디지털 PCM 오디오 신호 라인 |
---

### 2. SG90 서보모터 배선
| SG90 선 색상 | ESP32 연결 핀 | 기능 | 설명 |
|:---:|:---:|:---:|---|
| **갈색/흑색 (GND)** | GND | 접지 | **마이크 GND와 공유 (공통 접지 필수)** |
| **적색 (VCC)** | 5V (VIN) | 전원 공급 | 서보모터 동작 전원 |
| **황색/등색 (Signal)** | **GPIO 13** | 제어 신호 | PWM 각도 제어 신호 선 |

>**공통 접지(Common GND)**: 모터 구동 시 발생하는 역기전력 및 전기 노이즈를 방지하고 제어 신호 전위차를 일치시키기 위해 마이크와 서보모터의 GND는 반드시 하나로 묶여서 ESP32 GND로 연결되어야 합니다.

---

<!-- _class: section-divider -->

# 3단계
## 8kHz 수음 대역폭 제약

---

# 3단계: 8kHz 수음 대역폭 제약 분석

## 하드웨어 및 전송 제약
* **수음 주파수 제한**: ESP32의 실시간 처리 능력 및 무선 UDP 오디오 데이터 전송(16kHz, 16-bit PCM)의 대역폭 최적화 문제로 인해, 실시간 오디오 입력의 유효 분석 주파수는 **8kHz 대역폭 (Nyquist 주파수 8kHz)**으로 실질적으로 제한됩니다.

## 타당성 검토 및 최적화
* **주요 물소리 대역**: 컵에 물이 차오름에 따라 기둥의 공명이 증가하며 주파수가 변동(피치 상승)하는 주 대역은 **1kHz ~ 4kHz**에 고르게 분산되어 있습니다.
* **학습 및 추론 모델 일치**: 8kHz 상한선 내에서 충분히 수위 판단이 가능하므로, 오디오 처리부와 Mel Spectrogram 생성 필터의 최대 주파수를 `fmax = 8000` (8kHz)으로 정교하게 매핑하여 고주파 간섭 노이즈를 제거하고 연산 효율을 높였습니다.

```python
# 8kHz 나이퀴스트 제한에 맞춘 Mel Spectrogram 필터링
mel = librosa.feature.melspectrogram(
    y=chunk, sr=16000, n_mels=64, fmax=8000
)
```

---

<!-- _class: section-divider -->

# 4단계
## Python 실시간 제어 및 버그 수정

---

# 4단계: 버그 — RMS 기반 동적 침묵 게이트

## 문제: 마이크 교체로 인한 배경 노이즈 상승 및 오인식
* 테스트 장비 변경 및 마이크(하드웨어) 교체 과정에서 **배경 잡음 레벨(RMS)이 대폭 상승**
* 물을 따르기 전 대기 상태인데도 플로팅 전압이나 주변 정적 노이즈를 AI가 **"물소리(수위 차오름)"**로 오인식하여 밸브가 먼저 닫히는 오작동 발생

## 해결: 동적 RMS 침묵 게이트 (Silence Gate)
* 마이크 하드웨어 변경(감도, 노이즈 플로어)에 유연하게 대응하기 위해, 프로그램 시작 시 2초간 주변 환경 노이즈의 RMS 평균값(`measured_noise`)을 동적 측정
* 측정된 기준치의 1.5배(`measured_noise * 1.5`) 미만 신호는 **묵음**으로 간주하여 AI 추론을 완벽 차단 함으로써 하드웨어 차이에 상관없이 견고하게 보정함

```python
# 1. 주변 소음 측정 (시작 시 2초간 데시벨 수집)
ok, measured_noise = check_mic()
SILENCE_GATE = measured_noise * 1.5

# 2. 실시간 루프 내 제어
while True:
    audio_chunk = receive_udp_packet()
    chunk_rms = np.sqrt(np.mean(audio_chunk ** 2))
    
    if chunk_rms < SILENCE_GATE:
        continue  # 무음 상태면 AI 연산을 건너뛰어 오동작 방지
```

---

<!-- _class: section-divider -->

# 5단계
## 유량 지연 보상

---

# 5단계: 유량 지연 보상 및 FILL_RATIO 튜닝

## 물리적 지연과 넘침 현상
* **지연 원인**: AI 분석 처리(1초 간격), HTTP 네트워크 정지 명령 전송 지연, 서보모터 작동 속도, 밸브 차단 후 배관 내 잔류 유량의 관성 유입.
* **현상**: AI가 70% 충전을 타겟팅했을 때 물리적 지연에 의해 실제 컵 수위는 **95% 이상으로 오버플로우 위험**이 존재함.

## 해결: 목표 비율 하향 조정 (FILL_RATIO = 0.55)
* 노이즈 방어력을 훼손하는 판단 주기(Interval)나 횟수(Confirm Count)를 타협하는 대신, **제어 목표 비율을 0.55 (55%)로 선제 적용**하여 최종 물리적 도달점을 제어.

| 설정 값 | 이전 값 | 조정 값 | 실제 정지 결과 |
|:---:|:---:|:---:|:---:|
| `FILL_RATIO` | 0.70 (70%) | **0.55 (55%)** | **80% ~ 85% (안전 차단)** |

> [!TIP]
> 임계값 연속 일치 횟수(2회) 및 분석 주기(1초) 필터 속성은 원본을 유지하여 오인식 방어력은 완벽하게 보존하였습니다.

---

<!-- _class: section-divider -->

# 6단계
## 실전 동작 및 시연 영상

---

# 6단계: 실전 동작 통합 테스트 성공

## 시스템 엔드투엔드 시연



> 추가적인 물리 수위 센서 없이 오직 **물 따르는 소리(Pitch)**만을 분석하여 밸브를 제어하는 데 성공했습니다.

---

---

<!-- _class: section-divider -->

# 7단계
## 무선 통신 지연(RTT) 사전 진단

---

# 7단계: 무선 통신 지연(RTT) 사전 진단 시스템

## 무선 환경(WiFi)의 변동성 문제
* **레이턴시 가변성:** 무선 네트워크 혼잡도나 공유기 상태에 따라 PC와 ESP32 간의 신호 전달 지연 시간(RTT)이 가변적으로 늘어날 수 있음.
* **오버플로우 위험:** 지연 시간이 증가하면 AI가 "정지" 판단을 내려도 실제 ESP32에 차단 신호(`/stop`)가 도달하기 전에 물이 컵 밖으로 넘치게 됨.

## 해결: 사전 핑퐁(Ping-Pong) 테스트 시스템 도입
* **초경량 응답 추가:** ESP32 웹서버에 즉각 `"pong"`을 회신하는 `/ping` 라우트 구축.
* **지연 자가 진단:** 시연 시작 시 PC에서 5회 핑 요청을 쏴 왕복 시간 측정 및 신뢰성 평가.

```python
# RTT 평균에 따른 연결 신뢰 등급 판정
if avg_rtt < 50.0:
    status_str = "🟢 Excellent (안전)"
elif avg_rtt < 150.0:
    status_str = "🟡 Good (양호 - 반응 미세 지연 가능)"
else:
    status_str = "🔴 Danger (경고 - 지연에 의한 오버플로우 위험)"
```

---

# 최종 시스템 구성 및 상태 요약

| 구성 요소 | 상태 | 상세 구현 및 세부 사항 |
|---|:---:|---|
| **WiFi UDP 통신** | ✅ | ESP32 `192.168.0.250` ↔ PC `192.168.0.206` 오디오 스트리밍 완료 |
| **수음 대역폭** | ✅ | Nyquist 8kHz 대역 내에서 16kHz PCM 오디오 데이터 전송 |
| **침묵 게이트** | ✅ | 동적 RMS 노이즈 측정 기법 도입 (`measured_noise * 1.5`)으로 오작동 차단 |
| **AI 추론 루프** | ✅ | wav2vec2 실시간 수위 비례 추론 (1초 간격 분석, 2회 연속 감지) |
| **서보 제어** | ✅ | HTTP GET `/open` 및 `/stop` 라우트를 통한 밸브 물리 차단 |
| **지연 보상** | ✅ | 목표 비율 `FILL_RATIO = 0.55` 조정으로 실제 80~85%에서 완벽 자동 정지 |
| **통신 지연 진단** | ✅ | `/ping` 핑퐁 통신을 통한 RTT 사전 측정 및 연결 안정성 등급 가이드 |

---

<!-- _class: title -->

# 감사합니다

<br>

**기반 논문**
*The Sound of Water: Inferring Physical Properties from Pouring Liquids*
Piyush Bagad · Makarand Tapaswi · Cees G.M. Snoek · Andrew Zisserman, 2024

