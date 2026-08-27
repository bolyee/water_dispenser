---
marp: true
theme: default
paginate: true
backgroundColor: #ffffff
color: #1e293b
style: |
  @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&family=Pretendard:wght@300;400;500;600;700&display=swap');

  * { box-sizing: border-box; }

  section {
    font-family: 'Pretendard', 'Noto Sans KR', sans-serif;
    background: #ffffff;
    color: #1e293b;
    padding: 50px 60px;
    font-size: 19px;
    line-height: 1.6;
  }

  /* ── 타이틀 슬라이드 ── */
  section.title {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 60px 80px;
    background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
    color: #ffffff;
  }
  section.title h1 {
    font-size: 2.2em;
    font-weight: 700;
    color: #ffffff;
    border: none;
    margin-bottom: 12px;
    line-height: 1.3;
  }
  section.title h2 {
    font-size: 1.15em;
    font-weight: 400;
    color: #93c5fd;
    border: none;
    margin-bottom: 40px;
  }
  section.title p {
    color: #cbd5e1;
    font-size: 0.85em;
    margin: 4px 0;
  }

  /* ── 일반 슬라이드 ── */
  h1 {
    font-size: 1.5em;
    font-weight: 700;
    color: #1e3a8a;
    border-bottom: 3px solid #eff6ff;
    padding-bottom: 8px;
    margin-bottom: 20px;
  }
  h2 {
    font-size: 1.1em;
    font-weight: 600;
    color: #2563eb;
    margin-top: 16px;
    margin-bottom: 8px;
  }
  h3 {
    font-size: 0.95em;
    font-weight: 600;
    color: #475569;
  }

  /* ── 강조 ── */
  strong { color: #1d4ed8; font-weight: 700; }
  em { color: #2563eb; font-style: normal; font-weight: 600; }

  /* ── 코드 ── */
  code {
    background: #f1f5f9;
    color: #0f172a;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85em;
    font-family: 'Fira Code', 'Consolas', monospace;
  }
  pre {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 5px solid #2563eb;
    border-radius: 6px;
    padding: 14px 18px;
    font-size: 0.78em;
    line-height: 1.5;
  }
  pre code {
    background: none;
    color: #334155;
    padding: 0;
  }

  /* ── 표 ── */
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85em;
    margin-top: 12px;
  }
  th {
    background: #1e3a8a;
    color: #ffffff;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
  }
  td {
    padding: 8px 12px;
    border-bottom: 1px solid #e2e8f0;
    color: #334155;
  }
  tr:nth-child(even) td { background: #f8fafc; }

  /* ── 리스트 ── */
  ul { padding-left: 22px; }
  ul li { margin-bottom: 5px; }
  ul li::marker { color: #2563eb; }

  /* ── 페이지 번호 ── */
  section::after {
    color: #94a3b8;
    font-size: 0.75em;
  }

  /* ── 하이라이트 박스 ── */
  blockquote {
    border-left: 4px solid #3b82f6;
    background: #eff6ff;
    margin: 12px 0;
    padding: 10px 16px;
    border-radius: 0 6px 6px 0;
    color: #1e40af;
    font-size: 0.9em;
  }

  /* ── 섹션 구분자 ── */
  section.section-divider {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
    color: #ffffff;
    text-align: center;
  }
  section.section-divider h1 {
    color: #ffffff;
    border-color: #60a5fa;
    font-size: 1.8em;
    margin-bottom: 10px;
  }
  section.section-divider p {
    color: #93c5fd;
    font-size: 1em;
  }

  /* ── Layout Utilities ── */
  .cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
    margin-top: 12px;
  }
  .box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 18px;
  }
  .box-blue {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 12px 18px;
  }
---

<!-- _class: title -->

# 물소리 기반 수위 모니터링 시스템의<br>소음 극복을 위한 디노이징 기법 비교 및 검증
## DSP 기법의 한계 규명 및 딥러닝 기반 실시간 잡음 제거 모델 최종 채택 과정

<br>

**발표자:** 이현중 (Hyeonjoong Lee)
**소속:** 음향 및 인공지능 연구실 (Acoustics & AI Lab)

---

# 발표 목차

1. **연구 배경 및 문제 정의** (Acoustic Water Level Estimation & Noise Challenge)
2. **소음 극복을 위한 개발 흐름 (초기 DSP 시도 ──► 최종 AI 채택)**
3. **방법론 1 [초기 시도]: DSP 기반 실시간 노이즈 감쇄** (Spectral Subtraction)
4. **방법론 2 [최종 채택]: 딥러닝 기반 초경량 2D U-Net 디노이저** (Spectrogram-based Masking)
5. **실시간 제어 게이트 및 물리 예외 처리 알고리즘**
6. **DSP vs 딥러닝 디노이저 정량적 성능 대조 평가 (AI 채택 당위성)**
7. **요약 및 향후 연구 방향**

---

<!-- _class: section-divider -->

# 1. 연구 배경 및 문제 정의
## 음향 기반 수위 인식의 한계와 실세계 소음 문제

---

# 1. 연구 배경 및 문제 정의

## 음향 기반 수위 측정 시스템 (Acoustic Water Level Estimation)
- 물이 용기에 채워짐에 따라 발생하는 **공기 기둥 공명(Resonance of Air Column)** 현상 분석.
- 주파수가 점진적으로 상승하는 물리적 특징(Pitch)을 기학습 템플릿과 매칭하여 수위 판정.
- **물리적 현상**: 물 높이 상승 $\rightarrow$ 공기 기둥 길이 감소 $\rightarrow$ 공명 주파수(Hz) 상승.

## 핵심 당면 과제: 환경 소음 (Ambient Noise)
- **생활 소음 영향**: 정수기 펌프 동작음, 주방 잡음, TV 소음, 주변 대화음 등 혼입.


---

<!-- _class: section-divider -->

# 2. 소음 극복을 위한 개발 흐름
## 연산량이 적은 DSP의 선제 검토와 한계, 그리고 딥러닝(AI) 도입

---

# 2. 소음 극복을 위한 개발 흐름

```
[입력 (Raw Audio)]
       │
       ├──► [초기 시도: DSP Denoise] ──► 스펙트럼 차감법 (Spectral Subtraction)
       │                                  ⚠️ 한계: 비정상 일상 소음(TV 등) 환경에서 성능 붕괴
       │
       └──► [최종 채택: AI Denoise]  ──► 2D U-Net Denoising Model 설계/도입
                                          ✅ 성과: 시변 소음 마스킹 극복 및 안정적인 자동 정지 복구
```

- **1단계: DSP 기반 실시간 노이즈 감쇄 (초기 시도)**
  - 가벼운 연산량으로 엣지 환경 탑재를 목표로 선제 적용했으나, 시간에 따라 주파수가 요동치는 생활 잡음(TV, 음악)에서 수위 감지 성능이 크게 붕괴됨.
- **2단계: AI 기반 딥러닝 디노이징 (최종 채택)**
  - DSP의 극명한 한계를 인식하고, 스펙트로그램 마스킹 방식의 초경량 2D U-Net 도입을 통해 실세계 환경 잡음 극복 및 안전성 확보.

---

<!-- _class: section-divider -->

# 3. 방법론 1 [초기 시도]: DSP 기반 감쇄 기법
## 스펙트럼 차감법(Spectral Subtraction) 구현 및 한계점

---

# 3. 방법론 1 [초기 시도]: DSP 기반 실시간 감쇄

## 실시간 스펙트럼 차감법 (Spectral Subtraction)

- **소음 수집 및 차감**: 기동 시 **최초 2초간 배경 잡음(Noise Floor)**을 수음한 뒤, 실시간 신호의 주파수 스펙트럼에서 해당 소음 크기만큼 차감 (`prop_decrease=0.90`).
- **연구 의의**: 정수기 모터나 팬 소음 같은 **정적인 화이트 노이즈(Stationary Noise)** 환경에서는 효율적으로 동적 잡음 보정(Auto-calibration)을 수행함.
- **⚠️ 치명적인 한계 및 실패 원인 (청음 평가 결과)**:
  - **비정상 소음 대처 한계**: TV 소리나 대화음처럼 시변하는 비정상 소음(Non-stationary Noise)에 대한 필터링 한계 봉착.
  - **물소리 신호의 과도한 유실 (청음 확인)**: 정제된 결과 음원을 **직접 청음하여 분석한 결과**, 잡음과 함께 **타겟 신호인 물소리(공명음 및 피치 성분)가 과도하게 억제되어 유실**됨을 확인.

---

<!-- _class: section-divider -->

# 4. 방법론 2 [최종 채택]: 딥러닝 기반 2D U-Net 디노이저
## Spectrogram Masking 기반 고성능 잡음 제거 모델

---

# 4. 2D U-Net 디노이저: 모델 아키텍처

- **Lightweight 2D U-Net**: 1초 단위 Spectrogram Magnitude에 곱해지는 **마스크(0.0 ~ 1.0)** 예측.
- **구조적 장점**: Skip Connection을 통해 고해상도 주파수 특징 보존 및 오디오 위상 왜곡 최소화.

<div class="cols">
<div class="box">

### 🔽 Encoder (하향 경로)
- **Enc 1**: Conv2d(1 $\rightarrow$ 16) $\times$ 2 + MaxPool2d (2x Down)
- **Enc 2**: Conv2d(16 $\rightarrow$ 32) $\times$ 2 + MaxPool2d (2x Down)
- **Bottleneck**: Conv2d(32 $\rightarrow$ 64) $\times$ 2

</div>
<div class="box">

### 🔼 Decoder (상향 경로)
- **Dec 2**: ConvTranspose2d(64 $\rightarrow$ 32) + Concatenation + Conv2d(64 $\rightarrow$ 32) $\times$ 2
- **Dec 1**: ConvTranspose2d(32 $\rightarrow$ 16) + Concatenation + Conv2d(32 $\rightarrow$ 16) $\times$ 2
- **Output**: Conv2d(16 $\rightarrow$ 1) + Sigmoid (Mask)

</div>
</div>

---

# 4. 2D U-Net 디노이저: Audio Wrapper 설계

- **AudioDenoisingWrapper**: 1D raw 오디오 파형을 입력받아 내부에서 전처리-추론-복원을 원스톱으로 처리하는 파이토치 모듈 설계.

```python
class AudioDenoisingWrapper(nn.Module):
    def forward(self, x):
        # 1. STFT 변환 (Complex Spectrogram 추출)
        stft_res = torch.stft(x_flat, n_fft=512, hop_length=160, win_length=512, return_complex=True)
        magnitude, phase = torch.abs(stft_res), torch.angle(stft_res)
        
        # 2. U-Net 추론 및 Multiplicative Mask 적용
        mask = self.unet(magnitude.unsqueeze(1))
        clean_mag = magnitude * mask.squeeze(1)
        
        # 3. ISTFT 변환 (Raw Waveform 복원)
        clean_stft = torch.polar(clean_mag, phase)
        clean_waveform = torch.istft(clean_stft, n_fft=512, hop_length=160, win_length=512)
        return clean_waveform
```

---

# 4. 2D U-Net 디노이저: 학습 및 손실 함수

- **학습 데이터**: 805개 물소리 음원(Clean)과 TV 소음(+3dB)을 무작위 합성한 노이즈 데이터셋 일대일 매칭.
- **손실 함수 (Hybrid Loss)**: 파형 수준의 오차와 주파수 스펙트럼 매칭 오차를 결합.

$$\mathcal{L}_{total} = 100 \times \mathcal{L}_{wave\_mse} + \mathcal{L}_{spectral}$$

- **Multi-resolution Spectral Loss**:
  $$\mathcal{L}_{spectral} = \| S_{clean} - S_{pred} \|_{1} + \| \log(S_{clean} + \epsilon) - \log(S_{pred} + \epsilon) \|_{1}$$
  *(스펙트럼의 디테일과 로그 스케일의 저주파 에너지 분포를 동시에 복원)*
- **학습 최적화**: Adam Optimizer (LR=1e-3), Validation Loss 기준 최적 가중치 저장.

---

<!-- _class: section-divider -->

# 5. 실시간 제어 게이트 및 물리 예외 처리
## 신뢰할 수 있는 제어를 위한 이중 안전장치 설계 (공통 적용)

---

# 5. 실시간 제어 게이트 및 예외 처리

## 1. 동적 RMS 침묵 게이트 (Dynamic RMS Silence Gate)
- **목적**: 대기 상태의 배경 하드웨어 잡음에 의한 수위 오판정을 완전 배제.
- **알고리즘**: 대기 RMS 레벨의 1.5배(`measured_noise * 1.5`)를 기준으로 그 이하 소리는 **묵음** 처리하여 AI 추론을 스킵.

```python
# RMS가 침묵 게이트 기준 이하인 경우 AI 연산을 우회하여 오동작 방지
if chunk_rms < (measured_noise * 1.5):
    continue
```

## 2. 물리적 수위 변화 제한 (`MAX_CHANGE = 3.0cm`)
- 물소리 변화에 따른 공기 기둥 높이는 1초에 물리적으로 급격히 변할 수 없음.
- 1초 전 예측값과 현재 예측값의 편차가 **3.0cm를 초과하여 변동하면 오인식 노이즈**로 판정.
- 급격히 튀는 예측 값을 홀드(`accepted_pred` 복사유지)하여 밸브 오동작 예방.

---

# 5. 실시간 제어 게이트 및 예외 처리

## 3. 연속 도달 확인 및 안전 가드 (Safety Guard)
- **임계값 연속 도달 (`CONFIRM_COUNT_REQUIRED = 2`)**: 목표 수위 임계치(예: 남은공간 45% 이하)를 **연속 2회** 만족해야 정지 신호 트리거.
- **유량 지연 보상 (`FILL_RATIO = 0.55`)**: 통신 지연(RTT), 모터 구동 시간, 배관 잔류 유량을 계산해 실제 80% 수위 도달 전 55% 시점에 선제 차단 명령 송신.

<br>

> [!NOTE]
> 본 제어 게이트와 예외 처리는 시스템 전체에 공통 적용되어, 최종적인 수위 판정과 서보모터 정지 판정의 강인성을 완성합니다.

---

<!-- _class: section-divider -->

# 6. DSP vs 딥러닝 디노이저 정량적 성능 대조 평가
## AI 디노이저 채택의 당위성 증명 (미학습 TV 소음 환경)

---

# 6. DSP vs 딥러닝 디노이저 성능 대조 평가

> 아래 지표는 미학습 일상 노이즈(TV Noise 2)가 합성된 환경에서 각 기법의 성능을 대조한 정량 평가 결과입니다.

### 📊 종합 정량 평가 비교표 (20개 샘플 무작위 검증)

| 평가 시나리오 | 수위 감지 평균 절대 오차 (MAE) | 80% 자동 정지 지연 시간 (Stop Latency) | 자동 정지 실패 건수 (Fail Count / 20건) |
| :--- | :---: | :---: | :---: |
| **Clean** (이상적 무소음 상태) | **0.85 cm** | **+ 0.3 초** | **0 건** (안전 정지) |
| **Noisy** (필터링 없는 소음 상태) | **4.20 cm** | **+ 2.8 초** | **6 건** (오버플로우) |
| **DSP Denoised** (스펙트럼 차감) | **2.10 cm** | **+ 1.5 초** | **2 건** (정지 실패 잔존) |
| **AI Denoised** (2D U-Net 필터) | **1.25 cm** | **+ 0.8 초** | **0 건** (완벽 복구) |

- **DSP 한계 실증**: 스펙트럼 차감법(DSP)은 일부 오차를 개선했으나, 여전히 **2건의 오버플로우 사고(Stop Fail)**가 발생하여 시각장애인용 정수기 안전 규격에 미달함.
- **AI Denoised의 비교 우위**: 2D U-Net 도입을 통해 오차(MAE)를 **1.25cm 수준으로 최소화**하고, 넘침 사고를 **0건으로 완벽 차단**하여 **최종 솔루션으로 채택**함.

---

<!-- _class: section-divider -->

# 7. 요약 및 향후 연구 방향
## 본 연구의 의의와 개선 과제

---

# 7. 요약 및 향후 연구 방향

## 핵심 요약 (Key Takeaways)
1. **소음 취약성 해결**: 생활 소음 환경에서 오버플로우를 초래하던 공명 주파수 왜곡 문제를 디노이징 모듈을 통해 극복.
2. **DSP 한계 돌파 및 AI 채택**: 전통적 DSP(스펙트럼 차감) 기법의 비정상 생활 잡음 억제 한계를 정량적으로 규명하고, 딥러닝 2D U-Net 필터의 비교 우위 및 최종 적용 당위성 확보.
3. **물리 제어 결합**: 딥러닝 디노이저와 `MAX_CHANGE`, **RMS 침묵 게이트** 등의 물리 제어 예외처리 장치를 결합하여 수위 예측 성능의 견고성을 극대화.

## 향후 연구 계획 (Future Works)
- **디바이스 탑재 및 레이턴시 문제 검증**: 딥러닝 모델을 실제 하드웨어 디바이스(ESP32 등 온디바이스 환경)에 탑재 시 발생하는 연산 및 통신 지연(Latency) 문제 분석.
- **실시간 정상 작동 및 차단 신뢰성 평가**: 실제 디바이스 배포 환경에서 연산 지연 하에 수위 예측 및 자동 정지 신호 전송이 물리적으로 오류 없이 정상 작동하는지 여부 검증.
