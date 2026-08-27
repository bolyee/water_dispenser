"""
simulate_mel_match.py
─────────────────────────────────────────────
Mel 스펙트로그램 윈도우 매칭 방식 검증 스크립트

1) 영상 전체 → AI 분석 → l_preds(정답) + mel_windows(매칭용) 계산·저장
2) 같은 영상을 1초 윈도우로 슬라이드 (실시간 마이크와 동일한 방식)
3) 각 윈도우를 캘리브레이션 mel_windows와 코사인 유사도 비교
4) AI 정답 vs Mel 예측을 OpenCV 화면에 나란히 표시

→ Mel 매칭이 실시간 마이크에서도 동작할지 사전 검증
"""

import os, sys
import numpy as np
import cv2
import librosa
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from demo.util import load_model, load_audio_tensor, get_model_output, visualise_args
import shared.utils as su

# ──────────────────────────────────────────────
FILL_RATIO    = 0.80
VIDEO_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media_assets")
CACHE_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "calibration_cache")
SR            = 16000
MEL_WINDOW_S  = 1.0    # 실시간 윈도우 크기 (초)
MEL_HOP_S     = 0.25   # 캘리브레이션 윈도우 간격 (초)
N_MELS        = 64
FMAX          = 8000
# ──────────────────────────────────────────────


def list_videos():
    exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    return sorted([
        os.path.join(VIDEO_DIR, f)
        for f in os.listdir(VIDEO_DIR)
        if os.path.splitext(f)[1].lower() in exts
    ]) if os.path.exists(VIDEO_DIR) else []


def select_video():
    videos = list_videos()
    if not videos:
        print("[ERROR] media_assets 폴더에 영상이 없습니다."); sys.exit(1)
    print("\n" + "="*56)
    for i, v in enumerate(videos):
        print(f"  [{i+1}] {os.path.basename(v)}")
    print("="*56)
    while True:
        try:
            ans = int(input(f"  영상 선택 (1~{len(videos)}): "))
            if 1 <= ans <= len(videos):
                return videos[ans - 1]
        except (ValueError, KeyboardInterrupt):
            pass


def get_cache_path(video_path):
    os.makedirs(CACHE_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(CACHE_DIR, f"{base}_mel_calibration.npz")


def calibrate(video_path, model):
    """영상 전체를 AI로 분석하고 mel_windows를 계산·저장합니다."""
    print("\n[1] 영상 전체 AI 분석 중...")
    audio_tensor = load_audio_tensor(video_path)
    audio_np, _ = librosa.load(video_path, sr=SR, mono=True)

    with torch.no_grad():
        _, y_audio = get_model_output(audio_tensor, model)
        wavelengths = y_audio @ torch.linspace(
            0, visualise_args['w_max'], visualise_args['n_bins']
        ).to(y_audio.device)
        l_preds = su.physics.estimate_length_of_air_column(wavelengths).numpy()

    l_max = float(np.max(l_preds))
    l_min = float(np.mean(l_preds[-10:]))
    n_frames = len(l_preds)
    timestamps_eval = librosa.frames_to_time(
        np.arange(n_frames),
        sr=visualise_args['sr'],
        n_fft=visualise_args['n_fft'],
        hop_length=visualise_args['hop_length'],
    )
    print(f"  AI 정답 범위: {l_max:.2f}cm(빈) ~ {l_min:.2f}cm(꽉 참)")

    # Mel 윈도우 계산
    print("[2] Mel 윈도우 계산 중...")
    win_samples = int(MEL_WINDOW_S * SR)
    hop_samples = int(MEL_HOP_S    * SR)
    mel_windows_list = []
    lpred_per_window = []

    for start in range(0, len(audio_np) - win_samples + 1, hop_samples):
        chunk  = audio_np[start : start + win_samples]
        mel    = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=N_MELS, fmax=FMAX)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        feat   = mel_db.mean(axis=1)                       # [N_MELS]
        t_c    = (start + win_samples / 2) / SR
        idx    = int(np.argmin(np.abs(timestamps_eval - t_c)))
        mel_windows_list.append(feat)
        lpred_per_window.append(l_preds[idx])

    mel_arr  = np.array(mel_windows_list, dtype=np.float32)
    lpred_arr= np.array(lpred_per_window, dtype=np.float32)

    # 코사인 유사도용 정규화
    norms = np.linalg.norm(mel_arr, axis=1, keepdims=True) + 1e-8
    mel_norm = mel_arr / norms

    cache_path = get_cache_path(video_path)
    np.savez(cache_path,
             timestamps_eval=timestamps_eval,
             l_pred=l_preds, l_max=l_max, l_min=l_min,
             audio_np=audio_np,
             mel_windows_norm=mel_norm,
             lpred_per_window=lpred_arr)
    print(f"  Mel 윈도우 {len(mel_arr)}개 저장 완료: {cache_path}\n")
    return timestamps_eval, l_preds, l_max, audio_np, mel_norm, lpred_arr


def simulate(video_path, timestamps_eval, l_preds, l_max, audio_np,
             mel_norm, lpred_per_window):
    """영상을 재생하면서 AI 정답과 Mel 매칭 예측을 동시에 표시합니다."""
    import decord
    decord.bridge.set_bridge('native')

    threshold = l_max * (1.0 - FILL_RATIO)
    win_samples = int(MEL_WINDOW_S * SR)

    vr  = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    LOG_INTERVAL_S = 0.5   # 콘솔 로그 출력 간격

    print(f"[3] 영상 재생 시작  (fps={fps:.1f} | 임계값={threshold:.2f}cm)")
    print(f"    컵 전체 높이: {l_max:.2f}cm | Mel 윈도우 수: {len(mel_norm)}개")
    print()
    print(f"{'t(s)':>6} | {'AI pred':>9} | {'Mel pred':>9} | {'오차':>7} | {'similarity':>10} | {'채워짐%':>7}")
    print("-" * 65)

    ai_stopped  = False
    mel_stopped = False
    last_frame  = None
    last_log_t  = -LOG_INTERVAL_S

    for frame_idx in range(len(vr)):
        t_video = frame_idx / fps

        # ── AI 정답 l_pred
        idx_ai  = int(np.argmin(np.abs(timestamps_eval - t_video)))
        l_ai    = l_preds[idx_ai]

        # ── Mel 매칭 예측
        audio_start = int(max(t_video - MEL_WINDOW_S, 0) * SR)
        audio_end   = int(t_video * SR)
        chunk       = audio_np[audio_start:audio_end]
        l_mel = None
        sim_val = 0.0
        if len(chunk) >= win_samples // 2:
            if len(chunk) < win_samples:
                chunk = np.pad(chunk, (win_samples - len(chunk), 0))
            mel    = librosa.feature.melspectrogram(y=chunk[-win_samples:], sr=SR, n_mels=N_MELS, fmax=FMAX)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            feat   = mel_db.mean(axis=1)
            norm   = np.linalg.norm(feat) + 1e-8
            feat_n = feat / norm
            sims   = mel_norm @ feat_n
            best   = int(np.argmax(sims))
            sim_val = float(sims[best])
            l_mel  = float(lpred_per_window[best])

        # ── 콘솔 로그 (LOG_INTERVAL_S 마다)
        if t_video - last_log_t >= LOG_INTERVAL_S:
            last_log_t = t_video
            if l_mel is not None:
                err = abs(l_ai - l_mel)
                fill_pct = min(max((l_max - l_mel) / max(l_max,1e-6), 0.0), 1.0) * 100
                err_mark = "✅" if err < 1.0 else ("⚠️ " if err < 2.5 else "❌")
                print(f"{t_video:>6.1f}s | {l_ai:>7.2f}cm | {l_mel:>7.2f}cm | "
                      f"{err:>5.2f}cm {err_mark} | {sim_val:>10.4f} | {fill_pct:>6.1f}%")
            else:
                print(f"{t_video:>6.1f}s | {l_ai:>7.2f}cm | {'대기중':>9} | {'':>7} | {'':>10} |")

        # ── 영상 프레임
        if not (ai_stopped and mel_stopped):
            frame = vr[frame_idx].asnumpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            last_frame = frame.copy()
        else:
            frame = last_frame.copy() if last_frame is not None else np.zeros((480, 640, 3), np.uint8)

        h, w = frame.shape[:2]
        panel = np.zeros((h, 420, 3), np.uint8)

        # ── 패널: AI 정답
        ff_ai = min(max((l_max - l_ai) / max(l_max, 1e-6), 0.0), 1.0)
        cv2.putText(panel, "AI (Ground Truth)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.putText(panel, f"{l_ai:.2f} cm  ({ff_ai*100:.0f}%)", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        bar_w_full = 380
        bx, by, bh = 10, 80, 30
        cv2.rectangle(panel, (bx, by), (bx+bar_w_full, by+bh), (60,60,60), -1)
        cv2.rectangle(panel, (bx, by), (bx+int(bar_w_full*ff_ai), by+bh), (0,200,80), -1)
        cv2.rectangle(panel, (bx, by), (bx+bar_w_full, by+bh), (200,200,200), 2)
        # 80% 임계선
        thresh_x = bx + int(bar_w_full * FILL_RATIO)
        cv2.line(panel, (thresh_x, by-4), (thresh_x, by+bh+4), (0,255,255), 2)

        # ── 패널: Mel 예측
        cv2.putText(panel, "Mel Match (Prediction)", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        if l_mel is not None:
            ff_mel = min(max((l_max - l_mel) / max(l_max, 1e-6), 0.0), 1.0)
            cv2.putText(panel, f"{l_mel:.2f} cm  ({ff_mel*100:.0f}%)", (10, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,220,255), 2)
            cv2.putText(panel, f"sim={sim_val:.4f}", (10, 215),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)
            by2 = 225
            cv2.rectangle(panel, (bx, by2), (bx+bar_w_full, by2+bh), (60,60,60), -1)
            cv2.rectangle(panel, (bx, by2), (bx+int(bar_w_full*ff_mel), by2+bh), (0,180,255), -1)
            cv2.rectangle(panel, (bx, by2), (bx+bar_w_full, by2+bh), (200,200,200), 2)
            cv2.line(panel, (thresh_x, by2-4), (thresh_x, by2+bh+4), (0,255,255), 2)

            # ── 오차 표시
            err = abs(l_ai - l_mel)
            col = (0,255,0) if err < 1.0 else ((0,165,255) if err < 2.5 else (0,0,255))
            cv2.putText(panel, f"Error: {err:.2f} cm", (10, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        else:
            cv2.putText(panel, "오디오 대기 중...", (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,100,100), 1)

        # ── STOP 판정 & 출력
        if not ai_stopped and l_ai <= threshold:
            ai_stopped = True
            t_ai_stop = t_video
            print(f"\n{'─'*65}")
            print(f"🛑 [AI STOP]  t={t_video:.1f}s | {l_ai:.2f}cm ≤ {threshold:.2f}cm  (80% 도달)")
        if l_mel is not None and not mel_stopped and l_mel <= threshold:
            mel_stopped = True
            t_mel_stop = t_video
            print(f"🛑 [Mel STOP] t={t_video:.1f}s | {l_mel:.2f}cm ≤ {threshold:.2f}cm  (80% 도달)")
            if ai_stopped:
                diff = t_mel_stop - t_ai_stop
                print(f"   → AI 대비 Mel 지연: {diff:+.1f}s")

        if ai_stopped:
            cv2.putText(panel, "AI STOP!", (130, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)
        if mel_stopped:
            cv2.putText(panel, "MEL STOP!", (110, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,100,255), 3)

        cv2.putText(panel, f"t = {t_video:.2f}s", (10, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)
        cv2.putText(panel, f"80% thr = {threshold:.2f}cm", (10, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 1)

        canvas = np.hstack([frame, panel])
        cv2.imshow("Mel Match Test (AI vs Mel)", canvas)

        if cv2.waitKey(max(1, int(1000/fps))) & 0xFF == ord('q'):
            break

    print(f"{'─'*65}")
    print("재생 완료.")
    cv2.destroyAllWindows()


def main():
    print("="*56)
    print("  Mel 스펙트로그램 매칭 검증 (영상 기반)")
    print("="*56)

    video_path = select_video()
    print(f"\n선택: {os.path.basename(video_path)}")

    cache_path = get_cache_path(video_path)
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        if 'mel_windows_norm' in data:
            print(f"\n캐시 발견! 재사용하시겠습니까? [Y/n]: ", end="")
            ans = input().strip().lower()
            if ans not in ('n', 'no'):
                print("[모델 로딩 중...] (AI 재분석 없음)")
                simulate(video_path,
                         data['timestamps_eval'], data['l_pred'],
                         float(data['l_max']), data['audio_np'],
                         data['mel_windows_norm'], data['lpred_per_window'])
                return

    print("\n[모델 로딩 중...]")
    model = load_model()
    t_eval, l_preds, l_max, audio_np, mel_norm, lpred_win = calibrate(video_path, model)
    simulate(video_path, t_eval, l_preds, l_max, audio_np, mel_norm, lpred_win)


if __name__ == "__main__":
    main()
