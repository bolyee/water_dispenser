# syntax=docker/dockerfile:1

# Base image ships Python 3.10 + PyTorch 2.1.0 built against CUDA 12.1.
# The torch/torchaudio/torchvision pins in requirements.txt are deliberately
# skipped below so this CUDA build is never replaced by a PyPI wheel.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# ---------------------------------------------------------------------------
# System dependencies
#   libsndfile1    -> soundfile / librosa    (wav I/O)
#   libportaudio2  -> sounddevice            (live mic capture)
#   ffmpeg         -> librosa, eva-decord    (mp3/mp4 decoding)
#   libgl1 + glib  -> opencv-python          (cv2 import)
#   curl           -> container HEALTHCHECK
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libportaudio2 \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

# Dependencies are installed before the source copy so edits to the code do
# not invalidate the (slow) pip layer.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    grep -vE '^(torch|torchaudio|torchvision)==' requirements.txt > /tmp/requirements-docker.txt \
    && pip install -r /tmp/requirements-docker.txt \
    && python -c "import torch; assert torch.version.cuda, 'CUDA build of torch was overwritten'"

COPY . .

# Model weights are excluded from the image (see .dockerignore) — they are
# ~360 MB and belong on a mounted volume. Create the mount points up front so
# they are owned by the unprivileged user.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/models /app/.cache/huggingface /app/calibration_cache \
    && chown -R app:app /app

USER app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:7860/ || exit 1

# demo/app.py resolves its example videos as "../media_assets/...", so it has
# to run from inside demo/. PYTHONPATH=/app keeps "from demo.util import *"
# working. Override working_dir to /app for a plain shell.
WORKDIR /app/demo

# The Gradio demo is the only entrypoint that runs headless; the realtime/
# scripts need host microphone/serial access (see README).
CMD ["python", "app.py"]
