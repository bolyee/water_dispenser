# Docker

## What is in the image

Base: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` (Python 3.10, CUDA 12.1).

The `torch`, `torchaudio`, and `torchvision` pins in `requirements.txt` are
stripped during the build so the base image's CUDA-linked PyTorch is not
replaced by a PyPI wheel. The build asserts `torch.version.cuda` is still set
afterwards, so a regression fails the build instead of silently shipping a
CPU-only image.

System libraries installed for the Python stack:

| Package | Needed by |
| --- | --- |
| `libsndfile1` | `soundfile`, `librosa` |
| `libportaudio2` | `sounddevice` (live mic capture) |
| `ffmpeg` | `librosa` mp3 decode, `eva-decord` mp4 decode |
| `libgl1`, `libglib2.0-0` | `opencv-python` |

The container runs as the unprivileged user `app` (uid 1000).

## Model weights are not baked in

`models/*.pth` is excluded via `.dockerignore` — the main checkpoint is
~360 MB and would land in every image layer. Download it to the host once and
mount it:

```bash
pip install huggingface_hub
huggingface-cli download bpiyush/sound-of-water-models --local-dir ./models
```

`demo/util.py` and `sound_of_water/audio_pitch/model.py` resolve the checkpoint
relative to the repo root, so `-v ./models:/app/models` is all that is needed.

`models/denoiser_best.pth` (the U-Net denoiser used by the `*_unet.py` scripts)
is a locally trained checkpoint and is **not** on HuggingFace — copy it into
`./models/` by hand if you need those scripts.

## Running

### Gradio demo

```bash
docker compose up demo          # http://localhost:7860
```

`GRADIO_SERVER_NAME=0.0.0.0` is set in the image; without it Gradio binds to
loopback inside the container and the published port goes nowhere.

Note that `demo/app.py` calls `demo.launch(..., share=True)`, which opens a
public `gradio.live` tunnel on every start. Change it to `share=False` if you
only want the demo on localhost.

### Shell for the realtime scripts

```bash
docker compose run --rm shell
python realtime/realtime_noesp_mic.py
```

This service bind-mounts the repo at `/app`, so host edits take effect without
a rebuild.

### Without compose

```bash
docker build -t sound-of-water .
docker run --rm --gpus all -p 7860:7860 -v "$PWD/models:/app/models:ro" sound-of-water
```

Drop `--gpus all` to run on CPU.

## Hardware access caveats

The `realtime/` scripts need host hardware that Docker does not forward by
default. The relevant lines are commented out in `docker-compose.yml`:

- **Microphone** — `--device /dev/snd` works on Linux hosts only. Docker
  Desktop on macOS and Windows cannot pass through audio input at all; run
  those scripts natively there.
- **ESP32 serial** — `--device /dev/ttyUSB0` (adjust for your port).
- **ESP32 UDP stream** — `firmware/esp32_servo_i2s_mic/` sends UDP to a hardcoded PC
  IP on port 5005. Use `network_mode: host` so the container sees that address,
  and update `pc_ip` in the sketch to match.
- **Camera** — `realtime/test_camera_cup_classification.py` and
  `realtime/realtime_noesp_camera.py` need `--device /dev/video0`, Linux only. They also
  call `cv2.imshow`, which needs an X11 socket mount
  (`-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix`) or they will crash on a
  headless host.

## Build context size

`.dockerignore` excludes `venv/` (1.3 GB), `models/*.pth` (360 MB), and the
notebook, keeping the context around 20 MB. `media_assets/` is deliberately
kept — `demo/app.py` loads its `.mp4` files as the demo examples.
