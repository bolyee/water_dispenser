# <img src="./media_assets/pouring-water-logo5.png" alt="Logo" width="40">  The Sound of Water: Inferring Physical Properties from Pouring Liquids

[Piyush Bagad](https://bpiyush.github.io/), [Makarand Tapaswi](https://makarandtapaswi.github.io/), [Cees G. M. Snoek](https://www.ceessnoek.info/), [Andrew Zisserman](https://www.robots.ox.ac.uk/~az/)

<p align="center">
  <a href="https://arxiv.org/abs/2411.11222" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-Paper-red" alt="arXiv">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a target="_blank" href="https://colab.research.google.com/github/bpiyush/SoundOfWater/blob/main/playground.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/spaces/bpiyush/SoundOfWater" target="_blank">
    <img src="https://img.shields.io/badge/Gradio-Demo-orange" alt="Gradio">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://huggingface.co/bpiyush/sound-of-water-models" target="_blank">
    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md-dark.svg" alt="Huggingface">
  </a>
  <a href="https://huggingface.co/datasets/bpiyush/sound-of-water" target="_blank">
    <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md-dark.svg" alt="Huggingface">
  </a>
</p>

<!-- Add a teaser image. -->
<p align="center">
  <img src="./media_assets/pitch_on_spectrogram-compressed.gif" alt="Teaser" width="100%">
</p>

*Key insight*: As water is poured, the fundamental frequency that we hear changes predictably over time as a function of physical properties (e.g., container dimensions).


**TL;DR**: We present a method to infer physical properties of liquids from *just* the sound of pouring. We show in theory how *pitch* can be used to derive various physical properties such as container height, flow rate, etc. Then, we train a pitch detection network (`wav2vec2`) using simulated and real data. The resulting model can predict the physical properties of pouring liquids with high accuracy. The latent representations learned also encode information about liquid mass and container shape.


## 📅 Updates

## 📑 Table of Contents

- [  The Sound of Water: Inferring Physical Properties from Pouring Liquids](#--the-sound-of-water-inferring-physical-properties-from-pouring-liquids)
  - [📅 Updates](#-updates)
  - [📑 Table of Contents](#-table-of-contents)
  - [✨ Highlights](#-highlights)
  - [📂 Dataset](#-dataset)
  - [🤖 Models](#-models)
  - [🎮 Playground](#-playground)
  - [📊 Results](#-results)
  - [📜 Citation](#-citation)
  - [🙏 Acknowledgements](#-acknowledgements)


## ✨ Highlights

1. We train a `wav2vec2` model to estimate the pitch of pouring water. We use supervision from simulated data and fine-tune on real data using visual co-supervision.
2. We show physical property estimation from pitch. For example, in estimating the height of the container, we achieve a mean absolute error of 2.2 cm, in radius estimation, 1.6 cm and in estimating length of air column, 0.6 cm.
3. We show strong generalisation to other datasets (e.g., [Wilson et al.](https://gamma.cs.unc.edu/PSNN/)) and some videos from YouTube.
4. We also show that the learned representations can be regressed to estimate the mass of the liquid and the shape of the container.
5. We release a clean dataset of 805 videos of water pouring with annotations for physical properties.

## 📂 Dataset

We collect a dataset of 805 clean videos that show the action of pouring water in a container. Our dataset spans over 50 unique containers made of 5 different materials, 4 different shapes and with hot and cold water. Some example containers are shown below.

<p align="center">
  <img width="650" alt="image" src="./media_assets/containers-v2.png">
</p>

The dataset is available to download [here]([.](https://huggingface.co/datasets/bpiyush/sound-of-water)).

**Option 1:** Download from `huggingface` 

```py
# Note: this shall take 5-10 mins.

# Optionally, disable progress bars
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = True

from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="bpiyush/sound-of-water",
    repo_type="dataset",
    local_dir="/path/to/dataset/SoundOfWater",
)
```
The total size of the dataset is 1.4 GB.

**Option 2:** Download from VGG servers

Coming soon!


## 🤖 Models

We provide trained models for pitch estimation.

<table style="font-size: 12px;">
<tr>
  <th>File link</th>
  <th>Description</th>
  <th>Size</th>
</tr>
<tr>
  <td> <a href="url">synthetic_pretrained.pth</a> </td>
  <td>Pre-trained on synthetic data &nbsp;&nbsp;&nbsp;</td>
  <td>361M</td>
</tr>
<tr>
  <td> <a href="url">real_finetuned_visual_cosupervision.pth</a> </td>
  <td>Trained with visual co-supervision &nbsp;&nbsp;&nbsp;</td>
  <td>361M</td>
</tr>
</table>

The models are available to download [here](https://huggingface.co/bpiyush/sound-of-water-models).


**Option 1:** Download from `huggingface`. Use this snippet to download the models:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bpiyush/sound-of-water-models",
    local_dir="/path/to/download/",
)
```

**Option 2:** Download from VGG servers

Coming soon!


## 🎮 Playground

We provide a single [notebook](./playground.ipynb) to run the model and visualise results.
We walk you through the following steps:
- Load data
- Demo the physics behind pouring water
- Load and run the model
- Visualise the results

Before running the notebook, be sure to install the required dependencies:

```bash
conda create -n sow python=3.8
conda activate sow

# Install desired torch version
# NOTE: change the version if you are using a different CUDA version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Additional packages
pip install lightning==2.1.2
pip install timm==0.9.10
pip install pandas
pip install decord==0.6.0
pip install librosa==0.10.1
pip install einops==0.7.0
pip install ipywidgets jupyterlab seaborn

# if you find a package is missing, please install it with pip
```

Remember to download the model in the previous step. Then, run the notebook.

You can checkout the demo [here](https://huggingface.co/spaces/bpiyush/SoundOfWater).

## 📊 Results

We show key results in this section. Please refer to the paper for more details.

<p align="center">
<img width="650" alt="image" src="https://github.com/user-attachments/assets/34b0ea66-5ee7-4338-bf04-f0b20f87d0de">

<img width="650" alt="image" src="https://github.com/user-attachments/assets/7193001b-1485-42b5-aa25-feab777e9921">

<img width="650" alt="image" src="https://github.com/user-attachments/assets/9cf2a960-af8b-4df3-b714-6755b5bb90f6">
</p>


<!-- Add a citation -->
## 📜 Citation

If you find this repository useful, please consider giving a star ⭐ and citation

```bibtex
@article{sound_of_water_bagad,
  title={The {S}ound of {W}ater: {I}nferring {P}hysical {P}roperties from {P}ouring {L}iquids},
  author={Bagad, Piyush and Tapaswi, Makarand and Snoek, Cees G. M. and Zisserman, Andrew},
  journal={arXiv},
  year={2024}
}

@inproceedings{
      bagad2024soundofwater,
      title={The {S}ound of {W}ater: {I}nferring {P}hysical {P}roperties from {P}ouring {L}iquids},
      author={Bagad, Piyush and Tapaswi, Makarand and Snoek, Cees G. M. and Zisserman, Andrew},
      booktitle={ICASSP},
      year={2025}
}
```

<!-- Add acknowledgements, license, etc. here. -->
## 🙏 Acknowledgements

* We thank Ashish Thandavan for support with infrastructure and Sindhu
Hegde, Ragav Sachdeva, Jaesung Huh, Vladimir Iashin, Prajwal KR, and Aditya Singh for useful
discussions.
* This research is funded by EPSRC Programme Grant VisualAI EP/T028572/1, and a Royal Society Research Professorship RP / R1 / 191132.

We also want to highlight closely related work that could be of interest:

* [Analyzing Liquid Pouring Sequences via Audio-Visual Neural Networks](https://gamma.cs.unc.edu/PSNN/). IROS (2019).
* [Human sensitivity to acoustic information from vessel filling](https://psycnet.apa.org/record/2000-13210-019). Journal of Experimental Psychology (2020).
* [See the Glass Half Full: Reasoning About Liquid Containers, Their Volume and Content](https://arxiv.org/abs/1701.02718). ICCV (2017).
* [CREPE: A Convolutional Representation for Pitch Estimation](https://arxiv.org/abs/1802.06182). ICASSP (2018).

---

## 🚰 Real-time Automatic Cup-Filler System (ESP32 + I2S Mic + Servo)

This repository has been extended to support a **real-time automatic cup-filler stop system** using an ESP32 microcontroller, an I2S digital microphone (INMP441), and an SG90 servo motor valve. The system streams audio via UDP to a PC running the trained `Wav2Vec2` model to estimate water level in real-time, sending back HTTP stop commands to shut the valve when the cup is filled to the target level.

### 🔌 1. Hardware Connections (Wiring Diagram)

Connect the components to your ESP32 board as follows:

#### INMP441 I2S Microphone
* **VDD** -> ESP32 **3.3V** (do not use 5V)
* **GND** -> ESP32 **GND**
* **L/R** -> ESP32 **GND** (selects Left Channel)
* **SD** -> ESP32 **GPIO 2** (D2)
* **WS** -> ESP32 **GPIO 15** (D15)
* **SCK** -> ESP32 **GPIO 4** (D4)

#### SG90 Servo Motor
* **Signal (Orange/White)** -> ESP32 **GPIO 13** (D13)
* **VCC (Red)** -> ESP32 **5V** (or VIN)
* **GND (Brown/Black)** -> ESP32 **GND** (tied together with Microphone GND)

---

### 💾 2. ESP32 Firmware Setup (`esp32_servo_i2s_mic.ino`)

1. Open the [esp32_servo_i2s_mic.ino](./esp32_servo_i2s_mic.ino) sketch in the Arduino IDE.
2. Configure your WiFi credentials and your PC's IP address:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   const char* pc_ip = "YOUR_PC_IP_ADDRESS"; // e.g., 192.168.0.206
   ```
3. Upload the sketch to the ESP32.
4. Open the Serial Monitor at **115200 baud**. When silent, you should see `Mic Volume` readings around `30-100` and changing when you speak/blow on the mic.

---

### 💻 3. PC Setup & Real-time Execution

Ensure you are in the python virtual environment and run the real-time script:

```bash
# Verify connection & record test
python tests_and_simulations/record_esp32_audio.py

# Run real-time monitoring and control
python realtime_esp32_mic.py
```

#### Running Steps:
1. **Choose New Cup (0)**: Input `0` to calibrate a new cup. Provide a name and record the water pouring sound from empty to full.
2. **Select Cup**: For subsequent runs, select your saved cup index (e.g. `juicy`) from the menu.
3. **Pour Water**: The system will automatically open the valve (`/open`). As you pour, it tracks the water level and fires a stop command (`/stop`) when it reaches the target threshold (`FILL_RATIO`, default 55% to compensate for fill latency, resulting in an 80-85% physical fill level).

---

### 🛠️ 4. Troubleshooting & Core Technical Rationale

* **32-Bit I2S Capture**: The INMP441 requires 32 clock cycles per channel to transmit its 24-bit audio properly. The driver is configured to `I2S_BITS_PER_SAMPLE_32BIT` and right-shifted by 12 (`>> 12`) to yield a 16x volume boost of clean, aligned 16-bit PCM.
* **Windows Firewall Block**: If the python script fails to receive UDP packets, you must allow `python.exe` through the Windows Defender Firewall. Run the following command in an **Administrator PowerShell**:
  ```powershell
  Set-NetFirewallRule -DisplayName "python.exe" -Action Allow
  ```
* **Dynamic Silence Gate**: To prevent premature stops when no water is pouring, the system calculates a dynamic threshold `max(0.0003, noise_rms * 1.5)` using the calibration baseline noise floor. It ignores any inputs quieter than this gate.

---

### 🛡️ 5. Real-time 2D U-Net Denoised Execution (Recommended for Noisy Rooms)

To handle ambient noise (speech, TV, background hubbub), two dedicated real-time scripts integrated with the trained **2D U-Net Denoising model** are provided. The model acts as a pre-processing filter to clean microphone signals before they are processed by the core AI model.

The necessary model weights (`models/denoiser_best.pth` and `models/dsr9mf13_ep100_step12423_real_finetuned_with_cosupervision.pth`) are already packaged in this zip archive, so **no external weights downloads are required**.

#### Step-by-Step Setup for ZIP Recipients:

1. **Extract** the zip package.
2. **Create and Activate a virtual environment**:
   ```bash
   # On Linux/macOS:
   python3 -m venv venv
   source venv/bin/activate
   
   # On Windows (PowerShell):
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. **Install Dependencies**:
   Ensure all libraries (including PyTorch, Librosa, and U-Net helper libraries) are installed:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the U-Net Denoised Real-time Monitors**:
   * **Local PC Microphone**:
     ```bash
     python3 realtime_mic_unet.py
     ```
   * **ESP32 UDP Microphone**:
     ```bash
     python3 realtime_esp32_mic_unet.py
     ```

#### Key Highlights:
- **Instant Processing**: The lightweight U-Net takes only ~5ms to process 1 second of audio, introducing negligible delay while ensuring high level estimation accuracy.
- **Latency Monitoring**: The scripts print `[Denoise Latency: X.Xms]` in the terminal for every 1-second interval so you can verify the execution speed in real-time.
- **Clean Calibration**: When teaching a new cup (`0`), the recording is automatically filtered through the U-Net before extracting the resonance templates, resulting in robust calibrations even in noisy rooms.


