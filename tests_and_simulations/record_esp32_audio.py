"""
record_esp32_audio.py — Record ESP32 UDP Audio Stream and Play it back
---------------------------------------------------------------------
Records incoming UDP audio data from ESP32 for a set duration,
saves it as a WAV file, and plays it back on the PC speakers.
"""

import os
import sys
import socket
import time
import numpy as np
import soundfile as sf
import sounddevice as sd

UDP_PORT = 5005
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
OUTPUT_FILENAME = "esp32_record_test.wav"

def main():
    print("="*60)
    print(" 🎙️  ESP32 WiFi Microphone Recorder & Playback")
    print("="*60)
    print(f"  - UDP Port: {UDP_PORT}")
    print(f"  - Format: 16kHz, Mono, 16-bit PCM")
    print(f"  - Record Duration: {RECORD_SECONDS} seconds")
    print(f"  - Output File: {OUTPUT_FILENAME}")
    print("="*60 + "\n")
    
    # 1. Bind Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", UDP_PORT))
        sock.settimeout(3.0)  # 3 seconds timeout for the first packet
    except Exception as e:
        print(f"[FAIL] Error binding to port {UDP_PORT}: {e}")
        print("       Make sure no other Python scripts (like test_mic_input.py) are running.")
        return

    # 2. Check Connection
    print("Waiting for ESP32 connection (check if ESP32 is powered on)...")
    try:
        data, addr = sock.recvfrom(4096)
        print(f"[OK] Connected! Receiving stream from ESP32 ({addr[0]}:{addr[1]})")
    except socket.timeout:
        print("\n[FAIL] Timeout: No packets received from ESP32 within 3 seconds.")
        print("\n💡 Troubleshooting Checklist:")
        print("  1. Windows Firewall: Did you allow 'python.exe' to bypass the firewall?")
        print("     -> Open Administrator PowerShell and run:")
        print("        Set-NetFirewallRule -DisplayName \"python.exe\" -Action Allow")
        print("  2. PC IP: Ensure 'pc_ip' in esp32_servo_i2s_mic.ino is set to this PC's IP.")
        sock.close()
        return

    # 3. Start Recording
    print(f"\n[RECORD] Starting {RECORD_SECONDS} seconds recording... Speak into the ESP32 mic!")
    
    # Calculate approximate number of packets expected
    # 512 samples per packet at 16000Hz = ~31.25 packets per second
    # For 5 seconds, we expect ~156 packets.
    audio_frames = []
    
    start_time = time.time()
    sock.settimeout(1.0)  # Short timeout during streaming
    
    while time.time() - start_time < RECORD_SECONDS:
        try:
            data, addr = sock.recvfrom(4096)
            # Convert 16-bit signed PCM bytes to float32 normalized array
            chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_frames.append(chunk)
        except socket.timeout:
            print("[WARN] Warning: Packet drop/timeout during recording.")
            
    sock.close()
    print("[RECORD] Finished recording!")

    if len(audio_frames) == 0:
        print("[FAIL] No audio frames were captured.")
        return

    # Concatenate all chunks
    audio_data = np.concatenate(audio_frames)
    
    # 4. Save to WAV
    try:
        sf.write(OUTPUT_FILENAME, audio_data, SAMPLE_RATE)
        print(f"[OK] Saved audio to: {os.path.abspath(OUTPUT_FILENAME)}")
    except Exception as e:
        print(f"[FAIL] Error saving WAV file: {e}")
        return

    # 5. Playback
    print(f"\n[PLAYBACK] Playing back the recorded audio on your PC speakers...")
    try:
        sd.play(audio_data, SAMPLE_RATE)
        sd.wait()  # Wait until playback is finished
        print("[OK] Playback finished!")
    except Exception as e:
        print(f"[FAIL] Error during audio playback: {e}")

if __name__ == "__main__":
    main()
