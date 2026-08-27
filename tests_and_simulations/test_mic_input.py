"""
test_mic_input.py — Microphone Input & UDP Streaming Diagnostic Script
----------------------------------------------------------------------
Usage:
    python test_mic_input.py
"""

import os
import sys
import time
import socket
import threading
import numpy as np

# Select standard sampling rate (16kHz)
SR = 16000
UDP_PORT = 5005

def draw_volume_bar(rms, max_bar_len=30):
    """Draws a visual text-based volume bar on the console."""
    # Scale RMS value for visibility (typically ranges from 0 to ~0.1 for normal speech)
    scaled = min(int(rms * 300), max_bar_len)
    bar = "#" * scaled + "." * (max_bar_len - scaled)
    return f"RMS: {rms:.5f} [{bar}]"

# ─────────────────────────────────────────────
#  Option 1: Local PC Microphone Test
# ─────────────────────────────────────────────
def test_local_pc_mic():
    try:
        import sounddevice as sd
    except ImportError:
        print("\n[FAIL] Error: 'sounddevice' library is not installed.")
        print("   Please install it by running: pip install sounddevice")
        return

    print("\n" + "="*58)
    print("  [MIC] Testing Local PC Microphone (Ctrl+C to stop)")
    print("="*58)
    
    # List available devices
    print("\nAvailable Audio Input Devices:")
    devices = sd.query_devices()
    has_input = False
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            default_mark = " (Default)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{default_mark} - Channels: {dev['max_input_channels']}, SR: {dev['default_samplerate']}Hz")
            has_input = True
            
    if not has_input:
        print("[FAIL] No input audio devices detected on this PC!")
        return

    print(f"\nOpening microphone stream at {SR}Hz (Mono)...")
    
    running = True
    
    def callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}", file=sys.stderr)
        # Calculate RMS
        audio_data = indata[:, 0].astype(np.float32)
        rms = np.sqrt(np.mean(audio_data ** 2))
        print(f"\r{draw_volume_bar(rms)}", end="", flush=True)

    try:
        stream = sd.InputStream(samplerate=SR, channels=1, callback=callback)
        with stream:
            print("\n[MIC] Listening live! Speak into the mic or pour water to see volume changes...")
            print("Press Enter or Ctrl+C to finish.")
            input()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n[FAIL] Failed to open microphone stream: {e}")
        print("   -> Make sure your microphone is not muted and check your system privacy settings.")
        return
        
    print("\n\n[OK] PC Microphone test finished.")


# ─────────────────────────────────────────────
#  Option 2: ESP32 UDP Microphone Test
# ─────────────────────────────────────────────
def test_esp32_udp_mic():
    print("\n" + "="*58)
    print("  [UDP] Testing ESP32 UDP Audio Stream (Ctrl+C to stop)")
    print("="*58)
    print(f"\nListening for UDP packets on port {UDP_PORT}...")
    print("Please make sure your ESP32 is powered on and streaming audio data to this PC's IP address.")
    
    # Setup UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", UDP_PORT))
        # 3 seconds timeout
        sock.settimeout(3.0)
    except Exception as e:
        print(f"\n[FAIL] Error binding to port {UDP_PORT}: {e}")
        print("   -> This port might be currently in use by another program (e.g. realtime_esp32_mic.py).")
        return

    print("\nWaiting for incoming packets (3 seconds timeout)...")
    
    packets_received = 0
    start_time = time.time()
    
    try:
        # First packet check to confirm link
        try:
            data, addr = sock.recvfrom(4096)
            print(f"[OK] Connection Established! Received packet from ESP32 IP: {addr[0]}:{addr[1]}")
            print("   - Packet Size:", len(data), "bytes")
            print("   - Stream Format: 16-bit PCM (Int16)")
            print("\n[UDP] Displaying Live UDP Audio volume. Pour water near the ESP32 mic!")
            print("Press Ctrl+C to stop.\n")
        except socket.timeout:
            print("\n[FAIL] Timeout: No packets received from ESP32 within 3 seconds.")
            print("\n[INFO] Troubleshooting Checklists:")
            print("  1. PC IP Setting inside ESP32 code: Check if the 'pc_ip' variable in your Arduino script matches this PC's local IP address.")
            print("  2. Shared WiFi Network: Make sure both your PC and ESP32 are connected to the EXACT SAME 2.4GHz WiFi network.")
            print("  3. Windows Firewall block: Try disabling your firewall temporarily or allow incoming UDP port 5005.")
            sock.close()
            return
            
        # Continuous volume print
        while True:
            try:
                data, addr = sock.recvfrom(4096)
                # Convert 16-bit PCM bytes to float numpy array normalized between -1.0 and 1.0
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                if len(audio_np) > 0:
                    rms = np.sqrt(np.mean(audio_np ** 2))
                    print(f"\rIP: {addr[0]} | {draw_volume_bar(rms)}", end="", flush=True)
            except socket.timeout:
                print("\n[WARN] Warning: Packet stream paused (Timeout). Check if ESP32 power was lost.")
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        
    print("\n\n[OK] ESP32 UDP test finished.")


# ─────────────────────────────────────────────
#  Main Menu
# ─────────────────────────────────────────────
def main():
    print("="*60)
    print(" [MIC] SoundOfWater Microphone Diagnostic Tool")
    print("="*60)
    print("  [1] Test Local PC Microphone (sounddevice)")
    print("  [2] Test Remote ESP32 Microphone (UDP Port 5005)")
    print("  [0] Exit")
    print("="*60)
    
    try:
        choice = input("Select an option (0-2): ").strip()
        if choice == '1':
            test_local_pc_mic()
        elif choice == '2':
            test_esp32_udp_mic()
        elif choice == '0' or choice == '':
            print("Exiting diagnostic tool.")
        else:
            print("Invalid option. Exiting.")
    except KeyboardInterrupt:
        print("\nExiting.")

if __name__ == "__main__":
    main()
