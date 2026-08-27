"""
download_dataset.py — SoundOfWater Dataset Downloader Script (Authenticated Version)
----------------------------------------------------------------------------------
Usage:
    .\venv\Scripts\python.exe download_dataset.py
"""

import os
import sys

local_dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sound_of_water_dataset")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("\n[FAIL] Error: 'huggingface_hub' library is not installed.")
    print("   Please install it by running: .\\venv\\Scripts\\pip.exe install huggingface_hub")
    sys.exit(1)

print("\n" + "="*60)
print("  📥 Downloading 1.4GB Dataset from Hugging Face...")
print("="*60)
print("  - Repository: bpiyush/sound-of-water")
print("  - Target Folder:", os.path.abspath(local_dataset_path))
print("="*60)

# Prompt for HF token to bypass anonymous rate limits
print("\n💡 Tip: Large datasets with many files can trigger Hugging Face rate limits on anonymous IPs.")
print("   To bypass this, you can use a free Hugging Face Access Token (read-only).")
print("   Create a free account at huggingface.co, go to Settings -> Access Tokens, and generate a token.")

hf_token = input("\n🔑 Enter Hugging Face Access Token (Press Enter to skip & try anonymously): ").strip()
if not hf_token:
    hf_token = None
    print("   -> Attempting anonymous download...")
else:
    print("   -> Access Token registered. Authenticating download...")

print("\n" + "="*60)
print("  Starting download. This might take 3 to 10 minutes.")
print("  Please do not close this terminal.")
print("="*60 + "\n")

try:
    # snapshot_download fully syncs LFS audio and video files automatically.
    # It supports incremental resuming, so it will not download already finished files!
    snapshot_download(
        repo_id="bpiyush/sound-of-water",
        repo_type="dataset",
        local_dir=local_dataset_path,
        max_workers=4,
        token=hf_token
    )
    
    print("\n" + "="*60)
    print("[OK] SUCCESS! Dataset fully downloaded.")
    print(f"     Path: {os.path.abspath(local_dataset_path)}")
    print("     You should now see 'videos' and 'audios' directories inside.")
    print("="*60)
    
except Exception as e:
    print(f"\n[FAIL] Download failed: {e}")
    print("\n🔍 Troubleshooting:")
    print("  1. If you got a Rate Limit error, please log in/create a free Hugging Face account")
    print("     and paste your Access Token above.")
    print("  2. The downloader supports resuming. Run this script again with a token,")
    print("     and it will pick up exactly where it left off without redownloading finished files!")
