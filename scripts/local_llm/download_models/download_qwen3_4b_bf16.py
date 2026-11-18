#!/usr/bin/env python3
"""
Download Qwen3-4B-Instruct (BF16) model.

This downloads the FULL PRECISION (BF16) version, NOT the FP8 quantized version.
BF16 works with SGLang on all GPUs, including Blackwell (RTX 5060 Ti).

Uses hf_hub_download() for FAST downloads (like the working FP8 script).
"""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def download_model():
    """Download Qwen3-4B-Instruct BF16 model files."""

    # Model configuration
    model_id = "Qwen/Qwen3-4B-Instruct-2507"

    # Auto-detect Windows vs WSL2
    if sys.platform == "win32":
        # Windows: Use D:/ path
        local_dir = Path("D:/python/ghostbyte/llm_models/shared_models/qwen3_4b_bf16")
    else:
        # Linux/WSL2: Use /mnt/d/ path
        local_dir = Path("/mnt/d/python/ghostbyte/llm_models/shared_models/qwen3_4b_bf16")

    # Files to download (BF16 model has 2 safetensors shards + config files)
    files_to_download = [
        "model-00001-of-00002.safetensors",  # ~4GB
        "model-00002-of-00002.safetensors",  # ~4GB
        "model.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]

    print("=" * 80)
    print("Qwen3-4B-Instruct BF16 Downloader (FAST MODE)")
    print("=" * 80)
    print(f"📦 Model: {model_id}")
    print(f"📁 Destination: {local_dir}")
    print(f"💾 Size: ~8GB (BF16 full precision)")
    print(f"🎯 Compatible: ALL GPUs (including Blackwell RTX 5060 Ti)")
    print(f"📄 Files: {len(files_to_download)}")
    print("")
    print("⏳ Download will take 2-5 minutes with 1Gbit connection...")
    print("")

    # Create directory
    local_dir.mkdir(parents=True, exist_ok=True)

    # Download each file
    print("🚀 Starting download...")
    print("")

    for i, filename in enumerate(files_to_download, 1):
        target_file = local_dir / filename

        # Skip if already exists
        if target_file.exists():
            print(f"[{i}/{len(files_to_download)}] ✅ Already exists: {filename}")
            continue

        try:
            print(f"[{i}/{len(files_to_download)}] ⬇️  Downloading: {filename}")
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            print(f"[{i}/{len(files_to_download)}] ✅ Downloaded: {filename}")
            print("")

        except Exception as e:
            print(f"[{i}/{len(files_to_download)}] ❌ Failed: {filename}")
            print(f"    Error: {e}")
            print("")
            print("Possible solutions:")
            print("1. Check your internet connection")
            print("2. Make sure you have enough disk space (~8GB)")
            print("3. Try again (already downloaded files will be skipped)")
            sys.exit(1)

    print("")
    print("=" * 80)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"📁 Model saved to: {local_dir}")
    print("")
    print("🚀 Next step: Start SGLang server with:")
    print("   wsl -d Ubuntu bash -c \"cd /mnt/d/python/ghostbyte && source venv/bin/activate && source ~/.bashrc && python scripts/start_sglang_server.py\"")
    print("")


if __name__ == "__main__":
    download_model()

