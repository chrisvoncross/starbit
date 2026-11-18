#!/usr/bin/env python3
"""
Download Qwen3-8B-FP8 model for H100/H200.

This downloads the FP8 QUANTIZED MODEL for maximum inference speed.
FP8 pre-quantized models are 2-3x faster than on-the-fly quantization!

USAGE:
    python scripts/download_qwen3_8b_fp8.py

SIZE: ~8GB (FP8 quantized)
TIME: ~3-5 minutes on RunPod
"""

import sys
from pathlib import Path
from huggingface_hub import snapshot_download


def download_model():
    """Download Qwen3-8B-FP8 model files."""

    # Model configuration - Official Qwen3-8B-FP8
    model_id = "Qwen/Qwen3-8B-FP8"

    # RunPod: Use /workspace/ path (persistent storage)
    local_dir = Path("/workspace/ghostbyte/llm_models/shared_models/qwen3_8b_fp8")

    print("=" * 80)
    print("Qwen3-8B-FP8 Downloader (H100/H200 WORKER MODEL)")
    print("=" * 80)
    print(f"📦 Model: {model_id}")
    print(f"📁 Destination: {local_dir}")
    print(f"💾 Size: ~8GB (FP8 quantized)")
    print(f"🎯 Target GPU: H100/H200")
    print("")
    print("⏳ Download will take 3-5 minutes on RunPod...")
    print("")
    print("🚀 EXPECTED PERFORMANCE:")
    print("   - Latency: 0.3-0.5s per query (200 tokens)")
    print("   - Throughput: 180-250 tokens/s")
    print("   - 2-3x FASTER than on-the-fly FP8 quantization!")
    print("")

    # Create directory
    local_dir.mkdir(parents=True, exist_ok=True)

    # Download entire model using snapshot_download
    try:
        print(f"📥 Downloading entire model repository...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Copy files instead of symlinks
        )
        print(f"   ✅ Model downloaded successfully!")

    except Exception as e:
        print(f"   ❌ Failed to download model: {e}")
        print("")
        print("Possible solutions:")
        print("1. Check your internet connection")
        print("2. Make sure you have enough disk space (~10GB)")
        print("3. Try again")
        sys.exit(1)

    print("")
    print("=" * 80)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"📁 Model saved to: {local_dir}")
    print("")
    print("🔧 NEXT STEPS:")
    print("")
    print("1. Update llm_config.yaml:")
    print("   Change worker.h200 from 'qwen3_8b' to 'qwen3_8b_fp8'")
    print("")
    print("2. Delete old BF16 model (optional, saves 16GB):")
    print("   rm -rf /workspace/ghostbyte/llm_models/shared_models/qwen3_8b")
    print("")
    print("3. REMOVE --quantization fp8 from start_sglang_server_h100.py")
    print("   (Model is already FP8, no need for on-the-fly quantization!)")
    print("")
    print("4. Start SGLang server:")
    print("   python scripts/start_sglang_server_h100.py")
    print("")
    print("5. Warmup:")
    print("   python scripts/warmup_sglang_server.py")
    print("")
    print("6. Run main:")
    print("   python main.py --mode micro_trade --symbol BTCUSDT")
    print("")
    print("=" * 80)


if __name__ == "__main__":
    download_model()

