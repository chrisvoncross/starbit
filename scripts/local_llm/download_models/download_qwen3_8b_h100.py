#!/usr/bin/env python3
"""
Download Qwen3-8B (BF16) Base model for H100.

This downloads the BASE MODEL (not Instruct-tuned) in BF16 precision.
Base model is better for complex financial reasoning with proper prompting.

USAGE:
    python scripts/download_qwen3_8b_h100.py

SIZE: ~16GB (BF16 full precision)
TIME: ~5-10 minutes on RunPod
"""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_model():
    """Download Qwen3-8B BF16 model files."""

    # Model configuration
    model_id = "Qwen/Qwen3-8B"

    # RunPod: Use /workspace/ path (persistent storage)
    local_dir = Path("/workspace/ghostbyte/llm_models/shared_models/qwen3_8b")

    # Files to download (8B model has 5 safetensors shards + config files)
    files_to_download = [
        "model-00001-of-00005.safetensors",  # ~4.00GB
        "model-00002-of-00005.safetensors",  # ~3.99GB
        "model-00003-of-00005.safetensors",  # ~3.96GB
        "model-00004-of-00005.safetensors",  # ~3.19GB
        "model-00005-of-00005.safetensors",  # ~1.24GB
        "model.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]

    print("=" * 80)
    print("Qwen3-8B BF16 Downloader (H100 MAIN MODEL)")
    print("=" * 80)
    print(f"📦 Model: {model_id}")
    print(f"📁 Destination: {local_dir}")
    print(f"💾 Size: ~16.4GB (BF16 full precision, 5 shards)")
    print(f"🎯 Target GPU: H100 80GB")
    print(f"📄 Files: {len(files_to_download)}")
    print("")
    print("⏳ Download will take 5-10 minutes on RunPod...")
    print("")
    print("🚀 H100 PERFORMANCE (with Qwen3-1.7B Draft):")
    print("   - Expected latency: 0.5-0.9s per query (200 tokens)")
    print("   - Throughput: 100-180 tokens/s")
    print("   - Concurrent requests: 50+ (with 80GB VRAM)")
    print("   - Speedup vs RTX 4060 Ti: 30-50x faster!")
    print("   - Accept Rate: 60-75% (with 1.7B draft)")
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
            print("2. Make sure you have enough disk space (~17GB)")
            print("3. Try again (already downloaded files will be skipped)")
            sys.exit(1)

    print("")
    print("=" * 80)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"📁 Model saved to: {local_dir}")
    print("")
    print("🚀 Next step: Start SGLang server with:")
    print("   python scripts/start_sglang_server_h100.py")
    print("")
    print("📊 Expected Performance on H100 (with 1.7B Draft):")
    print("   - Latency: 0.5-0.9s per query (vs 28-30s on RTX 4060 Ti)")
    print("   - Throughput: 100-180 tokens/s (vs 4-6 tokens/s)")
    print("   - Accept Rate: 60-75%")
    print("   - Speedup: 30-50x faster!")
    print("")


if __name__ == "__main__":
    download_model()

