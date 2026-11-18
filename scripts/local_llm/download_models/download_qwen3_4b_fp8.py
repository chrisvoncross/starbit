#!/usr/bin/env python3
"""
Download Qwen3-4B (BF16) Base model for H100 as DRAFT MODEL.

This downloads the BASE MODEL (not Instruct-tuned) in BF16 precision.
Base model is better for speculative decoding with Qwen3-8B.

USAGE:
    python scripts/download_qwen3_4b_h100.py

SIZE: ~8GB (BF16 full precision, 3 shards)
TIME: ~3-5 minutes on RunPod
"""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_model():
    """Download Qwen3-4B BF16 model files."""

    # Model configuration
    model_id = "Qwen/Qwen3-4B"

    # RunPod: Use /workspace/ path (persistent storage)
    local_dir = Path("/workspace/ghostbyte/llm_models/shared_models/qwen3_4b")

    # Files to download (4B model has 3 safetensors shards + config files)
    files_to_download = [
        "model-00001-of-00003.safetensors",  # ~2.67GB
        "model-00002-of-00003.safetensors",  # ~2.67GB
        "model-00003-of-00003.safetensors",  # ~2.67GB
        "model.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]

    print("=" * 80)
    print("Qwen3-4B BF16 Downloader (H100 DRAFT MODEL)")
    print("=" * 80)
    print(f"📦 Model: {model_id}")
    print(f"📁 Destination: {local_dir}")
    print(f"💾 Size: ~8.0GB (BF16 full precision, 3 shards)")
    print(f"🎯 Target GPU: H100 80GB")
    print(f"📄 Files: {len(files_to_download)}")
    print("")
    print("⏳ Download will take 3-5 minutes on RunPod...")
    print("")
    print("🚀 H100 PERFORMANCE (as Draft Model for Qwen3-8B):")
    print("   - Expected latency: 0.8-1.2s per query (200 tokens)")
    print("   - Throughput: 200-300 tokens/s")
    print("   - Accept Rate: 65-75% (vs 48.5% with 1.7B)")
    print("   - Speedup vs 1.7B Draft: 1.5-2x faster!")
    print("   - Speedup vs No Spec Decoding: 1.3-1.8x faster!")
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
            print("2. Make sure you have enough disk space (~9GB)")
            print("3. Try again (already downloaded files will be skipped)")
            sys.exit(1)

    print("")
    print("=" * 80)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"📁 Model saved to: {local_dir}")
    print("")
    print("🚀 Next steps:")
    print("   1. Update llm_config.yaml to use qwen3_4b as draft model")
    print("   2. Start SGLang server with speculative decoding:")
    print("      ENABLE_SPEC_DECODING=1 python scripts/start_sglang_server_h100.py")
    print("")
    print("📊 Expected Performance on H100 (with 4B Draft):")
    print("   - Latency: 0.8-1.2s per query (vs 1.5s without spec decoding)")
    print("   - Throughput: 200-300 tokens/s (vs 187-194 tokens/s)")
    print("   - Accept Rate: 65-75% (vs 48.5% with 1.7B)")
    print("   - Speedup: 1.3-1.8x faster!")
    print("")


if __name__ == "__main__":
    download_model()

