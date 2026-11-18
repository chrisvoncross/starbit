#!/usr/bin/env python3
"""
Download Qwen3-1.7B-FP8 model for H200 Worker LLM.

PRIORITY: FAST WORKER MODEL FOR PRODUCTION
- This is the WORKER MODEL for fast market analysis
- FP8 quantization: 2x faster than BF16 with minimal quality loss
- Perfect for simple classification tasks (BUY/SELL/HOLD)

WHY QWEN3-1.7B-FP8?
- Optimal size for worker tasks (1.7B params)
- Fast inference (~800-1200ms per query on H200)
- FP8 quantization (50% memory savings)
- Good enough for pre-processed market analysis
- 4-5x faster than 8B model!

USAGE:
    python scripts/download_qwen3_1.7b_fp8.py

SIZE: ~2GB (FP8 quantized)
TIME: ~2-3 minutes on RunPod
"""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_model():
    """Download Qwen3-1.7B-FP8 model files."""

    # Model configuration
    model_id = "Qwen/Qwen3-1.7B-FP8"

    # RunPod: Use /workspace/ path (persistent storage)
    local_dir = Path("/workspace/ghostbyte/llm_models/shared_models/qwen3_1.7b_fp8")

    # Files to download (1.7B-FP8 has 1 shard)
    files_to_download = [
        "model.safetensors",  # ~2GB (FP8 quantized)
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]

    print("=" * 80)
    print("Qwen3-1.7B-FP8 Downloader (H200 WORKER MODEL)")
    print("=" * 80)
    print(f"📦 Model: {model_id}")
    print(f"📁 Destination: {local_dir}")
    print(f"💾 Size: ~2GB (FP8 quantized)")
    print(f"🎯 Purpose: Fast Worker Model for Market Analysis")
    print(f"📄 Files: {len(files_to_download)}")
    print("")
    print("⏳ Download will take 2-3 minutes on RunPod...")
    print("")
    print("🚀 WORKER MODEL PERFORMANCE:")
    print("   - Latency: 800-1200ms per query (4-5x faster than 8B!)")
    print("   - Throughput: 150-200 tokens/s")
    print("   - Quality: Good for classification tasks (BUY/SELL/HOLD)")
    print("   - Memory: ~5GB VRAM (50% less than BF16)")
    print("   - Use Case: MarketIntelligenceAgent, DecisionOrchestrator, OrderExecution")
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
            print("2. Make sure you have enough disk space (~4GB)")
            print("3. Try again (already downloaded files will be skipped)")
            sys.exit(1)

    print("")
    print("=" * 80)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"📁 Model saved to: {local_dir}")
    print("")
    print("🚀 Next step: Update config to use 1.7B-FP8 worker")
    print("   Edit: bytedance/config/llm_config.yaml")
    print("   Change: models.worker.h200 = 'qwen3_1.7b_fp8'")
    print("")
    print("📊 Expected Performance:")
    print("   - Latency: 800-1200ms per query (vs 3-5s with 8B)")
    print("   - Total Flow: ~3.7s sequential, ~1.3s parallel")
    print("   - Quality: Good for worker tasks (classification)")
    print("")


if __name__ == "__main__":
    download_model()

