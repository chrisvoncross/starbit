#!/usr/bin/env python3
"""
Download Qwen3-235B-A22B-Instruct-2507-FP8 for H100 as MASTER MODEL.

This downloads the INSTRUCT-TUNED MODEL in FP8 precision (fine-grained quantization).
This is the MASTER MODEL for LLM Master Learning (Session 5.75).

FP8 BENEFITS:
- 2x smaller download size (~235GB vs ~470GB)
- 2x less VRAM usage (~250GB vs ~500GB)
- Minimal quality loss (see benchmarks - same performance as BF16)
- Faster inference (native FP8 support on H100)

USAGE:
    python scripts/download_qwen3_235b_h100.py

SIZE: ~235GB (FP8 quantized, 118 shards)
TIME: ~1-2 hours on RunPod (depending on network speed)
GPU REQUIREMENTS: 4x H100 80GB (320GB total VRAM) - HALF of BF16!
"""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_model():
    """Download Qwen3-235B-A22B-Instruct-2507-FP8 model files."""

    # Model configuration (FP8 version!)
    model_id = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

    # RunPod: Use /workspace/ path (persistent storage)
    local_dir = Path("/workspace/ghostbyte/llm_models/shared_models/qwen3_235b_fp8")

    # Files to download (235B model has 24 safetensors shards + config files)
    # Generate shard filenames programmatically
    safetensors_files = [
        f"model-{str(i).zfill(5)}-of-00024.safetensors" for i in range(1, 25)
    ]

    config_files = [
        "model.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ]

    files_to_download = safetensors_files + config_files

    print("=" * 80)
    print("Qwen3-235B-A22B-Instruct-2507-FP8 Downloader (H100 MASTER MODEL)")
    print("=" * 80)
    print(f"📦 Model: {model_id}")
    print(f"📁 Destination: {local_dir}")
    print(f"💾 Size: ~236GB (FP8 quantized, 24 shards)")
    print(f"🎯 Target GPU: 2x H100 80GB (160GB total VRAM) - HALF of BF16!")
    print(f"📄 Files: {len(files_to_download)} (24 shards + 7 config files)")
    print("")
    print("✨ FP8 ADVANTAGES:")
    print("   - 2x smaller download: ~236GB (vs ~470GB BF16)")
    print("   - 2x less VRAM: ~120GB (vs ~240GB BF16)")
    print("   - Same quality: Minimal performance loss (see benchmarks)")
    print("   - Faster inference: Native FP8 support on H100")
    print("   - Fits on 2x H100 instead of 4x H100! (or even 1x H200 141GB!)")
    print("")
    print("⏳ Download will take 1-2 hours on RunPod...")
    print("")
    print("🚀 H100 PERFORMANCE (as Master Model for Learning):")
    print("   - Expected latency: 10-30s per deep analysis (pattern discovery)")
    print("   - Throughput: 50-100 tokens/s (4x H100 TP)")
    print("   - Context length: 256K native")
    print("   - Use case: LLM Master Learning (Session 5.75)")
    print("   - Frequency: Every 5-10 minutes (deep pattern discovery)")
    print("")
    print("🧠 MASTER/WORKER HIERARCHY:")
    print("   - Master Model (235B FP8): Deep pattern discovery")
    print("   - Worker Model (4B/8B): Fast pattern application")
    print("   - Knowledge Base: PostgreSQL (master_rules table)")
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
            print("2. Make sure you have enough disk space (~500GB)")
            print("3. Try again (already downloaded files will be skipped)")
            sys.exit(1)

    print("")
    print("=" * 80)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"📁 Model saved to: {local_dir}")
    print("")
    print("🚀 Next steps:")
    print("   1. Update agent_llm_mapping.yaml to use qwen3_235b_fp8 as Master model")
    print("   2. Start SGLang server with 2x H100 TP (or 1x H200!):")
    print("      python -m sglang.launch_server \\")
    print("        --model-path /workspace/ghostbyte/llm_models/shared_models/qwen3_235b_fp8 \\")
    print("        --tp 2 \\")
    print("        --context-length 262144 \\")
    print("        --mem-frac 0.75")
    print("")
    print("   3. Alternative: Use vLLM with FP8:")
    print("      vllm serve /workspace/ghostbyte/llm_models/shared_models/qwen3_235b_fp8 \\")
    print("        --tensor-parallel-size 2 \\")
    print("        --max-model-len 262144")
    print("")
    print("⚠️  IMPORTANT: FP8 Known Issue with Transformers:")
    print("   - For multi-GPU inference, set: CUDA_LAUNCH_BLOCKING=1")
    print("   - This is only needed for transformers, not SGLang/vLLM")
    print("")
    print("📊 Expected Performance on 2x H100 or 1x H200 (235B FP8 Master):")
    print("   - Latency: 10-30s per deep analysis")
    print("   - Throughput: 50-100 tokens/s (2x H100 TP)")
    print("   - Context: 256K native")
    print("   - Use case: Pattern discovery every 5-10 minutes")
    print("   - Quality: Same as BF16 (minimal loss)")
    print("")
    print("🧠 LLM Master Learning Workflow:")
    print("   1. Master (235B FP8) discovers patterns from historical trades")
    print("   2. Master stores patterns in PostgreSQL (master_rules table)")
    print("   3. Worker (4B/8B) loads patterns and applies them in real-time")
    print("   4. Deep RL optimizes execution based on patterns")
    print("   5. Performance feedback updates Master Rules")
    print("")
    print("💰 COST SAVINGS with FP8:")
    print("   - GPU cost: 2x H100 instead of 4x H100 = 50% cheaper!")
    print("   - Storage: 236GB instead of 470GB = 50% cheaper!")
    print("   - Download time: 1-2h instead of 2-4h = 50% faster!")
    print("   - BONUS: Fits on 1x H200 141GB with CPU offloading!")
    print("")


if __name__ == "__main__":
    download_model()

