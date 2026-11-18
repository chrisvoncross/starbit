#!/usr/bin/env python3
"""
Download Qwen3-30B-A3B-Instruct-FP8 for H200 (ALL AGENTS - Development Phase).

This downloads the FP8 PRE-QUANTIZED MoE MODEL with 30B total parameters and 3B active.
Used for ALL AGENTS (Worker + Master) during development before switching to 235B.

FP8 BENEFITS:
- 50% smaller: ~32GB instead of ~64GB!
- 2-3x faster inference than BF16!
- 50% less VRAM: ~50-60GB instead of ~100GB!
- Minimal quality loss (<1%)

MoE BENEFITS:
- 30B total parameters, only 3B active per token
- Same architecture as 235B (easy upgrade path!)
- Fits easily on 1x H200 141GB

USAGE:
    python scripts/download_qwen3_30b_a3b.py

SIZE: ~32GB (FP8 MoE model, 128 experts, 8 active)
TIME: ~10-20 minutes on RunPod (50% faster than BF16!)
GPU REQUIREMENTS: 1x H200 141GB (or 1x H100 80GB)
"""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_model():
    """Download Qwen3-30B-A3B-Instruct model files."""

    # Model configuration (MoE version - FP8!)
    model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"  # ✅ FP8 pre-quantized!

    # RunPod: Use /workspace/ path (persistent storage)
    local_dir = Path("/workspace/ghostbyte/llm_models/shared_models/qwen3_30b_a3b_fp8")

    # Files to download (30B MoE model has fewer shards than 235B)
    # We'll download all safetensors files automatically
    # First, let's check the model structure on HuggingFace
    
    # For now, we'll download the essential config files first
    # Then download all safetensors files
    config_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
    ]

    print("=" * 80)
    print("Qwen3-30B-A3B-Instruct-FP8 Downloader (ALL AGENTS - DEVELOPMENT)")
    print("=" * 80)
    print(f"📦 Model: {model_id}")
    print(f"📁 Destination: {local_dir}")
    print(f"💾 Size: ~32GB (FP8 MoE: 30B total, 3B active)")
    print(f"🎯 Target GPU: 1x H200 141GB (or 1x H100 80GB)")
    print("")
    print("✨ FP8 ADVANTAGES:")
    print("   - 50% smaller: ~32GB instead of ~64GB BF16!")
    print("   - 2-3x faster inference than BF16!")
    print("   - 50% less VRAM: ~50-60GB instead of ~100GB!")
    print("   - Minimal quality loss (<1%)")
    print("")
    print("✨ MoE ADVANTAGES:")
    print("   - 30B total parameters, only 3B active per token")
    print("   - Same MoE architecture as 235B (easy upgrade!)")
    print("   - 128 experts, 8 active per token")
    print("")
    print("⏳ Download will take 10-20 minutes on RunPod (50% faster!)...")
    print("")
    print("🚀 H200 PERFORMANCE (ALL AGENTS):")
    print("   - Expected latency: 200-500ms per query (2-3x faster!)")
    print("   - Throughput: 200-400 tokens/s (1x H200)")
    print("   - Context length: 128K native")
    print("   - VRAM usage: ~50-60GB (fits easily!)")
    print("   - Use case: ALL agents (Worker + Master)")
    print("")
    print("🧠 UNIFIED MODEL ARCHITECTURE (DEVELOPMENT):")
    print("   - ALL Agents use 30B-A3B-FP8 (Worker + Master)")
    print("   - Centralized config: bytedance/config/llm_config.yaml")
    print("   - Knowledge Base: PostgreSQL (master_rules table)")
    print("")
    print("🔄 UPGRADE PATH:")
    print("   - Phase 1 (NOW): 30B-A3B-FP8 on 1x H200 (development)")
    print("   - Phase 2 (LATER): 235B-A22B-FP8 on 2x H200 (production)")
    print("   - Same MoE architecture = easy config switch!")
    print("")

    # Create directory
    local_dir.mkdir(parents=True, exist_ok=True)

    # Download config files first
    print("🚀 Starting download (config files first)...")
    print("")

    for i, filename in enumerate(config_files, 1):
        target_file = local_dir / filename

        # Skip if already exists
        if target_file.exists():
            print(f"[{i}/{len(config_files)}] ✅ Already exists: {filename}")
            continue

        try:
            print(f"[{i}/{len(config_files)}] ⬇️  Downloading: {filename}")
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            print(f"[{i}/{len(config_files)}] ✅ Downloaded: {filename}")
            print("")

        except Exception as e:
            print(f"[{i}/{len(config_files)}] ❌ Failed: {filename}")
            print(f"    Error: {e}")
            print("")
            # Config files are critical, but we can continue
            if filename == "config.json":
                print("❌ config.json is required! Exiting...")
                sys.exit(1)

    # Now download all safetensors files
    # We'll use the index file to determine how many shards there are
    print("")
    print("📦 Downloading model weights (safetensors)...")
    print("")

    # Read the index file to get the list of safetensors files
    import json
    index_file = local_dir / "model.safetensors.index.json"
    
    if index_file.exists():
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        # Get unique safetensors files from the index
        safetensors_files = sorted(set(index_data.get("weight_map", {}).values()))
        print(f"📄 Found {len(safetensors_files)} safetensors files in index")
        print("")
    else:
        # Fallback: try common patterns
        print("⚠️  Index file not found, trying common patterns...")
        safetensors_files = [
            f"model-{str(i).zfill(5)}-of-00008.safetensors" for i in range(1, 9)
        ]
        print(f"📄 Trying {len(safetensors_files)} safetensors files (estimated)")
        print("")

    # Download safetensors files
    for i, filename in enumerate(safetensors_files, 1):
        target_file = local_dir / filename

        # Skip if already exists
        if target_file.exists():
            print(f"[{i}/{len(safetensors_files)}] ✅ Already exists: {filename}")
            continue

        try:
            print(f"[{i}/{len(safetensors_files)}] ⬇️  Downloading: {filename}")
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            print(f"[{i}/{len(safetensors_files)}] ✅ Downloaded: {filename}")
            print("")

        except Exception as e:
            print(f"[{i}/{len(safetensors_files)}] ❌ Failed: {filename}")
            print(f"    Error: {e}")
            print("")
            print("Possible solutions:")
            print("1. Check your internet connection")
            print("2. Make sure you have enough disk space (~50GB)")
            print("3. Try again (already downloaded files will be skipped)")
            sys.exit(1)

    print("")
    print("=" * 80)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"📁 Model saved to: {local_dir}")
    print("")
    print("🚀 Next steps:")
    print("   1. Start SGLang server with optimizations:")
    print("      python scripts/start_sglang_server_qwen3_30b_a3b.py")
    print("")
    print("   2. Or manually with SGLang:")
    print("      python -m sglang.launch_server \\")
    print("        --model-path /workspace/ghostbyte/llm_models/shared_models/qwen3_30b_a3b \\")
    print("        --host 0.0.0.0 \\")
    print("        --port 30001 \\")
    print("        --mem-fraction-static 0.85 \\")
    print("        --tp-size 1 \\")
    print("        --attention-backend flashinfer \\")
    print("        --kv-cache-dtype fp8_e4m3")
    print("")
    print("📊 Expected Performance on 1x H200 (30B-A3B Master):")
    print("   - Latency: 500-1000ms per analysis")
    print("   - Throughput: 100-200 tokens/s")
    print("   - Context: 128K native")
    print("   - Use case: Development & Testing")
    print("   - Quality: Comparable to 32B Dense, 10x faster!")
    print("")
    print("🧠 Development Workflow:")
    print("   1. Test all agents with 30B-A3B (fast iteration)")
    print("   2. Fix bugs and optimize prompts")
    print("   3. Validate performance and quality")
    print("   4. Switch to 235B-A22B for production (just config change!)")
    print("")
    print("🔄 To upgrade to 235B later:")
    print("   1. Download 235B: python scripts/download_qwen3_235b_h100.py")
    print("   2. Update llm_config.yaml: master model = qwen3_235b_fp8")
    print("   3. Start with 2x H200: --tp-size 2")
    print("   4. Same MoE architecture = no code changes!")
    print("")


if __name__ == "__main__":
    download_model()

