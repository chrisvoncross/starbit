#!/usr/bin/env python3
"""
Download Qwen3.5-122B-A10B-FP8 for H200 (ALL AGENTS).

This downloads the FP8 PRE-QUANTIZED MoE MODEL with 122B total parameters and 10B active.

FP8 BENEFITS:
- 50% smaller than BF16!
- 2-3x faster inference than BF16!
- Minimal quality loss (<1%)

QWEN 3.5 ARCHITECTURE:
- 122B total parameters, only 10B active per token
- Hybrid: Gated DeltaNet (75%) + Gated Attention (25%) + MoE (256 experts, 8+1 active)
- 262K native context, extendable to 1M+
- Native Multi-Token Prediction (MTP) heads for speculative decoding
- Unified Vision-Language Foundation
- 201 languages supported

USAGE:
    python scripts/llm/download_qwen3.5_122b_a10b_fp8.py

SIZE: ~122GB (FP8 MoE model, Hybrid-Attention)
TIME: ~45-60 minutes on RunPod
GPU REQUIREMENTS: 1x H200 141GB (tp=1) or 2x GPU (tp=2)
"""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_model():
    """Download Qwen3.5-122B-A10B-FP8 model files."""

    # Model configuration (Qwen3.5-122B-A10B - FP8!)
    model_id = "Qwen/Qwen3.5-122B-A10B-FP8"  # ✅ FP8 pre-quantized Qwen3.5!

    # RunPod: Use /workspace/ path (persistent storage)
    local_dir = Path("/workspace/ghostbyte/llm_models/shared_models/qwen3.5_122b_a10b_fp8")

    # Files to download (Next-80B model)
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
        "preprocessor_config.json",  # Required by transformers 5.2.0 for multimodal processor init
    ]

    print("=" * 80)
    print("Qwen3.5-122B-A10B-FP8 Downloader (ALL AGENTS)")
    print("=" * 80)
    print(f"📦 Model: {model_id}")
    print(f"📁 Destination: {local_dir}")
    print(f"💾 Size: ~122GB (FP8 MoE: 122B total, 10B active)")
    print(f"🎯 Target GPU: 1x H200 141GB")
    print("")
    print("✨ FP8 ADVANTAGES:")
    print("   - 50% smaller than BF16!")
    print("   - 2-3x faster inference than BF16!")
    print("   - Minimal quality loss (<1%)")
    print("")
    print("✨ QWEN 3.5 ARCHITECTURE:")
    print("   - 122B total parameters, only 10B active per token")
    print("   - Hybrid: Gated DeltaNet (75%) + Gated Attention (25%)")
    print("   - MoE: 256 experts, 8 routed + 1 shared active per token")
    print("   - 262K native context, extendable to 1M+")
    print("   - Native MTP heads for speculative decoding (+60% throughput!)")
    print("   - Unified Vision-Language Foundation")
    print("   - 201 languages supported")
    print("")
    print("⏳ Download will take 45-60 minutes on RunPod...")
    print("")
    print("🚀 H200 PERFORMANCE:")
    print("   - Throughput: 200-500 tokens/s with MTP")
    print("   - Context length: 262K native")
    print("   - VRAM usage: ~122GB FP8 (fits on 1x H200 141GB!)")
    print("   - Use case: ALL agents (Worker + Master)")
    print("")
    print("🧠 CENTRALIZED CONFIG: bytedance/config/llm_config.yaml")
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
        # Fallback: try common patterns (Next-80B likely has more shards)
        print("⚠️  Index file not found, trying common patterns...")
        safetensors_files = [
            f"model-{str(i).zfill(5)}-of-00016.safetensors" for i in range(1, 17)
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
    print("   1. Start SGLang server:")
    print("      python scripts/start_sglang_server_qwen3.5_122b_a10b_fp8.py")
    print("")
    print("   2. Or manually with SGLang + MTP:")
    print("      python -m sglang.launch_server \\")
    print("        --model-path /workspace/ghostbyte/llm_models/shared_models/qwen3.5_122b_a10b_fp8 \\")
    print("        --host 0.0.0.0 \\")
    print("        --port 30001 \\")
    print("        --mem-fraction-static 0.85 \\")
    print("        --tp-size 1 \\")
    print("        --attention-backend flashinfer \\")
    print("        --kv-cache-dtype fp8_e4m3 \\")
    print("        --reasoning-parser qwen3 \\")
    print("        --speculative-algo NEXTN \\")
    print("        --speculative-num-steps 3 \\")
    print("        --speculative-eagle-topk 1 \\")
    print("        --speculative-num-draft-tokens 4")
    print("")
    print("📊 Expected Performance on 1x H200 (Qwen3.5-122B-A10B-FP8):")
    print("   - Throughput: 200-500 tokens/s with MTP")
    print("   - Context: 262K native")
    print("   - VRAM: ~122GB FP8")
    print("   - MTP: +60% throughput via native MTP heads")
    print("   - Architecture: Gated DeltaNet + MoE (256 experts)")
    print("")


if __name__ == "__main__":
    download_model()

