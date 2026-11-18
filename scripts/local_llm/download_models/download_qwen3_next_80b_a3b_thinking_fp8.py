#!/usr/bin/env python3
"""
Download Qwen3-Next-80B-A3B-Thinking-FP8 for H200 (ALL AGENTS - Development Phase).

This downloads the FP8 PRE-QUANTIZED MoE MODEL with 80B total parameters and 3B active.
THINKING MODE: Includes <think>...</think> reasoning blocks for complex analysis!

FP8 BENEFITS:
- 50% smaller: ~80GB instead of ~160GB!
- 2-3x faster inference than BF16!
- 50% less VRAM: ~80GB instead of ~160GB!
- Minimal quality loss (<1%)

THINKING MODE BENEFITS:
- Deep reasoning with <think> blocks for complex tasks
- Perfect for Market Thesis, Pattern Discovery, Deep Analysis
- Can also do fast responses (without <think>) when needed
- Approaches 235B quality with 10x speed on long context!

MoE BENEFITS:
- 80B total parameters, only 3B active per token
- Hybrid-Attention (75% Gated DeltaNet, 25% Gated Attention)
- 262K native context, extendable to 1M with YaRN
- Fits on 1x H200 141GB!

USAGE:
    python scripts/download_qwen3_next_80b_a3b_thinking_fp8.py

SIZE: ~80GB (FP8 MoE model, Hybrid-Attention)
TIME: ~30-40 minutes on RunPod
GPU REQUIREMENTS: 1x H200 141GB
"""

import sys
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_model():
    """Download Qwen3-Next-80B-A3B-Thinking-FP8 model files."""

    # Model configuration (Next-80B Thinking - FP8!)
    model_id = "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8"  # ✅ FP8 pre-quantized Thinking Mode!

    # RunPod: Use /workspace/ path (persistent storage)
    local_dir = Path("/workspace/ghostbyte/llm_models/shared_models/qwen3_next_80b_a3b_thinking_fp8")

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
    ]

    print("=" * 80)
    print("Qwen3-Next-80B-A3B-Thinking-FP8 Downloader (ALL AGENTS - DEVELOPMENT)")
    print("=" * 80)
    print(f"📦 Model: {model_id}")
    print(f"📁 Destination: {local_dir}")
    print(f"💾 Size: ~80GB (FP8 MoE: 80B total, 3B active)")
    print(f"🎯 Target GPU: 1x H200 141GB")
    print("")
    print("✨ FP8 ADVANTAGES:")
    print("   - 50% smaller: ~80GB instead of ~160GB BF16!")
    print("   - 2-3x faster inference than BF16!")
    print("   - 50% less VRAM: ~80GB instead of ~160GB!")
    print("   - Minimal quality loss (<1%)")
    print("")
    print("✨ THINKING MODE ADVANTAGES:")
    print("   - Deep reasoning with <think>...</think> blocks")
    print("   - Perfect for Market Thesis, Pattern Discovery")
    print("   - Can also do fast responses (without <think>)")
    print("   - Approaches 235B quality!")
    print("")
    print("✨ NEXT-GENERATION ARCHITECTURE:")
    print("   - 80B total parameters, only 3B active per token")
    print("   - Hybrid-Attention (75% Gated DeltaNet, 25% Gated Attention)")
    print("   - 262K native context, extendable to 1M with YaRN")
    print("   - 10x faster on long context (>32K tokens)!")
    print("")
    print("⏳ Download will take 30-40 minutes on RunPod...")
    print("")
    print("🚀 H200 PERFORMANCE (ALL AGENTS):")
    print("   - Expected latency: 800-1500ms per query (with <think>)")
    print("   - Fast mode: 200-500ms (without <think>)")
    print("   - Throughput: 200-400 tokens/s (1x H200)")
    print("   - Context length: 262K native (1M with YaRN)")
    print("   - VRAM usage: ~80GB (fits on 1x H200!)")
    print("   - Use case: ALL agents (Worker + Master)")
    print("")
    print("🧠 UNIFIED MODEL ARCHITECTURE (DEVELOPMENT):")
    print("   - ALL Agents use Next-80B-Thinking-FP8")
    print("   - Master Agent: Full <think> reasoning power!")
    print("   - Worker Agents: Can use fast mode or thinking mode")
    print("   - Centralized config: bytedance/config/llm_config.yaml")
    print("")
    print("🔄 UPGRADE PATH:")
    print("   - Phase 1 (NOW): Next-80B-Thinking-FP8 on 1x H200 (development)")
    print("   - Phase 2 (LATER): 235B-A22B-FP8 on 2x H200 (production)")
    print("   - Or: Add Next-80B-Instruct-FP8 for Worker Agents (2 models)")
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
    print("   1. Start SGLang server with optimizations:")
    print("      python scripts/start_sglang_server_qwen3_next_80b_a3b_thinking_fp8.py")
    print("")
    print("   2. Or manually with SGLang:")
    print("      python -m sglang.launch_server \\")
    print("        --model-path /workspace/ghostbyte/llm_models/shared_models/qwen3_next_80b_a3b_thinking_fp8 \\")
    print("        --host 0.0.0.0 \\")
    print("        --port 30000 \\")
    print("        --mem-fraction-static 0.85 \\")
    print("        --tp-size 1 \\")
    print("        --attention-backend flashinfer \\")
    print("        --kv-cache-dtype fp8_e4m3")
    print("")
    print("📊 Expected Performance on 1x H200 (Next-80B-Thinking):")
    print("   - Latency: 800-1500ms per analysis (with <think>)")
    print("   - Fast mode: 200-500ms (without <think>)")
    print("   - Throughput: 200-400 tokens/s")
    print("   - Context: 262K native (1M with YaRN)")
    print("   - Use case: ALL agents (Worker + Master)")
    print("   - Quality: Approaches 235B, 10x faster on long context!")
    print("")
    print("🧠 Development Workflow:")
    print("   1. Test all agents with Next-80B-Thinking (deep reasoning)")
    print("   2. Master Agent uses full <think> power for Market Thesis")
    print("   3. Worker Agents can use fast mode or thinking mode")
    print("   4. Validate performance and quality")
    print("   5. Optionally upgrade to 235B or add Instruct model later")
    print("")
    print("🔄 Future upgrade options:")
    print("   1. Add Next-80B-Instruct-FP8 for Worker Agents (2 models)")
    print("   2. Or upgrade to 235B-A22B-FP8 on 2x H200 (production)")
    print("   3. Update llm_config.yaml: model = qwen3_next_80b_a3b_thinking_fp8")
    print("")


if __name__ == "__main__":
    download_model()

