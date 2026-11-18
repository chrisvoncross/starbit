#!/usr/bin/env python3
"""
Download GPT-OSS-120B — OpenAI's first open-weight model for H200.

117B total parameters, only 5.1B active per token (MoE).
Ships with MXFP4 quantization baked in — fits on a single H100 80GB!
On our H200 141GB we have ~60GB headroom for KV-cache + Transformer.

ARCHITECTURE:
- MoE with MXFP4 quantized expert weights (post-trained)
- 117B total, 5.1B active per token
- Apache 2.0 license
- Configurable reasoning levels (low/medium/high)
- Native tool use, function calling, web browsing
- Harmony response format (required)

USAGE:
    python scripts/llm/download_gpt_oss_120b.py

SIZE: ~70-80GB (MXFP4 quantized weights)
TIME: ~30-40 minutes on RunPod
GPU REQUIREMENTS: 1x H100 80GB (minimum) / 1x H200 141GB (recommended)
"""

import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


def download_model():
    """Download GPT-OSS-120B model files from HuggingFace."""

    model_id = "openai/gpt-oss-120b"
    local_dir = Path("/workspace/ghostbyte/llm_models/shared_models/gpt-oss-120b")

    print("=" * 80)
    print("GPT-OSS-120B Downloader (OpenAI Open-Weight)")
    print("=" * 80)
    print(f"📦 Model: {model_id}")
    print(f"📁 Destination: {local_dir}")
    print(f"💾 Size: ~70-80GB (MXFP4 quantized MoE: 117B total, 5.1B active)")
    print(f"🎯 Target GPU: 1x H200 141GB")
    print("")
    print("✨ KEY ADVANTAGES:")
    print("   - OpenAI's first open-weight model (Apache 2.0)")
    print("   - MXFP4 quantization baked into training — no quality loss!")
    print("   - Fits on single H100 80GB, plenty of room on H200 141GB")
    print("   - Configurable reasoning: low / medium / high")
    print("   - Native tool use & function calling")
    print("   - 117B total params, only 5.1B active (MoE)")
    print("")
    print("⏳ Download will take ~30-40 minutes on RunPod...")
    print("")

    # Create directory
    local_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Discover all files in the repo
    print("🔍 Discovering model files on HuggingFace...")
    try:
        all_files = list(list_repo_files(model_id))
        print(f"   Found {len(all_files)} files in repo")
    except Exception as e:
        print(f"❌ Failed to list repo files: {e}")
        print("   Check your internet connection and HuggingFace access.")
        sys.exit(1)

    # Skip metal/ (Apple Silicon only) and original/ (OpenAI native format, redundant)
    # We only need top-level HuggingFace safetensors + configs for vLLM/SGLang
    skip_prefixes = ("metal/", "original/", ".")
    relevant_files = [f for f in all_files if not any(f.startswith(p) for p in skip_prefixes)]
    
    config_files = [f for f in relevant_files if not f.endswith(".safetensors")]
    weight_files = [f for f in relevant_files if f.endswith(".safetensors")]
    other_original = []  # skipped

    skipped = len(all_files) - len(relevant_files)
    print(f"   Config files: {len(config_files)}")
    print(f"   Weight files: {len(weight_files)}")
    print(f"   Skipped: {skipped} (metal/ + original/ — not needed for CUDA)")
    print("")

    # Step 2: Download config + metadata files
    print("📋 Downloading config & metadata files...")
    print("")

    all_non_weight = config_files + other_original
    for i, filename in enumerate(all_non_weight, 1):
        target_file = local_dir / filename
        if target_file.exists():
            print(f"  [{i}/{len(all_non_weight)}] ✅ Already exists: {filename}")
            continue

        try:
            print(f"  [{i}/{len(all_non_weight)}] ⬇️  Downloading: {filename}")
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            print(f"  [{i}/{len(all_non_weight)}] ✅ Done: {filename}")
        except Exception as e:
            print(f"  [{i}/{len(all_non_weight)}] ⚠️  Skipped: {filename} ({e})")

    # Step 3: Download weight files
    print("")
    print("📦 Downloading model weights (safetensors)...")
    print("   This is the big download — ~70-80GB total")
    print("")

    for i, filename in enumerate(weight_files, 1):
        target_file = local_dir / filename
        if target_file.exists():
            print(f"  [{i}/{len(weight_files)}] ✅ Already exists: {filename}")
            continue

        try:
            print(f"  [{i}/{len(weight_files)}] ⬇️  Downloading: {filename}")
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
            )
            print(f"  [{i}/{len(weight_files)}] ✅ Done: {filename}")
            print("")
        except Exception as e:
            print(f"  [{i}/{len(weight_files)}] ❌ Failed: {filename}")
            print(f"     Error: {e}")
            print("")
            print("Possible solutions:")
            print("  1. Check your internet connection")
            print("  2. Make sure you have enough disk space (~80GB)")
            print("  3. Try again (already downloaded files will be skipped)")
            sys.exit(1)

    print("")
    print("=" * 80)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"📁 Model saved to: {local_dir}")
    print("")
    print("🚀 Next steps:")
    print("   1. Serve with vLLM:")
    print("      vllm serve openai/gpt-oss-120b \\")
    print("        --host 0.0.0.0 --port 30000 \\")
    print("        --gpu-memory-utilization 0.90 \\")
    print("        --max-model-len 32768")
    print("")
    print("   2. Or with Ollama:")
    print("      ollama run gpt-oss-120b")
    print("")
    print("   3. Or with the official gpt_oss package:")
    print("      pip install gpt-oss")
    print("      python -m gpt_oss.chat " + str(local_dir) + "/original/")
    print("")
    print("⚠️  IMPORTANT: GPT-OSS uses the 'harmony' response format.")
    print("   It will NOT work correctly without it.")
    print("   See: https://github.com/openai/harmony")
    print("")
    print("📊 Expected Performance on 1x H200:")
    print("   - VRAM: ~70-80GB (leaves ~60GB for KV-cache + Transformer)")
    print("   - Reasoning: Configurable low/medium/high via system prompt")
    print("   - Tool calling: Native support")
    print("   - License: Apache 2.0")
    print("")


if __name__ == "__main__":
    download_model()
