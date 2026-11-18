#!/usr/bin/env python3
"""
Download Qwen3.5-122B-A10B-GPTQ-Int4 for H200 / Hopper (SM90) LLM coach.

This downloads the OFFICIAL Qwen GPTQ 4-bit PRE-QUANTIZED MoE MODEL
(122B total parameters, 10B active). It is the Hopper-viable equivalent of
the Blackwell-only NVFP4 build: GPTQ Int4 MoE runs on Hopper via SGLang's
`moe_wna16` (Int4/Marlin weight-only) kernel — no SM100+ required.

WHY GPTQ-Int4 (not NVFP4, not FP8) on this pod:
- NVFP4 (`modelopt_fp4`) MoE requires NVIDIA Blackwell (SM100+); this pod is
  a Hopper H200 (SM90), so NVFP4 is a HARD blocker.
- FP8 weights are ~119 GB and leave no VRAM headroom for co-resident Orion
  training on a single 143 GB H200.
- GPTQ-Int4 weights are ~79 GB, keeping ~55-60 GB free for Orion while
  preserving Qwen-122B reasoning quality.

QWEN 3.5 ARCHITECTURE:
- 122B total parameters, only 10B active per token
- Hybrid: Gated DeltaNet (75%) + Gated Attention (25%) + MoE (256 experts, 8+1 active)
- 262K native context, extendable to 1M+
- Native Multi-Token Prediction (MTP) heads for speculative decoding
- Unified Vision-Language Foundation
- 201 languages supported
- Quantization: GPTQ 4-bit

USAGE:
    python scripts/local_llm/download_models/download_qwen3.5_122b_a10b_gptq_int4.py

SIZE: ~79GB (GPTQ Int4 MoE model)
TIME: ~30-45 minutes on RunPod (hf_transfer parallel chunked downloader)
GPU REQUIREMENTS: 1x H200 141GB (tp=1), co-resident with Orion training
"""

import os
import sys
from pathlib import Path

# Parallel chunked downloader (pinned in orion[llm-mesh-local]: hf-transfer>=0.1.9).
# Must be set BEFORE importing huggingface_hub for it to take effect.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import hf_hub_download


def download_model():
    """Download Qwen3.5-122B-A10B-GPTQ-Int4 model files."""

    # Model configuration (Qwen3.5-122B-A10B - official GPTQ Int4!)
    model_id = "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4"

    # RunPod: Use /workspace/ path (persistent storage)
    local_dir = Path("/workspace/ghostbyte/llm_models/shared_models/qwen3.5_122b_a10b_gptq_int4")

    # Config / tokenizer / template files (download before the weight shards).
    # Mirrors the official repo file listing; the safetensors shards are then
    # resolved from model.safetensors.index.json below.
    config_files = [
        "config.json",
        "configuration.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
        "preprocessor_config.json",       # multimodal processor init (transformers)
        "video_preprocessor_config.json", # VL video preprocessor
        "chat_template.jinja",            # Qwen3.5 thinking-aware chat template
        "LICENSE",
    ]

    print("=" * 80)
    print("Qwen3.5-122B-A10B-GPTQ-Int4 Downloader (LLM Coach — Hopper/H200)")
    print("=" * 80)
    print(f"📦 Model: {model_id}")
    print(f"📁 Destination: {local_dir}")
    print(f"💾 Size: ~79GB (GPTQ Int4 MoE: 122B total, 10B active)")
    print(f"🎯 Target GPU: 1x H200 141GB (tp=1), co-resident with Orion")
    print("")
    print("✨ GPTQ-Int4 ADVANTAGES on Hopper (SM90):")
    print("   - Runs via SGLang moe_wna16 (Int4/Marlin) — no Blackwell needed!")
    print("   - ~79GB weights → ~55-60GB VRAM stays free for Orion training")
    print("   - Preserves Qwen-122B reasoning quality (GPTQ 4-bit)")
    print("")
    print("✨ QWEN 3.5 ARCHITECTURE:")
    print("   - 122B total parameters, only 10B active per token")
    print("   - Hybrid: Gated DeltaNet (75%) + Gated Attention (25%)")
    print("   - MoE: 256 experts, 8 routed + 1 shared active per token")
    print("   - 262K native context, extendable to 1M+")
    print("   - Native MTP heads for speculative decoding")
    print("   - Unified Vision-Language Foundation, 201 languages")
    print("")
    print("⏳ Download will take ~30-45 minutes on RunPod (hf_transfer)...")
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
            # config.json + the weight index are the only hard-required files;
            # the rest (video preprocessor, LICENSE) are best-effort.
            if filename in ("config.json", "model.safetensors.index.json"):
                print(f"❌ {filename} is required! Exiting...")
                sys.exit(1)

    # Now download all safetensors files, driven by the weight index.
    print("")
    print("📦 Downloading model weights (safetensors)...")
    print("")

    import json
    index_file = local_dir / "model.safetensors.index.json"

    if index_file.exists():
        with open(index_file, "r") as f:
            index_data = json.load(f)

        # Get unique safetensors files from the index
        safetensors_files = sorted(set(index_data.get("weight_map", {}).values()))
        print(f"📄 Found {len(safetensors_files)} safetensors files in index")
        print("")
    else:
        print("❌ model.safetensors.index.json missing — cannot resolve shards. Exiting...")
        sys.exit(1)

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
            print("2. Make sure you have enough disk space (~80GB)")
            print("3. Try again (already downloaded files will be skipped)")
            sys.exit(1)

    print("")
    print("=" * 80)
    print("✅ DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"📁 Model saved to: {local_dir}")
    print("")
    print("🚀 Next step — start the SGLang coach (Hopper-native Int4 MoE):")
    print("   bash scripts/local_llm/acceleration/sglang/"
          "start_singular_sglang_servers_qwen3.5_122b_a10b_gptq_int4.sh")
    print("")
    print("📊 Expected on 1x H200 (Qwen3.5-122B-A10B-GPTQ-Int4, tp=1):")
    print("   - Quantization kernel: moe_wna16 (Int4/Marlin, Hopper-native)")
    print("   - VRAM (weights): ~79GB → ~55-60GB free for Orion training")
    print("   - Served model name: qwen3.5-122b-a10b-gptq-int4 "
          "(derived from model dir; Orion routes resolve it via model: \"auto\")")
    print("")


if __name__ == "__main__":
    download_model()
