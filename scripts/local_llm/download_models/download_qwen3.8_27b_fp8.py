#!/usr/bin/env python3
"""
Download Qwen/Qwen3.8-27B-FP8.

Official FP8 dense checkpoint, 27B. Weights are layers-*.safetensors plus
mtp.safetensors and outside.safetensors, listed in model.safetensors.index.json.

Does not load the model and does not touch the GPU.

USAGE:
    python scripts/local_llm/download_models/download_qwen3.8_27b_fp8.py

SIZE: about 28 GB
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import hf_hub_download, list_repo_files


def download_model():
    model_id = "Qwen/Qwen3.8-27B-FP8"
    local_dir = Path("/workspace/ghostbyte/llm_models/shared_models/qwen3.8_27b_fp8")

    config_files = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.jinja",
        "LICENSE",
        "README.md",
        "crc32.txt",
        "safetensors-md5sum.txt",
    ]

    print("=" * 80)
    print("Qwen3.8-27B-FP8 downloader")
    print("=" * 80)
    print(f"Model: {model_id}")
    print(f"Destination: {local_dir}")
    print("Size: about 28 GB")
    print("")

    local_dir.mkdir(parents=True, exist_ok=True)

    for i, filename in enumerate(config_files, 1):
        target_file = local_dir / filename
        if target_file.exists() and target_file.stat().st_size > 0:
            print(f"[{i}/{len(config_files)}] already exists: {filename}")
            continue
        try:
            print(f"[{i}/{len(config_files)}] downloading: {filename}")
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=str(local_dir),
            )
            print(f"[{i}/{len(config_files)}] downloaded: {filename}")
        except Exception as e:
            print(f"[{i}/{len(config_files)}] failed: {filename}")
            print(f"    {e}")
            if filename in ("config.json", "model.safetensors.index.json"):
                sys.exit(1)

    index_file = local_dir / "model.safetensors.index.json"
    if not index_file.exists():
        print("model.safetensors.index.json missing")
        sys.exit(1)

    with open(index_file, "r") as f:
        index_data = json.load(f)
    weight_files = sorted(set(index_data.get("weight_map", {}).values()))
    repo_shards = [
        name
        for name in list_repo_files(model_id)
        if name.endswith(".safetensors") and name not in weight_files
    ]
    safetensors_files = weight_files + repo_shards
    print(f"weight files: {len(safetensors_files)}")

    for i, filename in enumerate(safetensors_files, 1):
        target_file = local_dir / filename
        if target_file.exists() and target_file.stat().st_size > 0:
            print(f"[{i}/{len(safetensors_files)}] already exists: {filename}")
            continue
        try:
            print(f"[{i}/{len(safetensors_files)}] downloading: {filename}")
            hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=str(local_dir),
            )
            print(f"[{i}/{len(safetensors_files)}] downloaded: {filename}")
        except Exception as e:
            print(f"[{i}/{len(safetensors_files)}] failed: {filename}")
            print(f"    {e}")
            sys.exit(1)

    print("")
    print(f"saved to {local_dir}")
    print("server script is not started by this downloader")


if __name__ == "__main__":
    download_model()
