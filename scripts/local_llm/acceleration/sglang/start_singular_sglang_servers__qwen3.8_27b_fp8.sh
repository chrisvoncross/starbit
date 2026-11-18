#!/bin/bash
# ============================================================================
# SINGULAR SGLANG SERVER STARTUP SCRIPT (Qwen3.8-27B-FP8)
# ============================================================================
# Purpose: Start ONLY the Qwen3.8-27B-FP8 SGLang server (singular mode)
#
# MEMORY ALLOCATION (H100 80GB, GPU must be free):
#   - 70% VRAM (~56GB) - Port 30000
#   - Weights are the official FP8 checkpoint, about 28GB
#   - No Worker (singular mode)
#
# USE CASE:
#   - Decision Transformer Training mit Interactive LLM Coaching
#   - Port 30000 = Same as dual mode for code compatibility
#
# EXPECTED STARTUP TIME on SGLang 0.5.21 with JIT DeepGEMM off:
#   - Same pod, Triton and FlashInfer caches already warm: a few minutes.
#   - New pod, those caches restored from this directory: about 8 minutes.
#   - The old "~40 min DeepGEMM compile" line does not run. See STEP 0.
# ============================================================================

set -e  # Exit on error

# Pins live in lyra/scripts/boot/requirements-sglang.txt.
# SGLang 0.5.21 requires torch==2.13.0 built for CUDA 13 and a host
# driver >= 580. Do not pip-upgrade past those pins.

echo "============================================================================"
echo "🚀 SINGULAR SGLANG SERVER STARTUP (Qwen3.8-27B-FP8)"
echo "============================================================================"
echo ""
echo "   🎯 Mode: Qwen3.8-27B-FP8"
echo "   📊 Port: 30000"
echo "   💾 VRAM: 68% of the H100. Weights landed at 39GB beside the turn run."
echo "   ⚡ MTP: off until the upgraded SGLang recipe for this checkpoint is checked"
echo ""

# ============================================================================
# STEP 0: Restore/Compile DeepGEMM Kernels (Smart Caching!)
# ============================================================================
echo "🔍 STEP 0: Checking DeepGEMM kernel cache..."
echo ""

cd /workspace/ghostbyte
source venv/bin/activate

# Model path (122B)
MODEL_122B="/workspace/ghostbyte/llm_models/shared_models/qwen3.8_27b_fp8"

# 🔥 Per-model kernel cache namespace.
# DeepGEMM/Triton/MoE kernels are model-specific (shape + dtype + quant).
# A flat shared cache means switching models overwrites/pollutes the other
# model's kernels → ~40 min recompile on switch-back. Namespacing the
# persistent cache by model dir name isolates each model's kernels so we
# never lose another model's compiled cache.
MODEL_NAME="$(basename "$MODEL_122B")"

# 🔥 Served-model id (SSOT for the coaching client's dynamic resolution).
# Derived from MODEL_NAME so it is ALWAYS accurate per model — never a
# hand-typed alias that drifts from the real loaded weight. Underscores
# → dashes to match the OpenAI-model-id convention (e.g.
# qwen3.8_27b_fp8 → qwen3.8-27b-fp8). Orion's
# coaching routes ship `model: "auto"` and discover this exact id via
# GET /v1/models at boot (see llm_model_router_policy.md
# § "Dynamic served-model resolution").
SERVED_MODEL_NAME="${MODEL_NAME//_/-}"

# ============================================================================
# SMART CACHING STRATEGY:
# ============================================================================
# Problem: DeepGEMM kernels are stored in TWO places, both lost on pod kill:
#   1. MoE Configs: venv/.../sglang/.../configs/ (JSON configs)
#   2. Triton Cache: /root/.triton/ (Compiled CUDA kernels .cubin)
#
# Solution: Cache BOTH in Network Volume, restore on new pod!
#
# Network Volume Cache: /workspace/ghostbyte/.deepgemm_cache/
#   - triton_configs/ (MoE JSON configs)
#   - triton_cache/ (Compiled CUDA kernels)
#
# On new pod: Copy from Network Volume → local directories
# After compilation: Copy from local directories → Network Volume
# ============================================================================

# Network Volume cache directory (persistent across pods!)
# 🔥 Namespaced per model (see MODEL_NAME above) so different models keep
# their own kernel caches instead of clobbering each other.
PERSISTENT_CACHE="/workspace/ghostbyte/.deepgemm_cache/${MODEL_NAME}"
PERSISTENT_TRITON_CONFIGS="$PERSISTENT_CACHE/triton_configs"
PERSISTENT_TRITON_CACHE="$PERSISTENT_CACHE/triton_cache"
# This wheel loads an entry only when the folder is <tag>.<digest> and
# contains .committed. Folders named kernel.* are a previous template.
# They stay on disk and are not hits.
PERSISTENT_DEEPGEMM_CACHE="$PERSISTENT_CACHE/deep_gemm"
# FlashInfer JIT from this SGLang. Lost with the overlay unless restored.
PERSISTENT_FLASHINFER="$PERSISTENT_CACHE/flashinfer"
mkdir -p "$PERSISTENT_TRITON_CONFIGS"
mkdir -p "$PERSISTENT_TRITON_CACHE"
mkdir -p "$PERSISTENT_DEEPGEMM_CACHE"

# Local cache directories (lost when pod is killed)
VENV_CACHE="venv/lib/python3.12/site-packages/sglang/srt/layers/moe/fused_moe_triton/configs"
# 🔥 CRITICAL: Triton uses ~/.triton/cache by default, NOT ~/.triton!
TRITON_CACHE="/root/.triton/cache"
# This process writes DeepGEMM into SGLANG_DG_CACHE_DIR (the persistent
# tree). ~/.cache/deep_gemm is the old default and stays empty while that
# redirect is set. $HOME/.dj is the wheel default when the env is unset.
DEEPGEMM_CACHE="/root/.cache/deep_gemm"

# 🔥 CRITICAL: Detect installed triton version dynamically.
# SGLang's MoE configs are versioned per-Triton (triton_3_3_1/, triton_3_5_1/, ...).
# Hardcoding the version causes silent cache-restore failure after every
# SGLang upgrade. Strip any "+local" build tag (e.g. "3.5.1+gitabcd" -> "3_5_1").
TRITON_VERSION_DIR=$(python -c "import triton; v=triton.__version__.split('+')[0].split('.'); print(f'triton_{v[0]}_{v[1]}_{v[2]}')" 2>/dev/null)
if [ -z "$TRITON_VERSION_DIR" ]; then
    echo "   ⚠️  Could not detect triton version — falling back to triton_3_5_1"
    TRITON_VERSION_DIR="triton_3_5_1"
fi

# Get GPU info for validation
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)

echo "   🖥️  GPU: $GPU_NAME (Compute $GPU_COMPUTE_CAP)"
echo "   📁 Persistent Cache: $PERSISTENT_CACHE"
echo "   📁 Local Cache: $VENV_CACHE"
echo ""

# ============================================================================
# STEP 0.1: Try to restore kernels from Network Volume cache
# ============================================================================
echo "🔄 Checking for cached kernels in Network Volume..."
echo ""

# Qwen3.8-27B is dense. MoE JSON under triton_3_5_1 is not on this path.
# Installed Triton is 3.7.1, so a missing triton_3_7_1 config dir is not a
# cold DeepGEMM compile. The old gate (MoE>10 AND cubins>10 AND DeepGEMM>100)
# printed "~40 min" on every start: a previous wheel's cubins are not hits.
CACHED_CUBINS=0
if [ -d "$PERSISTENT_TRITON_CACHE" ]; then
    CACHED_CUBINS=$(find "$PERSISTENT_TRITON_CACHE" -name "*.cubin" 2>/dev/null | wc -l)
fi
CACHED_DEEPGEMM_HIT=0
CACHED_DEEPGEMM_OLD=0
if [ -d "$PERSISTENT_DEEPGEMM_CACHE/cache" ]; then
    CACHED_DEEPGEMM_HIT=$(find "$PERSISTENT_DEEPGEMM_CACHE/cache" -name ".committed" 2>/dev/null | wc -l)
    CACHED_DEEPGEMM_OLD=$(find "$PERSISTENT_DEEPGEMM_CACHE/cache" -mindepth 1 -maxdepth 1 -type d -name "kernel.*" 2>/dev/null | wc -l)
fi
CACHED_FLASHINFER=0
if [ -d "$PERSISTENT_FLASHINFER" ]; then
    CACHED_FLASHINFER=$(find "$PERSISTENT_FLASHINFER" -type f 2>/dev/null | wc -l)
fi

echo "   📦 Cached Triton cubins: $CACHED_CUBINS"
echo "   📦 Cached FlashInfer files: $CACHED_FLASHINFER"
echo "   📦 DeepGEMM kernels this wheel loads: $CACHED_DEEPGEMM_HIT"
echo "   📦 Previous-wheel DeepGEMM cubins, ignored: $CACHED_DEEPGEMM_OLD"
echo ""

# Triton looks up TRITON_CACHE_DIR. Without it a restored tree is ignored.
export TRITON_CACHE_DIR="$TRITON_CACHE"
echo "   🔧 Set TRITON_CACHE_DIR=$TRITON_CACHE_DIR"

if [ -d "$PERSISTENT_TRITON_CACHE/cache" ]; then
    mkdir -p /root/.triton
    cp -a "$PERSISTENT_TRITON_CACHE"/. /root/.triton/ 2>/dev/null || true
    RESTORED_CUBINS=$(find /root/.triton -name "*.cubin" 2>/dev/null | wc -l)
    echo "   ✅ Restored $RESTORED_CUBINS Triton cubins to /root/.triton"
fi

if [ -d "$PERSISTENT_FLASHINFER" ]; then
    mkdir -p /root/.cache/sglang/.cache
    rm -rf /root/.cache/sglang/.cache/flashinfer
    cp -a "$PERSISTENT_FLASHINFER" /root/.cache/sglang/.cache/flashinfer
    echo "   ✅ Restored FlashInfer cache to /root/.cache/sglang/.cache/flashinfer"
fi

# JIT reads SGLANG_DG_CACHE_DIR. A hit is a folder <tag>.<digest> with
# .committed. kernel.* folders from the previous wheel are not hits.
# A full M-grid precompile stays off; missing shapes compile on first
# capture into this directory.
export SGLANG_DG_CACHE_DIR="$PERSISTENT_DEEPGEMM_CACHE"
echo "   🔧 Set SGLANG_DG_CACHE_DIR=$SGLANG_DG_CACHE_DIR"
echo "   ✅ DeepGEMM JIT on. Cache is the persistent directory above."
echo ""

if [ ! -d "$MODEL_122B" ]; then
    echo "   ❌ Model not found: $MODEL_122B"
    echo "      Run: python scripts/local_llm/download_models/download_qwen3.8_27b_fp8.py"
    exit 1
fi

# ============================================================================
# STEP 1: Kill old servers
# ============================================================================
echo "🔪 STEP 1: Killing old SGLang servers..."
pkill -f "sglang.*30000" 2>/dev/null || echo "   No server on 30000"
pkill -f "sglang.*30000" 2>/dev/null || echo "   No server on 30000"
echo "   ✅ Old servers killed"
echo ""

# ============================================================================
# STEP 2: Wait for GPU to be free
# ============================================================================
echo "⏳ STEP 2: Waiting for GPU to be free (5 seconds)..."
sleep 5
echo "   ✅ GPU should be free now"
echo ""

# ============================================================================
# STEP 3: Start Qwen3.8-27B-FP8 (Port 30000)
# ============================================================================
echo "🎯 STEP 3: Starting Qwen3.8-27B-FP8 (Port 30000)..."
echo "   Memory: 68% of the H100. Weights are 39GB, so 55% left no room for the cache."
echo "   🏷️  Served model id: $SERVED_MODEL_NAME (from GET /v1/models)"
echo ""

cd /workspace/ghostbyte
source venv/bin/activate

python - <<'PY'
import sglang
print(f"sglang {sglang.__version__}")
PY

mkdir -p logs
# Official FP8 checkpoint. No --quantization moe_wna16 (that flag is the
# GPTQ MoE path of the 122B script this file was copied from).
# MTP stays off: the checkpoint has mtp.safetensors, the flag set is checked
# after the SGLang upgrade, not copied from the MoE script.
# sgl-deep-gemm 0.2.0's wheel ships layout.py. A partial install left it
# out and JIT aborted. The venv now has those files again.
# SGLang redirects the DeepGEMM cache to SGLANG_DG_CACHE_DIR, so the
# persistent cubins have to be that directory, not ~/.cache/deep_gemm.
# Precompile of every M is off: this server runs one request. The shapes
# it actually captures are compiled once into that cache.
nohup env TRITON_CACHE_DIR="$TRITON_CACHE" SGLANG_SINGULAR_MODE=true \
    SGLANG_ENABLE_JIT_DEEPGEMM=1 \
    SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
    SGLANG_DG_CACHE_DIR="$PERSISTENT_DEEPGEMM_CACHE" \
    python -m sglang.launch_server \
        --model-path "$MODEL_122B" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --host 0.0.0.0 \
        --port 30000 \
        --mem-fraction-static 0.68 \
        --tp-size 1 \
        --trust-remote-code \
        --attention-backend flashinfer \
        --kv-cache-dtype fp8_e4m3 \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --max-running-requests 1 \
        --cuda-graph-max-bs-decode 8 \
        --cuda-graph-max-bs-prefill 8 \
    2>&1 | tee logs/sglang_server_qwen3.8_27b_fp8_$(date +%Y%m%d_%H%M%S).log &

SERVER_PID=$!
echo "   Qwen3.8-27B-FP8 started (PID: $SERVER_PID)"
echo ""

# ============================================================================
# STEP 4: Wait for Qwen3.8-27B-FP8 to finish loading
# ============================================================================
echo "⏳ STEP 4: Waiting for Qwen3.8-27B-FP8 to finish loading..."
echo "   Checking health endpoint every 30 seconds..."
echo "   ⚠️ NO TIMEOUT - Will wait as long as needed!"
echo ""

ELAPSED=0
while true; do
    if curl -s http://127.0.0.1:30000/health > /dev/null 2>&1; then
        echo "   ✅ Qwen3.8-27B-FP8 is ready! (took ${ELAPSED}s = $((ELAPSED / 60))min)"
        break
    fi
    echo "   ⏳ Still loading... (${ELAPSED}s = $((ELAPSED / 60))min)"
    sleep 30
    ELAPSED=$((ELAPSED + 30))
done
echo ""

# ============================================================================
# STEP 5: Verify server is running
# ============================================================================
echo "✅ STEP 5: Verifying server..."
echo ""

# Check GPU memory usage
echo "📊 GPU Memory Usage:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
echo ""

# Check processes
echo "🔍 SGLang Processes:"
ps aux | grep sglang | grep -v grep
echo ""

# Check port
echo "🌐 Port Status:"
lsof -i :30000 | head -2
echo ""

echo "============================================================================"
echo "✅ SINGULAR 122B SGLANG SERVER STARTED SUCCESSFULLY!"
echo "============================================================================"
echo ""
echo "📊 Server Status:"
echo "   - Qwen3.8-27B-FP8: http://127.0.0.1:30000 (PID: $SERVER_PID)"
echo ""
echo "📝 Log:"
echo "   - sglang_server_qwen3.8_27b_fp8_*.log"
echo ""

# ============================================================================
# STEP 6: Warmup server
# ============================================================================
echo "============================================================================"
echo "🔥 STEP 6: Warming up 122B server..."
echo "============================================================================"
echo ""
echo "⏳ Running warmup queries to:"
echo "   - Compile CUDA graphs"
echo "   - Populate RadixCache"
echo "   - Optimize FP8 quantization"
echo ""
echo "⏱️  Expected time: 1-2 minutes"
echo ""

cd /workspace/ghostbyte
source venv/bin/activate

# Warmup single server on port 30000 (use --single to skip 1.7B warmup)
#
# `set +e` around the warmup is REQUIRED. warmup_sglang_server.py returns
# exit 1 whenever the warmup-query avg latency is >5s — a cold-start
# performance heuristic (its `is_ready = success_rate>=0.9 && avg<5000ms`
# gate), which fires on the FIRST warmup of a 122B MoE model even when
# 100% of queries succeed. Under the script-wide `set -e` (top of file)
# that benign non-zero would abort the script HERE, which:
#   1. makes the WARMUP_EXIT_CODE if/else below dead code, and
#   2. (the real bug) skips the "Backup compiled kernels to Network
#      Volume" final step — so a first-pod run would never seed the
#      Triton and FlashInfer caches. DeepGEMM JIT stays off on 0.5.21;
#      there is no 40-minute DeepGEMM compile to cache.
# Capture the real exit code, then restore `set -e` for the rest.
set +e
python scripts/local_llm/acceleration/sglang/warmup_sglang_server.py --single
WARMUP_EXIT_CODE=$?
set -e

echo ""
if [ $WARMUP_EXIT_CODE -eq 0 ]; then
    echo "✅ WARMUP SUCCESSFUL!"
else
    echo "⚠️ WARMUP COMPLETED WITH WARNINGS (exit code: $WARMUP_EXIT_CODE)"
    echo "   Server is running but may not be fully optimized."
fi
echo ""

# ============================================================================
# WAIT FOR SERVER TO BE READY
# ============================================================================
echo ""
echo "⏳ Waiting for server to be fully ready (including DeepGEMM warmup)..."
echo ""

# Wait for Qwen3.8-27B-FP8 to be ready
echo "   🔍 Checking Qwen3.8-27B-FP8 (port 30000)..."
MAX_WAIT=3600  # 60 minutes max
ELAPSED=0
SERVER_READY=false

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:30000/health > /dev/null 2>&1; then
        echo "   ✅ Qwen3.8-27B-FP8 is READY!"
        SERVER_READY=true
        break
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    if [ $((ELAPSED % 60)) -eq 0 ]; then
        echo "   ⏳ Still waiting... (${ELAPSED}s)"
    fi
done

if [ "$SERVER_READY" = false ]; then
    echo "   ⚠️  Warning: Qwen3.8-27B-FP8 not ready after ${MAX_WAIT}s"
fi

echo ""

# ============================================================================
# FINAL STEP: Backup compiled kernels to Network Volume
# ============================================================================
echo "💾 FINAL STEP: Backing up compiled kernels to Network Volume..."
echo ""

# Count current kernels (use entire /root/.triton structure)
CURRENT_CONFIGS=0
CURRENT_CUBINS=0
CURRENT_DEEPGEMM=0
if [ -d "$VENV_CACHE/${TRITON_VERSION_DIR}" ]; then
    CURRENT_CONFIGS=$(find "$VENV_CACHE/${TRITON_VERSION_DIR}" -name "*.json" 2>/dev/null | wc -l)
fi
if [ -d "/root/.triton" ]; then
    CURRENT_CUBINS=$(find /root/.triton -name "*.cubin" 2>/dev/null | wc -l)
fi
# Live DeepGEMM cache is the persistent tree. ~/.cache/deep_gemm is only
# a leftover default and is empty while SGLANG_DG_CACHE_DIR is set.
if [ -d "$PERSISTENT_DEEPGEMM_CACHE/cache" ]; then
    CURRENT_DEEPGEMM=$(find "$PERSISTENT_DEEPGEMM_CACHE/cache" -name ".committed" 2>/dev/null | wc -l)
fi
LOCAL_DEEPGEMM=0
if [ -d "/root/.cache/deep_gemm/cache" ]; then
    LOCAL_DEEPGEMM=$(find /root/.cache/deep_gemm/cache -name "*.cubin" 2>/dev/null | wc -l)
fi

# Count cached kernels
CACHED_CONFIGS=0
CACHED_CUBINS=0
CACHED_DEEPGEMM=0
if [ -d "$PERSISTENT_TRITON_CONFIGS/${TRITON_VERSION_DIR}" ]; then
    CACHED_CONFIGS=$(find "$PERSISTENT_TRITON_CONFIGS/${TRITON_VERSION_DIR}" -name "*.json" 2>/dev/null | wc -l)
fi
if [ -d "$PERSISTENT_TRITON_CACHE" ]; then
    CACHED_CUBINS=$(find "$PERSISTENT_TRITON_CACHE" -name "*.cubin" 2>/dev/null | wc -l)
fi
if [ -d "$PERSISTENT_DEEPGEMM_CACHE/cache" ]; then
    CACHED_DEEPGEMM=$(find "$PERSISTENT_DEEPGEMM_CACHE/cache" -name ".committed" 2>/dev/null | wc -l)
fi

echo "   📊 Current MoE configs: $CURRENT_CONFIGS (cached: $CACHED_CONFIGS)"
echo "   📊 Current CUDA kernels: $CURRENT_CUBINS (cached: $CACHED_CUBINS)"
echo "   📊 DeepGEMM kernels this wheel loads: $CURRENT_DEEPGEMM (committed: $CACHED_DEEPGEMM, ~/.cache/deep_gemm cubins: $LOCAL_DEEPGEMM)"
echo ""

UPDATED=false

# Update MoE configs if needed
if [ "$CURRENT_CONFIGS" -gt "$CACHED_CONFIGS" ]; then
    echo "   🔄 Updating MoE configs cache..."
    mkdir -p "$PERSISTENT_TRITON_CONFIGS/${TRITON_VERSION_DIR}"
    cp -r "$VENV_CACHE/${TRITON_VERSION_DIR}"/* "$PERSISTENT_TRITON_CONFIGS/${TRITON_VERSION_DIR}/" 2>/dev/null || true
    NEW_CACHED=$(find "$PERSISTENT_TRITON_CONFIGS/${TRITON_VERSION_DIR}" -name "*.json" 2>/dev/null | wc -l)
    echo "   ✅ Updated: $NEW_CACHED MoE configs"
    UPDATED=true
fi

# Update CUDA kernels if needed
if [ "$CURRENT_CUBINS" -gt "$CACHED_CUBINS" ]; then
    echo "   🔄 Updating CUDA kernels cache..."
    mkdir -p "$PERSISTENT_TRITON_CACHE"
    # Copy entire /root/.triton structure to preserve all metadata
    cp -r /root/.triton/* "$PERSISTENT_TRITON_CACHE/" 2>/dev/null || true
    NEW_CACHED=$(find "$PERSISTENT_TRITON_CACHE" -name "*.cubin" 2>/dev/null | wc -l)
    echo "   ✅ Updated: $NEW_CACHED CUDA kernels"
    UPDATED=true
fi

# Only merge a non-empty ~/.cache/deep_gemm. The live kernels are already
# in the persistent tree. Copying an empty default over that tree would
# not add hits, and a replace would drop the .committed entries.
if [ "$LOCAL_DEEPGEMM" -gt 0 ]; then
    echo "   🔄 Merging ~/.cache/deep_gemm into the persistent DeepGEMM cache..."
    mkdir -p "$PERSISTENT_DEEPGEMM_CACHE"
    cp -a /root/.cache/deep_gemm/. "$PERSISTENT_DEEPGEMM_CACHE/" 2>/dev/null || true
    NEW_CACHED=$(find "$PERSISTENT_DEEPGEMM_CACHE/cache" -name ".committed" 2>/dev/null | wc -l)
    echo "   ✅ Persistent DeepGEMM entries with .committed: $NEW_CACHED"
    UPDATED=true
fi

if [ -d /root/.cache/sglang/.cache/flashinfer ]; then
    rm -rf "$PERSISTENT_FLASHINFER"
    mkdir -p "$PERSISTENT_CACHE"
    cp -a /root/.cache/sglang/.cache/flashinfer "$PERSISTENT_FLASHINFER"
    echo "   ✅ Saved FlashInfer cache"
    UPDATED=true
fi

if [ "$UPDATED" = false ]; then
    echo "   ✅ Cache is up-to-date (no new kernels)"
else
    echo ""
    echo "   🎉 Cache updated! Future pods will use these kernels!"
    echo "   📁 Cache location: $PERSISTENT_CACHE"
fi

echo ""
echo "============================================================================"
echo "🎉 SINGULAR 122B SGLANG SERVER READY FOR PRODUCTION!"
echo "============================================================================"
echo ""
echo "🎯 Server Info:"
echo "   - Model: Qwen3.8-27B-FP8"
echo "   - Port: 30000 (same as dual mode)"
echo "   - GPU: $GPU_NAME"
echo ""
echo "🎯 Next Steps:"
echo "   1. Run: python bytedance/main.py --micro-trade"
echo "   2. Monitor logs for performance"
echo "   3. Check GPU utilization: watch -n 1 nvidia-smi"
echo ""
echo "📊 Kernel Cache Status:"
echo "   - Location: $PERSISTENT_CACHE"
echo "   - MoE Configs: $(find "$PERSISTENT_TRITON_CONFIGS/${TRITON_VERSION_DIR}" -name "*.json" 2>/dev/null | wc -l || echo "0")"
echo "   - CUDA Kernels: $(find "$PERSISTENT_TRITON_CACHE" -name "*.cubin" 2>/dev/null | wc -l || echo "0")"
echo ""
echo "============================================================================"

