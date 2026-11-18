#!/bin/bash
# ============================================================================
# SINGULAR SGLANG SERVER STARTUP SCRIPT (Qwen3.5-122B-A10B-FP8)
# ============================================================================
# Purpose: Start ONLY the 122B SGLang server (singular mode)
#
# MEMORY ALLOCATION:
#   - 122B Master: 70% VRAM (~99GB) - Port 30000 (same as dual for compatibility!)
#   - No Worker (singular mode)
#
# USE CASE:
#   - Decision Transformer Training mit Interactive LLM Coaching
#   - Port 30000 = Same as dual mode for code compatibility
#
# EXPECTED STARTUP TIME:
#   - WITH pre-compiled DeepGEMM: ~9 minutes ✅
#   - WITHOUT pre-compilation: ~25 minutes ⚠️
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "🚀 SINGULAR SGLANG SERVER STARTUP (Qwen3.5-122B-A10B-FP8)"
echo "============================================================================"
echo ""
echo "   🎯 Mode: Single 122B Server (Qwen3.5-122B-A10B-FP8)"
echo "   📊 Port: 30000 (same as dual mode for compatibility)"
echo "   💾 VRAM: 70% allocated (~99GB)"
echo "   ⚡ MTP: ENABLED (+60% throughput via native MTP heads)"
echo ""

# ============================================================================
# STEP 0: Restore/Compile DeepGEMM Kernels (Smart Caching!)
# ============================================================================
echo "🔍 STEP 0: Checking DeepGEMM kernel cache..."
echo ""

cd /workspace/ghostbyte
source venv/bin/activate

# Model path (122B)
MODEL_122B="/workspace/ghostbyte/llm_models/shared_models/qwen3.5_122b_a10b_fp8"

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
# qwen3.5_122b_a10b_fp8 → qwen3.5-122b-a10b-fp8). Orion's coaching routes
# ship `model: "auto"` and discover this exact id via GET /v1/models at
# boot (see llm_model_router_policy.md § "Dynamic served-model resolution").
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
# 🔥 CRITICAL: DeepGEMM has its OWN cache directory!
PERSISTENT_DEEPGEMM_CACHE="$PERSISTENT_CACHE/deep_gemm"
mkdir -p "$PERSISTENT_TRITON_CONFIGS"
mkdir -p "$PERSISTENT_TRITON_CACHE"
mkdir -p "$PERSISTENT_DEEPGEMM_CACHE"

# Local cache directories (lost when pod is killed)
VENV_CACHE="venv/lib/python3.12/site-packages/sglang/srt/layers/moe/fused_moe_triton/configs"
# 🔥 CRITICAL: Triton uses ~/.triton/cache by default, NOT ~/.triton!
TRITON_CACHE="/root/.triton/cache"
# 🔥 CRITICAL: DeepGEMM uses ~/.cache/deep_gemm by default!
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

# Check for MoE configs
CACHED_CONFIGS=0
if [ -d "$PERSISTENT_TRITON_CONFIGS/${TRITON_VERSION_DIR}" ]; then
    CACHED_CONFIGS=$(find "$PERSISTENT_TRITON_CONFIGS/${TRITON_VERSION_DIR}" -name "*.json" 2>/dev/null | wc -l)
fi

# Check for Triton CUDA kernels
CACHED_CUBINS=0
if [ -d "$PERSISTENT_TRITON_CACHE" ]; then
    CACHED_CUBINS=$(find "$PERSISTENT_TRITON_CACHE" -name "*.cubin" 2>/dev/null | wc -l)
fi

# 🔥 CRITICAL: Check for DeepGEMM cache!
CACHED_DEEPGEMM=0
if [ -d "$PERSISTENT_DEEPGEMM_CACHE/cache" ]; then
    CACHED_DEEPGEMM=$(find "$PERSISTENT_DEEPGEMM_CACHE/cache" -name "*.cubin" 2>/dev/null | wc -l)
fi

echo "   📦 Cached MoE Configs: $CACHED_CONFIGS"
echo "   📦 Cached CUDA Kernels: $CACHED_CUBINS"
echo "   📦 Cached DeepGEMM Kernels: $CACHED_DEEPGEMM"

if [ "$CACHED_CONFIGS" -gt 10 ] && [ "$CACHED_CUBINS" -gt 10 ] && [ "$CACHED_DEEPGEMM" -gt 100 ]; then
    echo "   ✅ Found complete kernel cache!"
    echo "   🔄 Restoring to local directories..."
    echo ""

    # Restore MoE configs to venv
    if [ -d "$PERSISTENT_TRITON_CONFIGS/${TRITON_VERSION_DIR}" ]; then
        mkdir -p "$VENV_CACHE/${TRITON_VERSION_DIR}"
        cp -r "$PERSISTENT_TRITON_CONFIGS/${TRITON_VERSION_DIR}"/* "$VENV_CACHE/${TRITON_VERSION_DIR}/" 2>/dev/null || true
        echo "   ✅ Restored $CACHED_CONFIGS MoE configs to venv"
    fi

    # Restore Triton CUDA kernels to /root/.triton/cache
    # Note: PERSISTENT_TRITON_CACHE already contains the full /root/.triton structure
    if [ -d "$PERSISTENT_TRITON_CACHE" ]; then
        # Copy the entire .triton directory structure
        mkdir -p /root/.triton
        cp -r "$PERSISTENT_TRITON_CACHE"/* /root/.triton/ 2>/dev/null || true
        RESTORED_CUBINS=$(find /root/.triton -name "*.cubin" 2>/dev/null | wc -l)
        echo "   ✅ Restored $RESTORED_CUBINS CUDA kernels to /root/.triton"
    fi

    # 🔥 CRITICAL: Restore DeepGEMM cache to /root/.cache/deep_gemm
    if [ -d "$PERSISTENT_DEEPGEMM_CACHE" ]; then
        mkdir -p /root/.cache
        cp -r "$PERSISTENT_DEEPGEMM_CACHE" /root/.cache/ 2>/dev/null || true
        RESTORED_DEEPGEMM=$(find /root/.cache/deep_gemm/cache -name "*.cubin" 2>/dev/null | wc -l)
        echo "   ✅ Restored $RESTORED_DEEPGEMM DeepGEMM kernels to /root/.cache/deep_gemm"
    fi

    echo ""
    echo "   🚀 Kernels ready! Server startup will be FAST (~10 min)"
    echo ""

    # ========================================================================
    # 🔥 CRITICAL FIX: Set Triton cache environment variable!
    # ========================================================================
    # DeepGEMM/Triton uses TRITON_CACHE_DIR to find cached kernels
    # Without this, it will IGNORE the restored cache and recompile!
    export TRITON_CACHE_DIR="$TRITON_CACHE"
    echo "   🔧 Set TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
    echo ""

    KERNELS_RESTORED=true
else
    echo "   ⚠️  Incomplete or missing kernel cache"
    echo "   ⚠️  First-time setup - will compile kernels (~40 min)"
    echo ""

    KERNELS_RESTORED=false
fi

# ============================================================================
# STEP 0.2: Compile kernels if not restored from cache
# ============================================================================

if [ "$KERNELS_RESTORED" = false ]; then
    echo "============================================================================"
    echo "🔥 COMPILING DEEPGEMM KERNELS (ONE-TIME SETUP)"
    echo "============================================================================"
    echo ""
    echo "   💡 This is a ONE-TIME compilation (~30 min total)"
    echo "   💡 Kernels will be cached in Network Volume for future pods!"
    echo ""
    echo "   📊 What's happening:"
    echo "      - Compiling optimized CUDA kernels for $GPU_NAME"
    echo "      - Caching in Network Volume: $PERSISTENT_CACHE"
    echo "      - Future pods will restore from cache (instant!)"
    echo ""

    # 🔥 CRITICAL: Set Triton cache directory for compilation
    export TRITON_CACHE_DIR="$TRITON_CACHE"
    echo "   🔧 Set TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
    echo ""

    # Check if model exists
    if [ ! -d "$MODEL_122B" ]; then
        echo "   ❌ 122B Model not found: $MODEL_122B"
        echo "      Run: python scripts/llm/download_qwen3.5_122b_a10b_fp8.py"
        exit 1
    fi

    # 🔥 SGLang 0.5.10 + Qwen3.5 hybrid GDN/MoE compatibility note:
    # The dedicated `python -m sglang.compile_deep_gemm` step crashes with
    # `AssertionError: max_running_request is zero` on hybrid GDN models —
    # its default mem_fraction_static=0.84 leaves only ~22 GB for the
    # KV-cache after the 116 GB FP8 weights, and SGLang can't compute a
    # positive max_running_requests from that. The fix is to skip the
    # standalone compile and let the launch_server below do it via
    # `--warmups compile-deep-gemm` instead — same kernels, same cache
    # destinations (~/.cache/deep_gemm/ + /root/.triton/cache/), but with
    # the launch-server's correct memory configuration.
    echo "   🎯 Skipping standalone compile_deep_gemm (broken on Qwen3.5 GDN in SGLang 0.5.10)"
    echo "   🎯 Kernels will be compiled inside launch_server (Step 3) via --warmups flag"
    echo "   🎯 Same cache destinations, ~5-10 min added to first server boot"
    echo ""

    echo ""
    echo "   ✅ 122B Model kernels compiled!"
    echo ""

    # ========================================================================
    # CRITICAL: Save compiled kernels to Network Volume!
    # ========================================================================
    echo "   💾 Saving kernels to Network Volume cache..."
    echo ""

    # Use TRITON_CACHE_DIR if set
    ACTUAL_TRITON_CACHE="${TRITON_CACHE_DIR:-$TRITON_CACHE}"

    # Save MoE configs
    if [ -d "$VENV_CACHE/${TRITON_VERSION_DIR}" ]; then
        mkdir -p "$PERSISTENT_TRITON_CONFIGS/${TRITON_VERSION_DIR}"
        cp -r "$VENV_CACHE/${TRITON_VERSION_DIR}"/* "$PERSISTENT_TRITON_CONFIGS/${TRITON_VERSION_DIR}/" 2>/dev/null || true

        SAVED_CONFIGS=$(find "$PERSISTENT_TRITON_CONFIGS/${TRITON_VERSION_DIR}" -name "*.json" 2>/dev/null | wc -l)
        echo "   ✅ Saved $SAVED_CONFIGS MoE configs"
    else
        echo "   ⚠️  Warning: venv cache directory not found"
    fi

    # Save Triton CUDA kernels (save entire /root/.triton structure)
    if [ -d "/root/.triton" ]; then
        mkdir -p "$PERSISTENT_TRITON_CACHE"
        # Copy entire .triton directory structure to preserve all metadata
        cp -r /root/.triton/* "$PERSISTENT_TRITON_CACHE/" 2>/dev/null || true

        SAVED_CUBINS=$(find "$PERSISTENT_TRITON_CACHE" -name "*.cubin" 2>/dev/null | wc -l)
        echo "   ✅ Saved $SAVED_CUBINS CUDA kernels"
    else
        echo "   ⚠️  Warning: Triton cache directory not found at /root/.triton"
    fi

    echo ""
    echo "============================================================================"
    echo "   🎉 Kernels compiled and cached in Network Volume!"
    echo "   🚀 Future pods will restore from cache (instant startup!)"
    echo "   📁 Cache location: $PERSISTENT_CACHE"
    echo "============================================================================"
    echo ""
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
# STEP 3: Start 122B Master (Port 30000 - Same as dual!)
# ============================================================================
echo "🎯 STEP 3: Starting 122B Master (Port 30000)..."
echo "   Memory: 70% VRAM (~99GB)"
echo "   🏷️  Served model id: $SERVED_MODEL_NAME (from GET /v1/models)"
echo "   Expected time: ~9 min (with pre-compiled kernels)"
echo ""

cd /workspace/ghostbyte
source venv/bin/activate

# Start 122B Master in background (SINGULAR MODE - no Worker!)
# 🔥 CRITICAL: Set TRITON_CACHE_DIR and SGLANG_SINGULAR_MODE for proper logging
#
# Launch flags rationale:
#   --served-model-name $SERVED_MODEL_NAME  accurate served id derived from
#                                the model dir (e.g. qwen3.5-122b-a10b-fp8);
#                                Orion coaching routes ship model:"auto" and
#                                discover this exact id via GET /v1/models
#   --mem-fraction-static 0.70   ~99 GB on H200 (per script header)
#   --tp-size 1                  single H200, no tensor parallelism
#   --trust-remote-code          Qwen3.5 ships custom HF model classes (qwen3_5_moe)
#   --attention-backend flashinfer  H200/SM90 has full FlashInfer GDN support (per SGLang #18590)
#   --kv-cache-dtype fp8_e4m3    halves KV cache memory; verified safe on H200 SM90
#                                (silent-corruption issue from SGLang #19603 only affects
#                                 SM120/Blackwell-pro — H200 Hopper is unaffected)
#   --reasoning-parser qwen3     parses <think>...</think> blocks
#   --tool-call-parser qwen3_coder  Qwen3.5 tool-call format
#   --speculative-* NEXTN/3/1/4  MTP heads enabled → +2.75x decode speedup (SGLang #19603)
mkdir -p logs
# 🔥 SGLang 0.5.10 + Qwen3.5 hybrid GDN env vars:
#   SGLANG_ENABLE_SPEC_V2=1     required when combining speculative decoding (NEXTN)
#                                with RadixCache on Qwen3_5MoeForConditionalGeneration
#                                — without this, server_args raises ValueError at boot.
#                                (SGLang 0.5.10 introduced a new "Spec V2" path for
#                                hybrid GDN/MoE models; old-style spec_v1 only worked
#                                without RadixCache.)
# 🔥 H200 + Qwen3.5-122B-A10B-FP8 memory budget (143 GB total HBM3e):
#   --mem-fraction-static 0.92   → 132 GB static pool: 119 GB model + 13 GB
#                                  KV/Mamba/CUDA-graphs/Spec-V2 buffer.
#                                  Anything < 0.85 crashes pre-load with
#                                  "Not enough memory" (model alone needs 119 GB).
#                                  Anything > 0.93 risks OOM after weight load.
#   --max-running-requests 2     → was 4, but Spec-V2 buffer is ~2-3 GB/request
#                                  on hybrid GDN models. 2 concurrent is enough
#                                  for single-user coaching workload.
#   --cuda-graph-max-bs 64       → was default 256. Saves ~3 GB. Coaching
#                                  workload rarely batches > 64 anyway.
nohup env TRITON_CACHE_DIR="$TRITON_CACHE" SGLANG_SINGULAR_MODE=true SGLANG_ENABLE_SPEC_V2=1 \
    python -m sglang.launch_server \
        --model-path "$MODEL_122B" \
        --served-model-name "$SERVED_MODEL_NAME" \
        --host 0.0.0.0 \
        --port 30000 \
        --mem-fraction-static 0.92 \
        --tp-size 1 \
        --trust-remote-code \
        --attention-backend flashinfer \
        --kv-cache-dtype fp8_e4m3 \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --speculative-algorithm NEXTN \
        --speculative-num-steps 3 \
        --speculative-eagle-topk 1 \
        --speculative-num-draft-tokens 4 \
        --max-running-requests 2 \
        --cuda-graph-max-bs 64 \
        --mamba-scheduler-strategy extra_buffer \
        --warmups compile-deep-gemm \
    2>&1 | tee logs/sglang_server_qwen3.5_122b_a10b_fp8_$(date +%Y%m%d_%H%M%S).log &

SERVER_PID=$!
echo "   122B Master started (PID: $SERVER_PID)"
echo ""

# ============================================================================
# STEP 4: Wait for 122B Master to finish loading
# ============================================================================
echo "⏳ STEP 4: Waiting for 122B Master to finish loading..."
echo "   Checking health endpoint every 30 seconds..."
echo "   ⚠️ NO TIMEOUT - Will wait as long as needed!"
echo ""

ELAPSED=0
while true; do
    if curl -s http://127.0.0.1:30000/health > /dev/null 2>&1; then
        echo "   ✅ 122B Master is ready! (took ${ELAPSED}s = $((ELAPSED / 60))min)"
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
echo "   - 122B Master: http://127.0.0.1:30000 (PID: $SERVER_PID)"
echo ""
echo "📝 Log:"
echo "   - sglang_server_qwen3.5_122b_a10b_fp8_*.log"
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
#      Volume" final step — so a first-pod / cache-miss run would never
#      seed the persistent DeepGEMM cache and every future pod would
#      recompile from scratch (~40 min), defeating the smart-caching
#      this script is built around.
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

# Wait for 122B Master to be ready
echo "   🔍 Checking 122B Master (port 30000)..."
MAX_WAIT=3600  # 60 minutes max
ELAPSED=0
SERVER_READY=false

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:30000/health > /dev/null 2>&1; then
        echo "   ✅ 122B Master is READY!"
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
    echo "   ⚠️  Warning: 122B Master not ready after ${MAX_WAIT}s"
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
# 🔥 CRITICAL: Count DeepGEMM kernels!
if [ -d "/root/.cache/deep_gemm/cache" ]; then
    CURRENT_DEEPGEMM=$(find /root/.cache/deep_gemm/cache -name "*.cubin" 2>/dev/null | wc -l)
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
    CACHED_DEEPGEMM=$(find "$PERSISTENT_DEEPGEMM_CACHE/cache" -name "*.cubin" 2>/dev/null | wc -l)
fi

echo "   📊 Current MoE configs: $CURRENT_CONFIGS (cached: $CACHED_CONFIGS)"
echo "   📊 Current CUDA kernels: $CURRENT_CUBINS (cached: $CACHED_CUBINS)"
echo "   📊 Current DeepGEMM kernels: $CURRENT_DEEPGEMM (cached: $CACHED_DEEPGEMM)"
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

# 🔥 CRITICAL: Update DeepGEMM kernels if needed!
if [ "$CURRENT_DEEPGEMM" -gt "$CACHED_DEEPGEMM" ]; then
    echo "   🔄 Updating DeepGEMM kernels cache..."
    mkdir -p "$PERSISTENT_DEEPGEMM_CACHE"
    # Copy entire /root/.cache/deep_gemm structure
    cp -r /root/.cache/deep_gemm/* "$PERSISTENT_DEEPGEMM_CACHE/" 2>/dev/null || true
    NEW_CACHED=$(find "$PERSISTENT_DEEPGEMM_CACHE/cache" -name "*.cubin" 2>/dev/null | wc -l)
    echo "   ✅ Updated: $NEW_CACHED DeepGEMM kernels"
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
echo "   - Model: Qwen3.5-122B-A10B-FP8 (MoE: 122B total, 10B active)"
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

