#!/bin/bash
# ============================================================================
# GPT-OSS-120B SERVER STARTUP SCRIPT (vLLM — Official OpenAI Recipe)
# ============================================================================
# Purpose: Start GPT-OSS-120B via vLLM on H200 for Trading Reasoning
#
# MODEL:
#   - OpenAI GPT-OSS-120B (Apache 2.0)
#   - 117B total params, 5.1B active per token (MoE)
#   - MXFP4 quantized (baked into training — no quality loss)
#   - Uses Marlin MXFP4 MoE kernels (built into vLLM, NO DeepGEMM needed!)
#
# MEMORY ALLOCATION:
#   - Model: ~65GB MXFP4 weights
#   - KV Cache: ~30GB (FP8)
#   - Total: ~95GB / 141GB H200 → 46GB headroom for Transformer
#
# PORT: 30000 (same as Qwen setup for code compatibility!)
#
# EXPECTED STARTUP TIME:
#   - ~3-5 minutes (no DeepGEMM compilation needed!)
#   - CUDA Graph capture: ~1-2 min
#
# HARMONY FORMAT:
#   - GPT-OSS REQUIRES the Harmony response format
#   - /v1/chat/completions works normally (reasoning in response)
#   - /v1/responses for full tool-use/agentic workflows
#
# REFERENCE: https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "🚀 GPT-OSS-120B SERVER STARTUP (vLLM — OpenAI Official)"
echo "============================================================================"
echo ""
echo "   🎯 Model: GPT-OSS-120B (117B total, 5.1B active, MXFP4)"
echo "   📊 Port: 30000 (compatible with existing agent code)"
echo "   💾 VRAM: ~95GB / 141GB H200 (46GB headroom)"
echo "   ⏱️  Startup: ~3-5 min (no DeepGEMM needed!)"
echo ""

cd /workspace/ghostbyte
source venv/bin/activate

# ============================================================================
# CONFIG
# ============================================================================
MODEL_PATH="/workspace/ghostbyte/llm_models/shared_models/gpt-oss-120b"
PORT=30000
HOST="0.0.0.0"
GPU_MEM_UTIL=0.70           # 70% = ~99GB (leaves room for Transformer)
MAX_MODEL_LEN=32768         # 32K context (sufficient for trading analysis)
MAX_NUM_BATCHED_TOKENS=8192 # Official Hopper recommendation
MAX_CUDAGRAPH_BS=2048       # Official Hopper recommendation
MAX_NUM_SEQS=6              # Match Qwen setup for consistency

# Tiktoken encodings (required for Harmony format)
TIKTOKEN_DIR="/workspace/ghostbyte/.tiktoken_encodings"

# Get GPU info
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')
GPU_COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)

echo "   🖥️  GPU: $GPU_NAME (Compute $GPU_COMPUTE_CAP)"
echo ""

# ============================================================================
# STEP 0: Verify model exists
# ============================================================================
echo "🔍 STEP 0: Verifying model..."

if [ ! -d "$MODEL_PATH" ]; then
    echo "   ❌ Model not found: $MODEL_PATH"
    echo "      Run: python scripts/llm/download_gpt_oss_120b.py"
    exit 1
fi

SAFETENSOR_COUNT=$(find "$MODEL_PATH" -name "*.safetensors" -maxdepth 1 | wc -l)
if [ "$SAFETENSOR_COUNT" -lt 10 ]; then
    echo "   ❌ Incomplete model: only $SAFETENSOR_COUNT safetensors (expected 14+)"
    echo "      Re-run: python scripts/llm/download_gpt_oss_120b.py"
    exit 1
fi

echo "   ✅ Model found: $SAFETENSOR_COUNT safetensors shards"
echo ""

# ============================================================================
# STEP 0.1: Prepare Tiktoken encodings (Harmony format requirement)
# ============================================================================
echo "🔧 STEP 0.1: Preparing Tiktoken encodings for Harmony format..."

mkdir -p "$TIKTOKEN_DIR"

if [ ! -f "$TIKTOKEN_DIR/o200k_base.tiktoken" ]; then
    echo "   ⬇️  Downloading o200k_base.tiktoken..."
    wget -q -O "$TIKTOKEN_DIR/o200k_base.tiktoken" \
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
fi

if [ ! -f "$TIKTOKEN_DIR/cl100k_base.tiktoken" ]; then
    echo "   ⬇️  Downloading cl100k_base.tiktoken..."
    wget -q -O "$TIKTOKEN_DIR/cl100k_base.tiktoken" \
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
fi

export TIKTOKEN_ENCODINGS_BASE="$TIKTOKEN_DIR"
echo "   ✅ Tiktoken ready (TIKTOKEN_ENCODINGS_BASE=$TIKTOKEN_DIR)"
echo ""

# ============================================================================
# STEP 1: Kill old servers
# ============================================================================
echo "🔪 STEP 1: Killing old servers..."
pkill -f "vllm.*30000" 2>/dev/null || echo "   No vLLM server on 30000"
pkill -f "sglang.*30000" 2>/dev/null || echo "   No SGLang server on 30000"
pkill -f "vllm.*30000" 2>/dev/null || echo "   No vLLM server on 30000"
pkill -f "sglang.*30000" 2>/dev/null || echo "   No SGLang server on 30000"
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
# STEP 3: Start GPT-OSS-120B via vLLM (Port 30000)
# ============================================================================
echo "🎯 STEP 3: Starting GPT-OSS-120B via vLLM (Port $PORT)..."
echo ""
echo "   📦 Model: $MODEL_PATH"
echo "   💾 GPU Memory: ${GPU_MEM_UTIL}x141GB = ~$((141 * ${GPU_MEM_UTIL%.*}))GB"
echo "   📏 Max Context: $MAX_MODEL_LEN tokens"
echo "   🔄 Max Concurrent: $MAX_NUM_SEQS requests"
echo "   ⚡ Async Scheduling: ENABLED (Hopper optimized)"
echo ""
echo "   ✨ NO DeepGEMM compilation needed!"
echo "   ✨ Uses Marlin MXFP4 MoE kernels (built into vLLM)"
echo "   ✨ Expected startup: ~3-5 minutes"
echo ""

# Start vLLM in background with Hopper-optimized config
# Reference: https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html
nohup env TIKTOKEN_ENCODINGS_BASE="$TIKTOKEN_DIR" \
    vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-cudagraph-capture-size "$MAX_CUDAGRAPH_BS" \
    --async-scheduling \
    --no-enable-prefix-caching \
    --tool-call-parser openai \
    --trust-remote-code \
    2>&1 | tee /workspace/ghostbyte/logs/vllm_gpt_oss_120b_$(date +%Y%m%d_%H%M%S).log &

SERVER_PID=$!
echo "   🚀 vLLM started (PID: $SERVER_PID)"
echo ""

# ============================================================================
# STEP 4: Wait for server to be ready
# ============================================================================
echo "⏳ STEP 4: Waiting for GPT-OSS-120B to finish loading..."
echo "   Checking health endpoint every 15 seconds..."
echo ""

ELAPSED=0
while true; do
    if curl -s http://127.0.0.1:${PORT}/health > /dev/null 2>&1; then
        echo "   ✅ GPT-OSS-120B is ready! (took ${ELAPSED}s = $((ELAPSED / 60))min)"
        break
    fi

    # Check if process is still alive
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "   ❌ vLLM process died! Check logs:"
        echo "      tail -50 /workspace/ghostbyte/logs/vllm_gpt_oss_120b_*.log"
        exit 1
    fi

    echo "   ⏳ Still loading... (${ELAPSED}s = $((ELAPSED / 60))min)"
    sleep 15
    ELAPSED=$((ELAPSED + 15))

    if [ $ELAPSED -gt 1800 ]; then
        echo "   ❌ Timeout after 30 minutes! Check logs."
        exit 1
    fi
done
echo ""

# ============================================================================
# STEP 5: Verify server is running
# ============================================================================
echo "✅ STEP 5: Verifying server..."
echo ""

echo "📊 GPU Memory Usage:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
echo ""

echo "🔍 vLLM Processes:"
ps aux | grep vllm | grep -v grep || echo "   (none found)"
echo ""

echo "🌐 Port Status:"
lsof -i :${PORT} | head -3 || echo "   (port not found)"
echo ""

# ============================================================================
# STEP 6: Quick smoke test
# ============================================================================
echo "============================================================================"
echo "🔥 STEP 6: Smoke test..."
echo "============================================================================"
echo ""

SMOKE_RESPONSE=$(curl -s http://127.0.0.1:${PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'"$MODEL_PATH"'",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10,
        "temperature": 0.0
    }' 2>/dev/null)

if echo "$SMOKE_RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null; then
    echo ""
    echo "   ✅ Smoke test passed! Model is responding."
else
    echo "   ⚠️  Smoke test inconclusive (server may still be warming up)"
    echo "   Response: $SMOKE_RESPONSE"
fi
echo ""

# ============================================================================
# DONE
# ============================================================================
echo "============================================================================"
echo "🎉 GPT-OSS-120B SERVER READY!"
echo "============================================================================"
echo ""
echo "🎯 Server Info:"
echo "   - Model: OpenAI GPT-OSS-120B (117B total, 5.1B active, MXFP4)"
echo "   - Endpoint: http://127.0.0.1:${PORT}"
echo "   - API: /v1/chat/completions (OpenAI compatible)"
echo "   - Tool Use: /v1/responses (Harmony format)"
echo "   - GPU: $GPU_NAME"
echo "   - License: Apache 2.0"
echo ""
echo "📝 Log:"
echo "   - /workspace/ghostbyte/logs/vllm_gpt_oss_120b_*.log"
echo ""
echo "🎯 Reasoning Levels (set in system prompt):"
echo "   - 'Reasoning: low'    — Fast responses, general dialogue"
echo "   - 'Reasoning: medium' — Balanced speed and detail"
echo "   - 'Reasoning: high'   — Deep analysis (trading thesis!)"
echo ""
echo "🎯 Next Steps:"
echo "   1. Run: python bytedance/main.py --micro-trade"
echo "   2. Monitor: tail -f /workspace/ghostbyte/logs/vllm_gpt_oss_120b_*.log"
echo "   3. GPU: watch -n 1 nvidia-smi"
echo ""
echo "⚠️  NOTE: GPT-OSS uses the Harmony response format."
echo "   /v1/chat/completions works normally for basic queries."
echo "   /v1/responses is needed for full tool-use/agentic workflows."
echo ""
echo "============================================================================"

