#!/usr/bin/env bash
set -euo pipefail

# Start Horizon in raw_farmer mode (the retained legacy base viz — raw farmer
# Lance OHLC + NWE/Z-Score/Stoch re-derivation; NOT what the model sees).
# Manual launch (not supervised by runit/spire).
#
# Env (all optional; sensible defaults):
#   HORIZON_SYMBOL   (default BTC_USDT)   HORIZON_INTERVAL (default Min1)
#   HORIZON_DAYS     (default 31)         HORIZON_PORT     (default 8000)
#   HORIZON_FARMER_LANCE_PATH  (default <repo>/data/synced_market_farmer_data/lance)

WORKSPACE_ROOT="/workspace/ghostbyte"
HORIZON_ROOT="${WORKSPACE_ROOT}/horizon"
VENV_DIR="${WORKSPACE_ROOT}/venv"

# shellcheck disable=SC1091
[[ -f "${VENV_DIR}/bin/activate" ]] && source "${VENV_DIR}/bin/activate"

export HORIZON_MODE="raw_farmer"
export HORIZON_SYMBOL="${HORIZON_SYMBOL:-BTC_USDT}"
export HORIZON_INTERVAL="${HORIZON_INTERVAL:-Min1}"
export HORIZON_DAYS="${HORIZON_DAYS:-31}"
PORT="${HORIZON_PORT:-8000}"

cd "${HORIZON_ROOT}"
echo "[horizon] raw_farmer | ${HORIZON_SYMBOL} ${HORIZON_INTERVAL} | http://0.0.0.0:${PORT}"
exec uvicorn websocket_server:app --host 0.0.0.0 --port "${PORT}"
