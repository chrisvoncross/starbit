#!/usr/bin/env bash
set -euo pipefail

# Draw one symbol over one time span from the Tigris sealed lake.
#
#   HORIZON_SYMBOL=BTC_USDT \
#   HORIZON_FROM=2026-10-03T11 \
#   HORIZON_TO=2026-10-04T10 \
#   bash horizon/scripts/start_tigris_range.sh
#
# FROM is inclusive. TO as YYYY-MM-DDTHH includes that hour.
# A full timestamp for TO is exclusive.
# Open http://<host>:8000/  — the candles are the raw bars, not Heikin Ashi.

HORIZON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "${HORIZON_ROOT}/.." && pwd)"
VENV_DIR="${WORKSPACE_ROOT}/venv"

if [[ -f "${WORKSPACE_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${WORKSPACE_ROOT}/.env"
  set +a
fi

# shellcheck disable=SC1091
[[ -f "${VENV_DIR}/bin/activate" ]] && source "${VENV_DIR}/bin/activate"

: "${HORIZON_SYMBOL:?set HORIZON_SYMBOL, for example BTC_USDT}"
: "${HORIZON_FROM:?set HORIZON_FROM, for example 2026-10-03T11}"
: "${HORIZON_TO:?set HORIZON_TO, for example 2026-10-04T10}"

export HORIZON_MODE="tigris_range"
export HORIZON_INTERVAL="${HORIZON_INTERVAL:-Sec15}"
export PYTHONPATH="${WORKSPACE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PORT="${HORIZON_PORT:-8000}"

cd "${HORIZON_ROOT}"
echo "[horizon] tigris_range | ${HORIZON_SYMBOL} ${HORIZON_INTERVAL} ${HORIZON_FROM} .. ${HORIZON_TO} | http://0.0.0.0:${PORT}"
exec uvicorn websocket_server:app --host 0.0.0.0 --port "${PORT}"
