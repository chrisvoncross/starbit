#!/usr/bin/env bash
set -euo pipefail

# Start Horizon in transformer_view mode — renders EXACTLY what the Orion
# world-model transformer ingests (Orion reuse seam: preflight_assay +
# windowed_sampler + data_source_pin + tier_paths, verbatim). Degrades to a
# clean NOT-READY state if the R2 sealed partitions are not yet available.
# Manual launch (not supervised by runit/spire).
#
# Env:
#   HORIZON_INTERVAL        (default Sec15 — the live training interval)
#   HORIZON_NEXUS_LANCE_URI (the EXACT enriched-features gold the trainer pins
#                            via --nexus-lance-uri; when set it is opened
#                            VERBATIM, bypassing tier_paths so Horizon reads the
#                            live run's absolute path bit-identically)
#   HORIZON_BUCKET_PREFIX   (default "" — legacy tier_paths prefix; only used
#                            when HORIZON_NEXUS_LANCE_URI is unset)
#   HORIZON_SEQ_LEN         (default 8)     HORIZON_HORIZON     (default 8)
#   HORIZON_PANEL_WIDTH     (default 1; >=2 enables the cross-asset panel)
#   HORIZON_PIN_TAG | HORIZON_PIN_VERSION | HORIZON_PIN_TIMESTAMP_MS (as-of pin)
#   HORIZON_STRICT          (default 0)     HORIZON_PORT        (default 8000)
#
# Bound-to-the-LIVE-run defaults (the training run in flight uses
# --nexus-lance-uri .../features/warm/sec15/dataset.lance
# --data-as-of-tag sealed-features-sec15-1783771200000, seq_len 8, horizon 8):
# just run this script to reproduce EXACTLY what that transformer sees.

WORKSPACE_ROOT="/workspace/ghostbyte"
HORIZON_ROOT="${WORKSPACE_ROOT}/horizon"
VENV_DIR="${WORKSPACE_ROOT}/venv"

# shellcheck disable=SC1091
[[ -f "${VENV_DIR}/bin/activate" ]] && source "${VENV_DIR}/bin/activate"

export HORIZON_MODE="transformer_view"
export HORIZON_INTERVAL="${HORIZON_INTERVAL:-Sec15}"
export HORIZON_NEXUS_LANCE_URI="${HORIZON_NEXUS_LANCE_URI:-${WORKSPACE_ROOT}/nexus/data/lance/features/warm/sec15/dataset.lance}"
export HORIZON_PIN_TAG="${HORIZON_PIN_TAG:-sealed-features-sec15-1783771200000}"
export HORIZON_BUCKET_PREFIX="${HORIZON_BUCKET_PREFIX:-}"
export HORIZON_SEQ_LEN="${HORIZON_SEQ_LEN:-8}"
export HORIZON_HORIZON="${HORIZON_HORIZON:-8}"
export HORIZON_PANEL_WIDTH="${HORIZON_PANEL_WIDTH:-1}"
export HORIZON_STRICT="${HORIZON_STRICT:-0}"
PORT="${HORIZON_PORT:-8000}"

cd "${HORIZON_ROOT}"
echo "[horizon] transformer_view | ${HORIZON_INTERVAL} seq=${HORIZON_SEQ_LEN} h=${HORIZON_HORIZON} panel=${HORIZON_PANEL_WIDTH} | http://0.0.0.0:${PORT}"
echo "[horizon] nexus_lance_uri: '${HORIZON_NEXUS_LANCE_URI:-}'"
echo "[horizon] pin: tag='${HORIZON_PIN_TAG:-}' version='${HORIZON_PIN_VERSION:-}' ts_ms='${HORIZON_PIN_TIMESTAMP_MS:-}'"
exec uvicorn websocket_server:app --host 0.0.0.0 --port "${PORT}"
