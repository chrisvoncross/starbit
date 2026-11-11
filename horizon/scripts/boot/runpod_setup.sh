#!/usr/bin/env bash
set -euo pipefail

# Horizon Viz — RunPod venv layering (nexus/orion-aware, additive, idempotent)
#
# Horizon is a READ-ONLY visualization consumer that runs INSIDE the SAME shared
# venv as Nexus + Orion (/workspace/ghostbyte/venv). This is the HORIZON half of
# that shared environment. Like the Orion boot (orion/scripts/boot/runpod_setup.sh
# — the pattern this mirrors), it is DELIBERATELY standalone + additive: it NEVER
# creates/recreates/mutates anything Nexus or Orion owns; it only LAYERS Horizon's
# own web-server Python deps on top and fail-loud smoke-checks the result.
#
#   ─ NOT owned here (owned by nexus/ then orion/ boot) ─
#     * venv creation, pylance/pyarrow/numpy/torch wheels, Rust, supervision (Nexus)
#     * the Orion control-plane deps (pydantic/pyyaml/jsonschema/click) + torch
#       runtime the transformer_view mode imports (Orion boot layers those)
#
#   ─ Horizon-owned surface (additive only) ─
#     * fastapi + uvicorn[standard] (the web server; Nexus/Orion don't install them)
#     * numpy / pyarrow / pylance floors are ALREADY satisfied by Nexus/Orion → no-op
#
# WHY additive (mirrors Orion's option (b)): the transformer_view mode reuses the
# Orion control-plane VERBATIM (orion.control.data.preflight_assay / windowed_sampler
# / storage.lance_dataplane), so Horizon MUST share Orion's venv (esp. pylance>=8,<9)
# rather than a private one. It assumes the Nexus+Orion base exists (asserts it,
# fail-loud) and layers only fastapi/uvicorn.
#
# Usage:
#   bash horizon/scripts/boot/runpod_setup.sh            # layer deps + smoke-check
#   bash horizon/scripts/boot/runpod_setup.sh --verify   # check-only; exit 0 iff ready
#   bash horizon/scripts/boot/runpod_setup.sh -h
#
# Idempotent — safe to re-run. Non-destructive to Nexus/Orion.

WORKSPACE_ROOT="/workspace/ghostbyte"
HORIZON_ROOT="${WORKSPACE_ROOT}/horizon"
VENV_DIR="${WORKSPACE_ROOT}/venv"
UV_CACHE_DIR="${WORKSPACE_ROOT}/.cache/uv"
PIP_CACHE_DIR="${WORKSPACE_ROOT}/.cache/pip"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${GREEN}[horizon]${NC} $*"; }
warn() { echo -e "${YELLOW}[horizon]${NC} $*"; }
err()  { echo -e "${RED}[horizon]${NC} $*" >&2; }
info() { echo -e "${CYAN}[horizon]${NC} $*"; }
fail() { err "FAIL: $*"; exit 1; }

VERIFY_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --verify)  VERIFY_ONLY=1; shift ;;
        -h|--help) sed -n '3,40p' "$0"; exit 0 ;;
        *)         echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Shared smoke check — the Horizon readiness assertion (fail-loud).
#
# raw_farmer needs only lance + fastapi + uvicorn. transformer_view additionally
# needs the Orion reuse seam importable. We assert BOTH the always-on web deps
# AND (as a warning-if-absent, NOT fatal) the Orion transformer path, because a
# CPU-only dev pod may legitimately run raw_farmer without the full torch stack.
# ---------------------------------------------------------------------------
_horizon_smoke_py() {
cat <<'PYEOF'
import sys

# --- always-required (both modes) ---
import lance
assert callable(getattr(lance, "dataset", None)), (
    "lance.dataset not callable — Nexus-owned pylance base is broken; "
    "re-run nexus/scripts/boot/runpod_setup.sh (do NOT pip install 'lance')"
)
import fastapi, uvicorn  # noqa: F401
import numpy, pyarrow    # noqa: F401

# --- transformer_view reuse seam (warn-only if the Orion runtime is absent) ---
tv = "unavailable"
try:
    sys.path.insert(0, "/workspace/ghostbyte")
    from orion.control.data.preflight_assay import run_preflight_assay  # noqa: F401
    from orion.control.data.windowed_sampler import FEATURE_COLUMNS      # noqa: F401
    tv = f"available (FEATURE_COLUMNS={len(FEATURE_COLUMNS)})"
except Exception as exc:  # noqa: BLE001
    tv = f"UNAVAILABLE ({type(exc).__name__}: {exc}) — raw_farmer still works"

print(
    f"OK lance {getattr(lance, '__version__', '?')} | "
    f"fastapi {fastapi.__version__} | uvicorn {uvicorn.__version__} | "
    f"transformer_view seam: {tv}"
)
PYEOF
}

if (( VERIFY_ONLY )); then
    rc=0
    note_ok()   { printf '  \033[0;32m✓\033[0m %s\n' "$*"; }
    note_bad()  { printf '  \033[0;31m✗\033[0m %s\n' "$*" >&2; rc=1; }
    info "verify: Horizon venv-layer readiness (nexus/orion-aware)"

    if [[ -f "${VENV_DIR}/pyvenv.cfg" ]]; then note_ok "venv present: ${VENV_DIR} (Nexus-owned)"
    else note_bad "venv missing at ${VENV_DIR} — run nexus/ then orion/ boot FIRST"; fi

    # shellcheck disable=SC1091
    if [[ -f "${VENV_DIR}/bin/activate" ]]; then
        source "${VENV_DIR}/bin/activate"
        if smoke_out="$(python3 -c "$(_horizon_smoke_py)" 2>&1)"; then note_ok "smoke: ${smoke_out}"
        else note_bad "smoke FAILED: ${smoke_out}"; fi
    else
        note_bad "no venv activate script — cannot run smoke check"
    fi

    if (( rc == 0 )); then
        info "verify: OK — Horizon is layered + importable on the shared venv"
        info "  raw chart:      bash ${HORIZON_ROOT}/scripts/start_farmer_chart.sh"
        info "  transformer:    bash ${HORIZON_ROOT}/scripts/start_transformer_view.sh"
    else
        err "verify: NOT ready — see ✗ above; re-run without --verify"
    fi
    exit "$rc"
fi

elapsed_start=$(date +%s)

# 0. Self-restore exec-bits (FUSE can drop +x on editor sync; chmod idempotent).
chmod_targets=()
for f in "${HORIZON_ROOT}/scripts/boot"/*.sh "${HORIZON_ROOT}/scripts"/*.sh; do
    [[ -e "$f" ]] && chmod_targets+=("$f")
done
if (( ${#chmod_targets[@]} > 0 )); then
    chmod +x "${chmod_targets[@]}"
    log "Ensured +x on ${#chmod_targets[@]} Horizon scripts."
fi

# 1. Assert the Nexus/Orion-owned base exists + is healthy (fail-loud).
if [[ ! -f "${VENV_DIR}/pyvenv.cfg" ]]; then
    fail "shared venv not found at ${VENV_DIR}.
  Horizon layers onto the Nexus-owned venv; it does not create it.
  Run nexus/scripts/boot/runpod_setup.sh (then orion/) FIRST, then re-run this."
fi
if [[ ! -d "${HORIZON_ROOT}" ]]; then
    fail "Horizon source not found at ${HORIZON_ROOT}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log "Activated shared venv: ${VENV_DIR}"

export UV_LINK_MODE=copy
export UV_CACHE_DIR PIP_CACHE_DIR
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}"

if ! python3 -c "import lance" >/dev/null 2>&1; then
    fail "Nexus-owned base (import lance) is not healthy in ${VENV_DIR}.
  Do NOT let Horizon 'fix' it — re-run nexus/scripts/boot/runpod_setup.sh."
fi
log "Nexus-owned base healthy (import lance OK) — layering Horizon web deps additively."

# 2. Layer Horizon's own web deps additively (NON-destructive).
#    --upgrade-strategy only-if-needed resolves ONLY the missing wheels
#    (fastapi, uvicorn[standard]); numpy/pyarrow/pylance floors are already
#    satisfied by Nexus/Orion → no-op, pins untouched.
log "Layering Horizon deps: pip install --upgrade-strategy only-if-needed -e '${HORIZON_ROOT}'"
python3 -m pip install --upgrade-strategy only-if-needed -e "${HORIZON_ROOT}"
log "Horizon deps layered."

# 3. Fail-loud smoke check.
log "Smoke check: lance + fastapi + uvicorn + (optional) Orion transformer seam..."
if ! smoke_out="$(python3 -c "$(_horizon_smoke_py)")"; then
    err "Horizon smoke check FAILED — refusing to declare the pod Horizon-ready."
    exit 1
fi
log "${smoke_out}"

elapsed=$(( $(date +%s) - elapsed_start ))
echo ""
log "========================================="
log "  Horizon venv-layer complete (${elapsed}s)"
log "========================================="
echo ""
info "  Launch (manual — NOT in the runit tree):"
info "    raw farmer chart : bash ${HORIZON_ROOT}/scripts/start_farmer_chart.sh"
info "    transformer view : bash ${HORIZON_ROOT}/scripts/start_transformer_view.sh"
echo ""
log "  Re-run any time (idempotent, non-destructive):"
log "    bash ${HORIZON_ROOT}/scripts/boot/runpod_setup.sh [--verify]"
echo ""
