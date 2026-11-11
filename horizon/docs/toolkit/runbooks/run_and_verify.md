---
id: rb_run_and_verify
kind: runbook
scope: horizon
status: accepted
last_reviewed: 2026-07-08
---

# Runbook — run Horizon + verify the drawn==ingested proof

## 0. Boot (one-time, additive to the shared venv)

```bash
bash horizon/scripts/boot/runpod_setup.sh            # layer fastapi/uvicorn + smoke-check
bash horizon/scripts/boot/runpod_setup.sh --verify   # check-only
```

The smoke check asserts `import lance`, `import fastapi, uvicorn`, and (warn-only
if absent) `from orion.control.data.preflight_assay import run_preflight_assay`.

## 1. Transformer view

```bash
# Point at the local features/warm prefix (R2-style relative prefix,
# resolved by tier_paths against the server cwd = horizon/):
HORIZON_BUCKET_PREFIX=<prefix> HORIZON_INTERVAL=Min1 \
  bash horizon/scripts/start_transformer_view.sh
# optional pin: HORIZON_PIN_TAG=... | HORIZON_PIN_VERSION=... | HORIZON_PIN_TIMESTAMP_MS=...
# optional panel: HORIZON_PANEL_WIDTH=5
```

Open `http://<host>:8000/`. If the gold is not minted you get the **NOT-READY**
overlay with actionable reasons (this is correct, not an error).

## 1b. One symbol, one time span, from Tigris

Raw 15s (or Min1/Min5/Min15) candles for a chosen span. `TO` written as
`YYYY-MM-DDTHH` includes that hour. A full timestamp for `TO` is exclusive.

```bash
HORIZON_SYMBOL=BTC_USDT HORIZON_FROM=2026-10-03T11 HORIZON_TO=2026-10-04T10 \
  bash horizon/scripts/start_tigris_range.sh
```

Open `http://<host>:8000/`. `curl -s localhost:8000/api/config` reports
`candles` and any missing hours in `gaps`.

## 2. Raw farmer chart (legacy base viz)

```bash
HORIZON_SYMBOL=BTC_USDT bash horizon/scripts/start_farmer_chart.sh
```

## 3. Quick endpoint smoke

```bash
curl -s localhost:8000/api/config
curl -s localhost:8000/api/preflight | python3 -m json.tool | head -40
```

## 4. The drawn==ingested cross-check (when a gold/tag exists)

The heatmap must equal Orion's `materialize_window` output byte-for-byte, and the
per-feature stats must reconcile with `preflight_assay.feature_norm_stats`:

```python
prov = TransformerViewProvider(cfg)                 # ready gold + pin
pack = prov.window(sym, None)
sampler = prov._ensure_sampler()
_, offs = prov._symbol_index(sampler, sym)
win = sampler.materialize_window(sym, offs[pack["start_index"]:pack["start_index"]+wl])
feats = np.asarray(win["feats"])                    # (wl, 203)
assert all(pack["heatmap"][j][t] == feats[t][j] for j in range(203) for t in range(wl))
```

Reference result (synthetic 2-symbol gold, seq_len=8 horizon=8): READY,
resolved_version=1, feature_version=4, heatmap 203×16, **0 mismatches / 3248
cells**, standardize f0 mean≈0.0000 std≈1.0000. Panel (4 symbols):
panel_shape `[1,4,8,203]`, 4 strips, `time_aligned=true`.

## 5. Expected NOT-READY reasons (first-class states)

* `features-gold GOLD Lance table not found at <uri>` — gold not minted yet.
* `Orion control-plane not importable (...)` — transformer runtime absent (the
  page still serves; `raw_farmer` unaffected).
* `coverage short: ...` / `no per-symbol window of length N` — insufficient rows.
* `cannot open the pinned gold table (...)` — the pin does not resolve.
