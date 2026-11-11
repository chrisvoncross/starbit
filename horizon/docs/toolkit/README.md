---
id: horizon_toolkit_readme
kind: toolkit_readme
scope: horizon visualization pillar (read-only Orion consumer)
status: accepted
last_reviewed: 2026-07-08
---

# HORIZON — "What the Transformer Sees" (toolkit README)

Horizon is a **visualization pillar** that renders, byte-for-byte, the input the
Orion world-model transformer actually ingests — and, in a retained legacy mode,
the raw self-farmed chart. It is a **READ-ONLY consumer of Orion**: it never
re-derives features, never re-implements normalization, never invents a feature
order. It imports Orion's production data control-plane and reuses it verbatim.

> If you only read one thing: the normalized 203 channels are **window-relative**
> (causal instance-norm over the lookback only). Horizon renders them per-window
> with the as-of pin, the resolved Lance version, the feature_version and the
> `FEATURE_COLUMNS` order on-screen (the **proof HUD**). Scrollback loads earlier
> windows of the **same immutable pinned snapshot** — the resolved version never
> changes as you scroll.

## 1. Two modes (`HORIZON_MODE`)

| Mode | What it shows | Data source | Needs |
|---|---|---|---|
| `raw_farmer` (default) | Legacy base viz — farmer OHLC + NWE/Z-Score/Stoch (the **raw** farmer re-derivation, **NOT** the model input) | `providers/farmer_lance_provider.py` (farmer Lance) | `market_data_sync` + farmer Lance (pod) |
| `transformer_view` | EXACTLY the transformer input: 203×T causal-normalized heatmap + per-policy channels + candles + cross-asset panel | `providers/transformer_view_provider.py` (Orion reuse seam) | importable `orion` + R2 sealed features dataset |

`raw_farmer` is behaviour-identical to the legacy `lightweight_charts/` base viz.
`lightweight_charts/` itself is **100% untouched** — Horizon is a clean fork.

## 2. The reuse seam (transformer_view)

Horizon imports and reuses VERBATIM (never copies):

* `orion.control.data.preflight_assay.run_preflight_assay` → `PreflightReport.to_payload()`
  — the READY / NOT-READY proof + the resolved pin/version/feature_version.
* `orion.control.data.windowed_sampler` — `WindowedSampler` /
  `MultiSymbolPanelSampler`, `materialize_window`, `causal_instance_norm`,
  `FEATURE_COLUMNS`, `FEATURE_NORM_POLICIES`, `FEATURE_DIM`.
* `orion.control.data.data_source_pin.DataSourcePin` — the as-of coordinate.
* `orion.control.data.tier_paths` — the canonical gold URI resolution.
* `orion.control.storage.lance_dataplane` — `open_at_pin` / `open_or_create`.

The FEATURE tensor is 100% Orion (`materialize_window` + `causal_instance_norm`).
Horizon only decides WHICH rows to show (the timeline adapter), never WHAT the
model sees for them. See ADR-20260708-H3 for the timeline adapter + the Orion
public-accessor follow-up.

## 3. Layout

```
horizon/
  index.html                       # v5.2 CDN, mode injection, proof HUD + NOT-READY overlay
  bootstrap.js                     # mode dispatcher (raw_farmer -> chart_realtime.js; tv -> transformer_view.js)
  chart_realtime.js                # VERBATIM legacy base-viz entry (raw_farmer)
  transformer_view.js              # transformer-view entry (reuses chart_bootstrap/legend)
  websocket_server.py              # FastAPI: mode switch + /api/preflight + tv_* WS frames
  core/                            # chart_bootstrap.js, legend.js, realtime_controller.js (verbatim)
                                   #   + transformer_view_controller.js (new)
  indicators/                      # verbatim base indicators + pane_manager.js
    feature_heatmap/feature_heatmap.js   # 203xT custom series (v5.2 hitTest)  [new]
    feature_channels/feature_channels.js # per-NormPolicy grouped lines        [new]
  panel/panel_view.js              # cross-asset small-multiples + alignment badge [new]
  plugins/ primitives/             # verbatim base plugins + Bloomberg-bars primitive
  providers/
    farmer_lance_provider.py       # retained raw mode (relocated from lance_data_provider.py)
    transformer_view_provider.py   # the reuse seam + timeline adapter          [new]
  scripts/boot/runpod_setup.sh     # additive venv layering (+ --verify)
  scripts/start_transformer_view.sh, start_farmer_chart.sh
  docs/toolkit/                    # this toolkit (README, AGENTS, adr/, building_blocks/, runbooks/)
```

Dead legacy files were NOT copied (verified unreferenced by the realtime path):
`data_provider.py`, `export_data.py`, `chart.js`, `data.json`, `start_server.py`.

## 4. Run it

```bash
# One-time (layers fastapi/uvicorn onto the shared Nexus/Orion venv):
bash horizon/scripts/boot/runpod_setup.sh            # or --verify

# Transformer view (degrades to NOT-READY if R2 sealed features unavailable):
bash horizon/scripts/start_transformer_view.sh       # HORIZON_PORT=8000 by default

# Retained raw farmer chart:
bash horizon/scripts/start_farmer_chart.sh
```

## 5. Pins & library

* Lightweight Charts **5.2.0** (was 5.0.9) — CDN pin in `index.html`; v5
  API-compatible; adds custom-series `hitTest` used by the heatmap.
* `pylance>=8.0,<9.0` — aligned with Orion (shared venv). See `pyproject.toml`
  comment + `orion/docs/toolkit/DEPENDENCIES.md` § "Lance cross-version rule".

## 6. Cross-linked Orion ADRs (cited, not duplicated)

* ADR-20260713-58 — on-demand training data serving from R2 sealed partitions.
* ADR-20260707-30 — strictly-causal per-window instance normalization.
* ADR-20260707-31 — the D=203 cross-asset panel feed.
* ADR-20260527-25 — the as-of `DataSourcePin`.
* ADR-20260708-32 — the training-data preflight assay (Horizon's `/api/preflight`).

## 7. Horizon ADRs

* `adr/ADR-20260708-H1-horizon-readonly-orion-reuse-seam.md`
* `adr/ADR-20260708-H2-window-relative-render-integrity.md`
* `adr/ADR-20260708-H3-timeline-adapter-and-orion-accessor-followup.md`
