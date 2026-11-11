---
id: bb_transformer_view_reuse_seam
kind: building_block
scope: horizon
status: accepted
last_reviewed: 2026-07-08
---

# Building block — the transformer-view reuse seam + payload contract

## Backend: `providers/transformer_view_provider.py`

`TransformerViewProvider(TransformerViewConfig)` — lazy, fail-soft. Construction
does no Orion work; the first data call resolves URIs (`tier_paths`), builds the
pin (`DataSourcePin`), opens at the pin (`lance_dataplane`), builds the Orion
sampler(s), and caches them.

All Orion access is through the **public** sampler API — including the
read-only time→offset index accessors of Orion `ADR-20260708-33`
(`WindowedSampler.symbol_time_index(symbol) -> SymbolTimeIndex` and
`MultiSymbolPanelSampler.panel_time_to_offset() -> Mapping`). No `_`-prefixed
sampler attribute is read; point-in-time correctness lives in Orion's
`SymbolTimeIndex.window_offsets_in_range` (inclusive upper bound).

| Method | Reuses (Orion, verbatim, public) | Returns |
|---|---|---|
| `preflight()` | `run_preflight_assay(...).to_payload()` | READY/NOT-READY proof dict (never raises) |
| `symbols()` | `WindowedSampler.eligible_symbols` | `list[int]` |
| `window(symbol, start_index=None)` | `symbol_time_index` (read-only views) + `materialize_window` | one window pack |
| `window_before(symbol, before_time_s)` | `symbol_time_index(...).window_offsets_in_range` (as-of PIT) + `materialize_window` | one window pack |
| `panel_window(start_index=None)` | `panel_grid` + `panel_time_to_offset` + per-symbol `materialize_window` | panel pack |

### Window pack (single-symbol) — the client contract

```json
{
  "symbol": 0, "start_index": 24, "seq_len": 8, "horizon": 8, "window_len": 16,
  "total_windows": 25, "has_earlier": true, "has_later": false,
  "lookback_boundary_time": 1777446000,
  "times": [<wl unix-s>],
  "candles": [{"time","open","high","low","close"}, ...],          // absolute prices
  "columns": [<203 FEATURE_COLUMNS names>],
  "policies": [<203 NormPolicy strings>],
  "feature_dim": 203,
  "heatmap": [[<wl values> ] x 203]                                // [feature][time], window-relative
}
```

`heatmap[j][t] == materialize_window(...)["feats"][t][j]` exactly (no re-scaling).

### Panel pack

```json
{ "panel_symbols":[...], "times":[...], "window_len":16, "seq_len":8,
  "columns":[...203...], "time_aligned": true,
  "strips":[{"symbol":0,"heatmap":[[...] x 203]}, ...] }
```

## WebSocket frames (`HORIZON_MODE=transformer_view`)

Server → client: `tv_preflight` (always first), then `tv_not_ready` (if not
ready) OR `tv_initial` (`{symbols, window}`) / `tv_panel`. On navigation:
`tv_more` (single) / `tv_panel`. Errors: `tv_error`.

Client → server: `{type:"tv_window", symbol, start_index}`,
`{type:"tv_before", symbol, before}`, `{type:"tv_panel_window", start_index}`,
`{type:"ping"}`.

## HTTP

* `GET /` — index.html with `%%HORIZON_MODE%%` + symbol injected.
* `GET /api/config` — `{mode, symbol, interval}`.
* `GET /api/preflight` — `run_preflight_assay(...).to_payload()` (transformer_view)
  or a not-applicable note (raw_farmer). Never raises.
* `GET /{asset}` — allowlisted static assets (`ASSET_PATHS`).

## Frontend

* `bootstrap.js` dispatches on `window.__HORIZON_MODE__`.
* `core/transformer_view_controller.js` — WS wiring, proof HUD, NOT-READY
  overlay, candle rendering, scrollback, v5.2 `hitTest` inspector.
* `indicators/feature_heatmap/feature_heatmap.js` — 203×T custom series.
* `indicators/feature_channels/feature_channels.js` — per-NormPolicy lines.
* `panel/panel_view.js` — cross-asset small-multiples + alignment badge.
