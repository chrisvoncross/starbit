# Horizon driftlog

Chronological record of drift, deltas, and explicit follow-ups for the
`horizon/` pillar. Newest first.

---

## 2026-07-11 — Transformer-Visibility: live-surface binding + central swap controls + dropout overlay

**What.** Bound the transformer view to the EXACT live training surface and made
the "what the transformer sees" guarantee machine-checked + on-screen, plus a
central Symbol-Swap + Timeframe-Swap and a data-gap (dropout) overlay.

**Deltas (all inside `horizon/`; Orion/Nexus untouched, read-only reuse):**
- **URI/surface binding (KNOWN GAP closed).** `TransformerViewConfig` gained
  `nexus_lance_uri`; `_resolve_uris` opens that explicit enriched-features gold
  VERBATIM (the live run's absolute `--nexus-lance-uri`), bypassing `tier_paths`
  (whose `prefix_segment` strips a leading `/` and so could never emit the run's
  absolute path). Wired `HORIZON_NEXUS_LANCE_URI` through `websocket_server` +
  `start_transformer_view.sh` (now defaulting to the live sec15 dataset + tag).
- **Surface-equivalence gate (fail-loud).** New
  `TransformerViewProvider._apply_surface_equivalence` proves, per preflight,
  that Horizon's render surface == the trainer's ingest surface: explicit-URI
  binding exists, as-of pin is the trainer's TAG and resolves to a concrete
  Lance version, feature_dim==203 / `FEATURE_COLUMNS` order / feature_version
  match Orion's pinned contract. Any divergence flips `ready=False` with an
  actionable reason and shows in the proof HUD (`GUARANTEED ✓` / `NOT GUARANTEED`).
- **Central Symbol-Swap + Timeframe-Swap.** New reusable ES modules
  `core/symbol_switcher.js` + `core/timeframe_switcher.js` (mounted in a
  `#hz-toolbar`), backed by `providers/symbol_registry.py` (id↔name via the SAME
  `market_data_sync` R2 map the farmer provider binds; `symbol_id <N>` fallback).
  Timeframe surface includes **Sec15** (previously missing) + Min1/Min5/Min15,
  each probed for on-disk availability; `/api/intervals` + WS `tv_set_interval`
  rebind the provider per interval (only the bound interval carries the live pin).
- **Dropout overlay.** Provider `_detect_gaps` marks Δt>cadence holes per window
  (mirror of `prove_seamless_timeseries.py` cadence logic); `plugins/dropout_bands/`
  is an `ISeriesPrimitive` background band + missing-bucket label so a 30s hole
  in a 15s series is no longer mis-read as a normal 15s step.

**Verification (2026-07-11):**
- `horizon/tests/providers/test_surface_and_swap.py` (9 tests) — binding,
  surface-equivalence guaranteed + every divergence class, gap detection,
  registry fallback. Existing `test_transformer_view_provider.py` (6) still green.
- Live smoke (read-only, no scan): dataset opens at tag → version 2,
  `feature_vec[203]`, tag registered; `/api/config` + `/api/intervals` bind the
  live sec15 surface (has_live_pin) and derive min1/min5/min15 siblings; all new
  assets served `200 application/javascript`.

**Limitation (panel).** The live run uses `RotatingCrossSectionPanelSampler`
(rotating 48-symbol cross-sections). Horizon's panel (`panel_width>=2`) keeps the
deterministic `MultiSymbolPanelSampler` (first-N eligible, time-aligned), which
proves the `(N,T,203)` alignment contract but is NOT a per-window replay of the
rotating draw. The single-symbol anchor replay is the guaranteed core; the
rotating-panel replay is deferred (see report).

---

## 2026-07-08 — Horizon pillar bootstrapped (transformer-view reuse seam)

**What.** Created `horizon/` as a clean 1:1 fork of the `lightweight_charts/`
base viz plus a `transformer_view` mode that renders EXACTLY the Orion model
input by reusing Orion's data control-plane verbatim. `lightweight_charts/` is
untouched (frozen legacy). See ADR-20260708-H1/H2/H3.

**Deltas vs the legacy base viz (raw_farmer stays behaviour-identical):**
- Lightweight Charts CDN 5.0.9 → **5.2.0** (`index.html`); v5 API-compatible.
- `pylance>=6,<7` → **`>=8,<9`** in `horizon/pyproject.toml` (shared Nexus/Orion
  venv alignment; Lance format forward-compatible). Added fastapi/uvicorn/numpy/
  pyarrow explicitly.
- `lance_data_provider.py` relocated to `providers/farmer_lance_provider.py`
  (project-root path fixed: `parents[2]`).
- `websocket_server.py` gained `HORIZON_MODE` switch, lazy providers,
  `/api/config`, `/api/preflight`, and `tv_*` WS frames. Farmer streaming +
  R2 symbol-map load wrapped fail-soft so a dev box without `market_data_sync`
  still boots + serves the page (pod behaviour unchanged; the import succeeds
  there).
- Dead legacy files NOT copied (verified unreferenced): `data_provider.py`,
  `export_data.py`, `chart.js`, `data.json`, `start_server.py`.

**Verification (reproduced 2026-07-08):**
- Server boots in both modes; `/` serves; `/api/config` + `/api/preflight` OK.
- transformer_view with an UNMINTED gold → clean NOT-READY (`ready=false`,
  actionable reasons) over HTTP + WS (`tv_preflight` then `tv_not_ready`).
- transformer_view with a synthetic gold → READY, resolved_version=1,
  feature_version=4; `tv_initial` heatmap 203×16 + 16 candles + 203 columns;
  scrollback `tv_more` on the SAME pinned snapshot (version stable).
- **drawn==ingested: 0 mismatches / 3248 cells** vs `materialize_window`;
  `feature_norm_stats` f0 mean≈0/std≈1 (causal norm).
- Panel (4 symbols): panel_shape `[1,4,8,203]`, 4 strips, `time_aligned=true`.
- Python import-clean + `py_compile` clean; no linter errors.

---

## FOLLOW-UPS

### FU-H3-1 — Orion public timeline accessors (ADR-20260708-H3) — RESOLVED 2026-07-08

**Resolved by Orion `ADR-20260708-33`** (public read-only time→offset index
accessors), landed as an uncommitted reviewable follow-up paired with the
horizon commit `b7819dd`.

The timeline adapter previously read Orion privates in two contained helpers
(`TransformerViewProvider._symbol_index`, `._panel_offset_map`):
`WindowedSampler._symbol_times` / `._symbol_offsets` and
`MultiSymbolPanelSampler._panel_time_to_offset`.

Orion now exposes (shipped STRONGER than the raw-tuple originally requested —
immutable, read-only views that also close a latent mutable-alias bug):
- `WindowedSampler.symbol_time_index(symbol) -> SymbolTimeIndex` (read-only
  `.times`/`.offsets` views + as-of `window_offsets_in_range`).
- `MultiSymbolPanelSampler.panel_time_to_offset() -> Mapping` (nested
  `MappingProxyType`).

`_symbol_index` / `_panel_offset_map` / `window_before` were swapped to the
public accessors; NO `_`-prefixed sampler attribute is read anymore (PIT lives
in Orion's `window_offsets_in_range`). Backed by
`horizon/tests/providers/test_transformer_view_provider.py`.
