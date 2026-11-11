---
id: ADR-20260708-H1
kind: adr
status: accepted
date: 2026-07-08
scope: horizon
supersedes: none
relates_to: [ADR-20260706-27, ADR-20260707-30, ADR-20260707-31, ADR-20260527-25, ADR-20260708-32]
---

# ADR-20260708-H1 — Horizon is a read-only Orion reuse-seam viz (clean fork of the base chart)

## Context

The existing `lightweight_charts/` viz renders the **raw self-farmed** chart:
it reads farmer Lance and re-derives its own indicators from
`mega_fusion_features_N` columns via a private `FEATURE_INDEX_MAP`. That is
**not** what the Orion world-model transformer ingests. The transformer input is
the pinned R2 sealed `feature_vec[203]`, drawn through the
production `WindowedSampler` and normalized by strictly-causal per-window
instance-norm (ADR-20260707-30). The raw viz differs in data source, feature
map, feature order, and normalization — so it cannot be trusted as a picture of
the model input.

We want a visualization that is provably the model input, without disturbing the
existing viz and without risk to the concurrently-edited `orion/` / `nexus/`
trees.

## Decision

Create a new pillar `horizon/` as a **clean 1:1 fork** of the base viz plus a
**transformer-view** layer, under a strict contract:

1. **Isolation.** Files are created/modified ONLY under `horizon/`.
   `lightweight_charts/` stays 100% untouched (frozen legacy). `orion/` and
   `nexus/` are never edited.
2. **Read-only Orion reuse.** `transformer_view` imports the Orion data
   control-plane and reuses it VERBATIM — `preflight_assay.run_preflight_assay`,
   `windowed_sampler` (`WindowedSampler` / `MultiSymbolPanelSampler`,
   `materialize_window`, `causal_instance_norm`, `FEATURE_COLUMNS`,
   `FEATURE_NORM_POLICIES`, `FEATURE_DIM`), `data_source_pin.DataSourcePin`,
   `tier_paths`, `storage.lance_dataplane`. Horizon never re-derives features,
   re-implements normalization, or hardcodes a feature order.
3. **Two modes, one server.** `HORIZON_MODE=raw_farmer` (default) is
   behaviour-identical to the legacy base viz via the relocated
   `providers/farmer_lance_provider.py`; `HORIZON_MODE=transformer_view` serves
   the reuse seam.
4. **Lazy Orion import + fail-soft.** All Orion imports live inside
   `transformer_view_provider`, so the server boots in `raw_farmer` mode without
   torch/lance/orion, and every transformer failure (Orion unavailable, gold
   unminted, pin unresolvable, coverage short) is captured as a structured
   NOT-READY payload — never a stack trace that takes the server down.

Dead legacy files (`data_provider.py`, `export_data.py`, `chart.js`,
`data.json`, `start_server.py`) were verified unreferenced by the realtime path
and NOT copied.

## Consequences

* The model-input picture is now provable and reproduces the trainer's exact
  data path (same open/pin/sample/normalize surfaces the launch-gate assay and
  the Ray loader use).
* Two code paths share one server; the legacy viz remains available and
  unchanged for operators who want the raw farmer chart.
* Horizon inherits Orion's data-plane semantics (pin resolution, coverage
  thresholds, feature-version guard) for free — no drift possible by design.
* Cost: the transformer path depends on the shared venv having the Orion
  runtime (torch etc.); documented + smoke-checked by the boot script.
