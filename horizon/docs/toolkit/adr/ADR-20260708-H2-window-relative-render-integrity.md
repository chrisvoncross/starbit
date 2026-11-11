---
id: ADR-20260708-H2
kind: adr
status: accepted
date: 2026-07-08
scope: horizon
relates_to: [ADR-20260707-30, ADR-20260708-32, ADR-20260527-25]
---

# ADR-20260708-H2 — Window-relative render integrity + the on-screen proof HUD

## Context

Under ADR-20260707-30 the transformer sees features normalized by **strictly-
causal per-window instance-norm**: for each window, each symbol, each feature,
the z-scoring statistics come ONLY from that window's lookback segment
`[:seq_len]`. Two consequences make naive visualization actively misleading:

* A feature's normalized value is **not** a global quantity — it is only
  meaningful relative to its own window's lookback. Plotting a continuous
  normalized ribbon across many windows would imply a shared scale that does not
  exist, and would silently re-scale as the window changed.
* The picture is only trustworthy if the viewer can see the exact data
  coordinate it was drawn from: which pinned snapshot, which resolved Lance
  version, which feature_version, and which feature-column order.

## Decision

The visualization renders normalized channels **per-window** and makes the
integrity contract visible on screen:

1. **One window at a time.** The 203×T heatmap and the per-policy channels show a
   single contiguous window (`seq_len` lookback + `horizon`), with a
   `lookback | horizon` divider. Scrollback/step loads an EARLIER window of the
   **same immutable pinned snapshot** — never a re-scaled continuous ribbon.
2. **The proof HUD** (top-right) always shows: the as-of pin (kind + value), the
   **resolved Lance version** with a stability flag (it must NOT change across
   scrollback), the `feature_version`, the sampled batch shape + `feature_dim`
   (== 203), coverage, available tags, and that normalization is
   window-relative. `FEATURE_COLUMNS` order is served with every window and the
   heatmap hover inspector names the exact `(index, name, policy, value)` cell.
3. **Drawn == ingested.** The heatmap values are the literal output of Orion's
   `materialize_window` (i.e. what the sampler stacks into the model batch);
   Horizon applies NO client-side re-scaling. This is cross-checked at runtime
   (see the runbook: 0 mismatches across all 203×T cells against
   `materialize_window`, and the per-feature stats reconcile with
   `preflight_assay.feature_norm_stats`).
4. **NOT-READY is a first-class UI state.** When the R2 sealed features dataset
   is not available (or Orion is unavailable), `/api/preflight` returns
   `ready=false` with actionable reasons and the client shows a NOT-READY
   overlay — never a
   blank chart or a crash.

## Consequences

* The viewer can never mistake a window-relative value for a global one, and can
  always audit the exact pinned coordinate the picture came from.
* Scroll interactions are constrained to the same snapshot; the resolved version
  is asserted stable in the HUD, making PIT-correctness visually obvious.
* The heatmap is a custom series (v5.2 `hitTest`) rather than an image, so
  click/hover inspection of any exact ingested cell is possible.
