---
id: ADR-20260708-H3
kind: adr
status: accepted
date: 2026-07-08
scope: horizon
relates_to: [ADR-20260707-30, ADR-20260707-31, ADR-20260706-27]
---

# ADR-20260708-H3 — The timeline adapter, the LWC 5.2 / pylance pins, and the Orion public-accessor FOLLOW-UP

## Context

Orion's samplers are built for **random, seeded** window draws
(`plan_window` / `sample_batch` / `plan_panel_window` / `sample_panel_batch`) —
the training access pattern. A visualization needs the opposite: a **contiguous,
time-ordered** scan so the operator can page a chosen `(symbol, time)` and scroll
back/forward through neighbouring windows of the same pinned snapshot.

`WindowedSampler.build()` already computes exactly the index we need — the
per-symbol, open_time_ms-ordered `(times, offsets)` arrays — and
`MultiSymbolPanelSampler.build()` computes the shared `panel_grid` + a per-symbol
time→offset map. But these are **private** (`_symbol_times`, `_symbol_offsets`,
`_panel_time_to_offset`). We must not edit Orion (concurrent-edit isolation).

## Decision

1. **Timeline adapter (horizon-side).** In ONE contained place
   (`TransformerViewProvider._symbol_index` / `_panel_offset_map`) Horizon uses
   the per-symbol index that Orion's `build()` already materialized to map
   `(symbol, start_index)` → contiguous row offsets. Those offsets are then fed
   to Orion's **public** `materialize_window`, which produces the normalized
   `(window_len, 203)` tensor. Horizon decides WHICH rows; Orion decides WHAT the
   model sees for them. No feature/normalization logic is re-implemented.
   *(As of 2026-07-08 this reads Orion's PUBLIC accessors — see the RESOLVED
   FOLLOW-UP below; the original landing used a contained private read.)*
2. **~~This is a documented workaround, not a pattern.~~** *(RESOLVED.)* The
   private reads were confined to the two helpers above and flagged inline
   `FOLLOW-UP(orion-public-accessor)`; they are now retired in favour of Orion
   `ADR-20260708-33`'s public accessors.
3. **Library + venv pins.**
   * Lightweight Charts **5.0.9 → 5.2.0** (CDN pin in `index.html`). v5
     API-compatible (no base-viz behaviour change) and adds the custom-series
     `hitTest` the 203×T heatmap uses for click-to-inspect.
   * `pylance>=8.0,<9.0` in `horizon/pyproject.toml` — Horizon runs INSIDE the
     shared Nexus/Orion venv and imports the Orion control-plane, which pins
     `pylance>=8,<9`. Horizon must not downgrade pylance underneath Orion; Lance's
     on-disk format is forward-compatible with data written by the Nexus write
     side. (Distinct from the frozen legacy `lightweight_charts` `>=6,<7`.)

## FOLLOW-UP (requested Orion public accessors) — RESOLVED 2026-07-08

**RESOLVED by Orion `ADR-20260708-33`** (public read-only time→offset index
accessors). The private reads are gone; the two helpers now call Orion's
public accessors:

* `WindowedSampler.symbol_time_index(symbol) -> SymbolTimeIndex` — an
  IMMUTABLE value object whose `.times` / `.offsets` are zero-copy READ-ONLY
  views (`flags.writeable is False`), plus an inclusive-upper-bound as-of
  `window_offsets_in_range(t0, t1, *, window_len)` (PIT correctness now lives
  in Orion). Orion shipped this as an immutable view — strictly stronger than
  the raw `(times, offsets)` tuple originally requested here — which also
  closes a latent mutable-alias bug (horizon's `np.asarray(int64)` had held a
  LIVE writable alias of the sampler's window-draw index).
* `MultiSymbolPanelSampler.panel_time_to_offset() -> Mapping` — the nested
  read-only `MappingProxyType` `{symbol: {open_time_ms: row_offset}}` map.

`TransformerViewProvider._symbol_index` / `._panel_offset_map` /
`window_before` were swapped to these public accessors; no `_`-prefixed
sampler attribute is read anymore. Closed in `horizon/docs/driftlog.md`
(`FU-H3-1`).

## Consequences

* A random-access training sampler powers a contiguous, PIT-correct viz timeline
  with zero duplication of the feature pipeline.
* The coupling to Orion internals is GONE — horizon reads only public,
  read-only accessors (Orion `ADR-20260708-33`); the latent mutable-alias
  corruption path is eliminated.
