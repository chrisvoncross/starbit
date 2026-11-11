---
id: horizon_agents_router
kind: agent_contract
scope: horizon pillar entry-anchor for AI agents (router; duplicates nothing)
status: accepted
last_reviewed: 2026-07-08
---

# AGENTS — HORIZON pillar router

First stop for any agent working **inside** `horizon/`. This is a router, not
content — it points you at the right source and states the hard rules.

## 1. What Horizon is

A **read-only visualization consumer of Orion**. It renders EXACTLY what the
Orion transformer ingests (`transformer_view`) and, in a retained legacy mode,
the raw self-farmed chart (`raw_farmer`). It is a clean fork of the (frozen)
`lightweight_charts/` base viz + a transformer-view layer on top.

## 2. START HERE

Read `horizon/docs/toolkit/README.md` (architecture, the two modes, the reuse
seam, how to run). Then the three ADRs under `adr/`.

## 3. Hard rules (do not violate)

- **Isolation.** Modify files ONLY under `horizon/`. NEVER edit `lightweight_charts/`
  (frozen legacy), `orion/`, `nexus/`, or anything else. Other agents edit
  nexus/orion concurrently.
- **Read-only Orion reuse.** Import `orion.control.data.preflight_assay` /
  `windowed_sampler` / `data_source_pin` / `tier_paths` /
  `orion.control.storage.lance_dataplane` and reuse them VERBATIM. Never
  re-derive features, re-implement normalization, or hardcode a feature order.
- **All Orion imports are LAZY** (inside `transformer_view_provider`), so the
  server boots in `raw_farmer` mode without torch/lance/orion present, and a
  transformer NOT-READY state is a first-class response, never a crash.
- **Window-relative integrity is non-negotiable.** The normalized 203 channels
  are rendered per-window with the as-of pin / resolved version / feature_version
  / `FEATURE_COLUMNS` order visible (the proof HUD). See ADR-20260708-H2.
- **Private Orion accessors are a documented workaround, not a pattern.** The
  timeline adapter reads Orion privates (`_symbol_times` / `_symbol_offsets` /
  `_panel_time_to_offset`) in ONE contained place with a FOLLOW-UP recorded in
  the driftlog + ADR-20260708-H3. Do NOT spread private reads; do NOT "fix" it
  by editing Orion here.

## 4. Sole source of truth per pillar

If you work inside another pillar, its toolkit governs, not this one:
`nexus/docs/toolkit/README.md`, `orion/docs/toolkit/README.md`. The repo root
router is `/workspace/ghostbyte/AGENTS.md`.

## 5. Binding upstream context (cited, not duplicated)

- Orion `preflight_assay` (ADR-20260708-32) — Horizon's `/api/preflight` is a
  thin wrapper over `run_preflight_assay(...).to_payload()`.
- Orion ADR-20260707-30 (causal norm), ADR-20260707-31 (panel feed),
  ADR-20260713-58 (on-demand R2 sealed features), ADR-20260527-25 (`DataSourcePin`).
