"""Horizon transformer-view provider — the READ-ONLY reuse seam over Lyra/Orion.

This module renders EXACTLY what the Phase-1 world-model transformer ingests by
re-using the production data path VERBATIM — it never re-derives features,
never re-implements the normalization, and never invents a feature order.

M0 import order: prefer ``vega.control.*`` (ORION-M1 clean slate under
``lyra/`` (package ``vega``)), fall back to legacy ``orion.control.*`` (e.g. ``docs/altes/orion``
on ``PYTHONPATH``) so Horizon can still point at Dreamer Orion when needed.

* dataset open at the as-of pin  -> ``{lyra|orion}.control.storage.lance_dataplane``
  (:meth:`LanceTrainingDataset.open_at_pin` / ``open_or_create``);
* the as-of coordinate           -> ``…data.data_source_pin.DataSourcePin``;
* the dataset URI resolution      -> ``…data.tier_paths``
  (``sealed_features_dataset_uri`` — the true ``feature_vec[203]`` surface);
* windowing + causal instance-norm -> ``…data.windowed_sampler``
  (:class:`WindowedSampler` / :class:`MultiSymbolPanelSampler`,
  :func:`causal_instance_norm`) — byte-untouched here;
* the READY / NOT-READY proof    -> ``…data.preflight_assay``
  (:func:`run_preflight_assay` -> ``PreflightReport.to_payload()``).

Cross-project ADRs (Orion side — cited, never duplicated):
  * ADR-20260713-58  on-demand training data serving from R2 sealed partitions.
  * ADR-20260707-30  strictly-causal per-window instance normalization.
  * ADR-20260707-31  the D=203 cross-asset panel feed.
  * ADR-20260527-25  the as-of DataSourcePin.

Integrity rule this provider enforces at the data layer (the UI mirrors it):
the normalized 203 channels are **window-relative** — their scale is derived
ONLY from that window's lookback ``[:seq_len]`` (``causal_instance_norm``). So
this provider serves one WINDOW at a time; scrollback reads an EARLIER window
of the SAME immutable pinned snapshot (never a re-scaled continuous ribbon,
never leaked future data).

Timeline adapter over Orion's PUBLIC read-only accessors
--------------------------------------------------------
Orion's seeded sampler API (``plan_window`` / ``sample_batch`` /
``plan_panel_window`` / ``sample_panel_batch``) is built for RANDOM window
draws, not for a contiguous, time-ordered timeline scan. To map a chosen
``(symbol, time-range)`` -> contiguous, time-ordered row offsets we use Orion's
PUBLIC, read-only accessors (ADR-20260708-33, discharging the ADR-20260708-H3
FOLLOW-UP):

* ``WindowedSampler.symbol_time_index(symbol) -> SymbolTimeIndex`` — an
  immutable value object whose ``.times`` / ``.offsets`` are ZERO-COPY,
  READ-ONLY (``flags.writeable is False``) views of the per-symbol contiguous
  index Orion's ``build()`` already computed, plus a point-in-time
  ``window_offsets_in_range(t0, t1, *, window_len)`` (inclusive-upper-bound
  as-of query — PIT correctness lives in Orion, not here);
* ``MultiSymbolPanelSampler.panel_time_to_offset() -> Mapping`` — the nested
  read-only ``{symbol: {open_time_ms: row_offset}}`` grid resolver.

No Orion private is read: the two contained helpers (:meth:`_symbol_index` /
:meth:`_panel_offset_map`) call the public accessors, and the read-only views
mean horizon can NEVER corrupt the sampler's window-draw index. The FEATURE
tensor itself is 100% Orion (``materialize_window`` + ``causal_instance_norm``)
— horizon only decides WHICH rows to show, never WHAT the model sees for them.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import numpy as np


class HorizonOrionUnavailable(RuntimeError):
    """Raised when neither Lyra nor legacy Orion control-plane can be imported.

    Surfaced to the client as a first-class NOT-READY state (never a crash):
    ``transformer_view`` needs torch/lance/pyarrow + an importable ``lyra``
    (preferred) or ``orion`` package; ``raw_farmer`` mode does not, so the
    server must still boot.
    """


def _import_control_surface(pkg: str) -> dict[str, Any]:
    """Import the read-only reuse surface from ``pkg`` (``lyra`` or ``orion``)."""
    assay = __import__(f"{pkg}.control.data.preflight_assay", fromlist=["*"])
    tier = __import__(f"{pkg}.control.data.tier_paths", fromlist=["*"])
    pin_mod = __import__(
        f"{pkg}.control.data.data_source_pin", fromlist=["DataSourcePin"]
    )
    ws = __import__(f"{pkg}.control.data.windowed_sampler", fromlist=["*"])
    ldp = __import__(f"{pkg}.control.storage.lance_dataplane", fromlist=["*"])
    return {
        "pkg": pkg,
        "assay": assay,
        "tier": tier,
        "DataSourcePin": pin_mod.DataSourcePin,
        "FEATURE_COLUMNS": ws.FEATURE_COLUMNS,
        "FEATURE_DIM": ws.FEATURE_DIM,
        "FEATURE_NORM_POLICIES": ws.FEATURE_NORM_POLICIES,
        "MultiSymbolPanelSampler": ws.MultiSymbolPanelSampler,
        "WindowedSampler": ws.WindowedSampler,
        "WindowSpec": ws.WindowSpec,
        "ldp": ldp,
    }


def _import_orion() -> dict[str, Any]:
    """Lazily import Lyra (preferred) or legacy Orion reuse surface.

    Kept out of module import so the server boots in ``raw_farmer`` mode
    without torch/lance/lyra/orion present.
    """
    errors: list[str] = []
    for pkg in ("vega", "orion"):
        try:
            return _import_control_surface(pkg)
        except Exception as exc:  # noqa: BLE001 -- try next / NOT-READY
            errors.append(f"{pkg}: {type(exc).__name__}: {exc}")
    raise HorizonOrionUnavailable(
        "Neither vega nor orion control-plane importable "
        f"({'; '.join(errors)}). transformer_view needs `vega` (preferred) "
        "or legacy `orion` + torch/lance/pyarrow. "
        "pip install -e lyra/ or point PYTHONPATH at docs/altes/orion."
    )


# Bar cadence (grid step) in ms per interval token — the SAME lattice Nexus
# mints (mirror of ``nexus/scripts/health/prove_seamless_timeseries.py``
# ``CADENCE_MS`` and ``tier_paths._INTERVAL_URI_TOKENS``). Used by the dropout
# overlay: any consecutive-row Δt strictly greater than the cadence is a
# single-/multi-bucket data gap that would otherwise be mis-read on screen as
# a normal one-step move (Lightweight Charts spaces bars by INDEX, not wall
# time, so a 30s hole in a 15s series looks identical to a 15s step).
_CADENCE_MS: dict[str, int] = {
    "Sec15": 15_000,
    "Min1": 60_000,
    "Min5": 300_000,
    "Min15": 900_000,
}


@dataclasses.dataclass(frozen=True)
class TransformerViewConfig:
    """Inputs mirroring the ``HORIZON_*`` env / query surface."""

    bucket_prefix: str
    interval: str = "Min1"
    seq_len: int = 8
    horizon: int = 8
    panel_width: int = 5
    seed: int = 17
    strict: bool = False
    as_of_tag: str | None = None
    as_of_version: int | None = None
    as_of_timestamp_ms: int | None = None
    # The EXACT training surface binding (ADR-20260706-27 ``--nexus-lance-uri``).
    # When set, Horizon opens THIS enriched-features gold Lance table verbatim
    # instead of resolving through ``tier_paths`` (whose ``prefix_segment``
    # strips a leading ``/`` and so can never reproduce the live run's absolute
    # ``/workspace/.../features/warm/<token>/dataset.lance`` path). This is
    # the seam that makes Horizon read bit-identically what the trainer ingests.
    nexus_lance_uri: str | None = None
    # M2: optional Lyra ORION-M1 checkpoint for ``imagination_fan`` overlay.
    # When unset / unloadable the window pack still ships with a structured
    # ``imagination_fan: {status: not_ready, …}`` payload (never a crash).
    lyra_ckpt: str | None = None
    imagination_n: int = 8
    imagination_horizon: int | None = None

    @property
    def window_len(self) -> int:
        return int(self.seq_len) + int(self.horizon)

    @property
    def is_panel(self) -> bool:
        return int(self.panel_width) >= 2

    @property
    def cadence_ms(self) -> int | None:
        """Bar cadence in ms for this interval (``None`` if unknown)."""
        return _CADENCE_MS.get(self.interval)


def _finite(x: float) -> float | None:
    """JSON-safe float (NaN/Inf -> None) — mirrors websocket _sanitize_nans."""
    xf = float(x)
    if xf != xf or xf in (float("inf"), float("-inf")):
        return None
    return xf


class TransformerViewProvider:
    """Serve the exact model-input tensor for a pinned tag + symbol + window.

    Lazy + fail-soft: construction does NO Orion work. The first data call
    resolves the URI, builds the pin, opens at the pin, builds the Orion
    sampler(s). Any failure (Orion missing, dataset not available, pin
    unresolvable, coverage short) is captured as a structured NOT-READY
    payload rather than an exception that would take the server down.
    """

    def __init__(self, cfg: TransformerViewConfig) -> None:
        self.cfg = cfg
        self._orion: dict[str, Any] | None = None
        self._features_uri: str | None = None
        self._bars_uri: str | None = None
        self._pin: Any | None = None
        self._pin_built = False
        # Cached, built-once Orion samplers (single-symbol + optional panel).
        self._sampler: Any | None = None
        self._panel: Any | None = None
        # Cached per-symbol read-only time/offset index views (via the public
        # WindowedSampler.symbol_time_index accessor; see class docstring).
        self._index_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # M2: lazy Lyra WM for imagination_fan (None = not tried / unavailable).
        self._lyra_model: Any | None = None
        self._lyra_model_tried: bool = False
        self._lyra_load_error: str | None = None

    # ------------------------------------------------------------------
    # Resolution helpers (all Orion access lazy).
    # ------------------------------------------------------------------

    def _o(self) -> dict[str, Any]:
        if self._orion is None:
            self._orion = _import_orion()
        return self._orion

    def _resolve_uris(self) -> tuple[str, str | None]:
        """Resolve ``(features_uri, bars_uri)``.

        Binding precedence (ADR-20260713-58 parity with the trainer's CLI):

        1. an explicit ``cfg.nexus_lance_uri`` — the enriched-features
           dataset the live run pins via ``--nexus-lance-uri``. Used VERBATIM
           so Horizon opens the identical dataset directory the trainer
           streams. The bars twin is derived by swapping the
           ``features/warm`` path segment for ``warm`` (informational only —
           the candles Horizon renders come from the ``*_fp`` columns of the
           SAME features table via ``materialize_window``), dropped if absent.
        2. otherwise ``tier_paths`` (the versioned-token layout) — the
           standard resolution via ``sealed_features_dataset_uri`` /
           ``sealed_bars_dataset_uri``. NOTE this can NOT emit the live run's
           absolute path (``prefix_segment`` strips the leading ``/``), which
           is exactly the KNOWN GAP the ``nexus_lance_uri`` binding closes.
        """
        if self._features_uri is None:
            from pathlib import Path

            if self.cfg.nexus_lance_uri:
                self._features_uri = str(self.cfg.nexus_lance_uri)
                sibling = str(self.cfg.nexus_lance_uri).replace(
                    "features/warm", "warm"
                )
                self._bars_uri = (
                    sibling if sibling != self._features_uri and Path(sibling).exists() else None
                )
            else:
                tier = self._o()["tier"]
                self._features_uri = tier.sealed_features_dataset_uri(
                    bucket_prefix=self.cfg.bucket_prefix, interval=self.cfg.interval
                )
                self._bars_uri = tier.sealed_bars_dataset_uri(
                    bucket_prefix=self.cfg.bucket_prefix, interval=self.cfg.interval
                )
        return self._features_uri, self._bars_uri  # type: ignore[return-value]

    def _resolve_pin(self) -> Any | None:
        if not self._pin_built:
            DataSourcePin = self._o()["DataSourcePin"]
            self._pin = DataSourcePin.from_optional_fields(
                as_of_tag=self.cfg.as_of_tag,
                as_of_manifest_version=self.cfg.as_of_version,
                as_of_timestamp_ms=self.cfg.as_of_timestamp_ms,
            )
            self._pin_built = True
        return self._pin

    def _assay_config(self) -> Any:
        assay = self._o()["assay"]
        features_uri, bars_uri = self._resolve_uris()
        return assay.PreflightAssayConfig(
            features_uri=features_uri,
            bars_uri=bars_uri,
            pin=self._resolve_pin(),
            seq_len=self.cfg.seq_len,
            horizon=self.cfg.horizon,
            batch_size=1,
            panel_width=self.cfg.panel_width if self.cfg.is_panel else 1,
            seed=self.cfg.seed,
            strict=self.cfg.strict,
        )

    # ------------------------------------------------------------------
    # § 1. Preflight — the READY / NOT-READY proof (never raises).
    # ------------------------------------------------------------------

    def preflight(self) -> dict[str, Any]:
        """Return ``run_preflight_assay(...).to_payload()`` + a few UI extras.

        Fail-soft: an env/import problem is reported as a NOT-READY payload
        with ``ready == False`` and an actionable reason — the client renders
        it as a first-class state, never a stack trace.
        """
        base_meta = {
            "mode": "transformer_view",
            "interval": self.cfg.interval,
            "seq_len": self.cfg.seq_len,
            "horizon": self.cfg.horizon,
            "panel_width": self.cfg.panel_width,
            "cadence_ms": self.cfg.cadence_ms,
            "nexus_lance_uri": self.cfg.nexus_lance_uri,
            "as_of": {
                "tag": self.cfg.as_of_tag,
                "version": self.cfg.as_of_version,
                "timestamp_ms": self.cfg.as_of_timestamp_ms,
            },
        }
        try:
            assay = self._o()["assay"]
            report = assay.run_preflight_assay(self._assay_config())
            payload = report.to_payload()
            payload["horizon"] = base_meta
            payload["render_text"] = report.render()
            # Surface-equivalence gate: prove (or fail loud) that what Horizon
            # renders is bit-identical to what the trainer ingested.
            self._apply_surface_equivalence(payload)
            return payload
        except HorizonOrionUnavailable as exc:
            return {
                "ready": False,
                "not_ready_reasons": [str(exc)],
                "hard_failures": [],
                "warnings": [],
                "horizon": {**base_meta, "orion_available": False},
            }
        except Exception as exc:  # noqa: BLE001 -- last-resort fail-soft
            return {
                "ready": False,
                "not_ready_reasons": [
                    f"preflight failed unexpectedly ({type(exc).__name__}: {exc})."
                ],
                "hard_failures": [],
                "warnings": [],
                "horizon": base_meta,
            }

    # ------------------------------------------------------------------
    # § 1a. Surface-equivalence gate — the MATHEMATICAL guarantee.
    # ------------------------------------------------------------------

    def _apply_surface_equivalence(self, payload: dict[str, Any]) -> None:
        """Assert Horizon's render surface == the trainer's ingest surface.

        The guarantee is Iceberg/Lance time-travel: pinning the SAME dataset
        directory at the SAME as-of tag the trainer used yields the identical
        immutable snapshot version, and Orion's sampler + causal-norm code is
        reused VERBATIM — so a window Horizon materialises is byte-identical to
        the one the trainer ingested. This method turns that into a machine-
        checked verdict and FAILS LOUD (flips ``ready`` to ``False`` with an
        actionable reason) the moment any leg diverges:

        * the pinned dataset directory must be the explicit ``--nexus-lance-uri``
          binding and must exist on disk (tier_paths fallback can not reproduce
          the live absolute path — see :meth:`_resolve_uris`);
        * the as-of pin must be a TAG equal to the trainer's ``--data-as-of-tag``
          and that tag must be registered on the dataset and resolve to a
          concrete Lance version;
        * the feature surface must match Orion's pinned contract — feature_dim
          == 203, ``FEATURE_COLUMNS`` order intact, feature_version present
          (these three are already asserted by ``run_preflight_assay``; we
          re-read its verdicts so the surface block is one coherent proof).
        """
        features_uri, bars_uri = self._resolve_uris()
        from pathlib import Path

        pin = payload.get("pin") or {}
        checks = {c.get("name"): c for c in payload.get("checks", [])}
        cols_ok = bool(checks.get("feature_column_order", {}).get("passed"))
        fver_ok = bool(checks.get("feature_version_guard", {}).get("passed"))
        feature_dim = int((payload.get("batch") or {}).get("feature_dim") or 0)
        resolved_version = payload.get("resolved_lance_version")
        available_tags = payload.get("available_tags") or {}

        expected_tag = self.cfg.as_of_tag
        uri_explicit = bool(self.cfg.nexus_lance_uri)
        uri_exists = bool(features_uri) and Path(features_uri).exists()
        pin_is_expected_tag = (
            pin.get("kind") == "tag"
            and expected_tag is not None
            and pin.get("value") == expected_tag
        )
        tag_registered = expected_tag in available_tags if expected_tag else False

        divergences: list[str] = []
        if not uri_explicit:
            divergences.append(
                "no explicit nexus_lance_uri binding — Horizon can NOT prove it "
                "reads the live training dataset (tier_paths cannot emit the "
                "run's absolute path). Set HORIZON_NEXUS_LANCE_URI to the "
                "trainer's --nexus-lance-uri."
            )
        elif not uri_exists:
            divergences.append(
                f"bound nexus_lance_uri does not exist on disk: {features_uri}"
            )
        if expected_tag is None:
            divergences.append(
                "no as-of tag pin — the trainer pins --data-as-of-tag; without "
                "the same tag the snapshot version is not guaranteed identical."
            )
        elif not pin_is_expected_tag:
            divergences.append(
                f"as-of pin {pin.get('kind')}={pin.get('value')!r} != expected "
                f"tag {expected_tag!r} (the trainer's --data-as-of-tag)."
            )
        elif not tag_registered and resolved_version is None:
            divergences.append(
                f"as-of tag {expected_tag!r} is not registered on the dataset "
                f"and did not resolve to a version (available: {sorted(available_tags)})."
            )
        # Schema/columns/version legs — only meaningful once a REAL batch was
        # pulled through the production path (``batch.shape`` present). When the
        # assay bailed earlier (missing gold, short coverage, ...) its own
        # not_ready_reasons already explain it; we don't pile on a misleading
        # "column drift" that was never actually evaluated.
        if (payload.get("batch") or {}).get("shape") is not None:
            if feature_dim != 203:
                divergences.append(f"feature_dim {feature_dim} != 203 (feature_vec extent drift).")
            if not cols_ok:
                divergences.append("FEATURE_COLUMNS order drift vs Orion's pinned surface.")
            if not fver_ok:
                divergences.append("feature_version guard failed (wrong feature surface).")

        guaranteed = not divergences
        payload.setdefault("horizon", {})["surface"] = {
            "guaranteed": guaranteed,
            "features_uri": features_uri,
            "bars_uri": bars_uri,
            "uri_source": "explicit_nexus_lance_uri" if uri_explicit else "tier_paths",
            "uri_exists": uri_exists,
            "expected_as_of_tag": expected_tag,
            "pin_is_expected_tag": pin_is_expected_tag,
            "tag_registered": tag_registered,
            "resolved_lance_version": resolved_version,
            "feature_version": payload.get("feature_version"),
            "feature_columns_ok": cols_ok,
            "feature_dim": feature_dim or None,
            "cadence_ms": self.cfg.cadence_ms,
            "divergences": divergences,
        }
        if not guaranteed:
            payload["ready"] = False
            reasons = payload.setdefault("not_ready_reasons", [])
            for d in divergences:
                reasons.append(f"surface-equivalence: {d}")

    # ------------------------------------------------------------------
    # § 2. Sampler build (cached) + the timeline adapter.
    # ------------------------------------------------------------------

    def _open_dataset(self) -> Any:
        ldp = self._o()["ldp"]
        from pathlib import Path

        features_uri, _ = self._resolve_uris()
        contract = ldp.LancePolicyContract(
            time_key="open_time_ms",
            scoping_keys=("symbol", "interval"),
            btree_columns=("open_time_ms",),
        )
        pin = self._resolve_pin()
        path = Path(features_uri)
        if pin is not None:
            return ldp.LanceTrainingDataset.open_at_pin(path, contract=contract, pin=pin)
        return ldp.LanceTrainingDataset.open_or_create(path, contract=contract)

    def _spec(self) -> Any:
        WindowSpec = self._o()["WindowSpec"]
        FEATURE_DIM = self._o()["FEATURE_DIM"]
        return WindowSpec(
            seq_len=self.cfg.seq_len,
            horizon=self.cfg.horizon,
            batch_size=1,
            seed=self.cfg.seed,
            feature_dim=FEATURE_DIM,
        )

    def _ensure_sampler(self) -> Any:
        if self._sampler is None:
            WindowedSampler = self._o()["WindowedSampler"]
            ds = self._open_dataset()
            self._sampler = WindowedSampler(ds, self._spec()).build()
        return self._sampler

    def _ensure_panel(self) -> Any:
        if self._panel is None:
            MultiSymbolPanelSampler = self._o()["MultiSymbolPanelSampler"]
            ds = self._open_dataset()
            self._panel = MultiSymbolPanelSampler(
                ds, self._spec(), num_assets=int(self.cfg.panel_width)
            ).build()
        return self._panel

    def _symbol_index(self, sampler: Any, symbol: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(times, offsets)`` for ``symbol`` via Orion's PUBLIC
        read-only accessor ``WindowedSampler.symbol_time_index`` (ADR-33).

        The accessor returns an immutable ``SymbolTimeIndex`` whose ``.times`` /
        ``.offsets`` are zero-copy READ-ONLY views (``writeable=False``) of the
        per-symbol contiguous index Orion's ``build()`` computed — so horizon
        pages a time-ordered timeline without touching Orion privates and
        without any risk of corrupting the sampler's window-draw index. The
        feature tensor still flows through the public ``materialize_window``.
        """
        if symbol not in self._index_cache:
            idx = sampler.symbol_time_index(int(symbol))
            self._index_cache[symbol] = (idx.times, idx.offsets)
        return self._index_cache[symbol]

    # ------------------------------------------------------------------
    # § 3. Public data surface (single-symbol windows).
    # ------------------------------------------------------------------

    def symbols(self) -> list[int]:
        return list(self._ensure_sampler().eligible_symbols)

    def symbol_labels(self) -> list[dict[str, Any]]:
        """Return ``[{"id", "name"}]`` for every eligible symbol.

        Names resolve through the SAME ``market_data_sync`` R2 map the farmer
        provider binds (via :mod:`providers.symbol_registry`); unresolved ids
        degrade to ``symbol_id <N>`` so the Symbol-Swap element always lists
        the full eligible universe the sampler exposes."""
        from .symbol_registry import labels_for_ids

        return labels_for_ids(self.symbols())

    def _detect_gaps(self, times_ms: "np.ndarray") -> list[dict[str, Any]]:
        """Mark single-/multi-bucket data gaps INSIDE this window.

        A gap is any consecutive-row Δt strictly greater than the interval
        cadence — i.e. exactly the ``Δt > cadence`` holes ``prove_seamless_
        timeseries.py`` classifies. Orion's sampler windows over consecutive
        ROWS (not fixed-Δt time), so a 30s hole in a 15s series occupies ONE
        index step and, since Lightweight Charts spaces bars by index, would
        otherwise be mis-read as a normal 15s move. Each gap is reported as
        the (from,to) bar pair it sits between plus the count of missing
        buckets, so the overlay can band it and label it on screen.
        """
        cad = self.cfg.cadence_ms
        if not cad or times_ms is None or len(times_ms) < 2:
            return []
        gaps: list[dict[str, Any]] = []
        t = np.asarray(times_ms, dtype=np.int64)
        deltas = np.diff(t)
        for i in np.flatnonzero(deltas > cad):
            delta = int(deltas[i])
            gaps.append(
                {
                    "from_index": int(i),
                    "to_index": int(i) + 1,
                    "from_time": int(t[i] // 1000),
                    "to_time": int(t[i + 1] // 1000),
                    "delta_ms": delta,
                    "missing_buckets": int(delta // cad) - 1,
                }
            )
        return gaps

    def _window_pack(
        self, sampler: Any, symbol: int, start_index: int
    ) -> dict[str, Any]:
        """Materialise ONE contiguous window via Orion's public
        ``materialize_window`` and package it for the client."""
        FEATURE_COLUMNS = self._o()["FEATURE_COLUMNS"]
        FEATURE_NORM_POLICIES = self._o()["FEATURE_NORM_POLICIES"]
        _, offs = self._symbol_index(sampler, symbol)
        wl = self.cfg.window_len
        offsets = offs[start_index : start_index + wl]
        win = sampler.materialize_window(int(symbol), offsets)  # Orion, verbatim

        feats = np.asarray(win["feats"], dtype=np.float64)  # (wl, 203), normalized
        times = np.asarray(win["open_time_ms"], dtype=np.int64)
        o = np.asarray(win["open"], dtype=np.float64)
        h = np.asarray(win["high"], dtype=np.float64)
        low = np.asarray(win["low"], dtype=np.float64)
        c = np.asarray(win["close"], dtype=np.float64)

        candles = [
            {
                "time": int(times[i] // 1000),
                "open": _finite(o[i]),
                "high": _finite(h[i]),
                "low": _finite(low[i]),
                "close": _finite(c[i]),
            }
            for i in range(len(times))
        ]
        # Heatmap matrix: features (203 rows) x time (wl cols), JSON-safe.
        heatmap = [[_finite(feats[t, j]) for t in range(feats.shape[0])] for j in range(feats.shape[1])]
        return {
            "symbol": int(symbol),
            "start_index": int(start_index),
            "seq_len": int(self.cfg.seq_len),
            "horizon": int(self.cfg.horizon),
            "window_len": int(wl),
            "cadence_ms": self.cfg.cadence_ms,
            "gaps": self._detect_gaps(times),
            # The lookback|horizon boundary time (bar at index seq_len-1 is the
            # last bar whose info the norm may use; horizon rows never feed it).
            "lookback_boundary_time": int(times[self.cfg.seq_len - 1] // 1000)
            if len(times) >= self.cfg.seq_len
            else None,
            "times": [int(t // 1000) for t in times],
            "candles": candles,
            "columns": list(FEATURE_COLUMNS),
            "policies": [p.value for p in FEATURE_NORM_POLICIES],
            "feature_dim": int(feats.shape[1]),
            "heatmap": heatmap,
            # M2 L7 overlay — fail-soft (not_ready when no lyra ckpt).
            "imagination_fan": self.imagination_fan_for_features(feats),
        }

    def _resolve_lyra_ckpt(self) -> str | None:
        """Checkpoint path from config or ``LYRA_CKPT`` / ``HORIZON_LYRA_CKPT``."""
        if self.cfg.lyra_ckpt:
            return str(self.cfg.lyra_ckpt)
        import os

        return (
            os.environ.get("VEGA_CKPT")
            or os.environ.get("HORIZON_VEGA_CKPT")
            or os.environ.get("LYRA_CKPT")
            or os.environ.get("HORIZON_LYRA_CKPT")
        )

    def _ensure_lyra_model(self) -> Any | None:
        """Load Lyra ORION-M1 once; return ``None`` if unavailable (fail-soft)."""
        if self._lyra_model_tried:
            return self._lyra_model
        self._lyra_model_tried = True
        ckpt = self._resolve_lyra_ckpt()
        if not ckpt:
            self._lyra_model = None
            self._lyra_load_error = "no checkpoint path"
            return None
        try:
            import torch
            from vega.wm.model import OrionM1, OrionM1Config

            try:
                blob = torch.load(ckpt, map_location="cpu", weights_only=False)
            except TypeError:
                blob = torch.load(ckpt, map_location="cpu")

            cfg_kwargs: dict[str, Any] = {}
            state = blob
            if isinstance(blob, dict):
                if "config" in blob and isinstance(blob["config"], dict):
                    cfg_kwargs = {
                        k: v
                        for k, v in blob["config"].items()
                        if hasattr(OrionM1Config, k) or k in OrionM1Config.__dataclass_fields__
                    }
                if "model" in blob:
                    state = blob["model"]
                elif "state_dict" in blob:
                    state = blob["state_dict"]

            model = OrionM1(OrionM1Config(**cfg_kwargs)) if cfg_kwargs else OrionM1()
            model.load_state_dict(state)
            model.eval()
            self._lyra_model = model
            self._lyra_load_error = None
        except Exception as exc:  # noqa: BLE001 — degrade cleanly for Horizon
            self._lyra_model = None
            self._lyra_load_error = f"{type(exc).__name__}: {exc}"
        return self._lyra_model

    def imagination_fan_for_features(
        self, feats: "np.ndarray"
    ) -> dict[str, Any]:
        """Build M2 ``imagination_fan`` payload from a window feature matrix.

        Uses the lookback ``[:seq_len]`` as context and rolls
        ``imagination_horizon`` (default = cfg.horizon) steps. Without a
        loadable Lyra checkpoint returns a structured ``not_ready`` dict —
        never raises.
        """
        ckpt = self._resolve_lyra_ckpt()
        if not ckpt:
            return {
                "type": "imagination_fan",
                "status": "not_ready",
                "reason": (
                    "no Vega checkpoint — set VEGA_CKPT / LYRA_CKPT "
                    "or TransformerViewConfig.lyra_ckpt to enable the fan overlay"
                ),
            }
        model = self._ensure_lyra_model()
        if model is None:
            detail = self._lyra_load_error or "unknown load error"
            return {
                "type": "imagination_fan",
                "status": "not_ready",
                "reason": f"failed to load Lyra checkpoint at {ckpt!r}: {detail}",
                "ckpt": ckpt,
            }
        try:
            import torch
            from vega.wm.imagination import imagine_fan

            seq = int(self.cfg.seq_len)
            arr = np.asarray(feats, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 1:
                return {
                    "type": "imagination_fan",
                    "status": "not_ready",
                    "reason": "empty feature window",
                }
            ctx = arr[:seq]
            # Pad / trim feature dim to model expectation if surface differs.
            f_dim = int(model.config.feature_dim)
            if ctx.shape[1] < f_dim:
                pad = np.zeros((ctx.shape[0], f_dim - ctx.shape[1]), dtype=np.float32)
                ctx = np.concatenate([ctx, pad], axis=1)
            elif ctx.shape[1] > f_dim:
                ctx = ctx[:, :f_dim]
            x = torch.from_numpy(ctx).unsqueeze(0)  # (1, T, F)
            h_roll = int(
                self.cfg.imagination_horizon
                if self.cfg.imagination_horizon is not None
                else self.cfg.horizon
            )
            h_roll = max(1, h_roll)
            fan = imagine_fan(
                model,
                x,
                n_trajectories=max(1, int(self.cfg.imagination_n)),
                horizon=h_roll,
            )
            payload = fan.as_payload(max_trajectories=min(8, int(self.cfg.imagination_n)))
            payload["ckpt"] = ckpt
            payload["context_len"] = int(ctx.shape[0])
            return payload
        except Exception as exc:  # noqa: BLE001
            return {
                "type": "imagination_fan",
                "status": "not_ready",
                "reason": f"{type(exc).__name__}: {exc}",
                "ckpt": ckpt,
            }

    def window(self, symbol: int, start_index: int | None = None) -> dict[str, Any]:
        """Return one window pack + navigation info (newest window by default).

        ``start_index`` is an index into the symbol's contiguous
        open_time_ms-ordered rows. Scrollback decrements it; the SAME pinned
        snapshot is read (resolved version never changes).

        M2: attaches ``imagination_fan`` (ok or not_ready) for the L7
        "what the model dreams" overlay — see lyra README §Horizon fan.
        """
        sampler = self._ensure_sampler()
        times, _ = self._symbol_index(sampler, symbol)
        n = int(len(times))
        wl = self.cfg.window_len
        total_windows = max(0, n - wl + 1)
        if total_windows == 0:
            raise ValueError(
                f"symbol {symbol}: only {n} contiguous rows < window_len {wl}"
            )
        if start_index is None:
            start_index = total_windows - 1  # newest full window
        start_index = max(0, min(int(start_index), total_windows - 1))
        pack = self._window_pack(sampler, symbol, start_index)
        pack["total_windows"] = total_windows
        pack["has_earlier"] = start_index > 0
        pack["has_later"] = start_index < total_windows - 1
        return pack
    def window_before(self, symbol: int, before_time_s: int) -> dict[str, Any]:
        """Return the latest window whose bars all fall strictly BEFORE
        ``before_time_s`` (scrollback). Reads the same pinned snapshot.

        Point-in-time correctness lives in ORION: we ask the public
        ``SymbolTimeIndex.window_offsets_in_range`` for the latest full window
        within ``[first_bar, before)`` (inclusive upper bound ``before-1``ms),
        so no returned bar is at/after the requested instant — no hand-rolled
        as-of ``searchsorted`` on the horizon side."""
        sampler = self._ensure_sampler()
        idx = sampler.symbol_time_index(int(symbol))
        wl = self.cfg.window_len
        before_ms = int(before_time_s) * 1000
        offsets = idx.window_offsets_in_range(
            int(idx.times[0]), before_ms - 1, window_len=wl
        )
        if offsets.size == 0:
            raise ValueError("no earlier window before the requested time")
        # Map the PIT-selected window back to its contiguous-grid start index
        # so the nav pack (total_windows / has_earlier / has_later) stays
        # consistent with window(); dataset row offsets are unique.
        start_index = int(
            np.flatnonzero(np.asarray(idx.offsets) == int(offsets[0]))[0]
        )
        return self.window(symbol, start_index=start_index)

    # ------------------------------------------------------------------
    # § 4. Cross-asset panel (small multiples + alignment proof).
    # ------------------------------------------------------------------

    def _panel_offset_map(self, panel: Any) -> "Mapping[int, Mapping[int, int]]":
        """Return ``{symbol: {open_time_ms: row_offset}}`` via Orion's PUBLIC
        read-only accessor ``MultiSymbolPanelSampler.panel_time_to_offset``
        (ADR-33) — nested read-only ``MappingProxyType``, no reach into any
        private field."""
        return panel.panel_time_to_offset()

    def panel_window(self, start_index: int | None = None) -> dict[str, Any]:
        """Materialise ONE contiguous, time-aligned cross-symbol panel window.

        Uses the public ``panel_symbols`` + ``panel_grid`` and Orion's public
        ``materialize_window`` per symbol at the SAME grid timestamps, proving
        cross-symbol alignment on screen."""
        panel = self._ensure_panel()
        grid = np.asarray(panel.panel_grid, dtype=np.int64)  # public
        symbols = list(panel.panel_symbols)  # public
        wl = self.cfg.window_len
        total_windows = max(0, int(len(grid)) - wl + 1)
        if total_windows == 0:
            raise ValueError(
                f"panel grid has {len(grid)} aligned rows < window_len {wl}"
            )
        if start_index is None:
            start_index = total_windows - 1
        start_index = max(0, min(int(start_index), total_windows - 1))
        grid_times = grid[start_index : start_index + wl]
        t2o = self._panel_offset_map(panel)

        strips: list[dict[str, Any]] = []
        aligned = True
        for sym in symbols:
            offsets = [t2o[int(sym)][int(t)] for t in grid_times.tolist()]
            win = panel.materialize_window(int(sym), offsets)  # Orion, verbatim
            wtimes = np.asarray(win["open_time_ms"], dtype=np.int64)
            if not np.array_equal(wtimes, grid_times):
                aligned = False
            feats = np.asarray(win["feats"], dtype=np.float64)
            strips.append(
                {
                    "symbol": int(sym),
                    "heatmap": [
                        [_finite(feats[t, j]) for t in range(feats.shape[0])]
                        for j in range(feats.shape[1])
                    ],
                }
            )
        FEATURE_COLUMNS = self._o()["FEATURE_COLUMNS"]
        return {
            "panel_symbols": [int(s) for s in symbols],
            "start_index": int(start_index),
            "total_windows": total_windows,
            "has_earlier": start_index > 0,
            "has_later": start_index < total_windows - 1,
            "times": [int(t // 1000) for t in grid_times],
            "window_len": int(wl),
            "seq_len": int(self.cfg.seq_len),
            "columns": list(FEATURE_COLUMNS),
            "time_aligned": bool(aligned),
            "strips": strips,
        }
