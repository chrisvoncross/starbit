"""Transformer-view surface-binding + Symbol/Timeframe-swap + dropout tests.

These exercise the NEW deliverables WITHOUT importing Orion/torch or scanning
the 4.2M-row live dataset (so the suite runs anywhere and never competes with
the live training run for CPU):

* the ``--nexus-lance-uri`` binding fix (:meth:`TransformerViewProvider._resolve_uris`)
  — the explicit training surface is opened verbatim, bypassing tier_paths;
* the surface-equivalence gate (:meth:`_apply_surface_equivalence`) — proves the
  guarantee holds and FAILS LOUD on every divergence class;
* the dropout detector (:meth:`_detect_gaps`) — Δt>cadence holes are marked;
* the central symbol registry fallback (``symbol_id <N>`` when the R2 map is
  absent, as on non-pod boxes).

``_apply_surface_equivalence`` / ``_resolve_uris`` (with an explicit URI) /
``_detect_gaps`` touch NO Orion import, so construction + these calls are
dependency-free.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from providers.transformer_view_provider import (
    TransformerViewConfig,
    TransformerViewProvider,
)
from providers.symbol_registry import labels_for_ids, name_for_id


def _mk_dataset_dir(tmp_path: Path) -> Path:
    ds = tmp_path / "features" / "warm" / "sec15" / "dataset.lance"
    ds.mkdir(parents=True)
    return ds


def _ready_payload(tag: str) -> dict:
    """A minimal READY assay payload shaped like ``PreflightReport.to_payload``."""
    return {
        "ready": True,
        "pin": {"kind": "tag", "value": tag},
        "resolved_lance_version": 2,
        "feature_version": "feature_vec_v4",
        "available_tags": {tag: {"version": 2}},
        "batch": {"feature_dim": 203, "shape": [1, 8, 203]},
        "checks": [
            {"name": "feature_column_order", "passed": True},
            {"name": "feature_version_guard", "passed": True},
        ],
        "not_ready_reasons": [],
        "horizon": {},
    }


def test_resolve_uris_binds_explicit_nexus_lance_uri(tmp_path: Path) -> None:
    ds = _mk_dataset_dir(tmp_path)
    cfg = TransformerViewConfig(
        bucket_prefix="", interval="Sec15", nexus_lance_uri=str(ds)
    )
    prov = TransformerViewProvider(cfg)
    features_uri, _ = prov._resolve_uris()
    assert features_uri == str(ds)  # opened VERBATIM, no tier_paths mangling


def test_surface_equivalence_guaranteed(tmp_path: Path) -> None:
    tag = "canonical-features-sec15-1783771200000"
    ds = _mk_dataset_dir(tmp_path)
    cfg = TransformerViewConfig(
        bucket_prefix="", interval="Sec15", nexus_lance_uri=str(ds), as_of_tag=tag
    )
    prov = TransformerViewProvider(cfg)
    payload = _ready_payload(tag)
    prov._apply_surface_equivalence(payload)
    surface = payload["horizon"]["surface"]
    assert surface["guaranteed"] is True
    assert surface["divergences"] == []
    assert surface["uri_source"] == "explicit_nexus_lance_uri"
    assert surface["cadence_ms"] == 15_000
    assert payload["ready"] is True


def test_surface_equivalence_fails_on_missing_uri(tmp_path: Path) -> None:
    tag = "canonical-features-sec15-1783771200000"
    missing = tmp_path / "nope" / "dataset.lance"
    cfg = TransformerViewConfig(
        bucket_prefix="", interval="Sec15", nexus_lance_uri=str(missing), as_of_tag=tag
    )
    prov = TransformerViewProvider(cfg)
    payload = _ready_payload(tag)
    prov._apply_surface_equivalence(payload)
    assert payload["ready"] is False
    assert not payload["horizon"]["surface"]["guaranteed"]
    assert any("does not exist" in r for r in payload["not_ready_reasons"])


def test_surface_equivalence_fails_on_pin_mismatch(tmp_path: Path) -> None:
    ds = _mk_dataset_dir(tmp_path)
    cfg = TransformerViewConfig(
        bucket_prefix="", interval="Sec15", nexus_lance_uri=str(ds),
        as_of_tag="the-training-tag",
    )
    prov = TransformerViewProvider(cfg)
    payload = _ready_payload("a-DIFFERENT-tag")  # pin resolved to the wrong tag
    prov._apply_surface_equivalence(payload)
    assert payload["ready"] is False
    assert any("!= expected tag" in r for r in payload["not_ready_reasons"])


def test_surface_equivalence_fails_on_column_drift(tmp_path: Path) -> None:
    tag = "canonical-features-sec15-1783771200000"
    ds = _mk_dataset_dir(tmp_path)
    cfg = TransformerViewConfig(
        bucket_prefix="", interval="Sec15", nexus_lance_uri=str(ds), as_of_tag=tag
    )
    prov = TransformerViewProvider(cfg)
    payload = _ready_payload(tag)
    payload["checks"][0]["passed"] = False  # feature_column_order drift
    prov._apply_surface_equivalence(payload)
    assert payload["ready"] is False
    assert any("FEATURE_COLUMNS order drift" in r for r in payload["not_ready_reasons"])


def test_surface_equivalence_warns_when_unbound() -> None:
    # No nexus_lance_uri (tier_paths fallback) can never prove the live binding.
    cfg = TransformerViewConfig(bucket_prefix="", interval="Sec15")
    prov = TransformerViewProvider(cfg)
    # Bypass tier_paths resolution (which would need Orion) by pre-seeding URIs.
    prov._features_uri = "features/warm/sec15/dataset.lance"
    prov._bars_uri = None
    payload = _ready_payload("some-tag")
    payload["pin"] = {"kind": "tag", "value": "some-tag"}
    prov._apply_surface_equivalence(payload)
    assert payload["ready"] is False
    assert any("no explicit nexus_lance_uri" in r for r in payload["not_ready_reasons"])


def test_detect_gaps_marks_dropouts() -> None:
    cfg = TransformerViewConfig(bucket_prefix="", interval="Sec15")  # cadence 15s
    prov = TransformerViewProvider(cfg)
    base = 1_777_446_570_000
    # bars at 0,15,30 then a 30s HOLE (skip 45), 60, 75 ms-strided by 15s.
    times = np.array(
        [base, base + 15_000, base + 30_000, base + 60_000, base + 75_000],
        dtype=np.int64,
    )
    gaps = prov._detect_gaps(times)
    assert len(gaps) == 1
    g = gaps[0]
    assert g["from_index"] == 2 and g["to_index"] == 3
    assert g["delta_ms"] == 30_000
    assert g["missing_buckets"] == 1


def test_detect_gaps_none_on_uniform_grid() -> None:
    cfg = TransformerViewConfig(bucket_prefix="", interval="Sec15")
    prov = TransformerViewProvider(cfg)
    base = 1_777_446_570_000
    times = np.array([base + i * 15_000 for i in range(8)], dtype=np.int64)
    assert prov._detect_gaps(times) == []


def test_symbol_registry_fallback_label() -> None:
    # On a box without the market_data_sync R2 map, ids degrade to a stable
    # "symbol_id <N>" label (never dropped).
    assert name_for_id(1, id_to_symbol={}) == "symbol_id 1"
    labels = labels_for_ids([1, 2, 3])
    assert [entry["id"] for entry in labels] == [1, 2, 3]
    assert all(entry["name"] for entry in labels)
