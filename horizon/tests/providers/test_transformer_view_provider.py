"""Horizon transformer-view provider — public-accessor contract test.

Proves the timeline adapter drives Orion ENTIRELY through the public,
read-only accessors of Orion ``ADR-20260708-33`` (``symbol_time_index`` /
``SymbolTimeIndex.window_offsets_in_range`` / ``panel_time_to_offset``) plus
the pre-existing public surface (``eligible_symbols`` / ``panel_grid`` /
``panel_symbols`` / ``materialize_window``) — and NEVER reaches into a
``_``-prefixed sampler private (the retired reach-in that ADR-H3 flagged).

A ``_NoPrivateAccess`` proxy wraps each sampler and hard-fails if the provider
reads any of the three private index fields the old code touched. The provider
holds the proxy, so any private access from ``window()`` / ``window_before()``
/ ``panel_window()`` would raise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
# The shared enriched-gold Lance fixture builders live in the Orion test tree.
sys.path.insert(0, str(REPO_ROOT / "orion" / "tests" / "data"))

from _enriched_gold import (  # noqa: E402
    _BASE_OPEN_MS,
    _STRIDE_MS,
    build_enriched_gold_panel_dataset,
)

from orion.control.data.windowed_sampler import (  # noqa: E402
    FEATURE_DIM,
    MultiSymbolPanelSampler,
    WindowSpec,
    WindowedSampler,
)
from orion.control.storage import lance_dataplane as ldp  # noqa: E402

from providers.transformer_view_provider import (  # noqa: E402
    TransformerViewConfig,
    TransformerViewProvider,
)

_SEQ_LEN = 6
_HORIZON = 4
_WL = _SEQ_LEN + _HORIZON
_PANEL_WIDTH = 5
_PER_SYMBOL = 40


class _NoPrivateAccess:
    """Forwarding proxy that FAILS if the provider reads a sampler private."""

    _FORBIDDEN = ("_symbol_times", "_symbol_offsets", "_panel_time_to_offset")

    def __init__(self, target: object) -> None:
        object.__setattr__(self, "_target", target)

    def __getattr__(self, name: str):  # noqa: ANN204
        if name in _NoPrivateAccess._FORBIDDEN:
            raise AssertionError(
                f"provider reached into sampler private {name!r} — the public "
                "accessor seam (ADR-20260708-33) was bypassed"
            )
        return getattr(object.__getattribute__(self, "_target"), name)


def _open(path: Path) -> ldp.LanceTrainingDataset:
    contract = ldp.LancePolicyContract(
        time_key="open_time_ms",
        scoping_keys=("symbol", "interval"),
        btree_columns=("open_time_ms",),
    )
    return ldp.LanceTrainingDataset.open_or_create(path, contract=contract)


@pytest.fixture
def provider(tmp_path: Path) -> TransformerViewProvider:
    """A provider with PRE-BUILT samplers wrapped in the no-private guard.

    Bypasses URI/pin resolution (not under test) by injecting the built
    samplers directly; every DATA call still flows through the real provider
    methods and the real Orion samplers behind the guard.
    """
    ds_path = tmp_path / "nexus_lance" / "candles" / "interval=0"
    build_enriched_gold_panel_dataset(
        ds_path, n_symbols=_PANEL_WIDTH, per_symbol=_PER_SYMBOL
    )
    spec = WindowSpec(
        seq_len=_SEQ_LEN, horizon=_HORIZON, batch_size=1, seed=3, feature_dim=FEATURE_DIM
    )
    single = WindowedSampler(_open(ds_path), spec).build()
    panel = MultiSymbolPanelSampler(_open(ds_path), spec, num_assets=_PANEL_WIDTH).build()

    cfg = TransformerViewConfig(
        bucket_prefix="test/",
        interval="Sec15",
        seq_len=_SEQ_LEN,
        horizon=_HORIZON,
        panel_width=_PANEL_WIDTH,
        seed=3,
    )
    prov = TransformerViewProvider(cfg)
    prov._sampler = _NoPrivateAccess(single)
    prov._panel = _NoPrivateAccess(panel)
    return prov


def test_guard_itself_flags_a_private_read(provider: TransformerViewProvider) -> None:
    """Sanity: the guard actually fires on a forbidden attribute so the
    negative assertions below are meaningful."""
    with pytest.raises(AssertionError):
        _ = provider._panel._panel_time_to_offset
    with pytest.raises(AssertionError):
        _ = provider._sampler._symbol_offsets


def test_symbols_via_public_eligible(provider: TransformerViewProvider) -> None:
    assert provider.symbols() == [0, 1, 2, 3, 4]


def test_window_uses_public_accessors(provider: TransformerViewProvider) -> None:
    total = _PER_SYMBOL - _WL + 1
    pack = provider.window(0)  # newest by default → no private read
    assert pack["symbol"] == 0
    assert pack["window_len"] == _WL
    assert pack["total_windows"] == total
    assert pack["start_index"] == total - 1
    assert pack["has_later"] is False and pack["has_earlier"] is True
    assert len(pack["candles"]) == _WL
    assert len(pack["times"]) == _WL
    assert len(pack["heatmap"]) == FEATURE_DIM
    assert all(len(row) == _WL for row in pack["heatmap"])

    mid = provider.window(0, start_index=5)
    assert mid["start_index"] == 5
    assert mid["has_earlier"] is True and mid["has_later"] is True


def test_window_before_pit_via_orion_range(provider: TransformerViewProvider) -> None:
    """window_before must return a window whose bars all fall strictly BEFORE
    the requested instant — PIT enforced by Orion's window_offsets_in_range,
    not a horizon-side searchsorted."""
    before_ms = _BASE_OPEN_MS + 25 * _STRIDE_MS
    before_s = before_ms // 1000
    pack = provider.window_before(0, before_s)
    assert max(pack["times"]) < before_s, "returned a bar at/after the cutoff"
    assert len(pack["times"]) == _WL


def test_panel_window_uses_public_accessors(provider: TransformerViewProvider) -> None:
    pack = provider.panel_window()  # newest aligned window
    assert pack["panel_symbols"] == [0, 1, 2, 3, 4]
    assert pack["time_aligned"] is True
    assert len(pack["strips"]) == _PANEL_WIDTH
    assert len(pack["times"]) == _WL
    for strip in pack["strips"]:
        assert len(strip["heatmap"]) == FEATURE_DIM


def test_no_private_read_across_full_navigation(
    provider: TransformerViewProvider,
) -> None:
    """Exercise every data path; a private read anywhere raises AssertionError
    through the guard and fails the test."""
    provider.symbols()
    provider.window(2)
    provider.window(2, start_index=0)
    provider.window_before(2, (_BASE_OPEN_MS + 30 * _STRIDE_MS) // 1000)
    provider.panel_window()
    provider.panel_window(start_index=0)
