"""Central symbol_id <-> symbol-name resolver for ALL Horizon views.

Single source of truth for turning the integer ``symbol`` scoping key that
lives on the Lance datasets (both the raw farmer candles and the Orion
enriched-features gold) into a human label — and back. It reuses the EXACT
same map the providers already bind (``market_data_sync.config.symbol_ids``:
the R2 auto-increment ``SYMBOL_TO_ID`` / ``ID_TO_SYMBOL`` single-source-of-
truth), so the Symbol-Swap UI can never drift from what the data plane and
the farmer provider resolve.

Fail-soft by design: the R2 symbol map is only present on the trading pod
(``market_data_sync`` is a namespace package elsewhere and the map is loaded
lazily from R2 at startup). When it is unavailable, resolution degrades to a
stable ``symbol_id <N>`` label rather than crashing — the transformer-view
data path keys off the integer id, so a missing name is cosmetic, never
correctness-affecting.
"""

from __future__ import annotations

from typing import Any


def _load_symbol_maps() -> tuple[dict[int, str], dict[str, int]]:
    """Return ``(id_to_symbol, symbol_to_id)`` from the shared R2 map.

    Returns two empty dicts when ``market_data_sync`` is not importable in
    this environment (dev boxes without the farmer data plane) — callers
    fall back to the ``symbol_id <N>`` label.
    """
    try:
        from market_data_sync.config.symbol_ids import ID_TO_SYMBOL, SYMBOL_TO_ID
    except Exception:  # noqa: BLE001 -- no farmer data plane here => degrade
        return {}, {}
    return dict(ID_TO_SYMBOL), dict(SYMBOL_TO_ID)


def name_for_id(symbol_id: int, *, id_to_symbol: dict[int, str] | None = None) -> str:
    """Resolve one ``symbol_id`` to its canonical name (or ``symbol_id <N>``)."""
    table = id_to_symbol if id_to_symbol is not None else _load_symbol_maps()[0]
    name = table.get(int(symbol_id))
    return name if name else f"symbol_id {int(symbol_id)}"


def labels_for_ids(symbol_ids: "list[int] | tuple[int, ...]") -> list[dict[str, Any]]:
    """Return ``[{"id": int, "name": str}]`` for a list of symbol ids.

    Resolved through the SAME ``market_data_sync`` R2 map the farmer provider
    binds; unresolved ids degrade to ``symbol_id <N>`` (never dropped, so the
    Symbol-Swap element always lists every eligible id the sampler exposes).
    """
    id_to_symbol, _ = _load_symbol_maps()
    return [
        {"id": int(sid), "name": name_for_id(int(sid), id_to_symbol=id_to_symbol)}
        for sid in symbol_ids
    ]
