"""One scale per lane and symbol. The formula belongs to Nova. Door does not choose it.

The batch mixes symbols, and one lane holds prices from a few cents to BTC.
A pooled median would leave BTC's book prices near a hundred thousand spreads.
Each symbol therefore gets its own median and MAD per lane, fitted on every
``SAMPLE_EVERY``-th candle of the practice days, and its own typical candle
range for the four prices. The pooled fit stays for symbols the practice
days did not hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from orion.models.door.walk import feature_scales

from ..spec import CANDLE_MS, LANES, SAMPLE_EVERY


def fit_column(values: np.ndarray) -> tuple[float, float]:
    column = np.asarray(values, dtype=np.float64)
    column = column[np.isfinite(column)]
    if column.size == 0:
        return 0.0, 1.0
    center = float(np.median(column))
    mad = float(np.median(np.abs(column - center)))
    return center, 1.0 if mad == 0.0 else mad


def fit_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> float:
    """Median of high minus low over close, in basis points. One if there is none."""

    high, low, close = (np.asarray(item, dtype=np.float64) for item in (high, low, close))
    keep = np.isfinite(high) & np.isfinite(low) & np.isfinite(close) & (close > 0.0)
    if not np.any(keep):
        return 1.0
    spans = (high[keep] - low[keep]) / close[keep] * 1e4
    spans = spans[spans > 0.0]
    return 1.0 if spans.size == 0 else float(np.median(spans))


@dataclass
class Scale:
    center: np.ndarray
    spread: np.ndarray
    span: float
    symbols: dict[int, tuple[np.ndarray, np.ndarray, float]]

    def of(self, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lane center and spread ``(rows, LANES)``, and the price span ``(rows,)``."""

        rows = len(symbols)
        center = np.empty((rows, LANES), dtype=np.float64)
        spread = np.empty((rows, LANES), dtype=np.float64)
        span = np.empty(rows, dtype=np.float64)
        for row, symbol in enumerate(np.asarray(symbols, dtype=np.int64)):
            own = self.symbols.get(int(symbol))
            if own is None:
                center[row], spread[row], span[row] = self.center, self.spread, self.span
            else:
                center[row], spread[row], span[row] = own
        return center, spread, span

    def export(self) -> dict[str, torch.Tensor]:
        index = sorted(self.symbols)
        return {
            "center": torch.from_numpy(np.asarray(self.center, dtype=np.float64)),
            "spread": torch.from_numpy(np.asarray(self.spread, dtype=np.float64)),
            "span": torch.tensor(float(self.span), dtype=torch.float64),
            "index": torch.tensor(index, dtype=torch.int64),
            "centers": torch.from_numpy(np.stack([self.symbols[s][0] for s in index]) if index else np.zeros((0, LANES))),
            "spreads": torch.from_numpy(np.stack([self.symbols[s][1] for s in index]) if index else np.zeros((0, LANES))),
            "spans": torch.tensor([self.symbols[s][2] for s in index], dtype=torch.float64),
        }

    @classmethod
    def load(cls, blob: dict[str, torch.Tensor]) -> Scale:
        index = [int(item) for item in blob["index"].tolist()]
        centers, spreads, spans = blob["centers"].numpy(), blob["spreads"].numpy(), blob["spans"].numpy()
        symbols = {symbol: (centers[row], spreads[row], float(spans[row])) for row, symbol in enumerate(index)}
        return cls(blob["center"].numpy(), blob["spread"].numpy(), float(blob["span"]), symbols)


def fit(days: tuple[str, ...], read) -> Scale:
    features: dict[int, list[np.ndarray]] = {}
    prices: dict[int, list[np.ndarray]] = {}

    def sample(series) -> None:
        times = np.asarray(series.open_time_ms, dtype=np.int64)
        keep = (times // CANDLE_MS) % SAMPLE_EVERY == 0
        if not np.any(keep):
            return
        symbol = int(series.symbol)
        features.setdefault(symbol, []).append(np.asarray(series.features, dtype=np.float32)[keep])
        prices.setdefault(symbol, []).append(
            np.stack([np.asarray(series.high)[keep], np.asarray(series.low)[keep], np.asarray(series.close)[keep]])
        )

    columns = feature_scales(days, fit_column, on_series=sample, read=read)
    if len(columns) != LANES:
        raise RuntimeError(f"expected {LANES} scales, got {len(columns)}")
    center = np.array([item[0] for item in columns], dtype=np.float64)
    spread = np.array([item[1] for item in columns], dtype=np.float64)
    symbols = {}
    every = []
    for symbol in sorted(features):
        block = np.concatenate(features.pop(symbol), axis=0)
        fitted = [fit_column(block[:, lane]) for lane in range(LANES)]
        high, low, close = np.concatenate(prices.pop(symbol), axis=1)
        every.append((high, low, close))
        symbols[symbol] = (
            np.array([item[0] for item in fitted], dtype=np.float64),
            np.array([item[1] for item in fitted], dtype=np.float64),
            fit_range(high, low, close),
        )
    pooled = fit_range(*(np.concatenate(part) for part in zip(*every))) if every else 1.0
    return Scale(center, spread, pooled, symbols)
