"""Scale columns the way one Causilo estimator does.

The fit sees only the context rows. Query rows use that same fit.
Missing cells stay missing. Tails are compressed the way Causilo compresses them,
and the order of finite values in a column stays the same.
"""

from __future__ import annotations

import numpy as np
from causilo.data.normalization import Normalizer

METHOD = "none"


def fit(context: np.ndarray) -> Normalizer:
    """Fit on context rows only."""

    table = np.asarray(context, dtype=np.float64)
    if table.ndim != 2 or table.shape[0] < 2:
        raise ValueError(f"context must be a table with rows, got {table.shape}")
    return Normalizer.fit(table, METHOD)


def apply(scaler: Normalizer, table: np.ndarray) -> np.ndarray:
    """Transform with a scaler already fitted on context."""

    return scaler.transform(np.asarray(table, dtype=np.float64))


def audit(context: np.ndarray, scaler: Normalizer, scaled: np.ndarray) -> dict[str, float]:
    """Say whether the scaled context still carries the context.

    NaN positions must match. A column that varied must still vary.
    Finite order inside a column must not flip. Cells outside Causilo's
    own bounds are compressed, not dropped, and are counted here.
    """

    raw = np.asarray(context, dtype=np.float64)
    done = np.asarray(scaled, dtype=np.float64)
    if raw.shape != done.shape:
        raise ValueError(f"scaled shape {done.shape} does not match {raw.shape}")
    missing = np.isnan(raw)
    if not np.array_equal(missing, np.isnan(done)):
        raise ValueError("scaling changed which cells are missing")
    if not np.isfinite(done[~missing]).all():
        raise ValueError("scaling created a non-finite value")
    filled = np.where(missing, scaler.fill, raw)
    standardized = (filled - scaler.center) / scaler.spread
    outside = ((standardized < scaler.lower) | (standardized > scaler.upper)) & ~missing
    varied = 0
    flattened = 0
    compared = 0
    flipped = 0
    material = 0
    generator = np.random.default_rng(0)
    for column in range(raw.shape[1]):
        finite = ~missing[:, column]
        if int(finite.sum()) < 3:
            continue
        raw_column = raw[finite, column]
        if float(np.std(raw_column)) == 0.0:
            continue
        varied += 1
        scaled_column = done[finite, column]
        if float(np.std(scaled_column)) == 0.0:
            flattened += 1
            continue
        left = generator.integers(0, raw_column.shape[0], 256)
        right = generator.integers(0, raw_column.shape[0], 256)
        gap = raw_column[left] - raw_column[right]
        usable = gap != 0
        compared += int(usable.sum())
        scaled_gap = scaled_column[left][usable] - scaled_column[right][usable]
        disagree = np.sign(gap[usable]) != np.sign(scaled_gap)
        flipped += int(np.sum(disagree))
        # A tie is float32 collapse after a huge column spread, not a reversal.
        strict = disagree & (scaled_gap != 0)
        raw_gap = np.abs(gap[usable][strict])
        base = np.maximum(np.abs(raw_column[left][usable][strict]), np.abs(raw_column[right][usable][strict]))
        material += int(np.sum(raw_gap > np.maximum(1e-4, 1e-5 * base))) if strict.any() else 0
    return {
        "rows": float(raw.shape[0]),
        "columns": float(raw.shape[1]),
        "missing_share": float(missing.mean()),
        "compressed_share": float(outside.mean()),
        "varied_columns": float(varied),
        "flattened_columns": float(flattened),
        "order_pairs": float(compared),
        "order_flips": float(flipped),
        "order_flips_material": float(material),
    }


def _blocks(paths: list, block: int):
    for path in paths:
        table = np.load(path, mmap_mode="r")
        if table.ndim != 2:
            raise ValueError(f"{path} is not a table")
        for start in range(0, table.shape[0], block):
            yield np.asarray(table[start : start + block], dtype=np.float64)


def save_scaler(path, scaler: Normalizer) -> None:
    """Write the frozen column statistics. The official weights are not here."""

    np.savez(
        path,
        fill=np.asarray(scaler.fill),
        center=np.asarray(scaler.center),
        spread=np.asarray(scaler.spread),
        lower=np.asarray(scaler.lower),
        upper=np.asarray(scaler.upper),
    )


def load_scaler(path) -> Normalizer:
    """Read a scaler written by ``save_scaler``."""

    stored = np.load(path)
    return Normalizer(
        fill=stored["fill"],
        center=stored["center"],
        spread=stored["spread"],
        normalizer=None,
        quantile_scale=None,
        lower=stored["lower"],
        upper=stored["upper"],
    )


def fit_paths(paths: list, block: int = 65_536) -> Normalizer:
    """Causilo's ``none`` fit, read in blocks from ``.npy`` tables.

    The equations are the ones ``Normalizer.fit`` uses for method ``none``.
    The tables stay on disk.
    """

    if not paths:
        raise ValueError("the fit has no tables")
    width = int(np.load(paths[0], mmap_mode="r").shape[1])
    count = np.zeros(width, dtype=np.float64)
    total = np.zeros(width, dtype=np.float64)
    rows = 0
    for part in _blocks(paths, block):
        finite = np.isfinite(part)
        count += finite.sum(axis=0)
        total += np.where(finite, part, 0.0).sum(axis=0)
        rows += int(part.shape[0])
    if rows < 2:
        raise ValueError(f"the fit needs rows, got {rows}")
    fill = np.divide(total, count, out=np.zeros(width), where=count > 0)
    filled_sum = np.zeros(width, dtype=np.float64)
    filled_sq = np.zeros(width, dtype=np.float64)
    for part in _blocks(paths, block):
        finite = np.isfinite(part)
        filled = np.where(finite, part, fill)
        filled_sum += filled.sum(axis=0)
        filled_sq += np.square(filled).sum(axis=0)
    center = filled_sum / rows
    variance = filled_sq / rows - np.square(center)
    spread = np.sqrt(np.maximum(variance, 0.0))
    tolerance = np.finfo(np.float64).eps * np.maximum(np.abs(center), 1.0)
    spread = np.where(spread <= tolerance, 1.0, spread)
    ddof = 1 if rows > 1 else 0
    scaled_sum = np.zeros(width, dtype=np.float64)
    scaled_sq = np.zeros(width, dtype=np.float64)
    for part in _blocks(paths, block):
        finite = np.isfinite(part)
        filled = np.where(finite, part, fill)
        scaled = (filled - center) / spread
        scaled_sum += scaled.sum(axis=0)
        scaled_sq += np.square(scaled).sum(axis=0)
    scaled_mean = scaled_sum / rows
    preliminary = np.maximum(np.sqrt(np.maximum(scaled_sq / rows - np.square(scaled_mean), 0.0) * rows / (rows - ddof)), 1e-6)
    kept_sum = np.zeros(width, dtype=np.float64)
    kept_sq = np.zeros(width, dtype=np.float64)
    kept = np.zeros(width, dtype=np.float64)
    for part in _blocks(paths, block):
        finite = np.isfinite(part)
        filled = np.where(finite, part, fill)
        scaled = (filled - center) / spread
        central = np.abs(scaled - scaled_mean) <= 4.0 * preliminary
        kept += central.sum(axis=0)
        kept_sum += np.where(central, scaled, 0.0).sum(axis=0)
        kept_sq += np.where(central, np.square(scaled), 0.0).sum(axis=0)
    kept_mean = np.divide(kept_sum, kept, out=np.zeros(width), where=kept > 0)
    # Match ``np.nanstd(..., ddof=1)``: divisor is the non-NaN count minus one.
    divisor = np.maximum(kept - ddof, 1.0)
    kept_std = np.sqrt(np.maximum(kept_sq - np.square(kept_sum) / np.maximum(kept, 1.0), 0.0) / divisor)
    kept_std = np.maximum(np.where(kept > 0, kept_std, 1.0), 1e-6)
    lower = kept_mean - 4.0 * kept_std
    upper = kept_mean + 4.0 * kept_std
    return Normalizer(
        fill=fill,
        center=center,
        spread=spread,
        normalizer=None,
        quantile_scale=None,
        lower=lower,
        upper=upper,
    )
