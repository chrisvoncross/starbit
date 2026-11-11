"""
Lightweight Charts Lance data provider.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

console_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[console_handler],
)
logger = logging.getLogger(__name__)

# farmer_lance_provider.py lives at horizon/providers/; the workspace root
# (where ``market_data_sync`` is importable) is two parents up. Kept as
# parents[2] because the provider descended one directory level vs the legacy
# lightweight_charts/lance_data_provider.py location (parent.parent there).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from market_data_sync.config.symbol_ids import (  # noqa: E402
    ID_TO_SYMBOL,
    SYMBOL_TO_ID,
    async_load_from_r2,
    id_to_symbol as _r2_id_to_symbol,
    normalize_symbol as _r2_normalize,
    symbol_to_id as _r2_symbol_to_id,
)

SYMBOL_NAME_TO_ID = SYMBOL_TO_ID


def _build_feature_index_map() -> Dict[str, int]:
    """
    Build feature name -> mega_fusion_features_N mapping.

    Preferred source is DEVICE_REGISTRY. On CPU-only boxes this may be empty,
    so we keep a deterministic fallback for the indicators used by this chart.
    """
    try:
        from market_data_sync.cuda_features.mega_fusion.device_registry import DEVICE_REGISTRY
        from market_data_sync.cuda_features.mega_fusion.kernel_generator import MegaKernelGeneratorV2

        gen = MegaKernelGeneratorV2()
        gen._collect_device_functions()

        mapping: Dict[str, int] = {}
        for calc_name, info in gen._device_functions.items():
            cfg = DEVICE_REGISTRY[calc_name]
            for i, feature_name in enumerate(cfg.feature_names):
                mapping[feature_name] = info.offset + i

        if mapping:
            logger.info(f"🔥 LANCE FEATURE MAP: {len(mapping)} features mapped from DEVICE_REGISTRY")
            return mapping
    except Exception as e:
        logger.warning(f"⚠️ DEVICE_REGISTRY feature map unavailable: {e}")

    fallback = {
        # NWE bands
        "nwe_upper_far": 88,
        "nwe_upper_avg": 89,
        "nwe_upper_near": 90,
        "nwe_lower_near": 91,
        "nwe_lower_avg": 92,
        "nwe_lower_far": 93,
        "nwe_yhat": 94,
        "nwe_band_width": 95,
        # Z-Score
        "zscore_close_75": 33,
        "zscore_sma_75": 37,
        "zscore_price_sma": 38,
        # Stochastic
        "stoch_slow_k": 53,
        "stoch_slow_d": 54,
    }
    logger.info(f"🔥 LANCE FEATURE MAP: using fallback map ({len(fallback)} features)")
    return fallback


FEATURE_INDEX_MAP: Dict[str, int] = _build_feature_index_map()


def _feature_col(name: str) -> str:
    idx = FEATURE_INDEX_MAP.get(name)
    if idx is None:
        raise KeyError(f"Unknown feature: {name}. Available: {list(FEATURE_INDEX_MAP.keys())[:10]}...")
    return f"mega_fusion_features_{idx}"


def calculate_heikin_ashi(
    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(closes)
    ha_close = (opens + highs + lows + closes) / 4
    ha_open = np.zeros(n)
    ha_open[0] = (opens[0] + closes[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2
    ha_high = np.maximum(highs, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(lows, np.minimum(ha_open, ha_close))
    return ha_open, ha_high, ha_low, ha_close


class LanceDataProvider:
    INITIAL_CANDLES = 1000
    CHUNK_SIZE = 500

    def __init__(self, lance_base_path: str):
        import lance as _lance

        self._lance = _lance
        self._base_path = Path(lance_base_path)
        logger.info(f"🔥 LanceDataProvider initialized: {self._base_path}")
        logger.info(
            f"   Features: {len(FEATURE_INDEX_MAP)} mapped | Symbols: R2 auto-increment ({len(SYMBOL_TO_ID)} loaded)"
        )

    def _get_dataset(self, interval: str):
        ds_path = self._base_path / interval
        if not ds_path.exists():
            raise FileNotFoundError(f"Lance dataset not found: {ds_path}")
        return self._lance.dataset(str(ds_path))

    def _resolve_symbol_id(self, symbol: str) -> int:
        env_sid = os.environ.get("CHART_SYMBOL_ID")
        if env_sid is not None:
            return int(env_sid)
        try:
            return int(symbol)
        except (ValueError, TypeError):
            pass
        normalized = _r2_normalize(symbol)
        return _r2_symbol_to_id(normalized)

    def _load_symbol_data(
        self, symbol: str, interval: str, days: int, extra_columns: Optional[List[str]] = None
    ):
        import pyarrow.compute as pc

        ds = self._get_dataset(interval)
        symbol_id = self._resolve_symbol_id(symbol)

        now_ms = int(datetime.utcnow().timestamp() * 1000)
        cutoff_ms = now_ms - (days * 86400 * 1000)

        columns = ["timestamp_ms", "open", "high", "low", "close", "volume"]
        if extra_columns:
            columns.extend(extra_columns)

        scanner = ds.scanner(
            columns=columns,
            filter=(f"symbol_id = {symbol_id} AND timestamp_ms >= {cutoff_ms}"),
        )
        table = scanner.to_table()
        if table.num_rows == 0:
            logger.warning(f"⚠️ No data for {symbol} (id={symbol_id}) in {interval}")
            return None

        sort_idx = pc.sort_indices(table.column("timestamp_ms"))
        table = table.take(sort_idx)
        ts = table.column("timestamp_ms").to_numpy()
        _, unique_idx = np.unique(ts, return_index=True)
        if len(unique_idx) < len(ts):
            table = table.take(unique_idx)

        logger.info(
            f"🔥 LANCE READ: {interval} | {symbol} (id={symbol_id}) | {table.num_rows:,} rows | "
            f"{len(columns)} cols | {days}d window"
        )
        return table

    async def get_candles(
        self, symbol: str, interval: str, days: int, heikin_ashi: bool = True, limit: int = None
    ) -> List[Dict]:
        table = self._load_symbol_data(symbol, interval, days)
        if table is None:
            return []

        timestamps = table.column("timestamp_ms").to_numpy()
        opens = table.column("open").to_numpy().astype(np.float64)
        highs = table.column("high").to_numpy().astype(np.float64)
        lows = table.column("low").to_numpy().astype(np.float64)
        closes = table.column("close").to_numpy().astype(np.float64)

        if limit and len(timestamps) > limit:
            sl = slice(-limit, None)
            timestamps = timestamps[sl]
            opens = opens[sl]
            highs = highs[sl]
            lows = lows[sl]
            closes = closes[sl]

        if heikin_ashi and len(closes) > 0:
            opens, highs, lows, closes = calculate_heikin_ashi(opens, highs, lows, closes)

        result = [
            {
                "time": int(ts / 1000),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
            }
            for ts, o, h, l, c in zip(timestamps, opens, highs, lows, closes)
        ]
        logger.info(f"🔥 LANCE CANDLES: {len(result)} {'HA ' if heikin_ashi else ''}candles for {symbol} {interval}")
        return result

    async def get_candles_before(
        self,
        symbol: str,
        interval: str,
        days: int,
        before_timestamp: int,
        limit: int,
        heikin_ashi: bool = True,
    ) -> List[Dict]:
        table = self._load_symbol_data(symbol, interval, days)
        if table is None:
            return []

        ts_ms = table.column("timestamp_ms").to_numpy().astype(np.int64)
        before_ms = int(before_timestamp) * 1000
        mask = ts_ms < before_ms
        if not np.any(mask):
            return []

        ts_ms = ts_ms[mask]
        opens = table.column("open").to_numpy().astype(np.float64)[mask]
        highs = table.column("high").to_numpy().astype(np.float64)[mask]
        lows = table.column("low").to_numpy().astype(np.float64)[mask]
        closes = table.column("close").to_numpy().astype(np.float64)[mask]

        if len(ts_ms) > limit:
            sl = slice(-limit, None)
            ts_ms = ts_ms[sl]
            opens = opens[sl]
            highs = highs[sl]
            lows = lows[sl]
            closes = closes[sl]

        if heikin_ashi and len(closes) > 0:
            opens, highs, lows, closes = calculate_heikin_ashi(opens, highs, lows, closes)

        return [
            {
                "time": int(ts / 1000),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
            }
            for ts, o, h, l, c in zip(ts_ms, opens, highs, lows, closes)
        ]

    async def get_nwe_bands(self, symbol: str, interval: str, days: int, limit: int = None) -> Dict:
        feature_cols = [
            _feature_col("nwe_upper_far"),
            _feature_col("nwe_upper_avg"),
            _feature_col("nwe_upper_near"),
            _feature_col("nwe_lower_near"),
            _feature_col("nwe_lower_avg"),
            _feature_col("nwe_lower_far"),
            _feature_col("nwe_yhat"),
            _feature_col("nwe_band_width"),
        ]
        table = self._load_symbol_data(symbol, interval, days, extra_columns=feature_cols)
        if table is None:
            return {
                "upper_far": [],
                "upper_avg": [],
                "upper_near": [],
                "lower_near": [],
                "lower_avg": [],
                "lower_far": [],
                "yhat": [],
                "upper_band": [],
                "mid_band": [],
                "lower_band": [],
                "width": [],
                "position": [],
            }

        ts = table.column("timestamp_ms").to_numpy()
        close = table.column("close").to_numpy().astype(np.float64)
        uf = table.column(_feature_col("nwe_upper_far")).to_numpy().astype(np.float64)
        ua = table.column(_feature_col("nwe_upper_avg")).to_numpy().astype(np.float64)
        un = table.column(_feature_col("nwe_upper_near")).to_numpy().astype(np.float64)
        ln = table.column(_feature_col("nwe_lower_near")).to_numpy().astype(np.float64)
        la = table.column(_feature_col("nwe_lower_avg")).to_numpy().astype(np.float64)
        lf = table.column(_feature_col("nwe_lower_far")).to_numpy().astype(np.float64)
        yh = table.column(_feature_col("nwe_yhat")).to_numpy().astype(np.float64)
        bw = table.column(_feature_col("nwe_band_width")).to_numpy().astype(np.float64)

        if limit and len(ts) > limit:
            sl = slice(-limit, None)
            ts, close, uf, ua, un, ln, la, lf, yh, bw = (
                ts[sl],
                close[sl],
                uf[sl],
                ua[sl],
                un[sl],
                ln[sl],
                la[sl],
                lf[sl],
                yh[sl],
                bw[sl],
            )

        valid = np.isfinite(bw) & (np.abs(bw) > 1e-6)
        t = (ts[valid] / 1000).astype(int)
        c = close[valid]
        ufp = c * (1.0 + uf[valid])
        uap = c * (1.0 + ua[valid])
        unp = c * (1.0 + un[valid])
        lnp = c * (1.0 + ln[valid])
        lap = c * (1.0 + la[valid])
        lfp = c * (1.0 + lf[valid])
        yhp = c * (1.0 + yh[valid])
        bwp = bw[valid]

        def _series(values):
            return [{"time": int(tt), "value": float(v)} for tt, v in zip(t, values) if np.isfinite(v)]

        return {
            "upper_far": _series(ufp),
            "upper_avg": _series(uap),
            "upper_near": _series(unp),
            "lower_near": _series(lnp),
            "lower_avg": _series(lap),
            "lower_far": _series(lfp),
            "yhat": _series(yhp),
            # legacy aliases used by existing UI code
            "upper_band": _series(uap),
            "mid_band": _series(yhp),
            "lower_band": _series(lap),
            "width": _series(np.abs(bwp)),
            "position": _series(np.zeros_like(bwp)),
        }

    async def get_zscore_data(self, symbol: str, interval: str, days: int, limit: int = None) -> Dict:
        feature_cols = [
            _feature_col("zscore_close_75"),
            _feature_col("zscore_sma_75"),
            _feature_col("zscore_price_sma"),
        ]
        table = self._load_symbol_data(symbol, interval, days, extra_columns=feature_cols)
        if table is None:
            return {"zscore": [], "zscore_long": [], "price_levels": []}

        ts = table.column("timestamp_ms").to_numpy()
        zc = table.column(_feature_col("zscore_close_75")).to_numpy().astype(np.float64)
        zs = table.column(_feature_col("zscore_sma_75")).to_numpy().astype(np.float64)

        if limit and len(ts) > limit:
            sl = slice(-limit, None)
            ts = ts[sl]
            zc = zc[sl]
            zs = zs[sl]

        t = (ts / 1000).astype(int)
        mask_zc = np.isfinite(zc)
        mask_zs = np.isfinite(zs)
        return {
            "zscore": [{"time": int(tt), "value": float(v)} for tt, v in zip(t[mask_zc], zc[mask_zc])],
            "zscore_long": [{"time": int(tt), "value": float(v)} for tt, v in zip(t[mask_zs], zs[mask_zs])],
            "price_levels": [],
        }

    async def get_stochastic_data(self, symbol: str, interval: str, days: int, limit: int = None) -> Dict:
        feature_cols = [_feature_col("stoch_slow_k"), _feature_col("stoch_slow_d")]
        table = self._load_symbol_data(symbol, interval, days, extra_columns=feature_cols)
        if table is None:
            return {"stoch_k": [], "stoch_d": []}

        ts = table.column("timestamp_ms").to_numpy()
        sk = table.column(_feature_col("stoch_slow_k")).to_numpy().astype(np.float64)
        sd = table.column(_feature_col("stoch_slow_d")).to_numpy().astype(np.float64)

        if limit and len(ts) > limit:
            sl = slice(-limit, None)
            ts = ts[sl]
            sk = sk[sl]
            sd = sd[sl]

        sk100 = (sk + 1.0) / 2.0 * 100.0
        sd100 = (sd + 1.0) / 2.0 * 100.0
        valid = np.isfinite(sk100) & np.isfinite(sd100)
        t = (ts[valid] / 1000).astype(int)
        return {
            "stoch_k": [{"time": int(tt), "value": float(v)} for tt, v in zip(t, sk100[valid])],
            "stoch_d": [{"time": int(tt), "value": float(v)} for tt, v in zip(t, sd100[valid])],
        }

    async def get_indicators_before(
        self, symbol: str, interval: str, days: int, before_timestamp: int, limit: int
    ) -> Dict:
        before_ts = int(before_timestamp)

        def _slice(points: List[Dict], field: str = "time") -> List[Dict]:
            older = [p for p in points if int(p.get(field, 0)) < before_ts]
            return older[-limit:] if len(older) > limit else older

        nwe = await self.get_nwe_bands(symbol, interval, days, limit=None)
        zsc = await self.get_zscore_data(symbol, interval, days, limit=None)
        sto = await self.get_stochastic_data(symbol, interval, days, limit=None)

        sliced_nwe = {k: _slice(v) if isinstance(v, list) else v for k, v in nwe.items()}
        sliced_zsc = {k: _slice(v) if isinstance(v, list) else v for k, v in zsc.items()}
        sliced_sto = {k: _slice(v) if isinstance(v, list) else v for k, v in sto.items()}

        return {
            "nwe_bands": sliced_nwe,
            "zscore_data": sliced_zsc,
            "stoch_obv_data": sliced_sto,
        }
