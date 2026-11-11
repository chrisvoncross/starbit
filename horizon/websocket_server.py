"""Horizon WebSocket + HTTP server.

Two modes (HORIZON_MODE):

* ``raw_farmer``       — behaviour-identical to the legacy lightweight_charts
  base viz: streams farmer-Lance OHLC + NWE/Z-Score/Stoch from
  ``providers/farmer_lance_provider.py`` (the RAW farmer re-derivation, NOT
  what the model sees). This is the retained legacy path and the DEFAULT.
* ``tigris_range``     — one symbol, one time span, raw OHLC from the Tigris
  sealed partitions (``providers/tigris_range_provider.py``).
* ``transformer_view`` — renders EXACTLY what the Orion transformer ingests by
  reusing Orion's production sampler/normalization/pin path verbatim via
  ``providers/transformer_view_provider.py``. The as-of pin + resolved Lance
  version + feature_version + FEATURE_COLUMNS order are served as the on-screen
  proof. Normalized 203 channels are WINDOW-RELATIVE (rendered per-window).

Isolation: horizon is a READ-ONLY consumer of Orion. All Orion imports are
LAZY (inside the transformer provider) so the server boots in ``raw_farmer``
mode without torch/lance/orion present, and a NOT-READY transformer state is a
first-class response — never a crash.
"""
import asyncio
import json
import math
import logging
import io
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Set
import sys
import os

def _sanitize_nans(obj):
    """Recursively replace NaN/Inf floats with None for valid JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nans(v) for v in obj]
    return obj


# Add parent directory (workspace root) to path so `market_data_sync` (farmer
# provider) and `orion` (transformer provider) are importable.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# UTF-8 Logging für Emojis
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setStream(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[console_handler]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION (HORIZON_* env; CHART_* / LANCE_PATH kept as fallbacks)
# ============================================================================
def _env(*names, default=None):
    for n in names:
        v = os.environ.get(n)
        if v is not None:
            return v
    return default


HORIZON_MODE = _env('HORIZON_MODE', default='raw_farmer').strip().lower()

CHART_DAYS = int(_env('HORIZON_DAYS', 'CHART_DAYS', default='31'))
CHART_SYMBOL = _env('HORIZON_SYMBOL', 'CHART_SYMBOL', default='BTC_USDT')
CHART_INTERVAL = _env('HORIZON_INTERVAL', 'CHART_INTERVAL', default='Min1')
CHART_HOUR = _env('HORIZON_FROM') or None
LANCE_PATH = _env('HORIZON_FARMER_LANCE_PATH', 'LANCE_PATH', default=os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'synced_market_farmer_data', 'lance'
))

# TradingView-style lazy loading constants
INITIAL_CANDLES = 1000    # Initial load: last 1000 candles (~16h for 1min)
CHUNK_SIZE = 500          # Load 500 more when scrolling (~8h for 1min)

logger.info(
    f"🚀 Horizon Config: mode={HORIZON_MODE} | Symbol={CHART_SYMBOL} | "
    f"Interval={CHART_INTERVAL} | Days={CHART_DAYS} | Lazy={INITIAL_CANDLES}+{CHUNK_SIZE}"
)

# ============================================================================
# DATA PROVIDERS — both lazy so the server boots regardless of mode/env.
# ============================================================================
_lance_provider = None       # raw_farmer (farmer Lance)


def _get_farmer_provider():
    """Lazily construct the raw farmer-Lance provider (raw_farmer mode)."""
    global _lance_provider
    if _lance_provider is None:
        from providers.farmer_lance_provider import LanceDataProvider
        _lance_provider = LanceDataProvider(LANCE_PATH)
        logger.info(f"📡 Using Lance farmer provider: {LANCE_PATH}")
    return _lance_provider


# nexus ``IntervalKind::uri_token`` (mirror of tier_paths._INTERVAL_URI_TOKENS)
# — the producer-canonical PATH token. Used to derive a sibling interval's
# enriched-features gold dataset from the bound ``--nexus-lance-uri`` and to
# probe on-disk availability for the Timeframe-Swap element.
INTERVAL_TOKENS = {
    "Sec15": "sec15",
    "Min1": "min1",
    "Min5": "min5",
    "Min15": "min15",
}
# Deliberate order for the Timeframe-Swap UI (finest -> coarsest); Sec15 was
# previously missing from the interval surface and is restored here.
INTERVAL_ORDER = ["Sec15", "Min1", "Min5", "Min15"]


def _int_or_none(v):
    return int(v) if v not in (None, "") else None


def _derive_interval_uri(base_uri: str, base_interval: str, target_interval: str):
    """Swap the interval PATH token in a sealed-features URI.

    ``.../features/warm/sec15/dataset.lance`` -> ``.../features/warm/min1/dataset.lance``.
    Returns ``None`` if the base URI does not carry the base interval's token
    (so we never fabricate a wrong path)."""
    src = INTERVAL_TOKENS.get(base_interval)
    dst = INTERVAL_TOKENS.get(target_interval)
    if not base_uri or not src or not dst:
        return None
    needle = f"/{src}/"
    if needle not in base_uri:
        return None
    return base_uri.replace(needle, f"/{dst}/")


def _tv_config(interval: str = None):
    """Build a TransformerViewConfig for ``interval`` (defaults to the bound
    live interval). Switching to a NON-live interval derives its sibling
    dataset from the bound ``--nexus-lance-uri`` and drops the as-of tag (a
    tag is only guaranteed on the interval the trainer actually pinned)."""
    from providers.transformer_view_provider import TransformerViewConfig

    base_uri = _env('HORIZON_NEXUS_LANCE_URI')
    base_interval = CHART_INTERVAL
    interval = interval or base_interval

    nexus_lance_uri = base_uri
    as_of_tag = _env('HORIZON_PIN_TAG')
    if interval != base_interval and base_uri:
        nexus_lance_uri = _derive_interval_uri(base_uri, base_interval, interval)
        as_of_tag = None  # no live pin for a non-bound interval

    return TransformerViewConfig(
        bucket_prefix=_env('HORIZON_BUCKET_PREFIX', default=''),
        interval=interval,
        seq_len=int(_env('HORIZON_SEQ_LEN', default='8')),
        horizon=int(_env('HORIZON_HORIZON', default='8')),
        panel_width=int(_env('HORIZON_PANEL_WIDTH', default='1')),
        seed=int(_env('HORIZON_SEED', default='17')),
        strict=_env('HORIZON_STRICT', default='0') in ('1', 'true', 'True', 'yes'),
        as_of_tag=as_of_tag,
        as_of_version=_int_or_none(_env('HORIZON_PIN_VERSION')) if interval == base_interval else None,
        as_of_timestamp_ms=_int_or_none(_env('HORIZON_PIN_TIMESTAMP_MS')) if interval == base_interval else None,
        nexus_lance_uri=nexus_lance_uri,
    )


# Per-interval provider cache (each interval binds its own pinned dataset).
_tv_providers: dict = {}


def _get_tv_provider(interval: str = None):
    """Lazily construct/cache the transformer-view provider for ``interval``."""
    interval = interval or CHART_INTERVAL
    prov = _tv_providers.get(interval)
    if prov is None:
        from providers.transformer_view_provider import TransformerViewProvider
        prov = TransformerViewProvider(_tv_config(interval))
        _tv_providers[interval] = prov
    return prov


# Safe empty payloads when feature mappings are unavailable (e.g. CPU-only/no CUDA registry)
EMPTY_NWE_BANDS = {
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
EMPTY_ZSCORE = {"zscore": [], "zscore_long": [], "price_levels": []}
EMPTY_STOCH = {"stoch_k": [], "stoch_d": []}


async def _safe_async(name: str, fn, default):
    """Execute async provider calls without breaking websocket init."""
    try:
        return await fn()
    except Exception as e:
        logger.warning(f"⚠️ Indicator '{name}' unavailable: {type(e).__name__}: {e}")
        return default


app = FastAPI(title="GhostByte Horizon Server")

# Active WebSocket connections
active_connections: Set[WebSocket] = set()


# ============================================================================
# RAW_FARMER data helpers (behaviour-identical to the legacy base viz).
# ============================================================================
async def get_initial_candles(symbol: str = CHART_SYMBOL, interval: str = CHART_INTERVAL,
                               heikin_ashi: bool = True, limit: int = INITIAL_CANDLES):
    """Load initial candles from farmer Lance datasets (raw_farmer mode)."""
    return await _get_farmer_provider().get_candles(symbol, interval, CHART_DAYS, heikin_ashi, limit=limit)


async def get_simulated_trades(workflow_id: str = None, symbol: str = CHART_SYMBOL):
    """Load simulated trades. Currently returns empty — trades are not stored in Lance."""
    return []


async def get_nwe_bands(symbol: str = CHART_SYMBOL, interval: str = CHART_INTERVAL,
                        limit: int = INITIAL_CANDLES):
    """Load NWE bands from farmer Lance GPU features."""
    return await _get_farmer_provider().get_nwe_bands(symbol, interval, CHART_DAYS, limit=limit)


async def get_kyle_signals(symbol: str = CHART_SYMBOL, interval: str = CHART_INTERVAL,
                           limit: int = INITIAL_CANDLES):
    """Kyle signals are not a MegaFusion GPU feature — returns empty."""
    return []


async def get_zscore_data(symbol: str = CHART_SYMBOL, interval: str = CHART_INTERVAL,
                          limit: int = INITIAL_CANDLES):
    """Load Z-Score data from farmer Lance GPU features."""
    return await _get_farmer_provider().get_zscore_data(symbol, interval, CHART_DAYS, limit=limit)


async def get_stochastic_obv_data(symbol: str = CHART_SYMBOL, interval: str = CHART_INTERVAL,
                                   limit: int = INITIAL_CANDLES):
    """Load Stochastic data from farmer Lance GPU features."""
    return await _get_farmer_provider().get_stochastic_data(symbol, interval, CHART_DAYS, limit=limit)


async def stream_new_candles(symbol: str = CHART_SYMBOL, interval: str = 'Min1'):
    """
    Stream new candles + indicator updates by polling the farmer Lance dataset
    for new versions (raw_farmer mode only). Verbatim port of the legacy base
    viz streaming task.
    """
    import lance as _lance

    ds_path = str(Path(LANCE_PATH) / interval)
    last_version = 0
    last_timestamp = 0
    poll_count = 0

    # Feature column names for live indicator streaming
    from providers.farmer_lance_provider import FEATURE_INDEX_MAP, SYMBOL_NAME_TO_ID
    _fcol = lambda name: f"mega_fusion_features_{FEATURE_INDEX_MAP[name]}"

    feature_columns = []
    feature_names_available = []
    for fname in [
        "nwe_upper_far", "nwe_upper_avg", "nwe_upper_near",
        "nwe_lower_near", "nwe_lower_avg", "nwe_lower_far",
        "nwe_yhat", "nwe_band_width",
        "zscore_close_75", "zscore_sma_75",
        "stoch_slow_k", "stoch_slow_d",
    ]:
        if fname in FEATURE_INDEX_MAP:
            feature_columns.append(_fcol(fname))
            feature_names_available.append(fname)

    try:
        import pyarrow.compute as _pc
        ds = _lance.dataset(ds_path)
        last_version = ds.version
        # Start from the LATEST timestamp so we only stream truly new data
        _max_table = ds.scanner(columns=["timestamp_ms"]).to_table()
        if _max_table.num_rows > 0:
            last_timestamp = int(_pc.max(_max_table.column("timestamp_ms")).as_py())
        logger.info(
            f"🔥 LANCE STREAM INIT: {interval} v{last_version} | "
            f"{len(feature_columns)} feature cols | baseline_ts={last_timestamp} | "
            f"mode=RAW_OHLCV (client-side HA)"
        )
    except Exception as e:
        logger.warning(f"⚠️ LANCE STREAM: Could not open {interval}: {e}")
        return

    while True:
        try:
            ds = _lance.dataset(ds_path)
            current_version = ds.version

            # 🔥 VERSION RESET DETECTION: Dataset was recreated (resync, compaction, etc.)
            if current_version < last_version:
                import pyarrow.compute as _pc_reset
                logger.warning(
                    f"⚠️ LANCE STREAM: Version reset detected! "
                    f"v{last_version} → v{current_version} — re-syncing baseline"
                )
                _reset_table = ds.scanner(columns=["timestamp_ms"]).to_table()
                if _reset_table.num_rows > 0:
                    last_timestamp = int(_pc_reset.max(_reset_table.column("timestamp_ms")).as_py())
                else:
                    last_timestamp = 0
                last_version = current_version
                logger.info(
                    f"🔥 LANCE STREAM: Baseline reset → v{last_version} | ts={last_timestamp}"
                )

            if current_version > last_version:
                import pyarrow.compute as pc
                import numpy as np

                symbol_id = SYMBOL_NAME_TO_ID.get(symbol, 0)

                # Read OHLC + feature columns in one scan
                scan_columns = ["timestamp_ms", "open", "high", "low", "close"] + feature_columns
                # Only request columns that exist in the schema
                schema_names = set(ds.schema.names)
                scan_columns = [c for c in scan_columns if c in schema_names]

                scanner = ds.scanner(
                    columns=scan_columns,
                    filter=f"symbol_id = {symbol_id} AND timestamp_ms > {last_timestamp}",
                )
                table = scanner.to_table()

                if table.num_rows > 0:
                    sort_idx = pc.sort_indices(table.column("timestamp_ms"))
                    table = table.take(sort_idx)

                    timestamps = table.column("timestamp_ms").to_numpy()
                    opens = table.column("open").to_numpy().astype(np.float64)
                    highs = table.column("high").to_numpy().astype(np.float64)
                    lows = table.column("low").to_numpy().astype(np.float64)
                    closes = table.column("close").to_numpy().astype(np.float64)

                    # Load feature arrays (if available in schema)
                    feat = {}
                    feat_available_count = 0
                    for fname in feature_names_available:
                        col_name = _fcol(fname)
                        if col_name in table.schema.names:
                            arr = table.column(col_name).to_numpy().astype(np.float64)
                            # Semantic NaN guard: feature column exists but is all-NaN = not enriched yet
                            if not np.all(np.isnan(arr)):
                                feat[fname] = arr
                                feat_available_count += 1

                    # Broadcast RAW OHLCV candles — client computes Heikin Ashi
                    candle_count = 0
                    indicator_count = 0
                    for i in range(len(timestamps)):
                        time_s = int(timestamps[i] / 1000)
                        close_val = float(closes[i])

                        # RAW OHLCV — no server-side HA (TradingView pattern)
                        candle = {
                            'time': time_s,
                            'open': float(opens[i]),
                            'high': float(highs[i]),
                            'low': float(lows[i]),
                            'close': float(closes[i]),
                        }
                        last_timestamp = int(timestamps[i])
                        await broadcast_candle(candle)
                        candle_count += 1

                        # Build indicator updates — each channel independent
                        indicators = _build_indicator_update(time_s, close_val, feat, i)
                        if indicators:
                            await broadcast_indicator_update(indicators)
                            indicator_count += 1

                    logger.info(
                        f"🔥 LANCE STREAM #{poll_count}: v{last_version}→v{current_version} | "
                        f"{candle_count} candles (RAW) | {indicator_count} indicator updates | "
                        f"{feat_available_count}/{len(feature_names_available)} features available | "
                        f"last_ts={last_timestamp}"
                    )
                else:
                    logger.debug(
                        f"🔍 LANCE STREAM #{poll_count}: v{last_version}→v{current_version} | "
                        f"0 new rows for symbol_id={SYMBOL_NAME_TO_ID.get(symbol, 0)}"
                    )

                last_version = current_version
                poll_count += 1

        except Exception as e:
            import traceback
            logger.error(f"⚠️ LANCE STREAM error: {e}\n{traceback.format_exc()}")

        await asyncio.sleep(5)


def _build_indicator_update(time_s: int, close: float, feat: dict, idx: int) -> dict:
    """
    Build indicator update dict from GPU feature arrays at row index.
    (Verbatim from the legacy base viz — each indicator channel independent,
    NaN-semantic guards.)
    """
    import math
    indicators = {'time': time_s}

    # ─── CHANNEL 1: NWE bands (independent) ─────────────────────────────
    nwe_names = ["nwe_upper_far", "nwe_upper_avg", "nwe_upper_near",
                 "nwe_lower_near", "nwe_lower_avg", "nwe_lower_far", "nwe_yhat"]
    nwe_bw = feat.get("nwe_band_width")
    if nwe_bw is not None and not math.isnan(nwe_bw[idx]) and abs(nwe_bw[idx]) > 0.0001:
        nwe = {}
        for name in nwe_names:
            arr = feat.get(name)
            if arr is not None and not math.isnan(arr[idx]):
                nwe[name.replace("nwe_", "")] = float(close * (1.0 + arr[idx]))
        if nwe:
            indicators['nwe'] = nwe

    # ─── CHANNEL 2: Z-Score (independent) ───────────────────────────────
    zc = feat.get("zscore_close_75")
    zs = feat.get("zscore_sma_75")
    if zc is not None and not math.isnan(zc[idx]):
        indicators['zscore'] = {'time': time_s, 'value': float(zc[idx])}
    if zs is not None and not math.isnan(zs[idx]):
        indicators['zscore_long'] = {'time': time_s, 'value': float(zs[idx])}

    # ─── CHANNEL 3: Stochastic (independent) ────────────────────────────
    sk = feat.get("stoch_slow_k")
    sd = feat.get("stoch_slow_d")
    if sk is not None and not math.isnan(sk[idx]):
        indicators['stoch_k'] = {'time': time_s, 'value': float((sk[idx] + 1.0) / 2.0 * 100.0)}
    if sd is not None and not math.isnan(sd[idx]):
        indicators['stoch_d'] = {'time': time_s, 'value': float((sd[idx] + 1.0) / 2.0 * 100.0)}

    has_data = len(indicators) > 1
    return indicators if has_data else None


async def broadcast_indicator_update(indicators: dict):
    """Broadcast indicator update to all connected WebSocket clients."""
    if not active_connections:
        return

    message = json.dumps(_sanitize_nans({
        'type': 'indicator_update',
        'data': indicators
    }))

    disconnected = set()
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except Exception:
            disconnected.add(connection)

    active_connections.difference_update(disconnected)


async def broadcast_candle(candle: dict):
    """Broadcast candle update to all connected WebSocket clients"""
    if not active_connections:
        return

    message = json.dumps(_sanitize_nans({
        'type': 'candle_update',
        'data': candle
    }))

    disconnected = set()
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            disconnected.add(connection)

    active_connections.difference_update(disconnected)


_tigris_range = None


def _ensure_tigris_range():
    """Load the requested span once. Later sockets reuse it."""
    global _tigris_range
    if _tigris_range is not None:
        return _tigris_range
    start = _env("HORIZON_FROM")
    end = _env("HORIZON_TO")
    if not start or not end:
        raise RuntimeError("tigris_range needs HORIZON_FROM and HORIZON_TO")
    from providers.tigris_range_provider import load_range

    chart = load_range(
        CHART_SYMBOL,
        start,
        end,
        interval=CHART_INTERVAL,
        uri=_env("HORIZON_FEATURES_URI"),
    )
    logger.info(
        f"📡 Tigris range: {chart.symbol} id={chart.symbol_id} "
        f"candles={len(chart.candles)} gaps={len(chart.gaps)} uri={chart.uri}"
    )
    _tigris_range = chart
    return chart


@app.on_event("startup")
async def startup_event():
    """Start background tasks on server startup (raw_farmer streaming only)."""
    if HORIZON_MODE == "tigris_range":
        await asyncio.to_thread(_ensure_tigris_range)
        return
    if HORIZON_MODE != 'raw_farmer':
        logger.info(f"🛰️  Mode={HORIZON_MODE}: farmer streaming task NOT started (pinned/replay-first).")
        return

    # The farmer path depends on `market_data_sync` + farmer Lance being present
    # (they are on the trading pod). Wrap in fail-soft so a dev box without the
    # farmer data plane still boots + serves the page (the WS chart path will
    # surface the missing dependency per-request instead of crashing startup).
    try:
        # Load R2 symbol map (async — Uvicorn event loop is running)
        from providers.farmer_lance_provider import async_load_from_r2, SYMBOL_TO_ID
        if not SYMBOL_TO_ID:
            r2_ok = await async_load_from_r2()
            if r2_ok:
                logger.info(f"📋 R2 symbol map loaded: {len(SYMBOL_TO_ID)} symbols")
            else:
                logger.warning("⚠️ R2 symbol map load failed — continuing with local symbols only")
        else:
            logger.info(f"📋 Symbol map already loaded: {len(SYMBOL_TO_ID)} symbols")

        # Inject Binance isolated symbols (not in R2, separate from MEXC system)
        from market_data_sync.config.symbol_ids import ID_TO_SYMBOL, _STRIPPED_TO_CANONICAL
        _BINANCE_SYMBOLS = {"BTC_USDT_BINANCE": 9999}
        for sym, sid in _BINANCE_SYMBOLS.items():
            if sym not in SYMBOL_TO_ID:
                SYMBOL_TO_ID[sym] = sid
                ID_TO_SYMBOL[sid] = sym
                _STRIPPED_TO_CANONICAL[sym.replace("_", "")] = sym
                logger.info(f"📋 Injected Binance symbol: {sym} → {sid}")

        # Start candle streaming task
        asyncio.create_task(stream_new_candles())
        logger.info("🚀 Background streaming task started")
    except Exception as e:
        logger.warning(
            f"⚠️ raw_farmer streaming NOT started ({type(e).__name__}: {e}). "
            "The page still serves; the WS/candle path needs `market_data_sync` "
            "+ the farmer Lance data plane (present on the trading pod)."
        )


# ============================================================================
# WebSocket endpoint — dispatches on HORIZON_MODE.
# ============================================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    logger.info(f"✅ Client connected. Total connections: {len(active_connections)}")
    try:
        if HORIZON_MODE == 'transformer_view':
            await _handle_transformer_ws(websocket)
        elif HORIZON_MODE == 'tigris_range':
            await _handle_tigris_range_ws(websocket)
        else:
            await _handle_raw_farmer_ws(websocket)
    except WebSocketDisconnect:
        active_connections.discard(websocket)
        logger.info(f"❌ Client disconnected. Total connections: {len(active_connections)}")
    except Exception as e:
        import traceback
        logger.error(f"WebSocket error: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        active_connections.discard(websocket)


async def _handle_tigris_range_ws(websocket: WebSocket):
    """Send the preloaded raw OHLC span. No live tail, no indicator bands."""
    chart = _ensure_tigris_range()
    payload = json.dumps(_sanitize_nans({
        "type": "initial_data",
        "data": chart.candles,
        "raw_data": chart.candles,
        "nwe_bands": EMPTY_NWE_BANDS.copy(),
        "kyle_signals": [],
        "zscore_data": EMPTY_ZSCORE.copy(),
        "stoch_obv_data": EMPTY_STOCH.copy(),
        "has_more": False,
        "chunk_size": len(chart.candles),
    }))
    logger.info(f"📦 Tigris range payload: {len(chart.candles)} candles, {len(payload) / 1024:.1f} KB")
    await websocket.send_text(payload)
    while True:
        data = await websocket.receive_text()
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            continue
        if message.get("type") == "load_more":
            await websocket.send_text(json.dumps({
                "type": "more_data",
                "data": [],
                "raw_data": [],
                "nwe_bands": EMPTY_NWE_BANDS.copy(),
                "zscore_data": EMPTY_ZSCORE.copy(),
                "stoch_obv_data": EMPTY_STOCH.copy(),
                "has_more": False,
            }))


async def _hold_after_farmer_failure(websocket: WebSocket, exc: BaseException) -> None:
    """Keep the socket open. A dropped upgrade looks like a browser WebSocket error."""
    logger.error(f"WebSocket farmer load failed: {type(exc).__name__}: {exc}")
    await websocket.send_text(json.dumps(_sanitize_nans({
        "type": "initial_data",
        "data": [],
        "raw_data": [],
        "nwe_bands": EMPTY_NWE_BANDS.copy(),
        "kyle_signals": [],
        "zscore_data": EMPTY_ZSCORE.copy(),
        "stoch_obv_data": EMPTY_STOCH.copy(),
        "has_more": False,
        "chunk_size": CHUNK_SIZE,
        "error": f"{type(exc).__name__}: {exc}",
    })))
    while True:
        await websocket.receive_text()


async def _handle_raw_farmer_ws(websocket: WebSocket):
    """Legacy base-viz WS handler (verbatim behaviour)."""
    logger.info(f"📊 Loading initial data for {CHART_SYMBOL} {CHART_INTERVAL}...")
    try:
        initial_candles = await get_initial_candles()
        raw_candles = await get_initial_candles(heikin_ashi=False)
    except Exception as exc:
        await _hold_after_farmer_failure(websocket, exc)
        return
    logger.info(f"📊 Initial candles: {len(initial_candles)} HA + {len(raw_candles)} Raw")

    nwe_bands = await _safe_async("nwe_bands", get_nwe_bands, EMPTY_NWE_BANDS.copy())
    kyle_signals = await _safe_async("kyle_signals", get_kyle_signals, [])
    zscore_data = await _safe_async("zscore_data", get_zscore_data, EMPTY_ZSCORE.copy())
    stoch_obv_data = await _safe_async("stoch_obv_data", get_stochastic_obv_data, EMPTY_STOCH.copy())
    logger.info(
        f"📊 Indicators loaded: NWE={len(nwe_bands.get('upper_band', []))}, "
        f"ZScore={len(zscore_data.get('zscore', []))}, Stoch={len(stoch_obv_data.get('stoch_k', []))}"
    )

    payload = json.dumps(_sanitize_nans({
        'type': 'initial_data',
        'data': initial_candles,
        'raw_data': raw_candles,
        'nwe_bands': nwe_bands,
        'kyle_signals': kyle_signals,
        'zscore_data': zscore_data,
        'stoch_obv_data': stoch_obv_data,
        'has_more': True,
        'chunk_size': CHUNK_SIZE
    }))
    logger.info(f"📦 JSON payload size: {len(payload) / 1024:.1f} KB")
    await websocket.send_text(payload)

    while True:
        data = await websocket.receive_text()
        try:
            message = json.loads(data)
            msg_type = message.get('type')

            if msg_type == 'load_more':
                before_timestamp = message.get('before')
                limit = message.get('limit', CHUNK_SIZE)
                if before_timestamp:
                    logger.info(f"📜 Load more requested: before={before_timestamp}, limit={limit}")
                    provider = _get_farmer_provider()
                    more_candles = await provider.get_candles_before(
                        CHART_SYMBOL, CHART_INTERVAL, CHART_DAYS,
                        before_timestamp, limit, heikin_ashi=True
                    )
                    more_raw = await provider.get_candles_before(
                        CHART_SYMBOL, CHART_INTERVAL, CHART_DAYS,
                        before_timestamp, limit, heikin_ashi=False
                    )
                    try:
                        more_indicators = await provider.get_indicators_before(
                            CHART_SYMBOL, CHART_INTERVAL, CHART_DAYS,
                            before_timestamp, limit
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ Historical indicators unavailable: {type(e).__name__}: {e}")
                        more_indicators = {
                            "nwe_bands": EMPTY_NWE_BANDS.copy(),
                            "zscore_data": EMPTY_ZSCORE.copy(),
                            "stoch_obv_data": EMPTY_STOCH.copy(),
                        }
                    await websocket.send_text(json.dumps(_sanitize_nans({
                        'type': 'more_data',
                        'data': more_candles,
                        'raw_data': more_raw,
                        'nwe_bands': more_indicators['nwe_bands'],
                        'zscore_data': more_indicators['zscore_data'],
                        'stoch_obv_data': more_indicators['stoch_obv_data'],
                        'has_more': len(more_candles) == limit
                    })))
                    logger.info(f"📜 Sent {len(more_candles)} more candles + indicators")
                else:
                    await websocket.send_text(json.dumps(_sanitize_nans({
                        'type': 'more_data', 'data': [], 'has_more': False
                    })))
            elif msg_type == 'ping':
                pass
            else:
                logger.debug(f"Received from client: {data}")
        except json.JSONDecodeError:
            logger.debug(f"Received non-JSON from client: {data}")


def _probe_intervals():
    """Report the Timeframe-Swap surface: every interval + whether its
    enriched-features gold dataset is materialized on disk, which one is the
    ACTIVE bound interval, and which carries the live as-of pin.

    Sibling datasets are derived from the bound ``--nexus-lance-uri`` token so
    availability reflects what Nexus actually produced (Sec15 is included —
    it was missing from the legacy interval surface)."""
    base_uri = _env('HORIZON_NEXUS_LANCE_URI')
    has_pin = bool(_env('HORIZON_PIN_TAG') or _env('HORIZON_PIN_VERSION') or _env('HORIZON_PIN_TIMESTAMP_MS'))
    out = []
    for interval in INTERVAL_ORDER:
        uri = base_uri if interval == CHART_INTERVAL else _derive_interval_uri(base_uri, CHART_INTERVAL, interval)
        available = bool(uri) and Path(uri).exists()
        out.append({
            "interval": interval,
            "token": INTERVAL_TOKENS[interval],
            "available": available,
            "is_active": interval == CHART_INTERVAL,
            "has_live_pin": interval == CHART_INTERVAL and has_pin,
        })
    return out


async def _handle_transformer_ws(websocket: WebSocket):
    """Transformer-view WS handler: pinned window packs + scrollback + panel +
    central Symbol-Swap / Timeframe-Swap.

    Fail-soft: a NOT-READY provider (gold not minted / Orion unavailable /
    coverage short / surface-equivalence divergence) is sent as a
    ``tv_not_ready`` frame the client renders as a first-class state — never an
    unhandled exception. The interactive loop stays reachable even when
    NOT-READY, so a Timeframe-Swap can recover onto an available interval.
    """
    loop = asyncio.get_event_loop()
    state = {"interval": CHART_INTERVAL, "provider": _get_tv_provider(CHART_INTERVAL), "symbol": None}

    async def _emit_surface(provider):
        """Send preflight (the proof/NOT-READY gate) + the initial window pack,
        the symbol universe (id+name) and the interval surface. Returns True if
        the provider is READY."""
        report = await loop.run_in_executor(None, provider.preflight)
        await websocket.send_text(json.dumps(_sanitize_nans({'type': 'tv_preflight', 'data': report})))
        intervals = _probe_intervals()
        await websocket.send_text(json.dumps(_sanitize_nans({
            'type': 'tv_intervals', 'data': {'intervals': intervals, 'active': provider.cfg.interval}
        })))
        if not report.get('ready', False):
            await websocket.send_text(json.dumps(_sanitize_nans({
                'type': 'tv_not_ready',
                'data': {
                    'reasons': report.get('not_ready_reasons', []),
                    'hard_failures': report.get('hard_failures', []),
                }
            })))
            return False
        cfg = provider.cfg
        try:
            if cfg.is_panel:
                pack = await loop.run_in_executor(None, provider.panel_window, None)
                await websocket.send_text(json.dumps(_sanitize_nans({'type': 'tv_panel', 'data': pack})))
            else:
                labels = await loop.run_in_executor(None, provider.symbol_labels)
                symbols = [entry['id'] for entry in labels]
                state['symbol'] = symbols[0]
                pack = await loop.run_in_executor(None, provider.window, symbols[0], None)
                await websocket.send_text(json.dumps(_sanitize_nans({
                    'type': 'tv_initial',
                    'data': {
                        'symbols': symbols,
                        'symbol_labels': labels,
                        'window': pack,
                        'intervals': intervals,
                        'interval': cfg.interval,
                    },
                })))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"⚠️ transformer initial window failed: {type(e).__name__}: {e}")
            await websocket.send_text(json.dumps(_sanitize_nans({
                'type': 'tv_not_ready', 'data': {'reasons': [f'{type(e).__name__}: {e}']}
            })))
            return False
        return True

    await _emit_surface(state['provider'])

    # Interactive loop: scrollback / step / panel nav / symbol-swap / timeframe-swap.
    while True:
        data = await websocket.receive_text()
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            continue
        msg_type = message.get('type')
        provider = state['provider']
        try:
            if msg_type == 'ping':
                continue
            elif msg_type == 'tv_set_interval':
                interval = str(message.get('interval') or '').strip()
                if interval not in INTERVAL_TOKENS:
                    raise ValueError(f"unknown interval {interval!r}")
                state['interval'] = interval
                state['provider'] = _get_tv_provider(interval)
                await _emit_surface(state['provider'])
            elif msg_type == 'tv_window':
                sym = int(message['symbol'])
                state['symbol'] = sym
                start_index = message.get('start_index')
                pack = await loop.run_in_executor(
                    None, provider.window, sym,
                    (int(start_index) if start_index is not None else None),
                )
                await websocket.send_text(json.dumps(_sanitize_nans({'type': 'tv_more', 'data': pack})))
            elif msg_type == 'tv_before':
                sym = int(message['symbol'])
                pack = await loop.run_in_executor(
                    None, provider.window_before, sym, int(message['before'])
                )
                await websocket.send_text(json.dumps(_sanitize_nans({'type': 'tv_more', 'data': pack})))
            elif msg_type == 'tv_panel_window':
                start_index = message.get('start_index')
                pack = await loop.run_in_executor(
                    None, provider.panel_window,
                    (int(start_index) if start_index is not None else None),
                )
                await websocket.send_text(json.dumps(_sanitize_nans({'type': 'tv_panel', 'data': pack})))
        except Exception as e:
            logger.warning(f"⚠️ transformer nav failed: {type(e).__name__}: {e}")
            await websocket.send_text(json.dumps(_sanitize_nans({
                'type': 'tv_error', 'data': {'reason': f'{type(e).__name__}: {e}'}
            })))


# ============================================================================
# HTTP API
# ============================================================================
@app.get("/api/trades")
async def api_get_trades(workflow_id: str = None, symbol: str = CHART_SYMBOL):
    """API endpoint to get simulated trades (raw_farmer)."""
    trades = await get_simulated_trades(workflow_id=workflow_id, symbol=symbol)
    return {'trades': trades}


@app.get("/api/candles/before")
async def api_get_candles_before(timestamp: int, limit: int = 60, symbol: str = CHART_SYMBOL, interval: str = CHART_INTERVAL):
    """API endpoint for lazy loading: Get candles BEFORE a specific timestamp (raw_farmer)."""
    candles = await _get_farmer_provider().get_candles_before(
        symbol, interval, CHART_DAYS, timestamp, limit, heikin_ashi=True
    )
    logger.info(f"📜 LANCE API: {len(candles)} candles before {timestamp}")
    return candles


_ARCUS_CACHE: dict | None = None
_ARCUS_PYTHON = "/opt/conda/bin/python3"
_ARCUS_CACHE_PATH = Path("/tmp/horizon-arcus-line.json")


def _arcus_key() -> str:
    return "|".join(
        os.environ.get(name, "")
        for name in ("HORIZON_SYMBOL", "HORIZON_FROM", "HORIZON_TO", "HORIZON_INTERVAL")
    )


def _arcus_line() -> dict:
    """One stroke per stretch for the span this process was started with."""
    global _ARCUS_CACHE
    key = _arcus_key()
    if (
        _ARCUS_CACHE is not None
        and _ARCUS_CACHE.get("key") == key
        and _ARCUS_CACHE.get("picture") == "extension"
    ):
        return _ARCUS_CACHE
    if _ARCUS_CACHE_PATH.is_file():
        try:
            stored = json.loads(_ARCUS_CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stored = None
        if (
            isinstance(stored, dict)
            and stored.get("key") == key
            and stored.get("picture") == "extension"
            and "error" not in stored
        ):
            _ARCUS_CACHE = stored
            return stored
    script = Path(__file__).resolve().parent / "indicators" / "arcus" / "build_line.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/ghostbyte" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(
        [_ARCUS_PYTHON, str(script)],
        cwd=str(Path(__file__).resolve().parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "arcus line failed")[-800:]
        logger.error("arcus line failed: %s", tail)
        return {"key": key, "points": [], "error": tail}
    payload = json.loads(proc.stdout)
    if payload.get("key") == key and payload.get("picture") == "extension" and "error" not in payload:
        _ARCUS_CACHE = payload
        tmp = _ARCUS_CACHE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(_ARCUS_CACHE_PATH)
    return payload


@app.get("/api/arcus")
async def api_arcus():
    """The Arcus stretch for the open chart. Computed on CPU, beside the train."""
    return await asyncio.to_thread(_arcus_line)


_UMBRA_CACHE: dict | None = None
_UMBRA_CACHE_PATH = Path("/tmp/horizon-umbra-cloud.json")


def _umbra_cloud() -> dict:
    """Heat on the candles where the next ATR zigzag extreme is likely."""
    global _UMBRA_CACHE
    key = _arcus_key()
    if (
        _UMBRA_CACHE is not None
        and _UMBRA_CACHE.get("key") == key
        and _UMBRA_CACHE.get("picture") == "density"
        and _UMBRA_CACHE.get("version") == "0.1.4"
        and not _UMBRA_CACHE.get("waiting")
    ):
        return _UMBRA_CACHE
    if _UMBRA_CACHE_PATH.is_file():
        try:
            stored = json.loads(_UMBRA_CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stored = None
        if (
            isinstance(stored, dict)
            and stored.get("key") == key
            and stored.get("picture") == "density"
            and stored.get("version") == "0.1.4"
            and not stored.get("waiting")
            and "error" not in stored
        ):
            _UMBRA_CACHE = stored
            return stored
    script = Path(__file__).resolve().parent / "indicators" / "umbra" / "build_cloud.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace/ghostbyte" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    proc = subprocess.run(
        [_ARCUS_PYTHON, str(script)],
        cwd=str(Path(__file__).resolve().parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "umbra cloud failed")[-800:]
        logger.error("umbra cloud failed: %s", tail)
        return {"key": key, "picture": "density", "cells": [], "error": tail}
    payload = json.loads(proc.stdout)
    if (
        payload.get("key") == key
        and payload.get("picture") == "density"
        and payload.get("version") == "0.1.4"
        and not payload.get("waiting")
        and "error" not in payload
    ):
        _UMBRA_CACHE = payload
        tmp = _UMBRA_CACHE_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(_UMBRA_CACHE_PATH)
    return payload


@app.get("/api/umbra")
async def api_umbra():
    """The Umbra density for the open chart. Computed on CPU, beside the train."""
    return await asyncio.to_thread(_umbra_cloud)


@app.get("/api/phase1-model-btc")
async def api_phase1_model_btc():
    """Green marks from the Phase-1 BTC exam. Missing file draws nothing."""
    path = Path(__file__).resolve().parent / "indicators" / "turns" / "phase1_btc_oos.json"
    if not path.is_file():
        return {"kind": "phase1_model_exam", "marks": []}
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/starbit-btc")
async def api_starbit_btc():
    """Two rows under the price. Missing file draws nothing. Old rings are not read."""
    path = Path(__file__).resolve().parent / "indicators" / "starbit" / "btc_hour.json"
    if not path.is_file():
        return {"kind": "starbit_btc", "points": []}
    body = json.loads(path.read_text(encoding="utf-8"))
    points = body.get("points") if isinstance(body, dict) else None
    return {"kind": "starbit_btc", "symbol": body.get("symbol") if isinstance(body, dict) else None, "points": points or []}


@app.get("/api/luma-btc")
async def api_luma_btc():
    """High and low marks Luma wrote on one unseen BTC hour. Missing file draws nothing."""
    path = Path(__file__).resolve().parent / "indicators" / "luma" / "btc_hour.json"
    if not path.is_file():
        return {"kind": "luma_btc", "marks": []}
    return json.loads(path.read_text(encoding="utf-8"))


_LUMA_SYMBOL: dict[str, int] = {}


def _luma_symbol_id(name: str) -> int:
    if name not in _LUMA_SYMBOL:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                row = line.strip()
                if not row or row.startswith("#") or "=" not in row:
                    continue
                key, value = row.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        from providers.tigris_range_provider import resolve_symbol_id
        from stardust.data.lake.open import _storage_options_from_env

        uri = "s3://ghostbyte/features/warm/sec15"
        _LUMA_SYMBOL[name] = int(resolve_symbol_id(name, uri=uri, storage_options=_storage_options_from_env()))
    return _LUMA_SYMBOL[name]


def _luma_train_paths() -> list[Path]:
    from stardust.models.luma import __version__

    root = Path(__file__).resolve().parents[1] / "stardust" / "artifacts" / "luma" / __version__
    return [root / "paare.jsonl", root / "d256n4" / "paare.jsonl"]


def _kopf_werte(blob: dict, key: str) -> list[dict]:
    """One number per candle. A missing key draws an empty line."""
    rows = []
    for row in blob.get(key) or []:
        if not isinstance(row, dict):
            continue
        zeit = row.get("zeit", row.get("time"))
        wert = row.get("wert", row.get("value"))
        if zeit is None or wert is None:
            continue
        try:
            rows.append({"zeit": int(zeit), "wert": float(wert)})
        except (TypeError, ValueError):
            continue
    return rows


def _koepfe_aus(blob: dict, hour: str | None) -> tuple[list[dict], list[dict]]:
    if hour and blob.get("hour") not in (None, hour):
        return [], []
    return _kopf_werte(blob, "lage"), _kopf_werte(blob, "vorstellung")


def _btc_koepfe(hour: str | None, extra: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Nearness and the imagination gap, when a drawn hour file carries them."""
    lage: list[dict] = []
    vorstellung: list[dict] = []
    blobs: list[dict] = []
    if isinstance(extra, dict):
        blobs.append(extra)
    path = Path(__file__).resolve().parent / "indicators" / "luma" / "btc_hour.json"
    if path.is_file():
        try:
            blobs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    for blob in blobs:
        found_lage, found_vorstellung = _koepfe_aus(blob, hour)
        if not lage and found_lage:
            lage = found_lage
        if not vorstellung and found_vorstellung:
            vorstellung = found_vorstellung
    return lage, vorstellung


def _mark_row(mark: dict) -> dict | None:
    """One booked mark. The later candle and the book's own distance stay on it."""
    if not isinstance(mark, dict):
        return None
    if mark.get("zeit") is None or mark.get("marke") not in ("hoch", "tief"):
        return None
    row = {"zeit": int(mark["zeit"]), "marke": mark["marke"], "urteil": mark.get("urteil")}
    if mark.get("gegen_zeit") is not None:
        try:
            row["gegen_zeit"] = int(mark["gegen_zeit"])
        except (TypeError, ValueError):
            pass
    if mark.get("weg") is not None:
        try:
            weg = float(mark["weg"])
        except (TypeError, ValueError):
            weg = None
        if weg is not None and weg == weg:
            row["weg"] = weg
    return row


def _btc_gesetzt(hour: str | None) -> list[dict]:
    """Marks the learner wrote on the chart hour. Luma Train draws these, not the rings."""
    path = Path(__file__).resolve().parent / "indicators" / "luma" / "btc_hour.json"
    if not path.is_file():
        return []
    blob = json.loads(path.read_text(encoding="utf-8"))
    if hour and blob.get("hour") not in (None, hour):
        return []
    marks = []
    for mark in blob.get("marks") or []:
        if mark.get("zeit") is None or mark.get("marke") not in ("hoch", "tief"):
            continue
        row = _mark_row(mark)
        if row is not None:
            marks.append(row)
    return marks


def _luma_train_body(symbol: str, hour: str | None = None) -> dict:
    """Places he pointed at for the open chart hour, else the latest hour."""
    from stardust.models.luma import __version__

    path = next((candidate for candidate in _luma_train_paths() if candidate.is_file()), _luma_train_paths()[0])
    empty = {
        "kind": "luma_train",
        "version": __version__,
        "symbol": symbol,
        "hour": hour,
        "paare": [],
        "gesetzt": [],
        "lage": [],
        "vorstellung": [],
    }
    if not path.is_file():
        empty["gesetzt"] = _btc_gesetzt(hour) if symbol == "BTC_USDT" else []
        if symbol == "BTC_USDT":
            empty["lage"], empty["vorstellung"] = _btc_koepfe(hour)
        empty["hour"] = hour
        return empty
    try:
        symbol_id = _luma_symbol_id(symbol)
    except Exception as exc:
        empty["fehler"] = f"{type(exc).__name__}: {exc}"
        return empty
    wanted = hour or None
    seen = None
    rows: list[dict] = []
    keep = (
        "rot_zeit",
        "rot_marke",
        "stelle_zeit",
        "stelle_marke",
        "wende_zeit",
        "wende_marke",
        "kerzen",
        "gang",
        "rot",
        "weiter",
        "ueber",
        "behalten",
        "kette",
    )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("symbol", -1)) != symbol_id:
                continue
            row_hour = row.get("hour")
            if wanted and row_hour != wanted:
                continue
            if row_hour != seen:
                seen = row_hour
                rows = []
            rot = row.get("rot_zeit")
            if rows and rot is not None and rows[-1].get("rot_zeit") is not None:
                if int(rot) + 60 < int(rows[-1]["rot_zeit"]):
                    rows = []
            rows.append({name: row.get(name) for name in keep})
    gesetzt_path = path.with_name("gesetzt_btc.json")
    gesetzt = []
    gesetzt_blob = None
    if symbol_id == 130 and gesetzt_path.is_file():
        gesetzt_blob = json.loads(gesetzt_path.read_text(encoding="utf-8"))
        gesetzt = [row for row in (_mark_row(mark) for mark in (gesetzt_blob.get("marken") or [])) if row]
    if symbol_id == 130 and not gesetzt:
        gesetzt = _btc_gesetzt(wanted)
    if not rows and isinstance(gesetzt_blob, dict):
        rows = [row for row in (gesetzt_blob.get("paare") or []) if isinstance(row, dict)]
    hour = seen or (gesetzt_blob.get("hour") if isinstance(gesetzt_blob, dict) else None) or wanted
    koepfe_path = path.with_name("koepfe.json")
    koepfe_blob = None
    if koepfe_path.is_file():
        try:
            koepfe_blob = json.loads(koepfe_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            koepfe_blob = None
    lage, vorstellung = [], []
    if symbol_id == 130:
        lage, vorstellung = _btc_koepfe(hour, gesetzt_blob if isinstance(gesetzt_blob, dict) else None)
    if isinstance(koepfe_blob, dict):
        found_lage, found_vorstellung = _koepfe_aus(koepfe_blob, hour)
        if found_lage:
            lage = found_lage
        if found_vorstellung:
            vorstellung = found_vorstellung
    return {
        "kind": "luma_train",
        "version": __version__,
        "symbol": symbol,
        "symbol_id": symbol_id,
        "hour": hour,
        "paare": rows[-2000:],
        "gesetzt": gesetzt,
        "lage": lage,
        "vorstellung": vorstellung,
    }


@app.get("/api/luma-train")
async def api_luma_train(symbol: str | None = None):
    """Red mark and the candle he pointed at, for the latest trained hour of this symbol."""
    name = symbol or CHART_SYMBOL
    return await asyncio.to_thread(_luma_train_body, name, CHART_HOUR)


@app.get("/api/cosmo-oos")
async def api_cosmo_oos():
    """Green Cosmo marks for one sealed-day symbol. Missing file draws nothing."""
    path = Path(__file__).resolve().parent / "indicators" / "cosmo" / "oos_symbol.json"
    if not path.is_file():
        return {"kind": "cosmo_oos", "marks": []}
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/nova-heat")
async def api_nova_heat():
    """One Nova temperature per candle. Missing file draws nothing."""
    path = Path(__file__).resolve().parent / "indicators" / "nova" / "heat.json"
    if not path.is_file():
        return {"kind": "nova_heat", "points": []}
    return json.loads(path.read_text(encoding="utf-8"))


_ladder = None


def _load_ladder():
    """Six clocks of the bound span. 2, 3 and 4 minutes are folded from 15s."""
    global _ladder
    if _ladder is not None:
        return _ladder
    start = _env("HORIZON_FROM")
    end = _env("HORIZON_TO")
    if not start or not end:
        raise RuntimeError("tigris_range needs HORIZON_FROM and HORIZON_TO")
    from providers.tigris_range_provider import load_ladder

    ladder = load_ladder(CHART_SYMBOL, start, end)
    counts = ", ".join(f"{frame['id']}={len(frame['candles'])}" for frame in ladder["frames"])
    logger.info(f"📐 Leiter: {ladder['symbol']} {start} .. {end} {counts}")
    _ladder = ladder
    return ladder


@app.get("/api/gegenueber")
async def api_gegenueber():
    """15s, 1–5 min of the bound span. 2, 3 and 4 min are aggregated from 15s."""
    if HORIZON_MODE != "tigris_range":
        raise HTTPException(status_code=409, detail="gegenueber needs tigris_range")
    return _sanitize_nans(await asyncio.to_thread(_load_ladder))


@app.get("/gegenueber")
async def gegenueber_page():
    page = Path(__file__).resolve().parent / "gegenueber.html"
    return HTMLResponse(page.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})


@app.get("/api/config")
async def api_config():
    """Client bootstrap config (mode + symbol + interval)."""
    body = {
        "mode": HORIZON_MODE,
        "symbol": str(CHART_SYMBOL),
        "interval": CHART_INTERVAL,
    }
    if HORIZON_MODE == "tigris_range" and _tigris_range is not None:
        body["from"] = _env("HORIZON_FROM")
        body["to"] = _env("HORIZON_TO")
        body["symbol_id"] = _tigris_range.symbol_id
        body["candles"] = len(_tigris_range.candles)
        body["gaps"] = _tigris_range.gaps
    return body


@app.get("/api/intervals")
async def api_intervals():
    """Timeframe-Swap surface: all intervals (incl. Sec15) + on-disk
    availability + which one is the active bound interval / carries the pin.

    In transformer_view mode availability is probed against the enriched-
    features gold datasets (siblings of the bound ``--nexus-lance-uri``); in
    raw_farmer mode it is probed against the farmer Lance interval dirs."""
    if HORIZON_MODE == 'transformer_view':
        return {"mode": HORIZON_MODE, "intervals": _probe_intervals(), "active": CHART_INTERVAL}
    out = []
    for interval in INTERVAL_ORDER:
        out.append({
            "interval": interval,
            "token": INTERVAL_TOKENS[interval],
            "available": (Path(LANCE_PATH) / interval).exists(),
            "is_active": interval == CHART_INTERVAL,
            "has_live_pin": False,
        })
    return {"mode": HORIZON_MODE, "intervals": out, "active": CHART_INTERVAL}


@app.get("/api/preflight")
async def api_preflight():
    """Return ``run_preflight_assay(...).to_payload()`` — the READY/NOT-READY
    proof. In raw_farmer mode this returns a not-applicable note (the farmer
    path has no as-of pin proof). Never raises."""
    if HORIZON_MODE != 'transformer_view':
        return {
            "ready": None,
            "mode": HORIZON_MODE,
            "note": "preflight applies to transformer_view mode only; "
                    "raw_farmer renders the raw farmer-Lance re-derivation.",
        }
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_tv_provider().preflight)


ASSET_MIME_TYPES = {
    ".js": "application/javascript",
    ".css": "text/css",
    ".html": "text/html",
}

ASSET_PATHS = {
    # --- bootstrap + entries ---
    "bootstrap.js",
    "chart_realtime.js",
    "transformer_view.js",
    # --- core ---
    "core/chart_bootstrap.js",
    "core/legend.js",
    "core/realtime_controller.js",
    "core/transformer_view_controller.js",
    # --- central, reusable Horizon UI controls ---
    "core/symbol_switcher.js",
    "core/timeframe_switcher.js",
    # --- plugins ---
    "plugins/fill-areas/fill-areas.js",
    "plugins/buy_sell_labels/labels.js",
    "plugins/price-level-labels/price-level-labels.js",
    "plugins/horizontal-lines/horizontal-lines.js",
    "plugins/zscore-table/zscore-table-plugin.js",
    "plugins/zscore-solid-fill/zscore-solid-fill.js",
    "plugins/buy_sell_positions/buy_sell_positions.js",
    # --- primitives ---
    "primitives/ohlc_bars_renderer.js",
    # --- indicators ---
    "indicators/indicator_panel.js",
    "indicators/indicator_registry.js",
    "indicators/pane_manager.js",
    "indicators/nwe/nwe.js",
    "indicators/kyle_algo/kyle.js",
    "indicators/zscore/zscore.js",
    "indicators/zscore/zscore-table.html",
    "indicators/zscore/zscore-table.css",
    "indicators/stoch_slow/stoch_slow.js",
    "indicators/turns/turns.js",
    "indicators/atr_zigzag/atr_zigzag.js",
    "indicators/luma/luma.js",
    "indicators/starbit/starbit.js",
    "indicators/luma_train/luma_train.js",
    "indicators/nordstern/nordstern.js",
    "indicators/zero_fees/zero_fees.js",
    "indicators/entropie/entropie.js",
    "indicators/entropiev2/entropiev2.js",
    "indicators/cosmo/cosmo.js",
    "indicators/nova/nova.js",
    "indicators/arcus/arcus.js",
    "indicators/umbra/umbra.js",
    "indicators/turns/phase1_model_exam.js",
    # --- transformer-view (new) ---
    "indicators/feature_heatmap/feature_heatmap.js",
    "indicators/feature_channels/feature_channels.js",
    "plugins/dropout_bands/dropout_bands.js",
    "panel/panel_view.js",
    # --- optionally vendored library ---
    "vendor/lightweight-charts.standalone.production.js",
    "gegenueber.js",
}


@app.get("/")
async def read_root():
    """Serve index.html with dynamic mode + symbol injection."""
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    html_content = html_content.replace("%%HORIZON_MODE%%", HORIZON_MODE)
    html_content = html_content.replace("DOGS_USDT", str(CHART_SYMBOL))
    return HTMLResponse(content=html_content, headers={"Cache-Control": "no-store"})


@app.get("/{asset_path:path}")
async def serve_assets(asset_path: str):
    if asset_path in ("", "/"):
        raise HTTPException(status_code=404, detail="Asset not found")
    if asset_path not in ASSET_PATHS:
        raise HTTPException(status_code=404, detail="Asset not found")
    full_path = Path(asset_path)
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="Asset file missing")
    suffix = full_path.suffix.lower()
    media_type = ASSET_MIME_TYPES.get(suffix, "text/plain")
    from fastapi.responses import Response

    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content=content, media_type=media_type, headers={"Cache-Control": "no-store"})
