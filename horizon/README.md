# Horizon starten

Kerzen kommen aus R2. `market_data_sync` wird dafür nicht gebraucht. Der Modus heißt `tigris_range`. Zeiten sind UTC. `FROM` ist inklusive, `TO` mit Minuten oder Sekunden ist exklusiv.

```bash
set -a && source /workspace-global/starbit/.env && set +a
export HORIZON_MODE=tigris_range HORIZON_SYMBOL=BTC_USDT HORIZON_INTERVAL=Sec15
export HORIZON_FROM=2026-10-14T19 HORIZON_TO=2026-10-14T19
cd /workspace-global/starbit/horizon
python3 -m uvicorn websocket_server:app --host 0.0.0.0 --port 8000
```

Seite: `http://<host>:8000/`. Stand prüfen: `curl -s localhost:8000/api/config`.

Nicht `HORIZON_MODE=raw_farmer`. Der liest eine lokale Lance-Datei und importiert `market_data_sync`.

# GhostByte Lightweight Charts

Realtime chart stack for GhostByte based on TradingView Lightweight Charts (frontend) and FastAPI + WebSocket (backend), backed by Lance datasets.

## What This Module Does

- Streams realtime OHLC candles and indicator updates over WebSocket.
- Loads historical chunks lazily (`initial_data` + `load_more`) to keep startup fast.
- Renders three synchronized price styles:
  - Heikin Ashi candles (default)
  - Raw OHLC candles
  - Bloomberg-style OHLC bars (custom primitive)
- Renders active indicators:
  - NWE (overlay)
  - Z-Score (pane)
  - Stochastic Slow %K (pane)
- Provides an interactive indicator panel for toggling indicators and sub-features.

## Architecture (Refactored)

### Frontend entry + core modules

- `chart_realtime.js`
  - Thin composition entrypoint.
  - Creates plugins, chart context, legend binding, realtime controller.
- `core/chart_bootstrap.js`
  - Chart creation (theme, time/price formatting, scaling).
  - Price series setup (HA/raw/bars).
  - Overlay primitive wiring and chart-style switching logic.
- `core/legend.js`
  - Crosshair-driven OHLC legend rendering.
- `core/realtime_controller.js`
  - WebSocket lifecycle, reconnect, ping.
  - Message handlers for `initial_data`, `more_data`, `candle_update`, `indicator_update`.
  - Lazy loading orchestration and pane/indicator registration.

### Indicator system

- `indicators/indicator_registry.js`
  - Static default indicator and group definitions.
- `indicators/indicator_panel.js`
  - UI rendering and toggle wiring.
  - Runtime state and callbacks.
- `indicators/pane_manager.js`
  - Bottom-pane create/destroy/rebuild orchestration.

### Backend

- `websocket_server.py`
  - FastAPI app, `/ws`, `/api/trades`, `/api/candles/before`, `/`.
  - Initial payload assembly and `load_more` handling.
  - Background Lance polling stream for realtime updates.
  - Controlled asset serving via allowlist (`ASSET_PATHS`).
- `lance_data_provider.py`
  - Candles and indicator retrieval from Lance data.
  - CPU-safe fallback mapping for selected feature indices.

## File Layout (Relevant)

```text
lightweight_charts/
├── README.md
├── index.html
├── chart_realtime.js
├── websocket_server.py
├── lance_data_provider.py
├── core/
│   ├── chart_bootstrap.js
│   ├── legend.js
│   └── realtime_controller.js
├── indicators/
│   ├── indicator_registry.js
│   ├── indicator_panel.js
│   ├── pane_manager.js
│   ├── nwe/nwe.js
│   ├── zscore/zscore.js
│   └── stoch_slow/stoch_slow.js
├── plugins/
│   ├── fill-areas/fill-areas.js
│   ├── buy_sell_labels/labels.js
│   ├── buy_sell_positions/buy_sell_positions.js
│   ├── zscore-solid-fill/zscore-solid-fill.js
│   └── zscore-table/zscore-table-plugin.js
└── primitives/
    └── ohlc_bars_renderer.js
```

## Runtime Flow

1. Client opens `/` and loads `index.html` + `chart_realtime.js`.
2. `chart_realtime.js` boots chart context and connects to `ws://localhost:8000/ws`.
3. Server sends `initial_data` (candles + indicator snapshots).
4. Client registers panes/indicators and renders chart.
5. As user scrolls left, client sends `load_more` and prepends historical chunks.
6. Server broadcasts `candle_update` and `indicator_update` from Lance version polling.

## Start

From project root, use the script:

```bash
./scripts/lightweight/start_binance_chart.sh
```

Or manually:

```bash
cd lightweight_charts
uvicorn websocket_server:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- `http://localhost:8000`
- `http://127.0.0.1:8000`

## Environment Variables

- `LANCE_PATH` (default: `data/synced_market_farmer_data/lance`)
- `CHART_SYMBOL` (default: `BTC_USDT`)
- `CHART_INTERVAL` (default: `Min1`)
- `CHART_DAYS` (default: `31`)

## Notes for Extending Indicators

- Add frontend visualization modules under `indicators/` or `plugins/`.
- Register defaults in `indicators/indicator_registry.js`.
- Wire toggles in `core/realtime_controller.js` via `registerIndicator(...)`.
- Keep backend payload contracts explicit in `websocket_server.py`.
- Keep allowlisted assets updated in `ASSET_PATHS` when adding new JS/CSS/HTML modules.
