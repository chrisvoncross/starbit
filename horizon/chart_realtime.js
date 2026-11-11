// GhostByte Realtime Chart - Lightweight Charts v5 + WebSocket
console.log('🚀 Lightweight Charts version:', LightweightCharts.version);

import { FillAreasPlugin } from './plugins/fill-areas/fill-areas.js';
import { LabelsPlugin } from './plugins/buy_sell_labels/labels.js';
import { BuySellPositionsPlugin } from './plugins/buy_sell_positions/buy_sell_positions.js';
import { OhlcBarsPlugin } from './primitives/ohlc_bars_renderer.js';

import { createChartContext } from './core/chart_bootstrap.js';
import { bindLegend } from './core/legend.js';
import { RealtimeController } from './core/realtime_controller.js?v=zerofees6';

const statusEl = document.getElementById('status');
const chartContainer = document.getElementById('chart-container');

const fillAreas = new FillAreasPlugin({ showLines: false });
const labels = new LabelsPlugin();
const buySellPositions = new BuySellPositionsPlugin([]);

const chartCtx = createChartContext({
    chartContainer,
    fillAreas,
    labels,
    buySellPositions,
    OhlcBarsPlugin,
});

bindLegend(chartCtx.chart, () => chartCtx.getActiveSeries(), {
    legendOpen: document.getElementById('legend-open'),
    legendHigh: document.getElementById('legend-high'),
    legendLow: document.getElementById('legend-low'),
    legendClose: document.getElementById('legend-close'),
    legendChange: document.getElementById('legend-change'),
});

const realtimeController = new RealtimeController({
    chartCtx,
    statusEl,
    FillAreasPluginClass: FillAreasPlugin,
});
realtimeController.init();

console.log('🎯 Chart initialized - waiting for data...');
