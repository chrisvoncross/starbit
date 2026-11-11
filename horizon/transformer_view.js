// transformer_view.js — entry point for HORIZON_MODE=transformer_view.
//
// Mirrors chart_realtime.js (the legacy base-viz composition root) but wires
// the TransformerViewController instead of the raw-farmer RealtimeController.
// The chart context, plugins and legend are REUSED verbatim from the base viz.

console.log('🚀 Lightweight Charts version:', LightweightCharts.version);

import { FillAreasPlugin } from './plugins/fill-areas/fill-areas.js';
import { LabelsPlugin } from './plugins/buy_sell_labels/labels.js';
import { BuySellPositionsPlugin } from './plugins/buy_sell_positions/buy_sell_positions.js';
import { OhlcBarsPlugin } from './primitives/ohlc_bars_renderer.js';

import { createChartContext } from './core/chart_bootstrap.js';
import { bindLegend } from './core/legend.js';
import { TransformerViewController } from './core/transformer_view_controller.js';

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

const controller = new TransformerViewController({ chartCtx, statusEl });
controller.init();

console.log('🎯 Transformer view initialized — awaiting pinned window packs...');
