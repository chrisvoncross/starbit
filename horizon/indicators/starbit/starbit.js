// Two own rows under the price. One point per candle, no line.
// The upper row is his end-reading on the full row.
// The lower row is the same reading after the deep row is removed.
// Nothing is drawn on the price.

const FULL = '#f2f2f2';
const BARE = '#8d8d8d';

const drawState = { chart: null, full: null, bare: null };
let payload = null;
let chartSymbol = null;
let inflight = null;

export function loadStarbit(onReady) {
    if (payload) {
        if (onReady) onReady();
        return;
    }
    if (inflight) return;
    inflight = Promise.all([
        fetch('/api/starbit-btc?v=starbit3').then((response) => (response.ok ? response.json() : { points: [] })),
        fetch('/api/config').then((response) => (response.ok ? response.json() : {})),
    ]).then(([body, config]) => {
        payload = body && Array.isArray(body.points) ? body : { points: [] };
        chartSymbol = config && config.symbol ? String(config.symbol) : null;
        inflight = null;
        if (onReady) onReady();
    }).catch(() => {
        payload = { points: [] };
        inflight = null;
    });
}

function bars(points, key, color) {
    const rows = [];
    for (const point of points) {
        const time = Number(point.time);
        const value = Number(point[key]);
        if (!Number.isFinite(time) || !Number.isFinite(value)) continue;
        rows.push({ time, value, color });
    }
    rows.sort((a, b) => a.time - b.time);
    const unique = [];
    for (const row of rows) {
        if (!unique.length || row.time !== unique[unique.length - 1].time) unique.push(row);
    }
    return unique;
}

function scale() {
    return {
        priceFormat: { type: 'volume' },
        autoscaleInfoProvider: () => ({
            priceRange: { minValue: 0, maxValue: 1 },
        }),
    };
}

function ensure(chart) {
    if (drawState.chart === chart && drawState.full && drawState.bare) return;
    detach();
    drawState.chart = chart;
    const fullPane = chart.panes().length;
    drawState.full = chart.addSeries(LightweightCharts.HistogramSeries, {
        ...scale(),
        color: FULL,
        title: 'volle Zeile',
        priceLineVisible: false,
        lastValueVisible: false,
    }, fullPane);
    const barePane = chart.panes().length;
    drawState.bare = chart.addSeries(LightweightCharts.HistogramSeries, {
        ...scale(),
        color: BARE,
        title: 'ohne Tiefe',
        priceLineVisible: false,
        lastValueVisible: false,
    }, barePane);
    for (const index of [fullPane, barePane]) {
        const pane = chart.panes()[index];
        if (pane && pane.setHeight) pane.setHeight(72);
    }
}

function detach() {
    const chart = drawState.chart;
    for (const series of [drawState.full, drawState.bare]) {
        if (chart && series) {
            try { chart.removeSeries(series); } catch (e) { /* already gone */ }
        }
    }
    drawState.chart = null;
    drawState.full = null;
    drawState.bare = null;
}

export function paintStarbit({ chart, enabled }) {
    if (!chart) return;
    if (!enabled || !payload || !Array.isArray(payload.points)) {
        detach();
        return;
    }
    if (payload.symbol && chartSymbol && payload.symbol !== chartSymbol) {
        detach();
        return;
    }
    ensure(chart);
    drawState.full.setData(bars(payload.points, 'voll', FULL));
    drawState.bare.setData(bars(payload.points, 'blind', BARE));
}
