// Nova cloud. One temperature per candle. The cloud sits on the candle's
// close and spreads above and below it only where the heat clears the
// resting temperature. Inside is red, then orange, the fringe is green.
// At or under the base, nothing is drawn. Coordinates come from the
// candle series, so the cloud cannot grow its own price scale.

const BASE = 0.40;
const SHOW_FROM = 0.46;

const drawState = {
    series: null,
    primitive: null,
};
let payload = null;
let chartSymbol = null;
let inflight = null;

export function loadNova(onReady) {
    if (payload) {
        if (onReady) onReady();
        return;
    }
    if (inflight) return;
    inflight = Promise.all([
        fetch('/api/nova-heat').then((response) => (response.ok ? response.json() : { points: [] })),
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

function heatByTime() {
    const map = new Map();
    if (!payload || !Array.isArray(payload.points)) return map;
    if (payload.symbol && chartSymbol && payload.symbol !== chartSymbol) return map;
    for (const point of payload.points) {
        const time = Number(point.time);
        const heat = Number(point.heat);
        if (!Number.isFinite(time) || !Number.isFinite(heat)) continue;
        map.set(time, heat);
    }
    return map;
}

function mix(a, b, t) {
    const k = Math.max(0, Math.min(1, t));
    return [
        Math.round(a[0] + (b[0] - a[0]) * k),
        Math.round(a[1] + (b[1] - a[1]) * k),
        Math.round(a[2] + (b[2] - a[2]) * k),
    ];
}

// 0 at the first visible heat, 1 at a hot wick.
function strength(heat) {
    return Math.max(0, Math.min(1, (heat - SHOW_FROM) / 0.40));
}

function coreColor(t) {
    const green = [70, 190, 80];
    const orange = [255, 150, 30];
    const red = [230, 40, 36];
    if (t < 0.45) return mix(green, orange, t / 0.45);
    return mix(orange, red, (t - 0.45) / 0.55);
}

class NovaCloudRenderer {
    constructor() {
        this._samples = [];
        this._chart = null;
        this._series = null;
        this._visible = true;
    }

    update(samples, chart, series, visible) {
        this._samples = samples || [];
        this._chart = chart;
        this._series = series;
        this._visible = visible;
    }

    draw() {}

    drawBackground(target) {
        if (!this._visible || !this._samples.length || !this._chart || !this._series) return;
        target.useMediaCoordinateSpace((scope) => {
            const ctx = scope.context;
            const width = scope.mediaSize.width;
            const height = scope.mediaSize.height;
            const ts = this._chart.timeScale();
            const spacing = (ts.options && ts.options().barSpacing) || 8;
            const colW = Math.max(4, Math.min(spacing * 1.25, 48));
            ctx.save();
            for (const sample of this._samples) {
                if (sample.heat < SHOW_FROM) continue;
                const x = ts.timeToCoordinate(sample.time);
                const y = this._series.priceToCoordinate(sample.price);
                if (x == null || y == null) continue;
                if (x < -colW || x > width + colW || y < -80 || y > height + 80) continue;
                const t = strength(sample.heat);
                const half = 14 + t * 58;
                const [r, g, b] = coreColor(t);
                const top = y - half;
                const grd = ctx.createLinearGradient(x, top, x, y + half);
                const edge = `rgba(70,190,80,0)`;
                const fringe = `rgba(90,180,70,${0.10 + t * 0.10})`;
                const mid = `rgba(255,150,30,${0.12 + t * 0.16})`;
                const core = `rgba(${r},${g},${b},${0.16 + t * 0.36})`;
                grd.addColorStop(0, edge);
                grd.addColorStop(0.16, fringe);
                grd.addColorStop(0.34, mid);
                grd.addColorStop(0.50, core);
                grd.addColorStop(0.66, mid);
                grd.addColorStop(0.84, fringe);
                grd.addColorStop(1, edge);
                ctx.fillStyle = grd;
                ctx.fillRect(x - colW / 2, top, colW, half * 2);
            }
            ctx.restore();
        });
    }
}

class NovaCloudPaneView {
    constructor(source) {
        this._source = source;
        this._renderer = new NovaCloudRenderer();
    }

    zOrder() {
        return 'bottom';
    }

    update() {
        this._renderer.update(
            this._source._samples,
            this._source._chart,
            this._source._series,
            this._source._visible,
        );
    }

    renderer() {
        return this._renderer;
    }
}

class NovaCloudPrimitive {
    constructor() {
        this._samples = [];
        this._chart = null;
        this._series = null;
        this._visible = true;
        this._paneView = new NovaCloudPaneView(this);
    }

    attached(params) {
        this._series = params.series;
        this._chart = params.chart;
    }

    detached() {
        this._series = null;
        this._chart = null;
    }

    setSamples(samples) {
        this._samples = samples || [];
        this.updateAllViews();
        if (this._chart) this._chart.applyOptions({});
    }

    setVisible(visible) {
        this._visible = visible;
        this.updateAllViews();
        if (this._chart) this._chart.applyOptions({});
    }

    paneViews() {
        return [this._paneView];
    }

    updateAllViews() {
        this._paneView.update();
    }

    autoscaleInfo() {
        return null;
    }
}

function samplesFrom(candles) {
    const heats = heatByTime();
    const rows = [];
    if (!heats.size || !candles) return rows;
    for (const candle of candles) {
        const time = Number(candle.time);
        if (!heats.has(time)) continue;
        const price = Number(candle.close);
        if (!Number.isFinite(price) || price <= 0) continue;
        rows.push({ time, price, heat: heats.get(time) });
    }
    rows.sort((a, b) => a.heat - b.heat);
    return rows;
}

function detach() {
    if (drawState.series && drawState.primitive) {
        try { drawState.series.detachPrimitive(drawState.primitive); } catch (e) { /* already gone */ }
    }
    drawState.series = null;
    drawState.primitive = null;
}

export function paintNova({ chart, series, candles, enabled }) {
    if (!series || !chart) return;
    if (drawState.series !== series) {
        detach();
        drawState.series = series;
        drawState.primitive = new NovaCloudPrimitive();
        series.attachPrimitive(drawState.primitive);
    }
    if (!enabled) {
        drawState.primitive.setVisible(false);
        return;
    }
    drawState.primitive.setVisible(true);
    drawState.primitive.setSamples(samplesFrom(candles));
}
