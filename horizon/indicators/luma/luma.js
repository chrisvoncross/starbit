// Marks Luma wrote. A hollow yellow ring held, or is still open.
// A hollow red ring was pierced. A hole in the series is not drawn.
// The red line runs from ring edge to ring edge.

const RING = '#ffe14a';
const BROKEN = '#ef3b3b';
const LINE = '#ef3b3b';
const RADIUS = 8;
const RING_WIDTH = 3;
const LINE_WIDTH = 3;

const drawState = { series: null, primitive: null };
let payload = null;
let chartSymbol = null;
let inflight = null;

export function loadLuma(onReady) {
    if (payload) {
        if (onReady) onReady();
        return;
    }
    if (inflight) return;
    inflight = Promise.all([
        fetch('/api/luma-btc?v=luma099').then((response) => (response.ok ? response.json() : { marks: [] })),
        fetch('/api/config').then((response) => (response.ok ? response.json() : {})),
    ]).then(([body, config]) => {
        payload = body && Array.isArray(body.marks) ? body : { marks: [] };
        chartSymbol = config && config.symbol ? String(config.symbol) : null;
        inflight = null;
        if (onReady) onReady();
    }).catch(() => {
        payload = { marks: [] };
        inflight = null;
    });
}

function modelPoints(marks, candles) {
    const byTime = new Map();
    for (const candle of candles || []) {
        const time = Number(candle.time);
        if (Number.isFinite(time)) byTime.set(time, candle);
    }
    const points = [];
    for (const mark of marks) {
        if (!mark || mark.kind === 'verpasst' || mark.kind === 'loch') continue;
        const time = Number(mark.time);
        const candle = byTime.get(time);
        if (!candle) continue;
        const price = Number(mark.turn === 'high' ? candle.high : candle.low);
        if (!Number.isFinite(price)) continue;
        points.push({ time, price, broken: mark.kind === 'bruch' });
    }
    points.sort((a, b) => a.time - b.time);
    const unique = [];
    for (const point of points) {
        if (!unique.length || point.time !== unique[unique.length - 1].time) unique.push(point);
    }
    return unique;
}

class LumaRenderer {
    constructor() {
        this._points = [];
        this._chart = null;
        this._series = null;
        this._visible = true;
    }

    update(points, chart, series, visible) {
        this._points = points || [];
        this._chart = chart;
        this._series = series;
        this._visible = visible;
    }

    draw(target) {
        if (!this._visible || this._points.length < 1 || !this._chart || !this._series) return;
        target.useMediaCoordinateSpace((scope) => {
            const ctx = scope.context;
            const placed = [];
            const timeScale = this._chart.timeScale();
            for (const point of this._points) {
                const x = timeScale.timeToCoordinate(point.time);
                const y = this._series.priceToCoordinate(point.price);
                if (x == null || y == null) continue;
                placed.push({ x, y, broken: point.broken });
            }
            if (!placed.length) return;
            ctx.save();
            ctx.lineCap = 'butt';
            ctx.strokeStyle = LINE;
            ctx.lineWidth = LINE_WIDTH;
            ctx.beginPath();
            for (let i = 0; i < placed.length - 1; i += 1) {
                const a = placed[i];
                const b = placed[i + 1];
                const dx = b.x - a.x;
                const dy = b.y - a.y;
                const len = Math.hypot(dx, dy);
                if (len <= RADIUS * 2) continue;
                const ux = dx / len;
                const uy = dy / len;
                ctx.moveTo(a.x + ux * RADIUS, a.y + uy * RADIUS);
                ctx.lineTo(b.x - ux * RADIUS, b.y - uy * RADIUS);
            }
            ctx.stroke();
            ctx.lineWidth = RING_WIDTH;
            for (const point of placed) {
                ctx.beginPath();
                ctx.strokeStyle = point.broken ? BROKEN : RING;
                ctx.arc(point.x, point.y, RADIUS, 0, Math.PI * 2);
                ctx.stroke();
            }
            ctx.restore();
        });
    }
}

class LumaPaneView {
    constructor(source) {
        this._source = source;
        this._renderer = new LumaRenderer();
    }

    zOrder() {
        return 'top';
    }

    update() {
        this._renderer.update(
            this._source._points,
            this._source._chart,
            this._source._series,
            this._source._visible,
        );
    }

    renderer() {
        return this._renderer;
    }
}

class LumaPrimitive {
    constructor() {
        this._points = [];
        this._chart = null;
        this._series = null;
        this._visible = true;
        this._paneView = new LumaPaneView(this);
    }

    attached(params) {
        this._series = params.series;
        this._chart = params.chart;
    }

    detached() {
        this._series = null;
        this._chart = null;
    }

    setPoints(points) {
        this._points = points || [];
        this._paneView.update();
        if (this._chart) this._chart.applyOptions({});
    }

    setVisible(visible) {
        this._visible = visible;
        this._paneView.update();
        if (this._chart) this._chart.applyOptions({});
    }

    paneViews() {
        return [this._paneView];
    }

    updateAllViews() {
        this._paneView.update();
    }
}

function detach() {
    if (drawState.series && drawState.primitive) {
        try { drawState.series.detachPrimitive(drawState.primitive); } catch (e) { /* already gone */ }
    }
    drawState.series = null;
    drawState.primitive = null;
}

export function paintLuma({ chart, series, candles, enabled }) {
    if (!series || !chart) return;
    if (drawState.series !== series) {
        detach();
        drawState.series = series;
        drawState.primitive = new LumaPrimitive();
        series.attachPrimitive(drawState.primitive);
    }
    if (!enabled || !payload || !Array.isArray(payload.marks)) {
        drawState.primitive.setVisible(false);
        return;
    }
    if (payload.symbol && chartSymbol && payload.symbol !== chartSymbol) {
        drawState.primitive.setVisible(false);
        return;
    }
    drawState.primitive.setVisible(true);
    drawState.primitive.setPoints(modelPoints(payload.marks, candles));
}
