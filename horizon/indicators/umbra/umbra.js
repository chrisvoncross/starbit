// Umbra. Die Farbe ist die Dichte, wann die nächste ATR-Zickzack-Kerze kommt.
// Heiß auf der Kerze, auf der die Masse sitzt. Der Fleck liegt auf dem
// Schluss dieser Kerze. Das ist kein Schlauch um den Preis.
// Rot bleiben die echten 5-bp-Punkte, die malt Wende.

const drawState = {
    chart: null,
    series: null,
    primitive: null,
    payload: null,
    inflight: null,
    ready: false,
    ticket: 0,
};

function heat(mass) {
    const stops = [
        [0.02, [20, 60, 160]],
        [0.06, [0, 140, 170]],
        [0.10, [40, 170, 70]],
        [0.16, [220, 190, 30]],
        [0.24, [230, 80, 20]],
        [0.36, [190, 20, 20]],
    ];
    const m = Math.max(stops[0][0], Math.min(0.95, mass));
    let left = stops[0];
    let right = stops[stops.length - 1];
    for (let i = 1; i < stops.length; i++) {
        if (m <= stops[i][0]) {
            left = stops[i - 1];
            right = stops[i];
            break;
        }
    }
    const span = right[0] - left[0] || 1;
    const t = Math.max(0, Math.min(1, (m - left[0]) / span));
    return left[1].map((channel, i) => Math.round(channel + (right[1][i] - channel) * t));
}

class CloudRenderer {
    constructor() {
        this._cells = [];
        this._chart = null;
        this._series = null;
        this._visible = true;
    }

    update(cells, chart, series, visible) {
        this._cells = cells || [];
        this._chart = chart;
        this._series = series;
        this._visible = visible;
    }

    draw() {}

    drawBackground(target) {
        if (!this._visible || !this._cells.length || !this._chart || !this._series) return;
        target.useMediaCoordinateSpace((scope) => {
            const ctx = scope.context;
            const ts = this._chart.timeScale();
            const spacing = (ts.options && ts.options().barSpacing) || 8;
            const rx = Math.max(7, spacing * 1.15);
            ctx.save();
            ctx.globalCompositeOperation = 'lighter';
            const ordered = this._cells.slice().sort((a, b) => a.mass - b.mass);
            for (const cell of ordered) {
                const x = ts.timeToCoordinate(cell.time);
                const y = this._series.priceToCoordinate(cell.price);
                if (x == null || y == null) continue;
                const mass = cell.mass;
                const ry = 10 + mass * 42;
                const [r, g, b] = heat(mass);
                const alpha = Math.min(0.72, 0.12 + mass * 1.6);
                const radius = Math.max(rx, ry);
                const glow = ctx.createRadialGradient(x, y, 0, x, y, radius);
                glow.addColorStop(0, `rgba(${r},${g},${b},${alpha})`);
                glow.addColorStop(1, `rgba(${r},${g},${b},0)`);
                ctx.fillStyle = glow;
                ctx.beginPath();
                ctx.ellipse(x, y, rx * 1.8, ry, 0, 0, Math.PI * 2);
                ctx.fill();
            }
            ctx.restore();
        });
    }
}

class CloudPaneView {
    constructor(source) {
        this._source = source;
        this._renderer = new CloudRenderer();
    }
    zOrder() { return 'bottom'; }
    update() {
        this._renderer.update(
            this._source._cells,
            this._source._chart,
            this._source._series,
            this._source._visible,
        );
    }
    renderer() { return this._renderer; }
}

class CloudPrimitive {
    constructor() {
        this._cells = [];
        this._chart = null;
        this._series = null;
        this._visible = true;
        this._paneView = new CloudPaneView(this);
    }
    attached(params) {
        this._series = params.series;
        this._chart = params.chart;
    }
    detached() {
        this._series = null;
        this._chart = null;
    }
    setCells(cells) {
        this._cells = cells || [];
        this.updateAllViews();
        if (this._chart) this._chart.applyOptions({});
    }
    setVisible(visible) {
        this._visible = visible;
        this.updateAllViews();
        if (this._chart) this._chart.applyOptions({});
    }
    paneViews() { return [this._paneView]; }
    updateAllViews() { this._paneView.update(); }
    autoscaleInfo() { return null; }
}

function destroy() {
    if (drawState.series && drawState.primitive) {
        try { drawState.series.detachPrimitive(drawState.primitive); } catch (e) { /* already gone */ }
    }
    drawState.primitive = null;
    drawState.ready = false;
}

function loadCloud() {
    if (drawState.payload) return Promise.resolve(drawState.payload);
    if (drawState.inflight) return drawState.inflight;
    drawState.inflight = fetch('/api/umbra')
        .then((response) => (response.ok ? response.json() : { cells: [] }))
        .then((body) => {
            drawState.payload = body && Array.isArray(body.cells) ? body : { cells: [] };
            drawState.inflight = null;
            return drawState.payload;
        })
        .catch(() => {
            drawState.payload = { cells: [] };
            drawState.inflight = null;
            return drawState.payload;
        });
    return drawState.inflight;
}

function cellsOf(body) {
    return (body.cells || [])
        .map((cell) => ({
            time: Number(cell.time),
            price: Number(cell.price),
            mass: Number(cell.mass),
        }))
        .filter((cell) => [cell.time, cell.price, cell.mass].every(Number.isFinite) && cell.price > 0 && cell.mass > 0);
}

export function paintUmbra({ chart, series, enabled, onReady }) {
    const ticket = ++drawState.ticket;
    if (drawState.chart !== chart || drawState.series !== series) {
        destroy();
        drawState.chart = chart;
        drawState.series = series;
        drawState.payload = null;
    }
    const finish = () => {
        if (ticket === drawState.ticket && onReady) onReady();
    };
    if (!series) {
        finish();
        return;
    }
    if (!enabled) {
        if (drawState.primitive) drawState.primitive.setVisible(false);
        finish();
        return;
    }
    if (drawState.ready && drawState.primitive) {
        drawState.primitive.setVisible(true);
        finish();
        return;
    }
    loadCloud().then((body) => {
        if (ticket !== drawState.ticket) return;
        if (!drawState.primitive) {
            drawState.primitive = new CloudPrimitive();
            series.attachPrimitive(drawState.primitive);
        }
        drawState.primitive.setCells(cellsOf(body));
        drawState.primitive.setVisible(true);
        drawState.ready = true;
        finish();
    });
}
