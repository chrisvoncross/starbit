// Cosmo on the sealed day. Green arrows on the extreme he marked,
// and the line that joins them. Same drawing as Entropie, other color.
// The file is one symbol. It draws only when that symbol is the chart.

const COLOR = '#00e676';

const drawState = { line: null, markers: null };
let payload = null;
let chartSymbol = null;
let inflight = null;

export function loadCosmo(onReady) {
    if (payload) {
        if (onReady) onReady();
        return;
    }
    if (inflight) return;
    inflight = Promise.all([
        fetch('/api/cosmo-oos').then((response) => (response.ok ? response.json() : { marks: [] })),
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

function activeMarks() {
    if (!payload || !Array.isArray(payload.marks)) return [];
    if (payload.symbol && chartSymbol && payload.symbol !== chartSymbol) return [];
    return payload.marks;
}

export function paintCosmo({ chart, enabled }) {
    if (drawState.line) {
        try { chart.removeSeries(drawState.line); } catch (e) { /* already gone */ }
        drawState.line = null;
        drawState.markers = null;
    }
    if (!enabled) return;
    const marks = activeMarks();
    const seen = new Set();
    const points = [];
    for (const mark of marks) {
        const time = Number(mark.time);
        const price = Number(mark.price);
        if (!Number.isFinite(time) || !Number.isFinite(price) || seen.has(time)) continue;
        seen.add(time);
        points.push({
            time,
            value: price,
            kind: mark.kind === 'low' ? 'low' : 'high',
        });
    }
    points.sort((a, b) => a.time - b.time);
    if (points.length < 2) return;
    const abs = Math.abs(points[0].value);
    const needed = abs > 0 && abs < 1 ? Math.ceil(-Math.log10(abs)) + 4 : 8;
    const precision = Math.max(8, Math.min(12, needed));
    const line = chart.addSeries(LightweightCharts.LineSeries, {
        color: COLOR,
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
        priceFormat: { type: 'price', precision, minMove: 10 ** -precision },
    });
    line.setData(points.map((point) => ({ time: point.time, value: point.value })));
    drawState.markers = LightweightCharts.createSeriesMarkers(line, points.map((point) => ({
        time: point.time,
        position: point.kind === 'high' ? 'aboveBar' : 'belowBar',
        color: COLOR,
        shape: point.kind === 'high' ? 'arrowDown' : 'arrowUp',
    })));
    drawState.line = line;
}
