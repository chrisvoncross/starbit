// =====================================================================
// PHASE 1 MODELL — AUSGABE DER BTC-PRÜFUNG
// Das hier ist nicht die rote 5-Basispunkt-Linie.
// Es zeichnet die Wendepunkte, die das Phase-1-Modell auf dem OOS-Tag
// von BTC gesetzt hat: Wahrscheinlichkeit >= 0.5 auf dem offenen Extrem.
// Punkt auf dem Extrem. Eine Linie von Extrem zu Extrem, wie bei 5 Basispunkten.
// Darf wieder raus. Nicht ins Modell zurückfüttern.
// =====================================================================

const GREEN_LINE = '#26a69a';
const GREEN_MARK = '#26a69a';

const drawState = { lines: [], markers: null, markerSeries: null };
let payload = null;
let inflight = null;

export function loadPhase1Model(onReady) {
    if (payload) {
        if (onReady) onReady();
        return;
    }
    if (inflight) return;
    inflight = fetch('/api/phase1-model-btc')
        .then((response) => (response.ok ? response.json() : { marks: [] }))
        .then((body) => {
            payload = body && Array.isArray(body.marks) ? body : { marks: [] };
            inflight = null;
            if (onReady) onReady();
        })
        .catch(() => {
            payload = { marks: [] };
            inflight = null;
        });
}

export function phase1ArrowMarkers(enabled) {
    const marks = enabled && payload && Array.isArray(payload.marks) ? payload.marks : [];
    const markers = [];
    const seen = new Set();
    for (const mark of marks) {
        const time = Number(mark.time);
        const price = Number(mark.price);
        if (!Number.isFinite(time) || !Number.isFinite(price) || seen.has(time)) continue;
        seen.add(time);
        markers.push({
            time,
            position: mark.kind === 'high' ? 'aboveBar' : 'belowBar',
            color: GREEN_MARK,
            shape: mark.kind === 'high' ? 'arrowDown' : 'arrowUp',
        });
    }
    return markers;
}

export function paintPhase1Model({ chart, enabled }) {
    for (const line of drawState.lines) {
        try { chart.removeSeries(line); } catch (e) { /* already gone */ }
    }
    drawState.lines = [];
    const marks = enabled && payload && Array.isArray(payload.marks) ? payload.marks : [];
    if (!marks.length) return;

    const seen = new Set();
    const points = [];
    for (const mark of marks) {
        const time = Number(mark.time);
        const price = Number(mark.price);
        if (!Number.isFinite(time) || !Number.isFinite(price) || seen.has(time)) continue;
        seen.add(time);
        points.push({ time, value: price });
    }
    points.sort((a, b) => a.time - b.time);
    if (points.length < 2) return;
    const line = chart.addSeries(LightweightCharts.LineSeries, {
        color: GREEN_LINE,
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
    });
    line.setData(points);
    drawState.lines.push(line);
}

// =====================================================================
// PHASE 1 MODELL — AUSGABE DER BTC-PRÜFUNG
// Das hier ist nicht die rote 5-Basispunkt-Linie.
// Es zeichnet die Wendepunkte, die das Phase-1-Modell auf dem OOS-Tag
// von BTC gesetzt hat: Wahrscheinlichkeit >= 0.5 auf dem offenen Extrem.
// Punkt auf dem Extrem. Eine Linie von Extrem zu Extrem, wie bei 5 Basispunkten.
// Darf wieder raus. Nicht ins Modell zurückfüttern.
// =====================================================================
