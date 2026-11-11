// Entropy-scaled SuperTrend. LuxAlgo, November 2024, describes the
// mechanism and does not publish the script constants: Shannon entropy of
// the recent direction widens the ATR multiple when the path is chaotic
// and tightens it when the path is one-way. The constants below are that
// rule, not their hidden defaults.
//
// Entropy is the binary entropy of up against down closes in a 30-bar
// window centered on the bar, so the window uses the finished path.
// The multiple runs from 2 (entropy 0) to 4 (entropy 1). ATR length is 10.
// The flip uses the wick, and the mark sits on the extreme of the leg.

const LENGTH = 10;
const RADIUS = 15;
const COLOR = '#ff8a65';

function formatFor(price) {
    const abs = Math.abs(price);
    const needed = abs > 0 && abs < 1 ? Math.ceil(-Math.log10(abs)) + 4 : 8;
    const precision = Math.max(8, Math.min(12, needed));
    return { type: 'price', precision, minMove: 10 ** -precision };
}

function wilderAtr(candles, length) {
    const n = candles.length;
    const tr = new Array(n);
    tr[0] = candles[0].high - candles[0].low;
    for (let i = 1; i < n; i++) {
        const high = candles[i].high;
        const low = candles[i].low;
        const prev = candles[i - 1].close;
        tr[i] = Math.max(high - low, Math.abs(high - prev), Math.abs(low - prev));
    }
    const atr = new Array(n).fill(NaN);
    if (n < length) return atr;
    let seed = 0;
    for (let i = 0; i < length; i++) seed += tr[i];
    atr[length - 1] = seed / length;
    for (let i = length; i < n; i++) atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length;
    return atr;
}

function entropy(rets, i) {
    const a = Math.max(1, i - RADIUS);
    const b = Math.min(rets.length - 1, i + RADIUS);
    let up = 0;
    let seen = 0;
    for (let t = a; t <= b; t++) {
        if (!rets[t]) continue;
        seen += 1;
        if (rets[t] > 0) up += 1;
    }
    if (seen < 2) return 1;
    const p = up / seen;
    if (p <= 0 || p >= 1) return 0;
    return -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p));
}

function pivotsFromDist(candles, dist) {
    const pivots = [];
    let lower = NaN;
    let upper = NaN;
    let trend = 1;
    let ext = candles[0].high;
    let extI = 0;
    let started = false;
    for (let i = 1; i < candles.length; i++) {
        const d = dist[i];
        if (!(d > 0)) continue;
        const mid = (candles[i].high + candles[i].low) / 2;
        const basicLower = mid - d;
        const basicUpper = mid + d;
        if (!started) {
            lower = basicLower;
            upper = basicUpper;
            started = true;
            ext = candles[i].high;
            extI = i;
            continue;
        }
        const prevHigh = candles[i - 1].high;
        const prevLow = candles[i - 1].low;
        lower = prevHigh > lower ? Math.max(basicLower, lower) : basicLower;
        upper = prevLow < upper ? Math.min(basicUpper, upper) : basicUpper;
        if (trend === 1) {
            if (candles[i].high >= ext) { ext = candles[i].high; extI = i; }
            if (candles[i].low < lower && extI !== i) {
                pivots.push({ time: candles[extI].time, price: ext, kind: 'high' });
                trend = -1;
                ext = candles[i].low;
                extI = i;
            }
        } else if (candles[i].low <= ext) {
            ext = candles[i].low;
            extI = i;
        } else if (candles[i].high > upper && extI !== i) {
            pivots.push({ time: candles[extI].time, price: ext, kind: 'low' });
            trend = 1;
            ext = candles[i].high;
            extI = i;
        }
    }
    return pivots;
}

export function entropiePivots(candles) {
    if (!candles || candles.length < LENGTH + 2) return [];
    const atr = wilderAtr(candles, LENGTH);
    const rets = new Array(candles.length).fill(0);
    for (let i = 1; i < candles.length; i++) {
        rets[i] = candles[i].close - candles[i - 1].close;
    }
    const dist = new Array(candles.length).fill(NaN);
    for (let i = LENGTH - 1; i < candles.length; i++) {
        if (!(atr[i] > 0)) continue;
        dist[i] = atr[i] * (2 + 2 * entropy(rets, i));
    }
    return pivotsFromDist(candles, dist);
}

const drawState = { line: null, markers: null };

export function paintEntropie({ chart, candles, enabled }) {
    if (drawState.line) {
        try { chart.removeSeries(drawState.line); } catch (e) { /* already gone */ }
        drawState.line = null;
        drawState.markers = null;
    }
    if (!enabled || !candles || !candles.length) return;
    const pivots = [];
    for (const p of entropiePivots(candles)) {
        if (!pivots.length || p.time > pivots[pivots.length - 1].time) pivots.push(p);
    }
    if (pivots.length < 2) return;
    const line = chart.addSeries(LightweightCharts.LineSeries, {
        color: COLOR,
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
        priceFormat: formatFor(pivots[0].price),
    });
    line.setData(pivots.map((p) => ({ time: p.time, value: p.price })));
    drawState.markers = LightweightCharts.createSeriesMarkers(line, pivots.map((p) => ({
        time: p.time,
        position: p.kind === 'high' ? 'aboveBar' : 'belowBar',
        color: COLOR,
        shape: p.kind === 'high' ? 'arrowDown' : 'arrowUp',
    })));
    drawState.line = line;
}
