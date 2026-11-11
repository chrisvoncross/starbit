// Entropy-scaled SuperTrend, v2. Same ATR(10), same 30-bar centered binary
// sign entropy H, same ratcheted band around (high+low)/2, wick flip, and
// vertex on the extreme of the leg. One change: the multiple is scaled by
// the percentile rank of H over the full loaded history, not by raw H.
// Multiple = 2 + 2 * rank(H), rank in [0,1], mid-rank for ties.
//
// Source of the change: the original script (BullVisionCapital, TradingView
// QIbCQopk, November 2024, mirrored by LuxAlgo) compares entropy to a
// percentile of its own history before it modulates the ATR multiple.
// Raw binary entropy of a 30-bar sign window sits near 1 on a random-sign
// path, so the raw multiple stays between 3.65 and 4 on almost every bar.

const LENGTH = 10;
const RADIUS = 15;
const COLOR = '#ba68c8';

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

function lowerBound(sorted, x) {
    let lo = 0;
    let hi = sorted.length;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (sorted[mid] < x) lo = mid + 1; else hi = mid;
    }
    return lo;
}

function upperBound(sorted, x) {
    let lo = 0;
    let hi = sorted.length;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (sorted[mid] <= x) lo = mid + 1; else hi = mid;
    }
    return lo;
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

export function entropiev2Pivots(candles) {
    if (!candles || candles.length < LENGTH + 2) return [];
    const atr = wilderAtr(candles, LENGTH);
    const rets = new Array(candles.length).fill(0);
    for (let i = 1; i < candles.length; i++) {
        rets[i] = candles[i].close - candles[i - 1].close;
    }
    const h = new Array(candles.length).fill(NaN);
    const pool = [];
    for (let i = LENGTH - 1; i < candles.length; i++) {
        if (!(atr[i] > 0)) continue;
        h[i] = entropy(rets, i);
        pool.push(h[i]);
    }
    if (!pool.length) return [];
    pool.sort((x, y) => x - y);
    const dist = new Array(candles.length).fill(NaN);
    for (let i = LENGTH - 1; i < candles.length; i++) {
        if (Number.isNaN(h[i])) continue;
        const below = lowerBound(pool, h[i]);
        const equal = upperBound(pool, h[i]) - below;
        const rank = (below + 0.5 * equal) / pool.length;
        dist[i] = atr[i] * (2 + 2 * rank);
    }
    return pivotsFromDist(candles, dist);
}

const drawState = { line: null, markers: null };

export function paintEntropiev2({ chart, candles, enabled }) {
    if (drawState.line) {
        try { chart.removeSeries(drawState.line); } catch (e) { /* already gone */ }
        drawState.line = null;
        drawState.markers = null;
    }
    if (!enabled || !candles || !candles.length) return;
    const pivots = [];
    for (const p of entropiev2Pivots(candles)) {
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
