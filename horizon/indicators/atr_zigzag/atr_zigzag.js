// ATR ZigZag. The reversal distance is a multiple of the recent true range,
// so a quiet hour and a loud hour keep a similar number of swings.
//
// A high is marked only after price has fallen by multiplier × ATR from that
// high. A low is marked only after price has risen by that distance. The
// mark sits on the extreme. The last leg stays open until the distance is
// met. This is not a forecast.
//
// Length 14 and multiplier 2 are the usual settings (Wilder ATR, the
// ZigZag-ATR default). On 15-second bars that length is only a few minutes,
// so multiplier 2 twitches on the small hooks. 2.5 is the top of the
// published balanced band and the small step up from 2. DeepEntropy,
// "ZigZag ATR": 2.0–2.5 for swings, 3 and above for major pivots only.
//
// Sources:
// - J. Welles Wilder Jr., New Concepts in Technical Trading Systems, 1978.
//   The ATR used here is his smoothed true range.
// - Sylvain Vervoort, "The 1-2-3 Wave Count", Technical Analysis of
//   Stocks & Commodities, June 2013. A high-low zigzag whose reversal
//   distance follows ATR, with or without a percent term.
//   http://traders.com/documentation/feedbk_docs/2013/06/Vervoort.html
// - Thinkorswim ZigZagHighLow, the Vervoort study. The last segment can
//   still change.
//   https://toslc.thinkorswim.com/center/reference/Tech-Indicators/studies-library/V-Z/ZigZagHighLow
// - LuxAlgo, "Swing Magnitude Filters". An ATR multiple loosens in a loud
//   regime and tightens in a quiet one, so the swing count stays steadier
//   than a fixed percent. The threshold chooses a scale. It does not find
//   one true reversal.
//   https://www.luxalgo.com/library/concept/swing-magnitude-filters/
// - DeepEntropy, "ZigZag ATR" (TradingView). ATR length 14, multiplier 2.
//   https://www.tradingview.com/script/v8XJuorH-ZigZag-ATR/

const LENGTH = 14;
const MULTIPLIER = 2.5;
const COLOR = '#ffd54f';

function formatFor(price) {
    const abs = Math.abs(price);
    const needed = abs > 0 && abs < 1 ? Math.ceil(-Math.log10(abs)) + 4 : 8;
    const precision = Math.max(8, Math.min(12, needed));
    return { type: 'price', precision, minMove: 10 ** -precision };
}

function trueRanges(candles) {
    const tr = new Array(candles.length);
    tr[0] = candles[0].high - candles[0].low;
    for (let i = 1; i < candles.length; i++) {
        const high = candles[i].high;
        const low = candles[i].low;
        const prev = candles[i - 1].close;
        const span = high - low;
        const up = Math.abs(high - prev);
        const down = Math.abs(low - prev);
        tr[i] = span > up ? (span > down ? span : down) : (up > down ? up : down);
    }
    return tr;
}

function wilderAtr(tr, length) {
    const atr = new Array(tr.length).fill(NaN);
    if (tr.length < length || !(length > 0)) return atr;
    let seed = 0;
    for (let i = 0; i < length; i++) seed += tr[i];
    atr[length - 1] = seed / length;
    for (let i = length; i < tr.length; i++) {
        atr[i] = (atr[i - 1] * (length - 1) + tr[i]) / length;
    }
    return atr;
}

function highestAfter(candles, from, to) {
    let price = -Infinity;
    let index = -1;
    for (let j = from + 1; j <= to; j++) {
        if (candles[j].high >= price) { price = candles[j].high; index = j; }
    }
    return { price, index };
}

function lowestAfter(candles, from, to) {
    let price = Infinity;
    let index = -1;
    for (let j = from + 1; j <= to; j++) {
        if (candles[j].low <= price) { price = candles[j].low; index = j; }
    }
    return { price, index };
}

// The distance changes bar by bar, so an earlier bar can reach further than
// the bar that confirms the turn. The opposite extreme is therefore tracked
// over the whole leg, not taken from the confirming bar.
export function atrZigZagPivots(candles, length = LENGTH, multiplier = MULTIPLIER) {
    if (!candles || candles.length < 2 || !(multiplier > 0)) return [];
    const atr = wilderAtr(trueRanges(candles), length);
    const pivots = [];
    let trend = 0;
    let hi = candles[0].high;
    let lo = candles[0].low;
    let hiI = 0;
    let loI = 0;
    for (let i = 1; i < candles.length; i++) {
        const bar = candles[i];
        const need = atr[i] * multiplier;
        if (trend === 0) {
            if (bar.high >= hi) { hi = bar.high; hiI = i; }
            if (bar.low <= lo) { lo = bar.low; loI = i; }
            if (need > 0 && hi - lo >= need && hiI !== loI) {
                if (hiI < loI) {
                    trend = -1;
                    ({ price: hi, index: hiI } = highestAfter(candles, loI, i));
                } else {
                    trend = 1;
                    ({ price: lo, index: loI } = lowestAfter(candles, hiI, i));
                }
            }
        } else if (trend === 1) {
            if (bar.high >= hi) {
                hi = bar.high;
                hiI = i;
                lo = Infinity;
                loI = -1;
            } else {
                if (bar.low <= lo) { lo = bar.low; loI = i; }
                if (hi - lo >= need) {
                    pivots.push({ time: candles[hiI].time, price: hi, kind: 'high' });
                    trend = -1;
                    ({ price: hi, index: hiI } = highestAfter(candles, loI, i));
                }
            }
        } else if (bar.low <= lo) {
            lo = bar.low;
            loI = i;
            hi = -Infinity;
            hiI = -1;
        } else {
            if (bar.high >= hi) { hi = bar.high; hiI = i; }
            if (hi - lo >= need) {
                pivots.push({ time: candles[loI].time, price: lo, kind: 'low' });
                trend = 1;
                ({ price: lo, index: loI } = lowestAfter(candles, hiI, i));
            }
        }
    }
    return pivots;
}

const drawState = { line: null, markers: null };

export function paintAtrZigZag({ chart, series, candles, enabled }) {
    if (drawState.line) {
        try { chart.removeSeries(drawState.line); } catch (e) { /* already gone */ }
        drawState.line = null;
        drawState.markers = null;
    }
    if (!enabled || !candles || !candles.length) return;
    const pivots = atrZigZagPivots(candles);
    if (!pivots.length) return;
    const line = chart.addSeries(LightweightCharts.LineSeries, {
        color: COLOR,
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
        priceFormat: formatFor(pivots[0].price),
    });
    line.setData(pivots.map((p) => ({ time: p.time, value: p.price })));
    const markers = pivots.map((p) => ({
        time: p.time,
        position: p.kind === 'high' ? 'aboveBar' : 'belowBar',
        color: COLOR,
        shape: p.kind === 'high' ? 'arrowDown' : 'arrowUp',
    }));
    drawState.markers = LightweightCharts.createSeriesMarkers(line, markers);
    drawState.line = line;
}
