// Swing turns on the loaded candles.
// A high is marked only after price has fallen by the chosen distance from
// that high. A low is marked only after price has risen by that distance.
// The dot sits on the extreme. It is not a forecast.

const LEVELS = [
    { id: 'bp1', bp: 1, line: '#ce93d8', marker: '#ce93d8' },
    { id: 'bp2', bp: 2, line: '#4fc3f7', marker: '#4fc3f7' },
    { id: 'bp3', bp: 3, line: '#aed581', marker: '#aed581' },
    { id: 'bp4', bp: 4, line: '#ff8a65', marker: '#ff8a65' },
    { id: 'bp5', bp: 5, line: '#e53935', marker: '#e53935' },
    { id: 'bp10', bp: 10, line: '#90a4ae', marker: '#90a4ae' },
    { id: 'bp20', bp: 20, line: '#ffb300', marker: '#ffb300' },
    { id: 'bp50', bp: 50, line: '#ffffff', marker: '#ffffff' },
];

function formatFor(price) {
    const abs = Math.abs(price);
    const needed = abs > 0 && abs < 1 ? Math.ceil(-Math.log10(abs)) + 4 : 8;
    const precision = Math.max(8, Math.min(12, needed));
    return { type: 'price', precision, minMove: 10 ** -precision };
}

export function swingPivots(candles, threshFrac) {
    if (!candles || candles.length < 2 || !(threshFrac > 0)) return [];
    const pivots = [];
    let trend = 0;
    let hi = candles[0].high;
    let lo = candles[0].low;
    let hiI = 0;
    let loI = 0;
    for (let i = 1; i < candles.length; i++) {
        const bar = candles[i];
        if (trend === 0) {
            if (bar.high >= hi) { hi = bar.high; hiI = i; }
            if (bar.low <= lo) { lo = bar.low; loI = i; }
            if (hi > 0 && (hi - lo) / hi >= threshFrac && hiI !== loI) {
                if (hiI < loI) {
                    trend = -1;
                    lo = bar.low;
                    loI = i;
                } else {
                    trend = 1;
                    hi = bar.high;
                    hiI = i;
                }
            }
        } else if (trend === 1) {
            if (bar.high >= hi) {
                hi = bar.high;
                hiI = i;
            } else if ((hi - bar.low) / hi >= threshFrac) {
                pivots.push({ time: candles[hiI].time, price: hi, kind: 'high' });
                trend = -1;
                lo = bar.low;
                loI = i;
            }
        } else if (bar.low <= lo) {
            lo = bar.low;
            loI = i;
        } else if (lo > 0 && (bar.high - lo) / lo >= threshFrac) {
            pivots.push({ time: candles[loI].time, price: lo, kind: 'low' });
            trend = 1;
            hi = bar.high;
            hiI = i;
        }
    }
    return pivots;
}

const drawState = { lines: [], markers: null, markerSeries: null };

export function paintSwingTurns({ chart, series, candles, enabled, activeLevels, overlayMarkers }) {
    for (const line of drawState.lines) {
        try { chart.removeSeries(line); } catch (e) { /* already gone */ }
    }
    drawState.lines = [];
    const markers = [];
    const seenTime = new Set();
    if (enabled && candles && candles.length) {
        for (const level of LEVELS) {
            if (!activeLevels[level.id]) continue;
            const pivots = swingPivots(candles, level.bp / 10000);
            if (!pivots.length) continue;
            const line = chart.addSeries(LightweightCharts.LineSeries, {
                color: level.line,
                lineWidth: level.bp >= 50 ? 2 : 1,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
                priceFormat: formatFor(pivots[0].price),
            });
            line.setData(pivots.map((p) => ({ time: p.time, value: p.price })));
            drawState.lines.push(line);
            for (const p of pivots) {
                if (seenTime.has(p.time)) continue;
                seenTime.add(p.time);
                markers.push({
                    time: p.time,
                    position: p.kind === 'high' ? 'aboveBar' : 'belowBar',
                    color: level.marker,
                    shape: p.kind === 'high' ? 'arrowDown' : 'arrowUp',
                });
            }
        }
    }
    if (overlayMarkers && overlayMarkers.length) {
        markers.push(...overlayMarkers);
    }
    markers.sort((a, b) => a.time - b.time);
    if (drawState.markers && drawState.markerSeries !== series) {
        drawState.markers.setMarkers([]);
        drawState.markers = null;
        drawState.markerSeries = null;
    }
    if (!drawState.markers) {
        drawState.markers = LightweightCharts.createSeriesMarkers(series, markers);
        drawState.markerSeries = series;
    } else {
        drawState.markers.setMarkers(markers);
    }
}
