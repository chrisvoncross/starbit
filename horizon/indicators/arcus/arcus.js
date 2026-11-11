// Arcus. Jede Prognose ist eine eigene Strecke: vom roten 5-bp-Punkt
// bis zum nächsten, mit einem grünen Pfeil dort.
// Die roten Pfeile bleiben auf der Kerzenserie von Wende.

const COLOR = '#39FF14';

const drawState = {
    chart: null,
    items: [],
    payload: null,
    inflight: null,
    ready: false,
    ticket: 0,
};

const SERIES_OPTIONS = {
    color: COLOR,
    lineWidth: 2,
    priceLineVisible: false,
    lastValueVisible: false,
    crosshairMarkerVisible: false,
    autoscaleInfoProvider: () => null,
};

function destroy(chart) {
    for (const item of drawState.items) {
        try { chart.removeSeries(item.line); } catch (e) { /* already gone */ }
    }
    drawState.items = [];
    drawState.ready = false;
}

function loadLine() {
    if (drawState.payload) return Promise.resolve(drawState.payload);
    if (drawState.inflight) return drawState.inflight;
    drawState.inflight = fetch('/api/arcus')
        .then((response) => (response.ok ? response.json() : { forecasts: [] }))
        .then((body) => {
            drawState.payload = body && Array.isArray(body.forecasts) ? body : { forecasts: [] };
            drawState.inflight = null;
            return drawState.payload;
        })
        .catch(() => {
            drawState.payload = { forecasts: [] };
            drawState.inflight = null;
            return drawState.payload;
        });
    return drawState.inflight;
}

function setVisible(visible) {
    for (const item of drawState.items) {
        item.line.applyOptions({ visible });
    }
}

function buildForecasts(chart, forecasts) {
    for (const forecast of forecasts) {
        const from = forecast.from;
        const to = forecast.to;
        if (!from || !to) continue;
        const start = Number(from.time);
        const end = Number(to.time);
        const startPrice = Number(from.value);
        const endPrice = Number(to.value);
        if (![start, end, startPrice, endPrice].every(Number.isFinite)) continue;
        if (end <= start || startPrice <= 0 || endPrice <= 0) continue;
        const line = chart.addSeries(LightweightCharts.LineSeries, SERIES_OPTIONS);
        line.setData([
            { time: start, value: startPrice },
            { time: end, value: endPrice },
        ]);
        const up = forecast.direction === 'up';
        if (LightweightCharts.createSeriesMarkers) {
            LightweightCharts.createSeriesMarkers(line, [{
                time: end,
                position: up ? 'belowBar' : 'aboveBar',
                color: COLOR,
                shape: up ? 'arrowUp' : 'arrowDown',
            }]);
        }
        drawState.items.push({ line });
    }
}

export function paintArcus({ chart, enabled, onReady }) {
    const ticket = ++drawState.ticket;
    if (drawState.chart !== chart) {
        if (drawState.chart) destroy(drawState.chart);
        drawState.chart = chart;
        drawState.ready = false;
        drawState.payload = null;
    }
    const finish = () => {
        if (ticket === drawState.ticket && onReady) onReady();
    };
    if (!enabled) {
        setVisible(false);
        finish();
        return;
    }
    if (drawState.ready) {
        setVisible(true);
        finish();
        return;
    }
    loadLine().then((body) => {
        if (ticket !== drawState.ticket) return;
        destroy(chart);
        drawState.chart = chart;
        buildForecasts(chart, body.forecasts || []);
        drawState.ready = true;
        finish();
    });
}
