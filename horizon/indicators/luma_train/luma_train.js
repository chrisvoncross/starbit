// Head A is the booked mark: yellow when it held, dull red when it was pierced.
// Head B is where he pointed: the start and the turn, in green.
// Head C is the nearness score, a count on the candle. Rising counts sit above the high, falling counts under the low. Nine is the full run.
// Head D is how far the step's picture beats staying still. The grey line is zero.
// Above zero the picture is closer than doing nothing. Missing rows draw nothing.
// Buch: Strecken and Buch: Liste use the same boxes, stops and table as Nordstern's positions and table.
// The turns are the model's booked marks. The money ruler is Nordstern's, so both panels read the same way.

import { dressTrades, paintIdealBook } from '../nordstern/nordstern.js?v=buchgleich';

const FAIL = '#8d4b4b';
const YELLOW = '#f5c518';
const GREEN = '#3ddc6a';
const NUMBER = '#dbe7ff';

const LAGE = '#7eb6ff';
const IMAGINATION = '#c084fc';
const ZERO = '#5c6570';

const drawState = { series: null, primitive: null, chart: null, lage: null, imagination: null, imaginationZero: null };
let payload = null;
let inflight = null;
let fetchedAt = 0;

export function loadLumaTrain(onReady) {
    const now = Date.now();
    if (payload && now - fetchedAt < 5000) {
        if (onReady) onReady();
        return;
    }
    if (inflight) return;
    inflight = Promise.all([
        fetch('/api/luma-train?v=buch').then((response) => (response.ok ? response.json() : { paare: [] })),
        fetch('/api/config').then((response) => (response.ok ? response.json() : {})),
    ]).then(([body, config]) => {
        payload = body && (Array.isArray(body.paare) || Array.isArray(body.gesetzt)) ? body : { paare: [], gesetzt: [] };
        payload.chartSymbol = config && config.symbol ? String(config.symbol) : null;
        fetchedAt = Date.now();
        inflight = null;
        if (onReady) onReady();
    }).catch(() => {
        payload = { paare: [] };
        inflight = null;
    });
}

function nearest(candles, time) {
    let best = null;
    let gap = 61;
    for (const candle of candles) {
        const stamp = Number(candle.time);
        if (!Number.isFinite(stamp)) continue;
        const delta = Math.abs(stamp - time);
        if (delta < gap) {
            gap = delta;
            best = candle;
        }
    }
    return best;
}

function lageSequence(rows, candles) {
    const points = [];
    for (const row of rows || []) {
        const candle = nearest(candles, Number(row.zeit));
        if (!candle) continue;
        const value = Number(row.wert);
        const time = Number(candle.time);
        const high = Number(candle.high);
        const low = Number(candle.low);
        if (!Number.isFinite(value) || !Number.isFinite(time) || !Number.isFinite(high) || !Number.isFinite(low)) continue;
        if (points.length && time <= points[points.length - 1].time) continue;
        points.push({ time, value, high, low });
    }
    let rising = 0;
    let falling = 0;
    for (let index = 1; index < points.length; index += 1) {
        const point = points[index];
        if (point.value > points[index - 1].value) {
            rising += 1;
            falling = 0;
            point.count = rising;
            point.up = true;
        } else if (point.value < points[index - 1].value) {
            falling += 1;
            rising = 0;
            point.count = falling;
            point.up = false;
        }
    }
    return points.filter((point) => point.count);
}

function paintLageCount(ctx, x, y, count, up) {
    const shown = Math.min(count, 9);
    ctx.font = count >= 9 ? 'bold 12px sans-serif' : '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = up ? 'bottom' : 'top';
    const py = up ? y - 3 : y + 3;
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#111111';
    ctx.strokeText(String(shown), x, py);
    ctx.fillStyle = count >= 9 ? '#ffe08a' : (up ? '#7eb6ff' : '#9aa7b8');
    ctx.fillText(String(shown), x, py);
}

function scored(rows, candles) {
    const points = [];
    for (const row of rows || []) {
        const candle = nearest(candles, Number(row.zeit));
        if (!candle) continue;
        const value = Number(row.wert);
        const time = Number(candle.time);
        if (!Number.isFinite(value) || !Number.isFinite(time)) continue;
        if (points.length && time <= points[points.length - 1].time) continue;
        points.push({ time, value });
    }
    return points;
}

function dropSeries(chart, series) {
    if (!series) return;
    try { chart.removeSeries(series); } catch (e) { /* already gone */ }
}

function lineOptions(color) {
    return {
        color,
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: true,
        crosshairMarkerVisible: false,
        priceFormat: { type: 'price', precision: 3, minMove: 0.001 },
    };
}

function holdPane(chart, key, color, points, withZero) {
    const slot = key;
    if (!points.length) {
        if (drawState[slot]) {
            dropSeries(chart, drawState[slot]);
            drawState[slot] = null;
        }
        if (withZero && drawState.imaginationZero) {
            dropSeries(chart, drawState.imaginationZero);
            drawState.imaginationZero = null;
        }
        return;
    }
    if (!drawState[slot]) {
        const index = chart.panes().length;
        drawState[slot] = chart.addSeries(LightweightCharts.LineSeries, lineOptions(color), index);
        if (withZero) {
            drawState.imaginationZero = chart.addSeries(LightweightCharts.LineSeries, {
                ...lineOptions(ZERO),
                lineWidth: 1,
                lastValueVisible: false,
            }, index);
        }
        const paneIndex = index;
        requestAnimationFrame(() => {
            const pane = chart.panes()[paneIndex];
            if (pane && pane.setHeight) pane.setHeight(96);
        });
    }
    drawState[slot].setData(points);
    if (withZero && drawState.imaginationZero) {
        drawState.imaginationZero.setData(points.map((point) => ({ time: point.time, value: 0 })));
    }
}

function placedMarks(marken, candles) {
    const points = [];
    for (const mark of marken || []) {
        const candle = nearest(candles, Number(mark.zeit));
        if (!candle) continue;
        const high = mark.marke === 'hoch';
        const price = Number(high ? candle.high : candle.low);
        if (!Number.isFinite(price)) continue;
        points.push({
            time: Number(candle.time),
            offset: 0,
            price,
            high,
            held: mark.urteil === 'gehalten',
        });
    }
    return points;
}

function logGap(top, bottom) {
    if (!(top > 0) || !(bottom > 0) || !Number.isFinite(top) || !Number.isFinite(bottom)) return null;
    return Math.log(top / bottom);
}

function bookWeg(high, held, marked, barHigh, barLow) {
    if (high && held) return logGap(marked, barLow);
    if (!high && held) return logGap(barHigh, marked);
    if (high && !held) return logGap(barHigh, marked) == null ? null : -logGap(barHigh, marked);
    if (!high && !held) return logGap(marked, barLow) == null ? null : -logGap(marked, barLow);
    return null;
}

function candleAt(candles, time) {
    let best = null;
    let gap = 16;
    for (const candle of candles) {
        const stamp = Number(candle.time);
        if (!Number.isFinite(stamp)) continue;
        const delta = Math.abs(stamp - time);
        if (delta < gap) {
            gap = delta;
            best = candle;
        }
    }
    return best;
}

function firstPierce(candles, entryTime, high, price, limit) {
    for (const candle of candles) {
        const stamp = Number(candle.time);
        if (!Number.isFinite(stamp) || stamp <= entryTime) continue;
        if (limit != null && stamp > limit) break;
        const barHigh = Number(candle.high);
        const barLow = Number(candle.low);
        const through = high ? barHigh > price : barLow < price;
        if (through) return candle;
    }
    return null;
}

function bookLegs(marken, candles) {
    const marks = (marken || []).filter((mark) => mark && (mark.marke === 'hoch' || mark.marke === 'tief') && mark.zeit != null);
    marks.sort((a, b) => Number(a.zeit) - Number(b.zeit));
    const series = (candles || []).slice().sort((a, b) => Number(a.time) - Number(b.time));
    const legs = [];
    marks.forEach((mark, index) => {
        const high = mark.marke === 'hoch';
        const entry = candleAt(series, Number(mark.zeit));
        if (!entry) return;
        const entryPrice = Number(high ? entry.high : entry.low);
        if (!Number.isFinite(entryPrice)) return;
        const entryTime = Number(entry.time);
        const next = marks[index + 1];
        const nextTime = next ? Number(next.zeit) : null;
        let kind = mark.urteil || 'offen';
        let held = kind === 'gehalten';
        let booked = kind === 'gehalten' || kind === 'bruch';
        let exit = null;
        if (kind === 'loch') {
            legs.push({
                side: high ? 'short' : 'long',
                high,
                entryTime,
                entryPrice,
                exitTime: null,
                exitPrice: null,
                kind,
                held: false,
                booked: false,
                weg: null,
                net: null,
            });
            return;
        }
        if (booked && mark.gegen_zeit != null) exit = candleAt(series, Number(mark.gegen_zeit));
        if (!exit && kind === 'gehalten') {
            const opposite = marks.slice(index + 1).find((item) => item.marke !== mark.marke);
            if (opposite) exit = candleAt(series, Number(opposite.zeit));
        }
        if (!exit && kind === 'bruch') exit = firstPierce(series, entryTime, high, entryPrice, nextTime);
        if (!booked) {
            const pierce = firstPierce(series, entryTime, high, entryPrice, null);
            if (pierce) {
                exit = pierce;
                held = false;
                kind = 'offen';
            } else {
                for (let i = series.length - 1; i >= 0; i -= 1) {
                    if (Number(series[i].time) > entryTime) {
                        exit = series[i];
                        held = true;
                        kind = 'offen';
                        break;
                    }
                }
            }
        }
        if (!exit) {
            legs.push({
                side: high ? 'short' : 'long',
                high,
                entryTime,
                entryPrice,
                exitTime: null,
                exitPrice: null,
                kind,
                held,
                booked,
                weg: null,
                net: null,
            });
            return;
        }
        const barHigh = Number(exit.high);
        const barLow = Number(exit.low);
        const exitPrice = held ? Number(high ? barLow : barHigh) : Number(high ? barHigh : barLow);
        const stored = Number(mark.weg);
        const distance = booked && Number.isFinite(stored) ? stored : bookWeg(high, held, entryPrice, barHigh, barLow);
        legs.push({
            side: high ? 'short' : 'long',
            high,
            entryTime,
            entryPrice,
            exitTime: Number(exit.time),
            exitPrice,
            kind,
            held,
            booked,
            weg: distance,
            net: null,
        });
    });
    return legs;
}

function placedPairs(paare, candles) {
    const points = [];
    const list = Array.isArray(paare) ? paare.slice() : [];
    list.sort((a, b) => Number(a.rot_zeit) - Number(b.rot_zeit));
    list.forEach((pair) => {
        if (pair.behalten === false) return;
        if (pair.kette) {
            const entryCandle = nearest(candles, Number(pair.stelle_zeit));
            const turnCandle = nearest(candles, Number(pair.wende_zeit));
            if (!entryCandle || !turnCandle) return;
            const entryHigh = pair.stelle_marke === 'hoch';
            const turnHigh = pair.wende_marke ? pair.wende_marke === 'hoch' : !entryHigh;
            const entryPrice = Number(entryHigh ? entryCandle.high : entryCandle.low);
            const turnPrice = Number(turnHigh ? turnCandle.high : turnCandle.low);
            if (!Number.isFinite(entryPrice) || !Number.isFinite(turnPrice)) return;
            points.push({
                leg: true,
                entryTime: Number(entryCandle.time),
                entryPrice,
                entryHigh,
                turnTime: Number(turnCandle.time),
                turnPrice,
                turnHigh,
            });
            return;
        }
        const redCandle = nearest(candles, Number(pair.rot_zeit));
        const greenCandle = nearest(candles, Number(pair.stelle_zeit));
        if (!redCandle || !greenCandle) return;
        const redPrice = Number(pair.rot_marke === 'hoch' ? redCandle.high : redCandle.low);
        const greenPrice = Number(pair.stelle_marke === 'hoch' ? greenCandle.high : greenCandle.low);
        if (!Number.isFinite(redPrice) || !Number.isFinite(greenPrice)) return;
        points.push({
            redTime: Number(redCandle.time),
            greenTime: Number(greenCandle.time),
            redPrice,
            greenPrice,
            redHigh: pair.rot_marke === 'hoch',
            greenHigh: pair.stelle_marke === 'hoch',
            over: Boolean(pair.ueber),
        });
    });
    return points;
}

function strokeHeld(ctx, timeScale, series, marks) {
    const held = (marks || []).filter((mark) => mark.held);
    held.sort((a, b) => a.time - b.time || a.offset - b.offset);
    ctx.beginPath();
    ctx.lineWidth = 4;
    ctx.strokeStyle = YELLOW;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    let open = false;
    for (const mark of held) {
        const x = timeScale.timeToCoordinate(mark.time);
        const y = series.priceToCoordinate(mark.price);
        if (x == null || y == null) {
            open = false;
            continue;
        }
        const px = x + mark.offset;
        if (!open) {
            ctx.moveTo(px, y);
            open = true;
        } else {
            ctx.lineTo(px, y);
        }
    }
    if (open) ctx.stroke();
}

function marked(ctx, x, y, high, color, text, out) {
    const dir = high ? -1 : 1;
    const away = 28 + (out || 0);
    const circleY = y + dir * away;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, circleY - dir * 9);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x - 5, y + dir * 8);
    ctx.lineTo(x + 5, y + dir * 8);
    ctx.closePath();
    ctx.fill();
    if (text != null) badge(ctx, x, circleY, text, color);
}

function arrow(ctx, x, y, high, color) {
    const tipY = y;
    const tailY = high ? y - 26 : y + 26;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, tailY);
    ctx.lineTo(x, tipY);
    ctx.stroke();
    ctx.beginPath();
    if (high) {
        ctx.moveTo(x, tipY);
        ctx.lineTo(x - 5, tipY - 8);
        ctx.lineTo(x + 5, tipY - 8);
    } else {
        ctx.moveTo(x, tipY);
        ctx.lineTo(x - 5, tipY + 8);
        ctx.lineTo(x + 5, tipY + 8);
    }
    ctx.closePath();
    ctx.fill();
    return tailY;
}

function circleAt(x, y, high, out) {
    const dir = high ? -1 : 1;
    return { x, y: y + dir * (28 + (out || 0)) };
}

function link(ctx, from, to) {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const len = Math.hypot(dx, dy);
    if (len < 8) return;
    const ux = dx / len;
    const uy = dy / len;
    const startPad = Math.min(12, len * 0.22);
    const tipPad = Math.min(14, len * 0.3);
    const startX = from.x + ux * startPad;
    const startY = from.y + uy * startPad;
    const tipX = to.x - ux * tipPad;
    const tipY = to.y - uy * tipPad;
    if (Math.hypot(tipX - startX, tipY - startY) < 6) return;
    ctx.save();
    ctx.setLineDash([8, 5]);
    ctx.lineCap = 'butt';
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.lineTo(tipX, tipY);
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 7;
    ctx.stroke();
    ctx.strokeStyle = GREEN;
    ctx.lineWidth = 3.5;
    ctx.stroke();
    ctx.setLineDash([]);
    const head = 11;
    const px = -uy;
    const py = ux;
    const paintHead = (scale, color) => {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(tipX - ux * head * scale + px * 6 * scale, tipY - uy * head * scale + py * 6 * scale);
        ctx.lineTo(tipX - ux * head * scale - px * 6 * scale, tipY - uy * head * scale - py * 6 * scale);
        ctx.closePath();
        ctx.fill();
    };
    paintHead(1.45, '#000000');
    paintHead(1, GREEN);
    ctx.restore();
}

function badge(ctx, x, y, text, color) {
    ctx.beginPath();
    ctx.fillStyle = '#10233f';
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.arc(x, y, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = NUMBER;
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(text), x, y);
}

function strokeChain(ctx, timeScale, series, legs) {
    const ordered = legs.slice().sort((a, b) => a.entryTime - b.entryTime);
    let number = 1;
    for (const leg of ordered) {
        const ax = timeScale.timeToCoordinate(leg.entryTime);
        const ay = series.priceToCoordinate(leg.entryPrice);
        const bx = timeScale.timeToCoordinate(leg.turnTime);
        const by = series.priceToCoordinate(leg.turnPrice);
        if (ax == null || ay == null || bx == null || by == null) continue;
        const draw = () => {
            ctx.beginPath();
            ctx.setLineDash([8, 5]);
            ctx.lineCap = 'round';
            ctx.moveTo(ax, ay);
            ctx.lineTo(bx, by);
        };
        draw();
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 7;
        ctx.stroke();
        draw();
        ctx.strokeStyle = GREEN;
        ctx.lineWidth = 3.5;
        ctx.stroke();
        ctx.setLineDash([]);
        marked(ctx, ax, ay, leg.entryHigh, GREEN, number, 0);
        number += 1;
        if (leg.turnTime !== leg.entryTime) {
            marked(ctx, bx, by, leg.turnHigh, GREEN, number, 0);
            number += 1;
        }
    }
}

class TrainRenderer {
    constructor() {
        this._points = [];
        this._marks = [];
        this._book = { trades: [], lines: false, list: false };
        this._chart = null;
        this._series = null;
        this._visible = true;
    }

    update(points, marks, book, chart, series, visible) {
        this._points = points || [];
        this._marks = marks || [];
        this._book = book || { trades: [], lines: false, list: false };
        this._chart = chart;
        this._series = series;
        this._visible = visible;
    }

    draw(target) {
        if (!this._visible || !this._chart || !this._series) return;
        const book = this._book || { trades: [], lines: false, list: false };
        const showBook = (book.lines || book.list) && (book.trades || []).length;
        const lage = book.lage || [];
        if (!this._points.length && !(this._marks || []).length && !showBook && !lage.length) return;
        target.useMediaCoordinateSpace((scope) => {
            const ctx = scope.context;
            const timeScale = this._chart.timeScale();
            ctx.save();
            strokeHeld(ctx, timeScale, this._series, this._marks);
            const legs = (this._points || []).filter((point) => point.leg);
            const arrows = (this._points || []).filter((point) => !point.leg);
            if (legs.length) strokeChain(ctx, timeScale, this._series, legs);
            const shown = [];
            const linked = new Set();
            for (const point of arrows) {
                const key = `${point.redTime}:${point.redHigh ? 1 : 0}`;
                linked.add(key);
                shown.push(point);
            }
            const counts = new Map();
            shown.forEach((point, index) => {
                point.n = index + 1;
                point.greenKey = `${point.greenTime}:${point.greenHigh ? 1 : 0}`;
                counts.set(point.greenKey, (counts.get(point.greenKey) || 0) + 1);
            });
            const placed = [];
            for (const point of shown) {
                const x = timeScale.timeToCoordinate(point.redTime);
                const y = this._series.priceToCoordinate(point.redPrice);
                const gx = timeScale.timeToCoordinate(point.greenTime);
                const gy = this._series.priceToCoordinate(point.greenPrice);
                if (x == null || y == null || gx == null || gy == null) continue;
                const from = circleAt(x, y, point.redHigh, 0);
                const to = circleAt(gx, gy, point.greenHigh, 22);
                placed.push({ point, x, y, gx, gy, from, to });
                link(ctx, from, to);
            }
            for (const mark of this._marks || []) {
                const key = `${mark.time}:${mark.high ? 1 : 0}`;
                if (linked.has(key)) continue;
                const x = timeScale.timeToCoordinate(mark.time);
                const y = this._series.priceToCoordinate(mark.price);
                if (x == null || y == null) continue;
                marked(ctx, x, y, mark.high, mark.held ? YELLOW : FAIL, null);
            }
            const seenGreen = new Set();
            for (const row of placed) {
                if (seenGreen.has(row.point.greenKey)) continue;
                seenGreen.add(row.point.greenKey);
                const text = counts.get(row.point.greenKey) === 1 ? row.point.n : null;
                marked(ctx, row.gx, row.gy, row.point.greenHigh, GREEN, text, 22);
            }
            for (const row of placed) {
                marked(ctx, row.x, row.y, row.point.redHigh, YELLOW, row.point.n);
            }
            for (const point of lage) {
                const x = timeScale.timeToCoordinate(point.time);
                const price = point.up ? point.high : point.low;
                const y = this._series.priceToCoordinate(price);
                if (x == null || y == null) continue;
                paintLageCount(ctx, x, y, point.count, point.up);
            }
            if (book.lines || book.list) {
                paintIdealBook(ctx, {
                    timeScale,
                    series: this._series,
                    trades: book.trades,
                    viewWidth: scope.mediaSize.width,
                    viewHeight: scope.mediaSize.height,
                    positions: book.lines,
                    table: book.list,
                    title: 'Buch',
                    anchor: 'right',
                });
            }
            ctx.restore();
        });
    }
}

class TrainPaneView {
    constructor(source) {
        this._source = source;
        this._renderer = new TrainRenderer();
    }

    zOrder() {
        return 'top';
    }

    update() {
        this._renderer.update(
            this._source._points,
            this._source._marks,
            this._source._book,
            this._source._chart,
            this._source._series,
            this._source._visible,
        );
    }

    renderer() {
        return this._renderer;
    }
}

class TrainPrimitive {
    constructor() {
        this._points = [];
        this._marks = [];
        this._book = { trades: [], lines: false, list: false };
        this._chart = null;
        this._series = null;
        this._visible = true;
        this._paneView = new TrainPaneView(this);
    }

    attached(params) {
        this._series = params.series;
        this._chart = params.chart;
    }

    detached() {
        this._series = null;
        this._chart = null;
    }

    setPoints(points, marks, book) {
        this._points = points || [];
        this._marks = marks || [];
        this._book = book || { trades: [], lines: false, list: false };
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

export function paintLumaTrain({
    chart,
    series,
    candles,
    enabled,
    showMarker = true,
    showExploration = true,
    showLage = true,
    showImagination = true,
    showStrecken = true,
    showListe = true,
}) {
    if (!series || !chart) return;
    if (drawState.chart !== chart) {
        dropSeries(chart, drawState.lage);
        dropSeries(chart, drawState.imagination);
        dropSeries(chart, drawState.imaginationZero);
        drawState.lage = null;
        drawState.imagination = null;
        drawState.imaginationZero = null;
        drawState.chart = chart;
    }
    if (drawState.series !== series) {
        detach();
        drawState.series = series;
        drawState.primitive = new TrainPrimitive();
        series.attachPrimitive(drawState.primitive);
    }
    const drawn = candles || [];
    const hideLines = () => {
        holdPane(chart, 'lage', LAGE, [], false);
        holdPane(chart, 'imagination', IMAGINATION, [], true);
    };
    if (!enabled || !payload) {
        drawState.primitive.setVisible(false);
        hideLines();
        return;
    }
    if (payload.symbol && payload.chartSymbol && payload.symbol !== payload.chartSymbol) {
        drawState.primitive.setVisible(false);
        hideLines();
        return;
    }
    const points = showExploration ? placedPairs(payload.paare, drawn) : [];
    const marks = showMarker ? placedMarks(payload.gesetzt, drawn) : [];
    const dressed = (showStrecken || showListe) ? dressTrades(drawn, bookLegs(payload.gesetzt, drawn)) : { trades: [] };
    const lage = showLage ? lageSequence(payload.lage, drawn) : [];
    drawState.primitive.setVisible(points.length > 0 || marks.length > 0 || dressed.trades.length > 0 || lage.length > 0);
    drawState.primitive.setPoints(points, marks, {
        trades: dressed.trades,
        lines: showStrecken,
        list: showListe,
        lage,
    });
    holdPane(chart, 'lage', LAGE, [], false);
    holdPane(chart, 'imagination', IMAGINATION, showImagination ? scored(payload.vorstellung, drawn) : [], true);
}
