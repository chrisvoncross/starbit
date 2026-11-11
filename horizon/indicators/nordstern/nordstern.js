// Nordstern draws the hindsight path that makes the most money on the candles
// already on the chart. One position at a time. Entry and exit sit on the
// wick. The fee is 0.05% of the position on the way in and 0.05% on the way
// out, the round trip of 0.10%. The stake and the leverage only scale that result into dollars.
// The 15-second candles have no bid, no ask and no fill, so spread and
// slippage stay out of the number.

const TAKER = 0.0005;
/** Median BTC L1 spread across the 1000 full hours on R2, as a fraction of price. 0.0131 bp, one tick. */
export const MEASURED_SPREAD = 0.0000013112235;
const BANK = 2000;
const RISK = 0.01;
const MAX_LEV = 150;
const TICK = 0.1;
const LONG = '#26a69a';
const SHORT = '#ef5350';

function quote(entry, exit, notional, side, feeRate = TAKER, minMove = 0) {
    if (!(entry > 0) || !(exit > 0)) return null;
    const up = side === 'long';
    if (up ? exit <= entry : exit >= entry) return null;
    const move = up ? (exit - entry) / entry : (entry - exit) / entry;
    if (!(move > minMove)) return null;
    const gross = notional * move;
    const fee = notional * feeRate * (1 + exit / entry);
    const net = gross - fee;
    if (!(net > 0)) return null;
    return { move, gross, fee, net };
}

function bestTrades(candles, feeRate = TAKER, minMove = 0) {
    const n = candles.length;
    const notional = BANK;
    const none = -1e100;
    const flat = new Array(n + 1).fill(0);
    const atLow = new Array(n).fill(none);
    const atHigh = new Array(n).fill(none);
    const parentFlat = new Array(n + 1).fill(null);
    const parentHigh = new Array(n).fill(null);
    const parentLow = new Array(n).fill(null);
    const keep = (cash, side, entry, exit, how, quoteRow) => {
        if (side === 'long') {
            if (cash > atHigh[exit]) {
                atHigh[exit] = cash;
                parentHigh[exit] = { entry, how, quote: quoteRow };
            }
        } else if (cash > atLow[exit]) {
            atLow[exit] = cash;
            parentLow[exit] = { entry, how, quote: quoteRow };
        }
        if (cash > flat[exit + 1]) {
            flat[exit + 1] = cash;
            parentFlat[exit + 1] = { kind: side === 'long' ? 'high' : 'low', index: exit };
        }
    };
    for (let exit = 1; exit < n; exit++) {
        if (flat[exit - 1] > flat[exit]) {
            flat[exit] = flat[exit - 1];
            parentFlat[exit] = { kind: 'skip', index: exit - 1 };
        }
        const outHigh = Number(candles[exit].high);
        const outLow = Number(candles[exit].low);
        for (let entry = 0; entry < exit; entry++) {
            const long = quote(Number(candles[entry].low), outHigh, notional, 'long', feeRate, minMove);
            if (long) {
                keep(flat[entry] + long.net, 'long', entry, exit, 'flat', long);
                if (atLow[entry] > none / 2) {
                    keep(atLow[entry] + long.net, 'long', entry, exit, 'flip', long);
                }
            }
            const short = quote(Number(candles[entry].high), outLow, notional, 'short', feeRate, minMove);
            if (short) {
                keep(flat[entry] + short.net, 'short', entry, exit, 'flat', short);
                if (atHigh[entry] > none / 2) {
                    keep(atHigh[entry] + short.net, 'short', entry, exit, 'flip', short);
                }
            }
        }
        if (flat[exit] > flat[exit + 1]) {
            flat[exit + 1] = flat[exit];
            parentFlat[exit + 1] = { kind: 'skip', index: exit };
        }
    }
    const trades = [];
    let node = { kind: 'flat', index: n };
    let guard = n + 5;
    while (node && guard > 0) {
        guard -= 1;
        if (node.kind === 'flat') {
            const parent = parentFlat[node.index];
            if (!parent) break;
            node = { kind: parent.kind === 'skip' ? 'flat' : parent.kind, index: parent.index };
            continue;
        }
        const parent = node.kind === 'high' ? parentHigh[node.index] : parentLow[node.index];
        if (!parent) break;
        const side = node.kind === 'high' ? 'long' : 'short';
        trades.push(row(candles, parent.entry, node.index, side, parent.quote));
        const flipped = parent.how === 'flip';
        if (side === 'long') node = { kind: flipped ? 'low' : 'flat', index: parent.entry };
        else node = { kind: flipped ? 'high' : 'flat', index: parent.entry };
    }
    trades.reverse();
    return { trades, total: flat[n], notional };
}

function row(candles, entry, exit, side, quoteRow) {
    const up = side === 'long';
    return {
        side,
        entryTime: Number(candles[entry].time),
        exitTime: Number(candles[exit].time),
        entryPrice: up ? Number(candles[entry].low) : Number(candles[entry].high),
        exitPrice: up ? Number(candles[exit].high) : Number(candles[exit].low),
        move: quoteRow.move,
        gross: quoteRow.gross,
        fee: quoteRow.fee,
        net: quoteRow.net,
        entryIndex: entry,
        exitIndex: exit,
    };
}

function medianCandle(candles) {
    const spans = [];
    for (const candle of candles) {
        const span = Number(candle.high) - Number(candle.low);
        if (span > 0) spans.push(span);
    }
    if (!spans.length) return TICK;
    spans.sort((a, b) => a - b);
    const mid = Math.floor(spans.length / 2);
    return spans.length % 2 ? spans[mid] : (spans[mid - 1] + spans[mid]) / 2;
}

function placeStops(candles, trades, feeRate = TAKER) {
    const floor = Math.max(medianCandle(candles), TICK);
    let total = 0;
    for (const trade of trades) {
        const up = trade.side === 'long';
        let worst = trade.entryPrice;
        const from = trade.entryIndex;
        const to = trade.exitIndex;
        if (from != null && to != null) {
            for (let i = from; i <= to; i++) {
                const px = up ? Number(candles[i].low) : Number(candles[i].high);
                if (up ? px < worst : px > worst) worst = px;
            }
        }
        const structural = Math.abs(trade.entryPrice - (up ? worst - TICK : worst + TICK));
        const stopDist = Math.max(structural, floor);
        const stop = up ? trade.entryPrice - stopDist : trade.entryPrice + stopDist;
        trade.floor = floor;
        const rewardDist = Math.abs(trade.exitPrice - trade.entryPrice);
        const stopFrac = trade.entryPrice > 0 ? stopDist / trade.entryPrice : 0;
        const rr = stopDist > 0 ? rewardDist / stopDist : 0;
        const needLev = stopFrac > 0 ? RISK / stopFrac : MAX_LEV;
        const lev = Math.min(needLev, MAX_LEV);
        const sized = BANK * lev;
        const riskDollars = sized * stopFrac;
        const gross = sized * trade.move;
        const fee = sized * feeRate * (1 + trade.exitPrice / trade.entryPrice);
        const net = gross - fee;
        trade.stop = stop;
        trade.rr = rr;
        trade.lev = lev;
        trade.riskDollars = riskDollars;
        trade.gross = gross;
        trade.fee = fee;
        trade.net = net;
        total += net;
    }
    return total;
}

function rrText(rr) {
    if (!(rr > 0)) return '0';
    if (rr >= 10) return String(Math.round(rr));
    return rr.toFixed(1).replace('.', ',');
}

function money(value) {
    const sign = value < 0 ? '-' : '';
    return `${sign}${Math.abs(value).toFixed(2).replace('.', ',')} $`;
}

function percent(value) {
    return `${(value * 100).toFixed(3).replace('.', ',')} %`;
}

function clock(time) {
    const ms = time > 10000000000 ? time : time * 1000;
    const date = new Date(ms);
    const part = (n) => String(n).padStart(2, '0');
    return `${part(date.getUTCHours())}:${part(date.getUTCMinutes())}:${part(date.getUTCSeconds())}`;
}

function paintStar(ctx, x, y, fill = '#ffe08a', stroke = '#8a6410') {
    const outer = 7;
    const inner = 2.8;
    ctx.beginPath();
    for (let i = 0; i < 10; i++) {
        const angle = -Math.PI / 2 + (i * Math.PI) / 5;
        const radius = i % 2 === 0 ? outer : inner;
        const px = x + Math.cos(angle) * radius;
        const py = y + Math.sin(angle) * radius;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.fill();
    ctx.stroke();
}

function paintGoldCandle(ctx, series, candle, x, tickHalf, color = '#e2b340') {
    const yHigh = series.priceToCoordinate(Number(candle.high));
    const yLow = series.priceToCoordinate(Number(candle.low));
    const yOpen = series.priceToCoordinate(Number(candle.open));
    const yClose = series.priceToCoordinate(Number(candle.close));
    if (yHigh == null || yLow == null || yOpen == null || yClose == null) return null;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, yHigh);
    ctx.lineTo(x, yLow);
    ctx.stroke();
    const top = Math.min(yOpen, yClose);
    const height = Math.max(Math.abs(yClose - yOpen), 2);
    ctx.fillStyle = color;
    ctx.fillRect(x - tickHalf, top, tickHalf * 2, height);
    ctx.strokeRect(x - tickHalf, top, tickHalf * 2, height);
    return { yHigh, yLow };
}

function fillOf(color) {
    if (color === LONG) return 'rgba(38, 166, 154, 0.18)';
    if (color === SHORT) return 'rgba(239, 83, 80, 0.18)';
    const hex = String(color || '').replace('#', '');
    if (hex.length !== 6) return 'rgba(255, 255, 255, 0.18)';
    const n = parseInt(hex, 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, 0.18)`;
}

function paintBox(ctx, ax, ay, bx, by, color) {
    const left = Math.min(ax, bx);
    const top = Math.min(ay, by);
    const width = Math.abs(bx - ax);
    const height = Math.abs(by - ay);
    ctx.fillStyle = fillOf(color);
    ctx.fillRect(left, top, Math.max(width, 1), Math.max(height, 1));
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
}

function paintTag(ctx, x, y, text, color) {
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#111111';
    ctx.strokeText(text, x + 6, y);
    ctx.fillStyle = color;
    ctx.fillText(text, x + 6, y);
}

function paintPanel(ctx, lines, viewWidth, viewHeight, anchor) {
    let lineH = 16;
    let fontSize = 12;
    const pad = 14;
    const fit = () => {
        ctx.font = `${fontSize}px sans-serif`;
        let boxWidth = 280;
        for (const line of lines) boxWidth = Math.max(boxWidth, ctx.measureText(line).width + 24);
        return { boxWidth, height: lines.length * lineH + pad };
    };
    let box = fit();
    if (viewHeight && box.height > viewHeight - 16) {
        lineH = Math.max(11, Math.floor((viewHeight - 24) / lines.length));
        fontSize = Math.max(10, lineH - 4);
        box = fit();
    }
    const right = Math.max(8, (viewWidth || box.boxWidth + 16) - box.boxWidth - 8);
    const x = anchor === 'left' ? 8 : right;
    ctx.fillStyle = 'rgba(12, 14, 18, 0.88)';
    ctx.fillRect(x, 8, box.boxWidth, box.height);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    lines.forEach((line, index) => {
        ctx.fillStyle = index === 4 ? '#3ddc6a' : '#e8e8e8';
        ctx.fillText(line, x + 12, 16 + index * lineH);
    });
}

export function dressTrades(candles, legs) {
    const byTime = new Map();
    (candles || []).forEach((candle, index) => byTime.set(Number(candle.time), index));
    const trades = [];
    for (const leg of legs || []) {
        if (leg.exitTime == null || leg.exitPrice == null || !(leg.entryPrice > 0)) continue;
        const entryIndex = byTime.get(Number(leg.entryTime));
        const exitIndex = byTime.get(Number(leg.exitTime));
        if (entryIndex == null || exitIndex == null || exitIndex < entryIndex) continue;
        const up = leg.side === 'long';
        const move = up
            ? (leg.exitPrice - leg.entryPrice) / leg.entryPrice
            : (leg.entryPrice - leg.exitPrice) / leg.entryPrice;
        if (!Number.isFinite(move)) continue;
        trades.push({
            side: leg.side,
            entryTime: Number(leg.entryTime),
            exitTime: Number(leg.exitTime),
            entryPrice: leg.entryPrice,
            exitPrice: leg.exitPrice,
            entryIndex,
            exitIndex,
            move,
        });
    }
    const total = placeStops(candles, trades);
    return { trades, total };
}

export function paintIdealBook(ctx, options) {
    const timeScale = options.timeScale;
    const series = options.series;
    const trades = options.trades || [];
    if (options.positions) {
        for (const trade of trades) {
            const ax = timeScale.timeToCoordinate(trade.entryTime);
            const ay = series.priceToCoordinate(trade.entryPrice);
            const bx = timeScale.timeToCoordinate(trade.exitTime);
            const by = series.priceToCoordinate(trade.exitPrice);
            if (ax == null || ay == null || bx == null || by == null) continue;
            const longColor = options.longColor || LONG;
            const shortColor = options.shortColor || SHORT;
            const sideColor = trade.side === 'long' ? longColor : shortColor;
            const exitColor = options.riskReward ? '#26a69a' : sideColor;
            const stopColor = options.riskReward ? '#ef5350' : '#ff8a65';
            const sy = series.priceToCoordinate(trade.stop);
            if (sy != null) {
                const left = Math.min(ax, bx);
                const top = Math.min(ay, sy);
                ctx.fillStyle = options.riskReward ? 'rgba(239, 83, 80, 0.28)' : 'rgba(255, 138, 101, 0.28)';
                ctx.fillRect(left, top, Math.max(Math.abs(bx - ax), 1), Math.max(Math.abs(sy - ay), 1));
                ctx.beginPath();
                ctx.setLineDash([3, 4]);
                ctx.moveTo(ax, sy);
                ctx.lineTo(bx, sy);
                ctx.strokeStyle = stopColor;
                ctx.lineWidth = 1.5;
                ctx.stroke();
                ctx.setLineDash([]);
                paintTag(ctx, bx, sy, `SL RR ${rrText(trade.rr)}`, stopColor);
            }
            paintBox(ctx, ax, ay, bx, by, exitColor);
            const name = trade.side === 'long' ? 'Long' : 'Short';
            paintTag(ctx, (ax + bx) / 2, (ay + by) / 2, `${name} ${money(trade.net)}`, exitColor);
        }
    }
    if (!options.table) return;
    const floor = trades[0] ? trades[0].floor : 0;
    const total = trades.reduce((sum, trade) => sum + (Number(trade.net) || 0), 0);
    const lines = [
        options.title || 'Nordstern',
        `Konto ${money(BANK)}   Risiko-Ziel ${money(BANK * RISK)}   Hebel höchstens ${MAX_LEV}`,
        `Stop mindestens eine normale Kerze (${money(floor)})`,
        options.feeLine || '0,05 % rein + 0,05 % raus = 0,10 % gesamt',
        `${trades.length} Fahrten   Summe ${money(total)}`,
    ];
    trades.forEach((trade, index) => {
        const name = trade.side === 'long' ? 'Long ' : 'Short';
        lines.push(
            `${index + 1}  ${name}  ${clock(trade.entryTime)}–${clock(trade.exitTime)}`
            + `  RR ${rrText(trade.rr)}  Hebel ${Math.round(trade.lev)}`
            + `  Risiko ${money(trade.riskDollars)}  ${money(trade.net)}`,
        );
    });
    paintPanel(ctx, lines, options.viewWidth, options.viewHeight, options.anchor);
}

class NordsternRenderer {
    constructor() {
        this._trades = [];
        this._candles = [];
        this._total = 0;
        this._notional = BANK;
        this._chart = null;
        this._series = null;
        this._visible = true;
        this._stars = true;
        this._positions = true;
        this._table = true;
        this._title = 'Nordstern';
        this._feeLine = '0,05 % rein + 0,05 % raus = 0,10 % gesamt';
        this._anchor = 'right';
        this._longColor = LONG;
        this._shortColor = SHORT;
        this._markColor = '#e2b340';
        this._starFill = '#ffe08a';
        this._starStroke = '#8a6410';
    }

    update(trades, candles, total, notional, chart, series, visible, stars, positions, table, style) {
        this._trades = trades || [];
        this._candles = candles || [];
        this._total = total;
        this._notional = notional;
        this._chart = chart;
        this._series = series;
        this._visible = visible;
        this._stars = stars;
        this._positions = positions;
        this._table = table;
        const look = style || {};
        this._title = look.title || 'Nordstern';
        this._feeLine = look.feeLine || '0,05 % rein + 0,05 % raus = 0,10 % gesamt';
        this._anchor = look.anchor || 'right';
        this._longColor = look.longColor || LONG;
        this._shortColor = look.shortColor || SHORT;
        this._markColor = look.markColor || '#e2b340';
        this._starFill = look.starFill || '#ffe08a';
        this._starStroke = look.starStroke || '#8a6410';
        this._riskReward = Boolean(look.riskReward);
        this._markCandle = look.markCandle !== false;
    }

    draw(target) {
        if (!this._visible || !this._chart || !this._series) return;
        target.useMediaCoordinateSpace((scope) => {
            const ctx = scope.context;
            ctx.save();
            paintIdealBook(ctx, {
                timeScale: this._chart.timeScale(),
                series: this._series,
                trades: this._trades,
                viewWidth: scope.mediaSize.width,
                viewHeight: scope.mediaSize.height,
                positions: this._positions,
                table: this._table,
                title: this._title,
                feeLine: this._feeLine,
                anchor: this._anchor,
                longColor: this._longColor,
                shortColor: this._shortColor,
                riskReward: this._riskReward,
            });
            if (this._stars) this._paintMarks(ctx, this._chart.timeScale());
            ctx.restore();
        });
    }

    _paintMarks(ctx, timeScale) {
        let barSpacing = 8;
        try {
            const opts = timeScale.options();
            if (opts && opts.barSpacing > 0) barSpacing = opts.barSpacing;
        } catch (e) { /* default */ }
        const tickHalf = Math.max(2, Math.min(7, barSpacing * 0.32));
        const seen = new Set();
        for (const trade of this._trades) {
            const ends = [
                { index: trade.entryIndex, highSide: trade.side === 'short' },
                { index: trade.exitIndex, highSide: trade.side === 'long' },
            ];
            for (const end of ends) {
                const candle = this._candles[end.index];
                if (!candle) continue;
                const key = `${candle.time}:${end.highSide ? 1 : 0}`;
                if (seen.has(key)) continue;
                seen.add(key);
                const x = timeScale.timeToCoordinate(Number(candle.time));
                if (x == null) continue;
                let yHigh;
                let yLow;
                if (this._markCandle) {
                    const span = paintGoldCandle(ctx, this._series, candle, x, tickHalf, this._markColor);
                    if (!span) continue;
                    yHigh = span.yHigh;
                    yLow = span.yLow;
                } else {
                    yHigh = this._series.priceToCoordinate(Number(candle.high));
                    yLow = this._series.priceToCoordinate(Number(candle.low));
                    if (yHigh == null || yLow == null) continue;
                }
                const y = end.highSide ? yHigh - 12 : yLow + 12;
                paintStar(ctx, x, y, this._starFill, this._starStroke);
            }
        }
    }
}

class NordsternPaneView {
    constructor(source) {
        this._source = source;
        this._renderer = new NordsternRenderer();
    }

    zOrder() {
        return 'top';
    }

    update() {
        this._renderer.update(
            this._source._trades,
            this._source._candles,
            this._source._total,
            this._source._notional,
            this._source._chart,
            this._source._series,
            this._source._visible,
            this._source._stars,
            this._source._positions,
            this._source._table,
            this._source._style,
        );
    }

    renderer() {
        return this._renderer;
    }
}

class NordsternPrimitive {
    constructor() {
        this._trades = [];
        this._candles = [];
        this._total = 0;
        this._notional = BANK;
        this._chart = null;
        this._series = null;
        this._visible = true;
        this._stars = true;
        this._positions = true;
        this._table = true;
        this._style = null;
        this._paneView = new NordsternPaneView(this);
    }

    attached(params) {
        this._series = params.series;
        this._chart = params.chart;
    }

    detached() {
        this._series = null;
        this._chart = null;
    }

    setTrades(trades, total, notional, candles, layers) {
        this._trades = trades || [];
        this._candles = candles || [];
        this._stars = !layers || layers.stars !== false;
        this._positions = !layers || layers.positions !== false;
        this._table = !layers || layers.table !== false;
        this._style = layers || null;
        this._total = total || 0;
        this._notional = notional || BANK;
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

const overlays = new Map();

function overlayState(id) {
    let state = overlays.get(id);
    if (!state) {
        state = { series: null, primitive: null };
        overlays.set(id, state);
    }
    return state;
}

export function paintCashOverlay(id, {
    chart,
    series,
    candles,
    enabled,
    showStars = true,
    showPositions = true,
    showTable = true,
    feeRate = TAKER,
    minMove = 0,
    title = 'Nordstern',
    feeLine = '0,05 % rein + 0,05 % raus = 0,10 % gesamt',
    anchor = 'right',
    longColor = LONG,
    shortColor = SHORT,
    markColor = '#e2b340',
    starFill = '#ffe08a',
    starStroke = '#8a6410',
    riskReward = false,
    markCandle = true,
}) {
    const state = overlayState(id);
    if (!series || !chart) return;
    if (state.series !== series) {
        if (state.series && state.primitive) {
            try { state.series.detachPrimitive(state.primitive); } catch (e) { /* already gone */ }
        }
        state.series = series;
        state.primitive = new NordsternPrimitive();
        series.attachPrimitive(state.primitive);
    }
    if (!enabled || !candles || candles.length < 2) {
        state.primitive.setVisible(false);
        return;
    }
    const found = bestTrades(candles, feeRate, minMove);
    const total = placeStops(candles, found.trades, feeRate);
    state.primitive.setVisible(true);
    state.primitive.setTrades(found.trades, total, BANK, candles, {
        stars: showStars,
        positions: showPositions,
        table: showTable,
        title,
        feeLine,
        anchor,
        longColor,
        shortColor,
        markColor,
        starFill,
        starStroke,
        riskReward,
        markCandle,
    });
}

export function mountNordstern({ chart, series, candles, showStars = true, showPositions = true, showTable = false }) {
    const primitive = new NordsternPrimitive();
    series.attachPrimitive(primitive);
    const found = bestTrades(candles || []);
    const total = placeStops(candles || [], found.trades);
    primitive.setTrades(found.trades, total, BANK, candles || [], {
        stars: showStars,
        positions: showPositions,
        table: showTable,
    });
    return { trades: found.trades, total };
}

export function paintNordstern(args) {
    paintCashOverlay('nordstern', args);
}
