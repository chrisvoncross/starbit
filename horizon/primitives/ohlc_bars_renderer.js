/**
 * Custom OHLC Bars Renderer — Bloomberg-style monochrome OHLC bars.
 *
 * Replaces LC's built-in BarSeries with a custom primitive that:
 *   - Draws OHLC bars (vertical high-low line + open/close ticks)
 *   - Caps line width so bars don't become infinitely thick on zoom
 *   - Scales open/close tick width with barSpacing
 *
 * Attached as a primitive to whichever series holds the Raw OHLC data.
 *
 * Usage:
 *   const ohlcBars = new OhlcBarsPlugin({ color: '#CCCCCC' });
 *   _rawCandleSeries.attachPrimitive(ohlcBars);
 */

// ═══════════════════════════════════════════════════════════════════════════════
// STYLE CONFIG — Central styling for Bloomberg OHLC bars
// ═══════════════════════════════════════════════════════════════════════════════

const BAR_STYLE = {
    color: '#CCCCCC',
    lineWidth: 1.5,           // Vertical HL line + open/close ticks
    maxLineWidth: 3,          // Cap — never thicker than this
    tickWidthFraction: 0.4,   // Open/close tick = 40% of barSpacing per side
    minTickHalf: 2,           // Min tick half-width in px
    maxTickHalf: 8,           // Max tick half-width in px — the cap
};

// ═══════════════════════════════════════════════════════════════════════════════
// RENDERER
// ═══════════════════════════════════════════════════════════════════════════════

class OhlcBarsRenderer {
    constructor(viewData) {
        this._viewData = viewData;
    }

    draw(target) {
        target.useBitmapCoordinateSpace(scope => {
            const ctx = scope.context;
            const hr = scope.horizontalPixelRatio;
            const vr = scope.verticalPixelRatio;
            ctx.scale(hr, vr);

            const { bars, style } = this._viewData;
            if (!bars || bars.length === 0) return;

            const s = style || BAR_STYLE;

            ctx.strokeStyle = s.color;
            ctx.lineWidth = s.lineWidth;

            for (const bar of bars) {
                const { x, yOpen, yClose, yHigh, yLow, tickHalf } = bar;

                // Vertical high-low line
                ctx.beginPath();
                ctx.moveTo(x, yHigh);
                ctx.lineTo(x, yLow);
                ctx.stroke();

                // Open tick (left side)
                ctx.beginPath();
                ctx.moveTo(x - tickHalf, yOpen);
                ctx.lineTo(x, yOpen);
                ctx.stroke();

                // Close tick (right side)
                ctx.beginPath();
                ctx.moveTo(x, yClose);
                ctx.lineTo(x + tickHalf, yClose);
                ctx.stroke();
            }
        });
    }

    drawBackground() {}
}

// ═══════════════════════════════════════════════════════════════════════════════
// PANE VIEW
// ═══════════════════════════════════════════════════════════════════════════════

class OhlcBarsPaneView {
    constructor(source) {
        this._source = source;
        this._data = { bars: [], style: BAR_STYLE };
    }

    update() {
        const series = this._source._series;
        const timeScale = this._source._chart.timeScale();
        if (!series || !timeScale || !this._source._visible) {
            this._data = { bars: [], style: BAR_STYLE };
            return;
        }

        // Get barSpacing for tick width scaling
        let barSpacing = 10;
        try {
            const tsOpts = timeScale.options();
            if (tsOpts && tsOpts.barSpacing > 0) barSpacing = tsOpts.barSpacing;
        } catch (_) {}

        // Compute capped tick half-width
        const s = this._source._style;
        const rawHalf = barSpacing * s.tickWidthFraction;
        const tickHalf = Math.max(s.minTickHalf, Math.min(s.maxTickHalf, rawHalf));

        // Scale lineWidth with barSpacing but cap it
        const scaledLW = Math.min(s.maxLineWidth, s.lineWidth + barSpacing * 0.02);

        const allData = series.data();
        const bars = [];

        for (const candle of allData) {
            const x = timeScale.timeToCoordinate(candle.time);
            if (x === null) continue;

            const yOpen = series.priceToCoordinate(candle.open);
            const yClose = series.priceToCoordinate(candle.close);
            const yHigh = series.priceToCoordinate(candle.high);
            const yLow = series.priceToCoordinate(candle.low);

            if (yOpen === null || yClose === null || yHigh === null || yLow === null) continue;

            bars.push({ x, yOpen, yClose, yHigh, yLow, tickHalf });
        }

        this._data = {
            bars,
            style: { ...s, lineWidth: scaledLW },
        };
    }

    renderer() {
        return new OhlcBarsRenderer(this._data);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PLUGIN
// ═══════════════════════════════════════════════════════════════════════════════

export class OhlcBarsPlugin {
    constructor(options = {}) {
        this._style = { ...BAR_STYLE, ...options };
        this._visible = false;  // Controlled by switchChartStyle
        this._chart = null;
        this._series = null;
        this._paneViews = [];
    }

    attached({ chart, series }) {
        this._chart = chart;
        this._series = series;
        this._paneViews = [new OhlcBarsPaneView(this)];
    }

    detached() {
        this._chart = null;
        this._series = null;
        this._paneViews = [];
    }

    paneViews() {
        return this._paneViews;
    }

    updateAllViews() {
        for (const pv of this._paneViews) pv.update();
    }

    // No price axis labels or crosshair primitives needed
    priceAxisViews() { return []; }
    priceAxisPaneViews() { return []; }
    autoscaleInfo() { return null; }
    timeAxisViews() { return []; }
}
