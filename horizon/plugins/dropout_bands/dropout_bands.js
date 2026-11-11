// dropout_bands.js — data-gap ("dropout") overlay for the transformer view.
//
// A Lightweight-Charts v5 SERIES PRIMITIVE (ISeriesPrimitive) that shades the
// span between two consecutive bars wherever the pinned dataset has a
// single-/multi-bucket data gap (Δt > cadence). This makes VISIBLE the exact
// trap the task calls out: because Lightweight Charts spaces bars by INDEX and
// not by wall-clock time, a 30s hole in a 15s series occupies one index step
// and looks identical to a normal 15s move — until it is banded here.
//
// The provider (transformer_view_provider._detect_gaps) computes each gap as
// the (from_time, to_time) bar pair it sits between plus the count of missing
// buckets; the renderer bands the midpoint region and labels the bucket count.
//
// Pattern: TradingView's session-highlighting / vertical-line plugin examples
// (background zones via IPrimitivePaneRenderer.drawBackground); see the Series
// Primitives docs — https://tradingview.github.io/lightweight-charts/docs/plugins/series-primitives

const BAND_FILL = 'rgba(239, 83, 80, 0.16)';   // translucent red (data missing)
const BAND_EDGE = 'rgba(239, 83, 80, 0.85)';
const LABEL_BG = 'rgba(239, 83, 80, 0.92)';
const LABEL_FG = '#ffffff';

class DropoutBandsRenderer {
    constructor() {
        this._gaps = [];
        this._chart = null;
        this._series = null;
    }

    update(gaps, chart, series) {
        this._gaps = gaps || [];
        this._chart = chart;
        this._series = series;
    }

    draw() {}

    // Bands sit BEHIND the candles (background), like session highlighting.
    drawBackground(target) {
        if (!this._gaps.length || !this._chart) return;
        target.useMediaCoordinateSpace((scope) => {
            const ctx = scope.context;
            const H = scope.mediaSize.height;
            const ts = this._chart.timeScale();
            for (const g of this._gaps) {
                const xa = ts.timeToCoordinate(g.from_time);
                const xb = ts.timeToCoordinate(g.to_time);
                if (xa === null || xb === null) continue;
                const left = Math.min(xa, xb);
                const right = Math.max(xa, xb);
                // Guarantee a visible band even when the two bars are adjacent
                // (index-spaced): pad to a minimum width around the midpoint.
                let x0 = left;
                let x1 = right;
                if (x1 - x0 < 6) {
                    const mid = (x0 + x1) / 2;
                    x0 = mid - 4;
                    x1 = mid + 4;
                }
                ctx.fillStyle = BAND_FILL;
                ctx.fillRect(x0, 0, x1 - x0, H);
                ctx.strokeStyle = BAND_EDGE;
                ctx.setLineDash([4, 3]);
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(x0, 0); ctx.lineTo(x0, H);
                ctx.moveTo(x1, 0); ctx.lineTo(x1, H);
                ctx.stroke();
                ctx.setLineDash([]);

                // Missing-bucket count label near the top.
                const miss = g.missing_buckets || 0;
                const secs = Math.round((g.delta_ms || 0) / 1000);
                const text = miss > 0 ? `⌀ ${miss} missing · Δ${secs}s` : `⌀ Δ${secs}s`;
                ctx.font = '10px ui-monospace, Menlo, monospace';
                const tw = ctx.measureText(text).width;
                const cx = (x0 + x1) / 2;
                const tx = Math.max(2, cx - tw / 2 - 4);
                ctx.fillStyle = LABEL_BG;
                ctx.fillRect(tx, 4, tw + 8, 15);
                ctx.fillStyle = LABEL_FG;
                ctx.fillText(text, tx + 4, 15);
            }
        });
    }
}

class DropoutBandsPaneView {
    constructor(source) {
        this._source = source;
        this._renderer = new DropoutBandsRenderer();
    }
    zOrder() { return 'bottom'; }
    update() {
        this._renderer.update(this._source._gaps, this._source._chart, this._source._series);
    }
    renderer() { return this._renderer; }
}

class DropoutBandsPrimitive {
    constructor(gaps = []) {
        this._gaps = gaps;
        this._chart = null;
        this._series = null;
        this._paneView = null;
    }
    attached(params) {
        this._series = params.series;
        this._chart = params.chart;
        this._paneView = new DropoutBandsPaneView(this);
    }
    detached() {
        this._series = null;
        this._chart = null;
        this._paneView = null;
    }
    setGaps(gaps) {
        this._gaps = gaps || [];
        this.updateAllViews();
    }
    paneViews() { return this._paneView ? [this._paneView] : []; }
    updateAllViews() { if (this._paneView) this._paneView.update(); }
    autoscaleInfo() { return null; }
}

/**
 * Attach a dropout-bands overlay to a series. Returns a handle whose setGaps()
 * accepts the provider's window `gaps` array.
 * @param {ISeriesApi} series
 * @returns {{ setGaps(gaps:Array):void, remove():void, primitive: DropoutBandsPrimitive }}
 */
export function attachDropoutBands(series) {
    const primitive = new DropoutBandsPrimitive([]);
    series.attachPrimitive(primitive);
    return {
        primitive,
        setGaps: (gaps) => primitive.setGaps(gaps),
        remove: () => {
            try { series.detachPrimitive(primitive); } catch (e) { /* detached */ }
        },
    };
}
