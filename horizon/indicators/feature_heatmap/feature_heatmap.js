// feature_heatmap.js — the 203×T "what the transformer sees" heatmap.
//
// Rendered as a Lightweight Charts v5 CUSTOM SERIES (ICustomSeriesPaneView):
// each data point carries the full window-relative normalized feature vector
// (values[FEATURE_DIM]); the renderer draws that column as a vertical stack of
// FEATURE_DIM colored cells at the point's x. The price scale spans the feature
// INDEX range [0, FEATURE_DIM] so feature j maps to y = priceToCoordinate(j).
//
// Integrity: values are exactly what providers/transformer_view_provider.py
// pulled from Orion's materialize_window (window-relative causal instance-norm).
// The heatmap draws them 1:1 — no client-side re-scaling.
//
// Click-to-inspect uses v5.2's custom-series hit testing: the renderer records
// per-column cell rects and answers hitTest(x, y) -> { feature, time, value }.

export const HEATMAP_COLORS = {
    // Diverging blue<->white<->red around 0 (normalized values are ~N(0,1)).
    clampAbs: 3.5,
};

function divergingColor(v, clampAbs) {
    if (v === null || v === undefined || Number.isNaN(v)) return '#111418';
    const c = Math.max(-clampAbs, Math.min(clampAbs, v)) / clampAbs; // [-1, 1]
    if (c >= 0) {
        // white -> red
        const t = c;
        const r = 255;
        const g = Math.round(255 * (1 - t));
        const b = Math.round(255 * (1 - t));
        return `rgb(${r},${g},${b})`;
    }
    // white -> blue
    const t = -c;
    const r = Math.round(255 * (1 - t));
    const g = Math.round(255 * (1 - t));
    const b = 255;
    return `rgb(${r},${g},${b})`;
}

class FeatureHeatmapRenderer {
    constructor() {
        this._data = null;
        this._options = null;
        this._featureDim = 203;
        // Cell layout captured at draw time for hit testing.
        this._cells = []; // { x0, x1, feature, time, value }
    }

    update(data, options) {
        this._data = data;
        this._options = options;
        if (options && options.featureDim) this._featureDim = options.featureDim;
    }

    draw(target, priceConverter) {
        target.useBitmapCoordinateSpace((scope) => {
            if (!this._data || this._data.bars.length === 0) return;
            const ctx = scope.context;
            const hpr = scope.horizontalPixelRatio;
            const vpr = scope.verticalPixelRatio;
            const bars = this._data.bars;
            const clampAbs = (this._options && this._options.clampAbs) || HEATMAP_COLORS.clampAbs;
            this._cells = [];

            // Column width in media px = spacing between consecutive bars.
            let colW = this._data.barSpacing || 6;
            const dim = this._featureDim;

            // y for feature index j (media coords) via price scale [0, dim].
            const yFor = (j) => priceConverter(j);

            for (let i = 0; i < bars.length; i++) {
                const bar = bars[i];
                const values = bar.originalData && bar.originalData.values;
                if (!values) continue;
                const xCenter = bar.x;
                const x0 = xCenter - colW / 2;
                const x1 = xCenter + colW / 2;
                for (let j = 0; j < dim; j++) {
                    const yTop = yFor(j);
                    const yBot = yFor(j + 1);
                    if (yTop === null || yBot === null) continue;
                    const v = values[j];
                    ctx.fillStyle = divergingColor(v, clampAbs);
                    ctx.fillRect(
                        Math.round(x0 * hpr),
                        Math.round(Math.min(yTop, yBot) * vpr),
                        Math.max(1, Math.round((x1 - x0) * hpr)),
                        Math.max(1, Math.round(Math.abs(yBot - yTop) * vpr)),
                    );
                    this._cells.push({ x0, x1, yTop: Math.min(yTop, yBot), yBot: Math.max(yTop, yBot), feature: j, time: bar.time, value: v });
                }
            }
        });
    }

    // v5.2 custom-series hit testing: map a pixel to a (feature, time, value).
    hitTest(x, y) {
        for (const cell of this._cells) {
            if (x >= cell.x0 && x <= cell.x1 && y >= cell.yTop && y <= cell.yBot) {
                return {
                    zOrder: 'top',
                    externalId: `feat-${cell.feature}@${cell.time}`,
                    _feature: cell.feature,
                    _time: cell.time,
                    _value: cell.value,
                };
            }
        }
        return null;
    }
}

export class FeatureHeatmapSeries {
    constructor() {
        this._renderer = new FeatureHeatmapRenderer();
    }

    priceValueBuilder(plotRow) {
        // Autoscale to the full feature-index range so feature j -> y(j).
        const dim = (plotRow && plotRow.values && plotRow.values.length) || 203;
        return [0, dim];
    }

    isWhitespace(data) {
        return !data.values;
    }

    renderer() {
        return this._renderer;
    }

    update(data, options) {
        this._renderer.update(data, options);
    }

    defaultOptions() {
        return {
            featureDim: 203,
            clampAbs: HEATMAP_COLORS.clampAbs,
            lastValueVisible: false,
            priceLineVisible: false,
        };
    }
}

/**
 * Attach a feature heatmap to a dedicated pane and return a small handle.
 * @param {IChartApi} chart
 * @param {number} paneIndex
 * @returns {{ series, view, setWindow(fn) }}
 */
export function createFeatureHeatmap(chart, paneIndex) {
    const view = new FeatureHeatmapSeries();
    const series = chart.addCustomSeries(view, {
        featureDim: 203,
        priceScaleId: 'heatmap',
    }, paneIndex);
    return {
        series,
        view,
        /**
         * @param {{times:number[], heatmap:number[][], feature_dim:number}} win
         * heatmap is [feature][time]; custom series wants [{time, values:[dim]}].
         */
        setWindow(win) {
            const T = win.times.length;
            const dim = win.feature_dim || win.heatmap.length;
            const data = [];
            for (let t = 0; t < T; t++) {
                const values = new Array(dim);
                for (let j = 0; j < dim; j++) values[j] = win.heatmap[j][t];
                data.push({ time: win.times[t], values });
            }
            series.applyOptions({ featureDim: dim });
            series.setData(data);
        },
    };
}
