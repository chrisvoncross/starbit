// feature_channels.js — selected normalized feature channels as line series,
// grouped/colored by their Orion NormPolicy.
//
// The 203 channels are window-relative (causal instance-norm over [:seq_len]);
// plotting all 203 as lines is unreadable, so we render a SELECTED subset
// (default: a representative feature per NormPolicy group) as lines in a
// dedicated pane. Colors encode the NormPolicy so the viewer can see how each
// normalization family behaves within the window.

// Stable-ish palette keyed by NormPolicy string value.
const POLICY_COLORS = {
    causal_zscore: '#26a69a',
    log_causal_zscore: '#4fc3f7',
    already_normalized: '#ffb74d',
    signed_log1p: '#ba68c8',
    identity: '#90a4ae',
    minmax_unit: '#ef5350',
    default: '#d1d4dc',
};

function colorFor(policy, fallbackIdx) {
    return POLICY_COLORS[policy] || POLICY_COLORS.default;
}

/**
 * @param {IChartApi} chart
 * @param {number} paneIndex
 */
export function createFeatureChannels(chart, paneIndex) {
    let _series = []; // { series, featureIndex }
    let _selection = null;

    function _clear() {
        for (const s of _series) {
            try { chart.removeSeries(s.series); } catch (e) { /* removed */ }
        }
        _series = [];
    }

    function _defaultSelection(win) {
        // One representative feature per distinct policy (first occurrence),
        // capped to keep the pane readable.
        const seen = new Set();
        const sel = [];
        for (let j = 0; j < win.policies.length; j++) {
            const p = win.policies[j];
            if (!seen.has(p)) {
                seen.add(p);
                sel.push(j);
            }
            if (sel.length >= 8) break;
        }
        return sel;
    }

    return {
        setSelection(indices) {
            _selection = indices;
        },
        /**
         * @param {{times:number[], heatmap:number[][], policies:string[], columns:string[]}} win
         */
        setWindow(win) {
            _clear();
            const sel = _selection || _defaultSelection(win);
            for (const j of sel) {
                const policy = win.policies[j];
                const series = chart.addSeries(LightweightCharts.LineSeries, {
                    color: colorFor(policy, j),
                    lineWidth: 1,
                    priceLineVisible: false,
                    lastValueVisible: false,
                    title: `${win.columns[j]} [${policy}]`,
                }, paneIndex);
                const data = win.times.map((t, ti) => ({ time: t, value: win.heatmap[j][ti] }));
                series.setData(data);
                _series.push({ series, featureIndex: j });
            }
            return sel;
        },
        clear: _clear,
        get selection() { return _selection; },
    };
}
