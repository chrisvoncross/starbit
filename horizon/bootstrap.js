// bootstrap.js — Horizon mode dispatcher.
//
// The server injects window.__HORIZON_MODE__ into index.html (read_root). We
// dynamically import the matching entry so the legacy base viz (chart_realtime.js)
// stays byte-identical while the transformer view lives in its own entry.
//
//   raw_farmer       -> chart_realtime.js  (verbatim legacy base viz)
//   tigris_range     -> chart_realtime.js  (one Tigris span, raw OHLC)
//   transformer_view -> transformer_view.js (Orion reuse seam + proof HUD)

async function resolveMode() {
    let mode = window.__HORIZON_MODE__;
    // If the placeholder was not substituted (opened as a static file), fall
    // back to /api/config, then to raw_farmer.
    if (!mode || mode === '__HORIZON_MODE__') {
        try {
            const res = await fetch('/api/config');
            if (res.ok) {
                const cfg = await res.json();
                mode = cfg.mode;
                window.__HORIZON_SYMBOL__ = cfg.symbol;
            }
        } catch (e) {
            console.warn('config fetch failed, defaulting to raw_farmer', e);
        }
    }
    return mode || 'raw_farmer';
}

(async () => {
    const mode = await resolveMode();
    console.log('🚀 Horizon bootstrap mode =', mode);
    if (mode === 'transformer_view') {
        await import('./transformer_view.js');
    } else {
        await import('./chart_realtime.js?v=zerofees6');
    }
})();
