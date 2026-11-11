// panel_view.js — cross-asset small-multiples for the D=203 panel feed.
//
// Renders one mini feature-heatmap per panel symbol (all sharing the SAME
// time-aligned grid) as a DOM canvas overlay, plus an ALIGNMENT badge proving
// every strip was materialized at identical open_time_ms (Orion panel_grid).
// Cross-links Orion ADR-20260707-31 (the cross-asset panel feed).

const CLAMP_ABS = 3.5;

function cellColor(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return '#111418';
    const c = Math.max(-CLAMP_ABS, Math.min(CLAMP_ABS, v)) / CLAMP_ABS;
    if (c >= 0) {
        const t = c;
        return `rgb(255,${Math.round(255 * (1 - t))},${Math.round(255 * (1 - t))})`;
    }
    const t = -c;
    return `rgb(${Math.round(255 * (1 - t))},${Math.round(255 * (1 - t))},255)`;
}

export function createPanelView(container) {
    const root = document.createElement('div');
    root.id = 'panel-view';
    root.style.cssText = [
        'position:absolute', 'inset:0', 'overflow:auto', 'padding:16px',
        'background:#05070b', 'z-index:800', 'display:none',
        'font-family:ui-monospace,SFMono-Regular,Menlo,monospace',
    ].join(';');
    container.appendChild(root);

    function _drawStrip(canvas, strip, T, dim) {
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;
        const cw = W / T;
        const ch = H / dim;
        for (let j = 0; j < dim; j++) {
            for (let t = 0; t < T; t++) {
                ctx.fillStyle = cellColor(strip.heatmap[j][t]);
                ctx.fillRect(Math.floor(t * cw), Math.floor(j * ch), Math.ceil(cw), Math.ceil(ch));
            }
        }
    }

    return {
        show() { root.style.display = 'block'; },
        hide() { root.style.display = 'none'; },
        /**
         * @param {{panel_symbols:number[], times:number[], strips:Array, time_aligned:boolean, columns:string[]}} pack
         */
        render(pack) {
            root.innerHTML = '';
            const T = pack.times.length;
            const dim = pack.columns.length;

            const header = document.createElement('div');
            header.style.cssText = 'margin-bottom:12px;color:#cfd6e0;font-size:12px;';
            const badge = pack.time_aligned
                ? '<span style="color:#26a69a;font-weight:700;">✔ TIME-ALIGNED</span>'
                : '<span style="color:#ef5350;font-weight:700;">x MISALIGNED</span>';
            header.innerHTML = `Cross-asset panel — ${pack.panel_symbols.length} symbols × ${dim} features × ${T} steps &nbsp; ${badge}`;
            root.appendChild(header);

            const grid = document.createElement('div');
            grid.style.cssText = 'display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px;';
            root.appendChild(grid);

            for (const strip of pack.strips) {
                const card = document.createElement('div');
                card.style.cssText = 'background:#0c1017;border:1px solid #232c38;border-radius:6px;padding:8px;';
                const label = document.createElement('div');
                label.style.cssText = 'color:#7fd1c1;font-size:11px;margin-bottom:6px;';
                label.textContent = `symbol_id ${strip.symbol}`;
                card.appendChild(label);
                const canvas = document.createElement('canvas');
                canvas.width = 200;
                canvas.height = 160;
                canvas.style.cssText = 'width:100%;height:160px;image-rendering:pixelated;';
                card.appendChild(canvas);
                grid.appendChild(card);
                _drawStrip(canvas, strip, T, dim);
            }
        },
    };
}
