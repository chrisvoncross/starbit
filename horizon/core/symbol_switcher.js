// symbol_switcher.js — CENTRAL, reusable Symbol-Swap element for Horizon.
//
// Not transformer-view-only: a self-contained control that any Horizon view
// can mount to let the user click through symbols (◀ / ▶) or jump via a
// filterable dropdown. It renders id + resolved name (the backend resolves
// symbol_id <-> name through the SAME market_data_sync R2 map the providers
// bind, so the label can never drift from the data plane), and calls back with
// the raw integer symbol_id the sampler keys off.
//
// Usage:
//   const sw = createSymbolSwitcher({ onSelect: (id) => controller.selectSymbol(id) });
//   container.appendChild(sw.el);
//   sw.setSymbols([{id: 1, name: 'BTC_USDT'}, ...], currentId);

export function createSymbolSwitcher({ onSelect } = {}) {
    let _symbols = [];          // [{id, name}]
    let _index = -1;            // position in _symbols of the current selection

    const el = document.createElement('div');
    el.className = 'hz-switcher hz-symbol-switcher';

    const label = document.createElement('span');
    label.className = 'hz-switcher-label';
    label.textContent = 'symbol';

    const prev = document.createElement('button');
    prev.className = 'hz-switcher-btn';
    prev.type = 'button';
    prev.textContent = '◀';
    prev.title = 'Previous symbol';

    const select = document.createElement('select');
    select.className = 'hz-switcher-select';
    select.title = 'Jump to symbol';

    const filter = document.createElement('input');
    filter.className = 'hz-switcher-filter';
    filter.type = 'text';
    filter.placeholder = 'filter…';
    filter.spellcheck = false;

    const next = document.createElement('button');
    next.className = 'hz-switcher-btn';
    next.type = 'button';
    next.textContent = '▶';
    next.title = 'Next symbol';

    el.append(label, prev, filter, select, next);

    function _renderOptions(query) {
        const q = (query || '').trim().toLowerCase();
        select.innerHTML = '';
        for (const s of _symbols) {
            const text = `${s.name} (id ${s.id})`;
            if (q && !text.toLowerCase().includes(q)) continue;
            const opt = document.createElement('option');
            opt.value = String(s.id);
            opt.textContent = text;
            select.appendChild(opt);
        }
        if (_index >= 0 && _symbols[_index]) select.value = String(_symbols[_index].id);
    }

    function _emit() {
        if (_index < 0 || !_symbols[_index]) return;
        select.value = String(_symbols[_index].id);
        if (typeof onSelect === 'function') onSelect(_symbols[_index].id);
    }

    function _step(delta) {
        if (!_symbols.length) return;
        _index = (_index + delta + _symbols.length) % _symbols.length;
        _emit();
    }

    prev.addEventListener('click', () => _step(-1));
    next.addEventListener('click', () => _step(1));
    filter.addEventListener('input', () => _renderOptions(filter.value));
    select.addEventListener('change', () => {
        const id = parseInt(select.value, 10);
        const idx = _symbols.findIndex((s) => s.id === id);
        if (idx >= 0) {
            _index = idx;
            _emit();
        }
    });

    return {
        el,
        /** @param {{id:number,name:string}[]} symbols @param {number} currentId */
        setSymbols(symbols, currentId) {
            _symbols = Array.isArray(symbols) ? symbols.slice() : [];
            _index = _symbols.findIndex((s) => s.id === currentId);
            if (_index < 0 && _symbols.length) _index = 0;
            _renderOptions(filter.value);
        },
        /** Reflect an externally-driven selection without re-emitting. */
        setCurrent(currentId) {
            const idx = _symbols.findIndex((s) => s.id === currentId);
            if (idx >= 0) {
                _index = idx;
                select.value = String(currentId);
            }
        },
        get currentId() {
            return _index >= 0 && _symbols[_index] ? _symbols[_index].id : null;
        },
    };
}
