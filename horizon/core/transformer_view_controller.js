// transformer_view_controller.js — drives the "what the transformer sees" mode.
//
// Consumes the pinned window packs from providers/transformer_view_provider.py
// over the WS and renders them with the NON-NEGOTIABLE integrity rule made
// visible: the normalized 203 channels are WINDOW-RELATIVE (causal instance-norm
// over the lookback only). The proof HUD shows the as-of pin, the RESOLVED Lance
// version, feature_version and the FEATURE_COLUMNS order on-screen; scrollback
// loads EARLIER windows of the SAME immutable pinned snapshot (the resolved
// version never changes — the HUD proves it).

import { createFeatureHeatmap } from '../indicators/feature_heatmap/feature_heatmap.js';
import { createFeatureChannels } from '../indicators/feature_channels/feature_channels.js';
import { createPanelView } from '../panel/panel_view.js';
import { createSymbolSwitcher } from './symbol_switcher.js';
import { createTimeframeSwitcher } from './timeframe_switcher.js';
import { attachDropoutBands } from '../plugins/dropout_bands/dropout_bands.js';

export class TransformerViewController {
    constructor({ chartCtx, statusEl }) {
        this.chartCtx = chartCtx;
        this.chart = chartCtx.chart;
        this.statusEl = statusEl;
        this.ws = null;

        this.symbols = [];
        this.currentSymbol = null;
        this.currentInterval = null;
        this.currentWindow = null;
        this.resolvedVersion = null; // pinned version — must NEVER change on scroll

        this.heatmap = null;
        this.channels = null;
        this.dropout = null; // dropout-bands overlay handle (attached lazily)
        this.panelView = createPanelView(document.getElementById('chart-container'));

        // Central, reusable Symbol-Swap + Timeframe-Swap controls.
        this.toolbar = document.getElementById('hz-toolbar');
        this.symbolSwitcher = createSymbolSwitcher({ onSelect: (id) => this.selectSymbol(id) });
        this.timeframeSwitcher = createTimeframeSwitcher({ onSelect: (iv) => this.setInterval(iv) });

        this.hud = document.getElementById('proof-hud');
        this.notReadyEl = document.getElementById('notready-overlay');
        this.notReadyReasons = document.getElementById('notready-reasons');
        this._loadingMore = false;
    }

    init() {
        if (this.toolbar) {
            this.toolbar.appendChild(this.timeframeSwitcher.el);
            this.toolbar.appendChild(this.symbolSwitcher.el);
        }
        this._connect();
        this._bindInteractions();
    }

    // Central controls -> backend requests.
    selectSymbol(id) {
        if (id == null) return;
        this.currentSymbol = id;
        this._send({ type: 'tv_window', symbol: id, start_index: null });
    }

    setInterval(interval) {
        if (!interval || interval === this.currentInterval) return;
        this._send({ type: 'tv_set_interval', interval });
    }

    _setStatus(text, cls) {
        if (!this.statusEl) return;
        this.statusEl.textContent = text;
        this.statusEl.className = `status ${cls || ''}`;
    }

    _connect() {
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        this.ws = new WebSocket(`${proto}://${location.host}/ws`);
        this._setStatus('Connecting...', 'loading');

        this.ws.onopen = () => this._setStatus('Connected (pinned)', 'connected');
        this.ws.onclose = () => {
            this._setStatus('Disconnected', 'disconnected');
            setTimeout(() => this._connect(), 2000);
        };
        this.ws.onerror = (e) => console.error('WS error', e);
        this.ws.onmessage = (ev) => {
            let msg;
            try { msg = JSON.parse(ev.data); } catch (e) { return; }
            this._onMessage(msg);
        };
    }

    _send(obj) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(obj));
        }
    }

    _onMessage(msg) {
        switch (msg.type) {
            case 'tv_preflight': return this._renderProof(msg.data);
            case 'tv_not_ready': return this._renderNotReady(msg.data);
            case 'tv_intervals': return this._onIntervals(msg.data);
            case 'tv_initial': return this._onInitial(msg.data);
            case 'tv_more': return this._onWindow(msg.data);
            case 'tv_panel': return this._onPanel(msg.data);
            case 'tv_error':
                console.error('transformer error:', msg.data && msg.data.reason);
                return;
            default:
                return;
        }
    }

    // ---- NOT-READY (first-class state) -------------------------------
    _renderNotReady(data) {
        const reasons = (data && data.reasons) || [];
        const hard = (data && data.hard_failures) || [];
        this.notReadyReasons.innerHTML = '';
        for (const r of [...reasons, ...hard]) {
            const li = document.createElement('li');
            li.textContent = r;
            this.notReadyReasons.appendChild(li);
        }
        this.notReadyEl.classList.add('visible');
        this._setStatus('NOT READY', 'disconnected');
    }

    // ---- The proof HUD (integrity rule made visible) -----------------
    _renderProof(p) {
        if (!this.hud) return;
        // Guard: the resolved version must be immutable across scrollback.
        if (p.resolved_lance_version != null) {
            if (this.resolvedVersion == null) {
                this.resolvedVersion = p.resolved_lance_version;
            }
        }
        const pin = p.pin || {};
        const cov = p.coverage || {};
        const batch = p.batch || {};
        const tags = p.available_tags || {};
        const versionStable = this.resolvedVersion != null &&
            p.resolved_lance_version != null &&
            this.resolvedVersion === p.resolved_lance_version;

        const surface = (p.horizon && p.horizon.surface) || {};
        const surfaceBadge = surface.guaranteed
            ? '<span class="ready">GUARANTEED ✓</span>'
            : '<span class="notready">NOT GUARANTEED</span>';
        const rows = [
            ['surface-equivalence', surfaceBadge],
            ['bound dataset', surface.features_uri || p.features_uri || '?'],
            ['uri source', surface.uri_source || '?'],
            ['expected tag', surface.expected_as_of_tag || '(none)'],
            ['as-of pin', `${pin.kind || '?'} = ${JSON.stringify(pin.value)}`],
            ['resolved version', `${p.resolved_lance_version} ${versionStable ? '(stable ✓)' : ''}`],
            ['feature_version', p.feature_version || '?'],
            ['feature_dim', `${batch.feature_dim} (== 203 required)`],
            ['cadence', surface.cadence_ms != null ? `${surface.cadence_ms} ms` : '?'],
            ['sampled shape', batch.shape ? `[${batch.shape.join(', ')}]` : '-'],
            ['eligible symbols', cov.eligible_symbols],
            ['coverage rows', `${cov.coverage_available} (ratio ${cov.coverage_ratio != null ? cov.coverage_ratio.toFixed(3) : '?'})`],
            ['available tags', Object.keys(tags).join(', ') || '(none)'],
            ['norm', 'causal instance-norm over lookback [:seq_len] — WINDOW-RELATIVE'],
        ];
        if (Array.isArray(surface.divergences) && surface.divergences.length) {
            rows.push(['divergences', surface.divergences.join(' | ')]);
        }
        const readyBadge = p.ready
            ? '<span class="ready">READY</span>'
            : '<span class="notready">NOT-READY</span>';
        this.hud.innerHTML =
            `<div class="hud-title">Transformer input proof &nbsp; ${readyBadge}</div>` +
            rows.map(([k, v]) => `<div class="kv"><span class="k">${k}</span><span class="v">${v ?? '-'}</span></div>`).join('') +
            `<div class="kv"><span class="k">FEATURE_COLUMNS</span><span class="v" id="hud-cols">order served per-window (203)</span></div>` +
            `<div class="kv"><span class="k">inspect</span><span class="v" id="hud-inspect">hover the heatmap</span></div>`;
        this.hud.classList.add('visible');
    }

    // ---- Timeframe surface -------------------------------------------
    _onIntervals(data) {
        if (!data) return;
        if (data.active) this.currentInterval = data.active;
        this.timeframeSwitcher.setIntervals(data.intervals || [], data.active);
    }

    // ---- Initial + window rendering ----------------------------------
    _onInitial(data) {
        this._clearNotReady();
        this.symbols = data.symbols || [];
        if (data.interval) this.currentInterval = data.interval;
        if (data.intervals) this.timeframeSwitcher.setIntervals(data.intervals, data.interval);
        this.currentSymbol = (data.window && data.window.symbol) ?? this.symbols[0];
        // Populate the central Symbol-Swap element (id + resolved name).
        const labels = data.symbol_labels || this.symbols.map((id) => ({ id, name: `symbol_id ${id}` }));
        this.symbolSwitcher.setSymbols(labels, this.currentSymbol);
        this._ensurePanes();
        this._onWindow(data.window);
    }

    _clearNotReady() {
        if (this.notReadyEl) this.notReadyEl.classList.remove('visible');
    }

    _ensurePanes() {
        if (!this.heatmap) this.heatmap = createFeatureHeatmap(this.chart, 1);
        if (!this.channels) this.channels = createFeatureChannels(this.chart, 2);
        if (!this.dropout) this.dropout = attachDropoutBands(this.chartCtx.series.rawSeries);
    }

    _onWindow(win) {
        if (!win) return;
        this._clearNotReady();
        this.currentWindow = win;
        this.currentSymbol = win.symbol;
        this._ensurePanes();
        // Keep the central Symbol-Swap element in sync with the served window.
        this.symbolSwitcher.setCurrent(win.symbol);

        // 1. Candles on the main pane (absolute prices — continuous, NOT normalized).
        this.chartCtx.setRawOhlcData(win.candles);
        this.chartCtx.switchChartStyle('raw');
        // Re-enable price autoscale before replacing data: once a user drags the
        // price scale it latches to manual, so a new symbol/window would render
        // off-screen without this (Lightweight Charts #1846 / setData contract).
        try { this.chartCtx.series.rawSeries.priceScale().applyOptions({ autoScale: true }); } catch (e) { /* noop */ }
        this.chartCtx.series.rawSeries.setData(win.candles);

        // 1b. Dropout overlay: band the Δt>cadence data gaps inside this window
        // so a 30s hole in a 15s series is not mis-read as a normal 15s step.
        if (this.dropout) this.dropout.setGaps(win.gaps || []);

        // 2. lookback | horizon divider marker.
        if (win.lookback_boundary_time != null && LightweightCharts.createSeriesMarkers) {
            try {
                LightweightCharts.createSeriesMarkers(this.chartCtx.series.rawSeries, [{
                    time: win.lookback_boundary_time,
                    position: 'aboveBar',
                    color: '#7fd1c1',
                    shape: 'arrowDown',
                    text: 'lookback | horizon',
                }]);
            } catch (e) { /* markers optional */ }
        }

        // 3. Heatmap + channels for THIS window (window-relative values).
        this.heatmap.setWindow(win);
        this.channels.setWindow(win);

        this.chart.timeScale().fitContent();

        // 4. HUD window nav (+ M2 imagination_fan status when present).
        const inspect = document.getElementById('hud-inspect');
        if (inspect) {
            const nGaps = (win.gaps || []).length;
            const gapTxt = nGaps ? ` | ⌀ ${nGaps} dropout${nGaps > 1 ? 's' : ''}` : ' | no dropouts';
            const fan = win.imagination_fan;
            let fanTxt = '';
            if (fan && fan.status === 'ok') {
                const st = fan.scenario_stats || {};
                fanTxt = ` | fan N=${fan.n_trajectories} H=${fan.horizon}` +
                    (st.fan_width_mean != null ? ` width≈${Number(st.fan_width_mean).toFixed(3)}` : '');
            } else if (fan) {
                fanTxt = ` | fan:${fan.status || 'n/a'}`;
            }
            inspect.textContent =
                `${this.currentInterval || ''} symbol ${win.symbol} | window ${win.start_index + 1}/${win.total_windows} ` +
                `| earlier:${win.has_earlier ? 'yes' : 'no'}${gapTxt}${fanTxt}`;
        }
        this._loadingMore = false;
    }

    _onPanel(pack) {
        this.panelView.render(pack);
        this.panelView.show();
        const inspect = document.getElementById('hud-inspect');
        if (inspect) inspect.textContent = `panel ${pack.start_index + 1}/${pack.total_windows} (aligned:${pack.time_aligned})`;
    }

    // ---- Interactions: scrollback, stepping, heatmap hover -----------
    _bindInteractions() {
        // Scrollback: when the user scrolls to the left edge, request an
        // EARLIER window of the SAME pinned snapshot.
        this.chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
            if (!range || this._loadingMore || !this.currentWindow) return;
            if (!this.currentWindow.has_earlier) return;
            if (range.from < 2) {
                this._loadingMore = true;
                const earliest = this.currentWindow.times[0];
                this._send({ type: 'tv_before', symbol: this.currentSymbol, before: earliest });
            }
        });

        // Keyboard stepping through windows.
        window.addEventListener('keydown', (e) => {
            if (!this.currentWindow) return;
            if (e.key === 'ArrowLeft' && this.currentWindow.has_earlier) {
                this._send({ type: 'tv_window', symbol: this.currentSymbol, start_index: this.currentWindow.start_index - 1 });
            } else if (e.key === 'ArrowRight' && this.currentWindow.has_later) {
                this._send({ type: 'tv_window', symbol: this.currentSymbol, start_index: this.currentWindow.start_index + 1 });
            }
        });

        // v5.2 custom-series hit testing: the heatmap renderer answers
        // hitTest(x,y) with an externalId "feat-<j>@<time>"; LWC surfaces it as
        // param.hoveredObjectId so we can inspect the exact ingested cell.
        this.chart.subscribeCrosshairMove((param) => {
            const inspectEl = document.getElementById('hud-inspect');
            if (!inspectEl || !param || !param.hoveredObjectId || !this.currentWindow) return;
            const m = /^feat-(\d+)@(\d+)$/.exec(String(param.hoveredObjectId));
            if (!m) return;
            const j = parseInt(m[1], 10);
            const t = parseInt(m[2], 10);
            const ti = this.currentWindow.times.indexOf(t);
            const v = ti >= 0 ? this.currentWindow.heatmap[j][ti] : null;
            const name = this.currentWindow.columns[j];
            const policy = this.currentWindow.policies[j];
            inspectEl.textContent = `[${j}] ${name} (${policy}) = ${v == null ? 'n/a' : v.toFixed(4)}`;
        });
    }
}
