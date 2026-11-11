/**
 * TradingView-Style Indicator Panel — with Group Nesting
 * 
 * Provides a floating panel to toggle indicators and individual features on/off.
 * Supports 3-level nesting: Group > Indicator > Features
 * 
 * Structure:
 * - INDICATORS (panel header)
 *   - NWE          (ungrouped indicator)
 *   - Z-Score      (ungrouped indicator)
 *   - Stoch %K     (ungrouped indicator)
 *   - Additional groups can be added later if needed
 */

import { DEFAULT_GROUP_REGISTRY, DEFAULT_INDICATOR_REGISTRY } from './indicator_registry.js?v=zerofees4';

// ============================================================================
// INDICATOR REGISTRY — Single Source of Truth for all chart indicators
// ============================================================================

const INDICATOR_REGISTRY = JSON.parse(JSON.stringify(DEFAULT_INDICATOR_REGISTRY));

// ============================================================================
// GROUP REGISTRY — Collapsible parent groups for related indicators
// ============================================================================

const GROUP_REGISTRY = JSON.parse(JSON.stringify(DEFAULT_GROUP_REGISTRY));

// ============================================================================
// PANEL STATE
// ============================================================================

let _panelEl = null;
let _isExpanded = {};  // Track which indicators have expanded feature lists
let _indicatorRefs = {};  // References to chart series/plugins for toggling
let _onToggleCallbacks = {};  // Callbacks for toggle events

// ============================================================================
// CREATE PANEL
// ============================================================================

/**
 * Create the indicator panel and inject it into the DOM
 * 
 * @param {HTMLElement} container - The chart container element
 * @returns {HTMLElement} - The panel element
 */
export function createIndicatorPanel(container) {
    _panelEl = document.createElement('div');
    _panelEl.id = 'indicator-panel';
    _panelEl.innerHTML = _buildPanelHTML();

    _injectStyles();
    container.appendChild(_panelEl);
    _bindEvents();

    console.log('✅ Indicator Panel created');
    return _panelEl;
}

// ============================================================================
// REGISTER INDICATOR REFERENCES (called from chart_realtime.js)
// ============================================================================

/**
 * Register an indicator's chart objects so the panel can toggle them
 * 
 * @param {string} id - Indicator ID (e.g. 'nwe', 'zscore', 'stoch_obv')
 * @param {Object} refs - Object containing series/plugin references
 * @param {Function} onToggle - Callback(visible) for whole-indicator toggle
 * @param {Object} featureToggleCallbacks - { featureId: callback(visible) }
 */
export function registerIndicator(id, refs, onToggle, featureToggleCallbacks = {}) {
    _indicatorRefs[id] = refs;
    _onToggleCallbacks[id] = {
        main: onToggle,
        features: featureToggleCallbacks,
    };
    console.log(`📊 Panel: Registered indicator "${id}"`);
}

// ============================================================================
// BUILD HTML
// ============================================================================

const _eyeSvg = (size, strokeW) => `<svg viewBox="0 0 24 24" width="${size}" height="${size}">
    <path class="ip-eye-open" d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" fill="none" stroke="currentColor" stroke-width="${strokeW}"/>
    <circle class="ip-eye-open" cx="12" cy="12" r="3" fill="none" stroke="currentColor" stroke-width="${strokeW}"/>
    <line class="ip-eye-slash" x1="1" y1="1" x2="23" y2="23" stroke="currentColor" stroke-width="${strokeW}" style="display:none"/>
</svg>`;

const _chevronSvg = (size) => `<svg viewBox="0 0 24 24" width="${size}" height="${size}">
    <polyline points="6 9 12 15 18 9" fill="none" stroke="currentColor" stroke-width="2"/>
</svg>`;

// SVG icons for chart style toggle
const _candleIconSvg = `<svg viewBox="0 0 20 20" width="14" height="14">
    <rect x="7" y="2" width="6" height="16" rx="1" fill="currentColor" opacity="0.9"/>
    <line x1="10" y1="0" x2="10" y2="2" stroke="currentColor" stroke-width="1.5"/>
    <line x1="10" y1="18" x2="10" y2="20" stroke="currentColor" stroke-width="1.5"/>
</svg>`;
const _barIconSvg = `<svg viewBox="0 0 20 20" width="14" height="14">
    <line x1="10" y1="1" x2="10" y2="19" stroke="currentColor" stroke-width="1.5"/>
    <line x1="4" y1="6" x2="10" y2="6" stroke="currentColor" stroke-width="1.5"/>
    <line x1="10" y1="14" x2="16" y2="14" stroke="currentColor" stroke-width="1.5"/>
</svg>`;

function _buildPanelHTML() {
    let html = `<div class="ip-header">
        <span class="ip-title">Indicators</span>
        <div class="ip-header-actions">
            <button class="ip-style-toggle" id="chart-style-toggle" title="Toggle chart style (Candles / OHLC Bars)">
                <span class="ip-style-icon ip-style-candles">${_candleIconSvg}</span>
                <span class="ip-style-icon ip-style-bars" style="display:none">${_barIconSvg}</span>
            </button>
            <button class="ip-collapse-btn" title="Collapse panel">▾</button>
        </div>
    </div>
    <div class="ip-body">`;

    // 1) Render ungrouped indicators first
    for (const [id, config] of Object.entries(INDICATOR_REGISTRY)) {
        if (config.group) continue;
        html += _buildIndicatorHTML(id, config);
    }

    // 2) Render groups with their child indicators
    for (const [groupId, group] of Object.entries(GROUP_REGISTRY)) {
        const childIds = Object.entries(INDICATOR_REGISTRY)
            .filter(([_, c]) => c.group === groupId)
            .map(([id]) => id);

        html += `
        <div class="ip-group" data-group="${groupId}">
            <div class="ip-group-header">
                <button class="ip-eye-btn ip-group-eye ${group.visible ? 'active' : ''}" data-group="${groupId}" title="Toggle all ${group.name}">
                    ${_eyeSvg(16, 2)}
                </button>
                <span class="ip-group-name">${group.name}</span>
                <span class="ip-group-count">${childIds.length}</span>
                <button class="ip-expand-btn ip-group-expand ${group.expanded ? 'expanded' : ''}" data-group="${groupId}" title="Expand ${group.name}">
                    ${_chevronSvg(12)}
                </button>
            </div>
            <div class="ip-group-body" data-group="${groupId}" style="display:${group.expanded ? 'block' : 'none'}">`;

        for (const childId of childIds) {
            html += _buildIndicatorHTML(childId, INDICATOR_REGISTRY[childId], true);
        }

        html += `
            </div>
        </div>`;
    }

    html += `</div>`;
    return html;
}

function _buildIndicatorHTML(id, config, nested = false) {
    const typeIcon = config.type === 'overlay' ? '◈' : '▤';
    const typeClass = config.type === 'overlay' ? 'ip-type-overlay' : 'ip-type-pane';
    const featureCount = Object.keys(config.features).length;
    const indent = nested ? ' ip-nested' : '';
    if (config.expanded) _isExpanded[id] = true;

    return `
    <div class="ip-indicator${indent}" data-id="${id}">
        <div class="ip-indicator-header">
            <button class="ip-eye-btn ${config.visible ? 'active' : ''}" data-id="${id}" title="Toggle ${config.shortName}">
                ${_eyeSvg(16, 2)}
            </button>
            <span class="ip-type-badge ${typeClass}">${typeIcon}</span>
            <span class="ip-indicator-name">${config.shortName}</span>
            ${featureCount > 0 ? `
            <button class="ip-expand-btn${config.expanded ? ' expanded' : ''}" data-id="${id}" title="Show features">
                ${_chevronSvg(12)}
            </button>` : ''}
        </div>
        <div class="ip-features" data-id="${id}" style="display:${config.expanded ? 'block' : 'none'}">
            ${_buildFeaturesHTML(id, config.features)}
        </div>
    </div>`;
}

function _buildFeaturesHTML(indicatorId, features) {
    let html = '';
    for (const [featureId, feature] of Object.entries(features)) {
        html += `
        <div class="ip-feature" data-indicator="${indicatorId}" data-feature="${featureId}">
            <button class="ip-feature-eye ${feature.visible ? 'active' : ''}" 
                    data-indicator="${indicatorId}" data-feature="${featureId}"
                    title="Toggle ${feature.name}">
                ${_eyeSvg(12, 1.5)}
            </button>
            <span class="ip-color-swatch" style="background:${feature.color}"></span>
            <span class="ip-feature-name">${feature.name}</span>
        </div>`;
    }
    return html;
}

// ============================================================================
// STYLES
// ============================================================================

function _injectStyles() {
    if (document.getElementById('indicator-panel-styles')) return;

    const style = document.createElement('style');
    style.id = 'indicator-panel-styles';
    style.textContent = `
        #indicator-panel {
            position: absolute;
            top: 68px;
            left: 10px;
            z-index: 1001;
            background: rgba(19, 23, 34, 0.92);
            border: 1px solid rgba(55, 59, 74, 0.6);
            border-radius: 6px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 12px;
            color: #d1d4dc;
            min-width: 180px;
            max-width: 280px;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
            user-select: none;
            pointer-events: auto;
        }

        #indicator-panel.collapsed .ip-body {
            display: none;
        }

        #indicator-panel.collapsed {
            min-width: auto;
        }

        .ip-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 10px;
            border-bottom: 1px solid rgba(55, 59, 74, 0.4);
            cursor: pointer;
        }

        .ip-title {
            font-size: 11px;
            font-weight: 600;
            color: #787b86;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .ip-header-actions {
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .ip-style-toggle {
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(55, 59, 74, 0.3);
            border: 1px solid rgba(55, 59, 74, 0.5);
            border-radius: 4px;
            color: #787b86;
            cursor: pointer;
            width: 24px;
            height: 20px;
            padding: 0;
            transition: all 0.15s;
        }

        .ip-style-toggle:hover {
            background: rgba(55, 59, 74, 0.6);
            color: #d1d4dc;
            border-color: rgba(100, 105, 120, 0.6);
        }

        .ip-style-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            line-height: 0;
        }

        .ip-collapse-btn {
            background: none;
            border: none;
            color: #787b86;
            cursor: pointer;
            font-size: 14px;
            padding: 0 2px;
            line-height: 1;
            transition: transform 0.2s;
        }

        #indicator-panel.collapsed .ip-collapse-btn {
            transform: rotate(-90deg);
        }

        .ip-body {
            padding: 4px 0;
        }

        /* ── Group level ── */
        .ip-group {
            border-bottom: 1px solid rgba(55, 59, 74, 0.2);
        }

        .ip-group:last-child {
            border-bottom: none;
        }

        .ip-group-header {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 5px 10px;
            transition: background 0.15s;
        }

        .ip-group-header:hover {
            background: rgba(55, 59, 74, 0.3);
        }

        .ip-group-name {
            flex: 1;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.3px;
        }

        .ip-group-count {
            font-size: 10px;
            color: #787b86;
            background: rgba(55, 59, 74, 0.4);
            padding: 1px 5px;
            border-radius: 8px;
            font-weight: 600;
        }

        .ip-group-body {
            padding-left: 8px;
            border-left: 2px solid rgba(55, 59, 74, 0.4);
            margin-left: 18px;
        }

        /* ── Indicator level ── */
        .ip-indicator {
            border-bottom: 1px solid rgba(55, 59, 74, 0.2);
        }

        .ip-indicator:last-child {
            border-bottom: none;
        }

        .ip-indicator.ip-nested {
            border-bottom: 1px solid rgba(55, 59, 74, 0.1);
        }

        .ip-indicator-header {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 5px 10px;
            transition: background 0.15s;
        }

        .ip-indicator-header:hover {
            background: rgba(55, 59, 74, 0.3);
        }

        .ip-eye-btn, .ip-feature-eye {
            background: none;
            border: none;
            color: #787b86;
            cursor: pointer;
            padding: 2px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 3px;
            transition: color 0.15s, background 0.15s;
        }

        .ip-eye-btn:hover, .ip-feature-eye:hover {
            background: rgba(55, 59, 74, 0.5);
        }

        .ip-eye-btn.active {
            color: #2962FF;
        }

        .ip-feature-eye.active {
            color: #d1d4dc;
        }

        .ip-eye-btn:not(.active) .ip-eye-open,
        .ip-feature-eye:not(.active) .ip-eye-open {
            opacity: 0.3;
        }

        .ip-eye-btn:not(.active) .ip-eye-slash,
        .ip-feature-eye:not(.active) .ip-eye-slash {
            display: inline !important;
            opacity: 0.5;
        }

        .ip-type-badge {
            font-size: 10px;
            padding: 1px 4px;
            border-radius: 3px;
            font-weight: 600;
        }

        .ip-type-overlay {
            color: #26a69a;
            background: rgba(38, 166, 154, 0.1);
        }

        .ip-type-pane {
            color: #7B61FF;
            background: rgba(123, 97, 255, 0.1);
        }

        .ip-indicator-name {
            flex: 1;
            font-size: 12px;
            font-weight: 500;
            color: #d1d4dc;
        }

        .ip-expand-btn {
            background: none;
            border: none;
            color: #787b86;
            cursor: pointer;
            padding: 2px 4px;
            display: flex;
            align-items: center;
            border-radius: 3px;
            transition: transform 0.2s, color 0.15s;
        }

        .ip-expand-btn:hover {
            color: #d1d4dc;
        }

        .ip-expand-btn.expanded {
            transform: rotate(180deg);
        }

        /* ── Feature sub-list ── */
        .ip-features {
            padding: 0 10px 4px 28px;
        }

        .ip-nested .ip-features {
            padding-left: 22px;
        }

        .ip-feature {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 3px 0;
        }

        .ip-color-swatch {
            width: 10px;
            height: 10px;
            border-radius: 2px;
            flex-shrink: 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .ip-feature-name {
            font-size: 11px;
            color: #787b86;
        }

        .ip-feature:hover .ip-feature-name {
            color: #d1d4dc;
        }
    `;
    document.head.appendChild(style);
}

// ============================================================================
// EVENT BINDING
// ============================================================================

function _bindEvents() {
    // Chart style toggle: candles (HA) → raw (Raw OHLC) → bars (Bloomberg) → candles
    const styleBtn = _panelEl.querySelector('#chart-style-toggle');
    if (styleBtn) {
        styleBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            const current = window._getChartStyle ? window._getChartStyle() : 'candles';
            const cycle = { 'candles': 'raw', 'raw': 'bars', 'bars': 'candles' };
            const next = cycle[current] || 'candles';
            if (window._switchChartStyle) window._switchChartStyle(next);
            // Toggle icon visibility + label
            const candleIcon = styleBtn.querySelector('.ip-style-candles');
            const barIcon = styleBtn.querySelector('.ip-style-bars');
            if (next === 'bars') {
                candleIcon.style.display = 'none';
                barIcon.style.display = '';
                styleBtn.title = 'Bloomberg OHLC Bars';
            } else if (next === 'raw') {
                candleIcon.style.display = '';
                barIcon.style.display = 'none';
                styleBtn.title = 'Raw OHLC Candles';
            } else {
                candleIcon.style.display = '';
                barIcon.style.display = 'none';
                styleBtn.title = 'Heikin Ashi Candles';
            }
        });
    }

    // Collapse/expand panel
    const header = _panelEl.querySelector('.ip-header');
    header.addEventListener('click', () => {
        _panelEl.classList.toggle('collapsed');
    });

    // Group eye buttons (toggle all indicators in group)
    _panelEl.querySelectorAll('.ip-group-eye').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            _toggleGroup(btn.dataset.group);
        });
    });

    // Group expand buttons
    _panelEl.querySelectorAll('.ip-group-expand').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            _toggleGroupExpand(btn.dataset.group);
        });
    });

    // Group header click = expand/collapse
    _panelEl.querySelectorAll('.ip-group-header').forEach(header => {
        header.addEventListener('click', (e) => {
            // Don't trigger if clicking buttons inside
            if (e.target.closest('button')) return;
            const groupId = header.closest('.ip-group').dataset.group;
            _toggleGroupExpand(groupId);
        });
    });

    // Indicator eye buttons
    _panelEl.querySelectorAll('.ip-eye-btn:not(.ip-group-eye)').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            _toggleIndicator(btn.dataset.id);
        });
    });

    // Indicator expand buttons (show/hide features)
    _panelEl.querySelectorAll('.ip-expand-btn:not(.ip-group-expand)').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            _toggleExpand(btn.dataset.id);
        });
    });

    // Feature eye buttons
    _panelEl.querySelectorAll('.ip-feature-eye').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            _toggleFeature(btn.dataset.indicator, btn.dataset.feature);
        });
    });
}

// ============================================================================
// TOGGLE LOGIC
// ============================================================================

function _toggleGroup(groupId) {
    const group = GROUP_REGISTRY[groupId];
    if (!group) return;

    group.visible = !group.visible;

    // Update group eye button
    const btn = _panelEl.querySelector(`.ip-group-eye[data-group="${groupId}"]`);
    if (btn) btn.classList.toggle('active', group.visible);

    // Toggle all child indicators
    for (const [id, config] of Object.entries(INDICATOR_REGISTRY)) {
        if (config.group !== groupId) continue;
        if (config.visible !== group.visible) {
            config.visible = group.visible;

            // Update child eye button
            const childBtn = _panelEl.querySelector(`.ip-eye-btn[data-id="${id}"]`);
            if (childBtn) childBtn.classList.toggle('active', config.visible);

            // Fire callback
            const callbacks = _onToggleCallbacks[id];
            if (callbacks && callbacks.main) {
                callbacks.main(config.visible);
            }
        }
    }

    // Auto-expand group when enabling, collapse when disabling
    if (group.visible && !group.expanded) {
        _toggleGroupExpand(groupId);
    }

    console.log(`📊 Panel: Group ${group.name} → ${group.visible ? 'ON' : 'OFF'}`);
}

function _toggleGroupExpand(groupId) {
    const group = GROUP_REGISTRY[groupId];
    if (!group) return;

    group.expanded = !group.expanded;

    const btn = _panelEl.querySelector(`.ip-group-expand[data-group="${groupId}"]`);
    const body = _panelEl.querySelector(`.ip-group-body[data-group="${groupId}"]`);

    if (btn) btn.classList.toggle('expanded', group.expanded);
    if (body) body.style.display = group.expanded ? 'block' : 'none';
}

function _toggleIndicator(id) {
    const config = INDICATOR_REGISTRY[id];
    if (!config) return;

    config.visible = !config.visible;

    // Update button visual
    const btn = _panelEl.querySelector(`.ip-eye-btn[data-id="${id}"]`);
    if (btn) btn.classList.toggle('active', config.visible);

    // If indicator is in a group, update group eye to reflect child state
    if (config.group) {
        _syncGroupEye(config.group);
    }

    // Call registered callback
    const callbacks = _onToggleCallbacks[id];
    if (callbacks && callbacks.main) {
        callbacks.main(config.visible);
    }

    console.log(`📊 Panel: ${config.shortName} → ${config.visible ? 'ON' : 'OFF'}`);
}

function _syncGroupEye(groupId) {
    const group = GROUP_REGISTRY[groupId];
    if (!group) return;

    // Group eye = ON if ANY child is visible
    const anyVisible = Object.entries(INDICATOR_REGISTRY)
        .filter(([_, c]) => c.group === groupId)
        .some(([_, c]) => c.visible);

    group.visible = anyVisible;
    const btn = _panelEl.querySelector(`.ip-group-eye[data-group="${groupId}"]`);
    if (btn) btn.classList.toggle('active', anyVisible);
}

function _toggleExpand(id) {
    _isExpanded[id] = !_isExpanded[id];

    const btn = _panelEl.querySelector(`.ip-expand-btn[data-id="${id}"]`);
    const features = _panelEl.querySelector(`.ip-features[data-id="${id}"]`);

    if (btn) btn.classList.toggle('expanded', _isExpanded[id]);
    if (features) features.style.display = _isExpanded[id] ? 'block' : 'none';
}

function _toggleFeature(indicatorId, featureId) {
    const config = INDICATOR_REGISTRY[indicatorId];
    if (!config || !config.features[featureId]) return;

    const feature = config.features[featureId];
    feature.visible = !feature.visible;

    // Update button visual
    const btn = _panelEl.querySelector(`.ip-feature-eye[data-indicator="${indicatorId}"][data-feature="${featureId}"]`);
    if (btn) btn.classList.toggle('active', feature.visible);

    // Call registered callback
    const callbacks = _onToggleCallbacks[indicatorId];
    if (callbacks && callbacks.features && callbacks.features[featureId]) {
        callbacks.features[featureId](feature.visible);
    }

    console.log(`📊 Panel: ${config.shortName}.${featureId} → ${feature.visible ? 'ON' : 'OFF'}`);
}

// ============================================================================
// PUBLIC API
// ============================================================================

/**
 * Check if an indicator is currently visible
 */
export function isIndicatorVisible(id) {
    return INDICATOR_REGISTRY[id]?.visible ?? false;
}

/**
 * Check if a specific feature within an indicator is visible
 */
export function isFeatureVisible(indicatorId, featureId) {
    return INDICATOR_REGISTRY[indicatorId]?.features[featureId]?.visible ?? false;
}

/**
 * Get the indicator registry (read-only)
 */
export function getIndicatorRegistry() {
    return { ...INDICATOR_REGISTRY };
}
