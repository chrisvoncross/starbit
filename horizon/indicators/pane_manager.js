/**
 * Central Pane Manager — Single Source of Truth for all bottom-pane indicators.
 *
 * Problem: Lightweight Charts auto-collapses empty panes but doesn't support
 * hiding/showing panes. Series with `visible: false` still occupy vertical space.
 *
 * Solution: Destroy/Recreate pattern. When hiding a pane, ALL its series are
 * removed via `chart.removeSeries()`. LC auto-collapses the empty pane.
 * When showing, series are recreated from stored data.
 *
 * Pane ordering: After any toggle, ALL bottom-pane series are destroyed and
 * only the visible ones are recreated in canonical order. This guarantees
 * consistent pane positions (e.g. Z-Score always above Stoch).
 *
 * Usage:
 *   import { paneManager } from './indicators/pane_manager.js';
 *
 *   // 1. Init with chart reference
 *   paneManager.init(chart);
 *
 *   // 2. Register panes (order of registration = canonical order)
 *   paneManager.register('zscore', {
 *       create: (chart) => { ... return [series1, series2, ...]; },
 *       destroy: (chart, seriesList) => { for (s of seriesList) chart.removeSeries(s); },
 *   });
 *
 *   // 3. Toggle visibility
 *   paneManager.show('zscore');    // creates pane
 *   paneManager.hide('zscore');    // destroys pane, LC auto-collapses
 *   paneManager.toggle('zscore', true/false);
 */

// ═══════════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════════

let _chart = null;

/**
 * Ordered list of pane IDs (registration order = canonical display order).
 * @type {string[]}
 */
const _paneOrder = [];

/**
 * Registry of pane definitions.
 * @type {Map<string, {create: Function, destroy: Function|null}>}
 */
const _registry = new Map();

/**
 * Which panes should be visible (desired state).
 * @type {Map<string, boolean>}
 */
const _visibility = new Map();

/**
 * Currently live series for each pane.
 * @type {Map<string, Array>}
 */
const _liveSeries = new Map();

/**
 * Desired pane heights (px). Set during register().
 * @type {Map<string, number>}
 */
const _heights = new Map();

// ═══════════════════════════════════════════════════════════════════════════════
// INTERNAL: Rebuild all visible panes in canonical order
// ═══════════════════════════════════════════════════════════════════════════════

function _destroyAllSeries() {
    for (const paneId of _paneOrder) {
        const seriesList = _liveSeries.get(paneId) || [];
        const reg = _registry.get(paneId);

        if (seriesList.length > 0) {
            if (reg && reg.destroy) {
                // Custom destroy (e.g. zscore needs to null out its indicator ref)
                reg.destroy(_chart, seriesList);
            } else {
                // Default: removeSeries for each
                for (const s of seriesList) {
                    try { _chart.removeSeries(s); } catch (e) { /* already removed */ }
                }
            }
            _liveSeries.set(paneId, []);
        }
    }
}

async function _rebuild() {
    if (!_chart) return;

    // 1) Destroy ALL bottom-pane series
    _destroyAllSeries();

    // 2) Force LC to reclaim empty panes
    _chart.applyOptions({});

    // 3) Recreate only visible panes, in canonical order (sequentially to guarantee pane order)
    //    paneIndex starts at 1 (0 = main chart) and increments for each visible pane
    let nextPaneIndex = 1;
    let createdCount = 0;
    const createdPanes = [];  // Track {paneIndex, height} for deferred height setting

    for (const paneId of _paneOrder) {
        if (_visibility.get(paneId)) {
            const reg = _registry.get(paneId);
            if (reg && reg.create) {
                try {
                    const paneIndex = nextPaneIndex;
                    const result = reg.create(_chart, paneIndex);
                    // Await async creates to guarantee sequential pane ordering
                    const seriesList = (result && typeof result.then === 'function')
                        ? await result
                        : result;
                    _liveSeries.set(paneId, seriesList || []);

                    const h = _heights.get(paneId);
                    if (h) createdPanes.push({ paneIndex, height: h });

                    nextPaneIndex++;
                    createdCount++;
                } catch (e) {
                    console.error(`❌ PaneManager: Failed to create ${paneId}:`, e);
                }
            }
        }
    }

    // 4) Force layout recalculation
    _chart.timeScale().applyOptions({});

    // 5) Apply pane heights AFTER LC layout pass completes (deferred to next frame)
    //    LC overrides setHeight() if called during the same frame as addSeries().
    //    requestAnimationFrame ensures the layout engine has settled first.
    if (createdPanes.length > 0) {
        requestAnimationFrame(() => {
            const allPanes = _chart.panes();
            for (const { paneIndex, height } of createdPanes) {
                if (allPanes[paneIndex]) {
                    allPanes[paneIndex].setHeight(height);
                }
            }
            _chart.timeScale().applyOptions({});
        });
    }

    console.log(`📊 PaneManager: rebuilt ${createdCount}/${_paneOrder.filter(id => _visibility.get(id)).length} panes`);
}

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════════

export const paneManager = {

    /**
     * Initialize with the Lightweight Charts instance.
     * Must be called once before any register/toggle calls.
     */
    init(chart) {
        _chart = chart;
        console.log('📊 PaneManager initialized');
    },

    /**
     * Register a bottom-pane indicator.
     * Registration order determines canonical display order.
     *
     * @param {string} paneId - Unique identifier (e.g. 'zscore', 'stoch', 'osc_alignment')
     * @param {Object} callbacks
     * @param {Function} callbacks.create - (chart, paneIndex) => series[] — creates series and returns them
     * @param {Function} [callbacks.destroy] - (chart, series[]) => void — custom cleanup (optional)
     * @param {number} [callbacks.height] - Desired pane height in px (prevents LC auto-distribute)
     * @param {boolean} [initiallyVisible=false] - Whether to show immediately
     */
    register(paneId, callbacks, initiallyVisible = false) {
        if (_registry.has(paneId)) {
            console.warn(`⚠️ PaneManager: ${paneId} already registered, overwriting`);
        }
        _registry.set(paneId, callbacks);
        _visibility.set(paneId, false);
        _liveSeries.set(paneId, []);

        // Store desired pane height
        if (callbacks.height) {
            _heights.set(paneId, callbacks.height);
        }

        // Maintain canonical order (append if new)
        if (!_paneOrder.includes(paneId)) {
            _paneOrder.push(paneId);
        }

        if (initiallyVisible) {
            _visibility.set(paneId, true);
        }
    },

    /**
     * Toggle a single pane's visibility.
     */
    toggle(paneId, visible) {
        if (!_registry.has(paneId)) {
            console.warn(`⚠️ PaneManager: unknown pane '${paneId}'`);
            return;
        }
        if (_visibility.get(paneId) === visible) return; // No change
        _visibility.set(paneId, visible);
        _rebuild();
    },

    /**
     * Show a pane.
     */
    show(paneId) {
        this.toggle(paneId, true);
    },

    /**
     * Hide a pane.
     */
    hide(paneId) {
        this.toggle(paneId, false);
    },

    /**
     * Toggle multiple panes at once (batch — single rebuild).
     * @param {Object} toggles - { paneId: boolean, ... }
     */
    toggleMultiple(toggles) {
        let changed = false;
        for (const [paneId, visible] of Object.entries(toggles)) {
            if (_registry.has(paneId) && _visibility.get(paneId) !== visible) {
                _visibility.set(paneId, visible);
                changed = true;
            }
        }
        if (changed) _rebuild();
    },

    /**
     * Check if a pane is currently visible.
     */
    isVisible(paneId) {
        return _visibility.get(paneId) || false;
    },

    /**
     * Get all registered pane IDs in canonical order.
     */
    getPaneOrder() {
        return [..._paneOrder];
    },

    /**
     * Get the live series for a pane (e.g. for sub-feature toggles).
     */
    getSeries(paneId) {
        return _liveSeries.get(paneId) || [];
    },

    /**
     * Force a full rebuild (e.g. after lazy-loading new data).
     */
    rebuild() {
        _rebuild();
    },
};
