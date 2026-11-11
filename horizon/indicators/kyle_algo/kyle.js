/**
 * Kyle Algo Buy/Sell Signals
 *
 * Displays buy/sell labels on the chart based on Kyle Algo signals
 */

/**
 * Kyle Algo Configuration
 */
export const KYLE_CONFIG = {
    // Colors (matching TradingView style exactly)
    buyBgColor: '#5D606B',        // Dark gray background for buy
    buyTextColor: '#FFFFFF',      // White text
    sellBgColor: '#2962FF',       // TradingView blue for sell
    sellTextColor: '#FFFFFF',     // White text

    // Label text
    buyText: 'Buy',
    sellText: 'Sell',
};

/**
 * Setup Kyle Algo Buy/Sell labels using Labels Plugin
 *
 * @param {LabelsPlugin} labelsPlugin - The Labels Plugin instance
 * @param {Array} kyleSignals - Kyle signals from server: [{time, buy_signal, sell_signal, price}, ...]
 */
export function setupKyleSignals(labelsPlugin, kyleSignals) {
    const labels = [];

    for (const signal of kyleSignals) {
        // Buy signal (gray rectangle below bar)
        if (signal.buy_signal) {
            labels.push({
                time: signal.time,
                price: signal.price,
                text: KYLE_CONFIG.buyText,
                position: 'belowBar',
                bgColor: KYLE_CONFIG.buyBgColor,
                textColor: KYLE_CONFIG.buyTextColor,
            });
        }

        // Sell signal (blue speech bubble above bar)
        if (signal.sell_signal) {
            labels.push({
                time: signal.time,
                price: signal.price,
                text: KYLE_CONFIG.sellText,
                position: 'aboveBar',
                bgColor: KYLE_CONFIG.sellBgColor,
                textColor: KYLE_CONFIG.sellTextColor,
                arrowSize: 6,
            });
        }
    }

    // Set all labels at once
    labelsPlugin.setLabels(labels);

    return labels.length;
}

/**
 * Clear all Kyle Algo labels
 *
 * @param {LabelsPlugin} labelsPlugin - The Labels Plugin instance
 */
export function clearKyleSignals(labelsPlugin) {
    labelsPlugin.clearLabels();
}

/**
 * Toggle Kyle Algo visibility
 *
 * @param {LabelsPlugin} labelsPlugin - The Labels Plugin instance
 * @param {boolean} visible - Show or hide signals
 * @param {Array} kyleSignals - Kyle signals data (needed if visible=true)
 */
export function toggleKyleSignals(labelsPlugin, visible, kyleSignals = []) {
    if (visible && kyleSignals.length > 0) {
        setupKyleSignals(labelsPlugin, kyleSignals);
    } else {
        clearKyleSignals(labelsPlugin);
    }
}

