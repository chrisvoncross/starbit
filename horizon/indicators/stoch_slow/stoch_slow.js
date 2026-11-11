/**
 * Stochastic Slow %K Indicator
 *
 * Displays:
 * - Stochastic Slow %K (orange line)
 * - Horizontal lines at 20, 50, 80
 */

import { HorizontalLinesPlugin } from '/plugins/horizontal-lines/horizontal-lines.js';

/**
 * Create Stochastic Slow %K Pane
 * @param {Object} chart - Lightweight Charts instance
 * @param {Object} stochObvData - Stochastic data
 * @returns {Object} - Pane and series references
 */
export function createStochObvPane(chart, stochObvData, paneIndex = 2) {
    console.log('📊 Creating Stochastic Slow %K Indicator pane...');

    // Create Stochastic %K Line Series (Orange)
    const stochKSeries = chart.addSeries(
        LightweightCharts.LineSeries,
        {
            color: '#FF8C00',  // Dark Orange
            lineWidth: 1,  // Thinner line (half of 2)
            title: '',  // No title - we only have one line
            priceFormat: {
                type: 'price',
                precision: 2,
                minMove: 0.01,
            },
            lastValueVisible: true,
            crosshairMarkerVisible: true,
            priceLineVisible: false,
        },
        paneIndex
    );

    // Set data
    console.log('📊 Stochastic Slow %K Data:', stochObvData);

    if (stochObvData.stoch_k && stochObvData.stoch_k.length > 0) {
        stochKSeries.setData(stochObvData.stoch_k);
        console.log(`✅ Loaded ${stochObvData.stoch_k.length} Stochastic Slow %K values`);
    } else {
        console.warn('⚠️ No Stochastic Slow %K data available');
    }

    // Add horizontal lines at 20, 50, 80
    const horizontalLinesPlugin = new HorizontalLinesPlugin(
        stochKSeries,
        chart,
        [
            { price: 80, color: '#757575', lineWidth: 1.5, lineStyle: 2 },  // Upper Band (dashed)
            { price: 50, color: '#757575', lineWidth: 1.5, lineStyle: 2 },  // Middle Band (dashed)
            { price: 20, color: '#757575', lineWidth: 1.5, lineStyle: 2 },  // Lower Band (dashed)
        ],
        []  // No zones
    );

    console.log('✅ Stochastic Slow %K pane created');

    return {
        stochKSeries,
        horizontalLinesPlugin,
    };
}


/**
 * Update Stochastic/OBV with new data
 * @param {Object} stochObvIndicator - Stochastic/OBV indicator object
 * @param {Object} data - New data (can be single update or batch for lazy loading)
 */
export function updateStochObv(stochObvIndicator, data) {
    // Batch update for lazy loading - set all data at once
    if (data.stoch_k && Array.isArray(data.stoch_k) && stochObvIndicator.stochKSeries) {
        stochObvIndicator.stochKSeries.setData(data.stoch_k);
        console.log(`📊 Stochastic: Batch updated ${data.stoch_k.length} values`);
        return;
    }

    // Single update (realtime) - placeholder for now
    console.log('📊 Updating Stochastic/OBV with new candle:', data);
}

