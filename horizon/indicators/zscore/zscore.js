/**
 * Z-Score Probability Indicator
 *
 * Bottom indicator showing Z-Score distribution with probability zones
 * Based on TradingView indicator by Steversteves
 */

import { PriceLevelLabelsPlugin } from '/plugins/price-level-labels/price-level-labels.js';
import { HorizontalLinesPlugin } from '/plugins/horizontal-lines/horizontal-lines.js';
import { ZScoreTablePlugin } from '/plugins/zscore-table/zscore-table-plugin.js';
import { ZScoreSolidFillPlugin } from '/plugins/zscore-solid-fill/zscore-solid-fill.js';

/**
 * Z-Score Configuration
 */
export const ZSCORE_CONFIG = {
    // Colors for probability zones (matching TradingView)
    colors: {
        // Filled areas
        greenFill: 'rgba(38, 166, 154, 0.25)',    // -1 to +1 SD (68.3%)
        yellowFill: 'rgba(255, 235, 59, 0.25)',   // -2 to -1 and +1 to +2 SD (27.2%)
        redFill: 'rgba(239, 83, 80, 0.25)',       // -3 to -2 and +2 to +3 SD (4.3%)
        
        // Z-Score line (area plot)
        green: '#26a69a',
        red: '#ef5350',
        
        // Z-Score SMA line
        smaLine: '#ffffff',
        
        // Horizontal lines
        horizontalLines: 'rgba(120, 123, 134, 0.3)',
        
        // Price labels
        labelBg: 'rgba(255, 255, 255, 0.0)',  // Transparent background
        labelText: '#ffffff',
    },
    
    // Line styling
    lineWidth: {
        sma: 2,
        horizontal: 1,
        zscore: 1,
    },
    
    // Standard deviation levels
    levels: {
        threeSD: 3,
        twoSD: 2,
        oneSD: 1,
        neutral: 0,
        negOneSD: -1,
        negTwoSD: -2,
        negThreeSD: -3,
    },
    
    // Z-Table probabilities
    zTable: {
        '-3': { probability: '0.0013', color: '#ef5350' },
        '-2': { probability: '0.0228', color: '#ffeb3b' },
        '-1': { probability: '0.1600', color: '#26a69a' },
        '0':  { probability: '0.5000', color: '#26a69a' },
        '1':  { probability: '0.8500', color: '#26a69a' },
        '2':  { probability: '0.9800', color: '#ffeb3b' },
        '3':  { probability: '0.9998', color: '#ef5350' },
    },
};

/**
 * Create Z-Score Indicator Pane
 *
 * @param {Chart} chart - Lightweight Charts instance
 * @param {Object} zscoreData - Z-Score data from server
 * @param {FillAreasPlugin} FillAreasPluginClass - Fill Areas Plugin class
 * @returns {Object} - { pane, series, priceLines }
 */
export async function createZScorePane(chart, zscoreData, FillAreasPluginClass, paneIndex = 1) {
    console.log('📊 Creating Z-Score Probability Indicator pane...');

    // Create base series for fill areas plugin (invisible)
    const baseSeries = chart.addSeries(
        LightweightCharts.LineSeries,
        {
            color: 'transparent',
            lineWidth: 0,
            priceFormat: {
                type: 'custom',
                formatter: (price) => price.toFixed(2),
            },
        },
        paneIndex
    );

    // Set base data (neutral line at 0)
    const constantData = zscoreData.zscore.map(point => ({ time: point.time, value: 0 }));
    baseSeries.setData(constantData);

    // Create Fill Areas Plugin for Z-Score pane
    const zscoreFillAreas = new FillAreasPluginClass({ showLines: false });
    baseSeries.attachPrimitive(zscoreFillAreas);

    // Create boundary line series for probability zones
    const oneSDUpper = chart.addSeries(LightweightCharts.LineSeries, { color: 'transparent', lineWidth: 0 }, paneIndex);
    const twoSDUpper = chart.addSeries(LightweightCharts.LineSeries, { color: 'transparent', lineWidth: 0 }, paneIndex);
    const threeSDUpper = chart.addSeries(LightweightCharts.LineSeries, { color: 'transparent', lineWidth: 0 }, paneIndex);
    const oneSDLower = chart.addSeries(LightweightCharts.LineSeries, { color: 'transparent', lineWidth: 0 }, paneIndex);
    const twoSDLower = chart.addSeries(LightweightCharts.LineSeries, { color: 'transparent', lineWidth: 0 }, paneIndex);
    const threeSDLower = chart.addSeries(LightweightCharts.LineSeries, { color: 'transparent', lineWidth: 0 }, paneIndex);

    // Set data for boundary lines
    oneSDUpper.setData(constantData.map(p => ({ ...p, value: 1 })));
    twoSDUpper.setData(constantData.map(p => ({ ...p, value: 2 })));
    threeSDUpper.setData(constantData.map(p => ({ ...p, value: 3 })));
    oneSDLower.setData(constantData.map(p => ({ ...p, value: -1 })));
    twoSDLower.setData(constantData.map(p => ({ ...p, value: -2 })));
    threeSDLower.setData(constantData.map(p => ({ ...p, value: -3 })));

    // Add probability zones using Fill Areas Plugin
    // Green zones (68.3% probability: -1σ to +1σ)
    zscoreFillAreas.addZone('zscore_green_upper', baseSeries, oneSDUpper, ZSCORE_CONFIG.colors.greenFill, { showLines: false });
    zscoreFillAreas.addZone('zscore_green_lower', oneSDLower, baseSeries, ZSCORE_CONFIG.colors.greenFill, { showLines: false });

    // Yellow zones (27.2% probability: -2σ to -1σ and +1σ to +2σ)
    zscoreFillAreas.addZone('zscore_yellow_upper', oneSDUpper, twoSDUpper, ZSCORE_CONFIG.colors.yellowFill, { showLines: false });
    zscoreFillAreas.addZone('zscore_yellow_lower', twoSDLower, oneSDLower, ZSCORE_CONFIG.colors.yellowFill, { showLines: false });

    // Red zones (4.3% probability: -3σ to -2σ and +2σ to +3σ)
    zscoreFillAreas.addZone('zscore_red_upper', twoSDUpper, threeSDUpper, ZSCORE_CONFIG.colors.redFill, { showLines: false });
    zscoreFillAreas.addZone('zscore_red_lower', threeSDLower, twoSDLower, ZSCORE_CONFIG.colors.redFill, { showLines: false });

    // ========================================
    // CREATE SEPARATE SERIES FOR SOLID FILL (so it renders AFTER probability zones)
    // ========================================
    const solidFillBaseSeries = chart.addSeries(
        LightweightCharts.LineSeries,
        {
            color: 'transparent',
            lineWidth: 0,
            priceFormat: {
                type: 'custom',
                formatter: (price) => price.toFixed(2),
            },
        },
        paneIndex
    );
    solidFillBaseSeries.setData(constantData);

    // ========================================
    // ADD 100% SOLID Z-SCORE FILL (Custom Plugin)
    // ========================================
    const zscoreSolidFill = new ZScoreSolidFillPlugin(zscoreData.zscore, {
        positiveColor: '#4CAF50',  // Green (100% solid)
        negativeColor: '#FF5252',  // Red (100% solid)
    });
    solidFillBaseSeries.attachPrimitive(zscoreSolidFill);

    // ========================================
    // NO Z-SCORE LINE - Just the solid fill!
    // ========================================
    // Use solidFillBaseSeries as the main series for plugins
    const zscoreLineSeries = solidFillBaseSeries;

    // Create Z-Score SMA line series in same pane
    const zscoreSMASeries = chart.addSeries(
        LightweightCharts.LineSeries,
        {
            color: ZSCORE_CONFIG.colors.smaLine,
            lineWidth: ZSCORE_CONFIG.lineWidth.sma,
            priceFormat: {
                type: 'custom',
                formatter: (price) => price.toFixed(2),
            },
            // Hide last value label (no current value on Y-axis)
            lastValueVisible: false,
            // Hide crosshair price line (no horizontal line when hovering)
            crosshairMarkerVisible: false,
            priceLineVisible: false,
        },
        paneIndex
    );

    // Set Z-Score long-window data (zscore_close_50)
    if (zscoreData.zscore_long) {
        zscoreSMASeries.setData(zscoreData.zscore_long);
    } else if (zscoreData.zscore_sma) {
        zscoreSMASeries.setData(zscoreData.zscore_sma);
    }

    // Create horizontal lines and zones plugin
    const horizontalLinesPlugin = createHorizontalLinesAndZones(baseSeries, chart);

    // Create price level labels plugin
    const priceLevelLabels = new PriceLevelLabelsPlugin(baseSeries, formatPrice, chart);

    // Initialize labels with current price levels if available
    if (zscoreData.price_levels && zscoreData.price_levels.length > 0) {
        const latestPriceLevels = zscoreData.price_levels[zscoreData.price_levels.length - 1];
        updatePriceLevelLabels(priceLevelLabels, latestPriceLevels);
    }

    // Add Z-Score Table Plugin
    const zscoreTablePlugin = new ZScoreTablePlugin();
    zscoreLineSeries.attachPrimitive(zscoreTablePlugin);

    // Initialize Z-Score Table with latest value
    if (zscoreData.zscore && zscoreData.zscore.length > 0) {
        const latestZScore = zscoreData.zscore[zscoreData.zscore.length - 1];
        zscoreTablePlugin.updateZScore(latestZScore.value);
    }

    console.log('✅ Z-Score pane created with probability zones and price labels');

    return {
        zscoreAreaSeries: zscoreLineSeries,
        baseSeries,
        zscoreSMASeries,
        horizontalLinesPlugin,
        zscoreFillAreas,
        priceLevelLabels,
        zscoreTablePlugin,
        zscoreSolidFill,  // For lazy loading updates
        // All series in this pane (needed for removeSeries/destroy)
        _allSeries: [baseSeries, oneSDUpper, twoSDUpper, threeSDUpper, oneSDLower, twoSDLower, threeSDLower, solidFillBaseSeries, zscoreSMASeries],
    };
}

/**
 * Format price for display (auto-detect decimal places needed)
 */
function formatPrice(price) {
    if (price === undefined || price === null) return '';

    // For very small numbers (< 0.01), use 8 decimal places for crypto
    if (Math.abs(price) < 0.01) {
        return price.toFixed(8);
    } else if (Math.abs(price) < 1) {
        return price.toFixed(4);
    } else {
        return price.toFixed(2);
    }
}

/**
 * Update price level labels with current price levels
 */
function updatePriceLevelLabels(priceLevelLabels, priceLevels) {
    if (!priceLevelLabels || !priceLevels) return;

    const levels = [
        { zScore: 3, price: priceLevels.threeSD, color: '#ffffff' },
        { zScore: 2, price: priceLevels.twoSD, color: '#ffffff' },
        { zScore: 1, price: priceLevels.oneSD, color: '#ffffff' },
        { zScore: 0, price: priceLevels.neutral, color: '#ffffff' },
        { zScore: -1, price: priceLevels.negOneSD, color: '#ffffff' },
        { zScore: -2, price: priceLevels.negTwoSD, color: '#ffffff' },
        { zScore: -3, price: priceLevels.negThreeSD, color: '#ffffff' },
    ];

    priceLevelLabels.updateLevels(levels);
}

/**
 * Create horizontal lines and colored zones using the custom plugin
 * Lines and zones only extend to the chart pane edge (not into price scale)
 */
function createHorizontalLinesAndZones(series, chart) {
    // Define the horizontal line levels
    const levels = [
        { price: -3, color: ZSCORE_CONFIG.colors.horizontalLines, width: ZSCORE_CONFIG.lineWidth.horizontal },
        { price: -2, color: ZSCORE_CONFIG.colors.horizontalLines, width: ZSCORE_CONFIG.lineWidth.horizontal },
        { price: -1, color: ZSCORE_CONFIG.colors.horizontalLines, width: ZSCORE_CONFIG.lineWidth.horizontal },
        { price: 0, color: ZSCORE_CONFIG.colors.horizontalLines, width: ZSCORE_CONFIG.lineWidth.horizontal },
        { price: 1, color: ZSCORE_CONFIG.colors.horizontalLines, width: ZSCORE_CONFIG.lineWidth.horizontal },
        { price: 2, color: ZSCORE_CONFIG.colors.horizontalLines, width: ZSCORE_CONFIG.lineWidth.horizontal },
        { price: 3, color: ZSCORE_CONFIG.colors.horizontalLines, width: ZSCORE_CONFIG.lineWidth.horizontal },
    ];

    // Define the colored zones (matching TradingView probability zones)
    const zones = [
        // Red zones (extreme)
        { topPrice: -2, bottomPrice: -3, color: ZSCORE_CONFIG.colors.redFill },
        { topPrice: 3, bottomPrice: 2, color: ZSCORE_CONFIG.colors.redFill },

        // Yellow zones (warning)
        { topPrice: -1, bottomPrice: -2, color: ZSCORE_CONFIG.colors.yellowFill },
        { topPrice: 2, bottomPrice: 1, color: ZSCORE_CONFIG.colors.yellowFill },

        // Green zones (normal)
        { topPrice: 0, bottomPrice: -1, color: ZSCORE_CONFIG.colors.greenFill },
        { topPrice: 1, bottomPrice: 0, color: ZSCORE_CONFIG.colors.greenFill },
    ];

    // Create and return the plugin
    return new HorizontalLinesPlugin(series, chart, levels, zones);
}

/**
 * Update Z-Score data with new candle (single update)
 */
export function updateZScoreData(indicator, newData) {
    // For batch updates (lazy loading), use setData instead of update
    if (Array.isArray(newData.zscore)) {
        // Batch update - set all data at once
        if (indicator.zscoreAreaSeries) {
            indicator.zscoreAreaSeries.setData(newData.zscore);
        }
        if (indicator.zscoreSMASeries && (newData.zscore_long || newData.zscore_sma)) {
            indicator.zscoreSMASeries.setData(newData.zscore_long || newData.zscore_sma);
        }
        if (indicator.baseSeries && newData.zscore.length > 0) {
            const baseData = newData.zscore.map(d => ({ time: d.time, value: 0 }));
            indicator.baseSeries.setData(baseData);
        }
        // Update the solid fill plugin with new data (critical for lazy loading!)
        if (indicator.zscoreSolidFill) {
            indicator.zscoreSolidFill.updateData(newData.zscore);
        }
        console.log(`📊 Z-Score: Batch updated ${newData.zscore.length} values`);
        return;
    }

    // Single update (realtime)
    if (newData.zscore && indicator.zscoreAreaSeries) {
        indicator.zscoreAreaSeries.update(newData.zscore);
        updateZScoreTable(newData.zscore.value);
    }

    if ((newData.zscore_long || newData.zscore_sma) && indicator.zscoreSMASeries) {
        indicator.zscoreSMASeries.update(newData.zscore_long || newData.zscore_sma);
    }

    if (newData.zscore && indicator.baseSeries) {
        indicator.baseSeries.update({ time: newData.zscore.time, value: 0 });
    }

    if (newData.price_levels && indicator.priceLevelLabels) {
        updatePriceLevelLabels(indicator.priceLevelLabels, newData.price_levels);
    }
}

