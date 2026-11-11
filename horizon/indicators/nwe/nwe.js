/**
 * NWE (Nadaraya-Watson Envelope) Bands Indicator
 * 
 * REAL Nadaraya-Watson Envelope via QuadFactor kernel regression (h=8, α=8, x₀=25)
 * + kernel ATR (RMA 60). 6 bands, 4 filled zones, 1:1 TradingView match.
 * 
 * Reads GPU-computed NWE features from Lance datasets via LanceDataProvider.
 * Features reconstructed from normalized MegaFusion GPU features (float64 precision).
 * 
 * 4 Zones (TradingView-style):
 *   - Outer upper (dark red):  upper_far  ↔ upper_avg
 *   - Inner upper (light red): upper_avg  ↔ upper_near
 *   - Inner lower (light green): lower_near ↔ lower_avg
 *   - Outer lower (dark green):  lower_avg  ↔ lower_far
 *   + Kernel regression line (yhat/mid_band)
 */

/**
 * NWE Bands Configuration
 */
export const NWE_CONFIG = {
    colors: {
        outerUpper: 'rgba(239, 83, 80, 0.20)',    // Dark red: upper_far → upper_avg
        innerUpper: 'rgba(239, 83, 80, 0.35)',    // Light red: upper_avg → upper_near
        innerLower: 'rgba(38, 166, 154, 0.35)',   // Light green: lower_near → lower_avg
        outerLower: 'rgba(38, 166, 154, 0.20)',   // Dark green: lower_avg → lower_far
    },
    lineColor: 'rgba(120, 123, 134, 0.3)',
    lineWidth: 1,
    showLines: false,
};

/**
 * Setup NWE Bands using Fill Areas Plugin
 * 
 * @param {FillAreasPlugin} fillAreasPlugin - The Fill Areas Plugin instance
 * @param {Object} nweData - NWE data from Lance with 6 bands + mid_band
 */
export function setupNWEBands(fillAreasPlugin, nweData) {
    fillAreasPlugin.clearZones();

    const lineOpts = {
        lineColor: NWE_CONFIG.lineColor,
        lineWidth: NWE_CONFIG.lineWidth,
        showUpperLine: NWE_CONFIG.showLines,
        showLowerLine: NWE_CONFIG.showLines,
    };

    // 4-zone rendering (if all 6 bands available)
    const has6 = nweData.upper_far && nweData.upper_avg && nweData.upper_near
              && nweData.lower_near && nweData.lower_avg && nweData.lower_far;

    if (has6 && nweData.upper_far.length > 0) {
        // Outer upper (dark red): upper_far → upper_avg
        fillAreasPlugin.addZone('nwe_outer_upper', nweData.upper_far, nweData.upper_avg,
            NWE_CONFIG.colors.outerUpper, lineOpts);

        // Inner upper (light red): upper_avg → upper_near
        fillAreasPlugin.addZone('nwe_inner_upper', nweData.upper_avg, nweData.upper_near,
            NWE_CONFIG.colors.innerUpper, lineOpts);

        // Inner lower (light green): lower_near → lower_avg
        fillAreasPlugin.addZone('nwe_inner_lower', nweData.lower_near, nweData.lower_avg,
            NWE_CONFIG.colors.innerLower, lineOpts);

        // Outer lower (dark green): lower_avg → lower_far
        fillAreasPlugin.addZone('nwe_outer_lower', nweData.lower_avg, nweData.lower_far,
            NWE_CONFIG.colors.outerLower, lineOpts);
    } else if (nweData.upper_band && nweData.mid_band && nweData.lower_band) {
        // Legacy 2-zone fallback
        if (nweData.upper_band.length > 0) {
            fillAreasPlugin.addZone('nwe_upper', nweData.upper_band, nweData.mid_band,
                NWE_CONFIG.colors.innerUpper, lineOpts);
        }
        if (nweData.lower_band.length > 0) {
            fillAreasPlugin.addZone('nwe_lower', nweData.mid_band, nweData.lower_band,
                NWE_CONFIG.colors.innerLower, lineOpts);
        }
    }
}

/**
 * Toggle NWE Bands visibility
 * 
 * @param {FillAreasPlugin} fillAreasPlugin - The Fill Areas Plugin instance
 * @param {boolean} visible - Show or hide NWE bands
 */
export function toggleNWEBands(fillAreasPlugin, visible) {
    if (!visible) {
        fillAreasPlugin.removeZone('nwe_outer_upper');
        fillAreasPlugin.removeZone('nwe_inner_upper');
        fillAreasPlugin.removeZone('nwe_inner_lower');
        fillAreasPlugin.removeZone('nwe_outer_lower');
        // Legacy
        fillAreasPlugin.removeZone('nwe_upper');
        fillAreasPlugin.removeZone('nwe_lower');
    }
}

/**
 * Update NWE Bands colors
 * 
 * @param {Object} colors - New colors
 */
export function updateNWEColors(colors) {
    Object.assign(NWE_CONFIG.colors, colors);
}
