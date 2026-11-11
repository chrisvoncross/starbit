/**
 * Price Level Labels Plugin for Lightweight Charts
 *
 * Renders floating price labels in the pane (similar to TradingView labels)
 * Shows actual price values at specific Z-Score levels
 */

/**
 * Renderer for price level labels
 */
class PriceLevelLabelsRenderer {
    constructor(levels, formatPrice, series, chart) {
        this._levels = levels;
        this._formatPrice = formatPrice;
        this._series = series;
        this._chart = chart;
    }

    /**
     * Draw the labels on the canvas
     */
    draw(target) {
        target.useMediaCoordinateSpace(scope => {
            const ctx = scope.context;

            // Get the X coordinate of the LAST DATA POINT in the series
            // (Same logic as horizontal-lines plugin - labels stick to where zones end)
            const timeScale = this._chart.timeScale();
            let lastBarX = null;

            // Get the actual data from the series
            const seriesData = this._series.data();

            if (seriesData && seriesData.length > 0) {
                // Get the time of the last data point
                const lastDataPoint = seriesData[seriesData.length - 1];
                const lastTime = lastDataPoint.time;

                // Convert time to X coordinate
                lastBarX = timeScale.timeToCoordinate(lastTime);
            }

            // Position labels EXACTLY at the zone end (no offset)
            // Text is right-aligned, so it will end exactly at lastBarX
            const rightEdgeX = (lastBarX !== null ? lastBarX : scope.mediaSize.width);

            this._levels.forEach(level => {
                // IMPORTANT: Calculate Y coordinate HERE, not in updateLevels!
                // This ensures the Y position is recalculated on every draw, staying fixed to the price level
                const y = this._series.priceToCoordinate(level.zScore);
                if (y === null) return; // Skip if not visible

                const text = this._formatPrice(level.price);
                if (!text) return;

                // Set font (in media coordinates)
                ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, "Helvetica Neue", sans-serif';
                ctx.textAlign = 'left';  // LEFT-aligned so text starts at the zone end and goes RIGHT
                ctx.textBaseline = 'middle';

                // Draw text starting at the zone end (with small padding to the right)
                const padding = 5;
                ctx.fillStyle = level.color;
                ctx.fillText(text, rightEdgeX + padding, y);
            });
        });
    }
}

/**
 * Pane View for price level labels
 */
class PriceLevelLabelsPaneView {
    constructor(levels, formatPrice, series, chart) {
        this._levels = levels;
        this._formatPrice = formatPrice;
        this._series = series;
        this._chart = chart;
        this._renderer = new PriceLevelLabelsRenderer(levels, formatPrice, series, chart);
    }

    /**
     * Update the levels
     */
    update(levels) {
        this._levels = levels;
        this._renderer = new PriceLevelLabelsRenderer(levels, this._formatPrice, this._series, this._chart);
    }

    /**
     * Return the renderer
     */
    renderer() {
        return this._renderer;
    }
}

/**
 * Price Level Labels Primitive (implements ISeriesPrimitive)
 */
class PriceLevelLabelsPrimitive {
    constructor(levels, formatPrice) {
        this._levels = levels;
        this._formatPrice = formatPrice;
        this._series = null;
        this._chart = null;
        this._paneView = null;
    }

    /**
     * Update the price levels
     */
    updateLevels(levels) {
        this._levels = levels;
        if (this._paneView) {
            this._paneView.update(levels);
        }
    }

    /**
     * Return pane views (required by ISeriesPrimitive)
     */
    paneViews() {
        if (!this._paneView) return [];
        return [this._paneView];
    }

    /**
     * Update all views (required by ISeriesPrimitive)
     */
    updateAllViews() {
        if (this._paneView) {
            this._paneView.update(this._levels);
        }
    }

    /**
     * Attached lifecycle method (optional)
     */
    attached(params) {
        this._series = params.series;
        this._chart = params.chart;
        this._paneView = new PriceLevelLabelsPaneView(this._levels, this._formatPrice, this._series, this._chart);
    }

    /**
     * Detached lifecycle method (optional)
     */
    detached() {
        this._series = null;
        this._chart = null;
        this._paneView = null;
    }

    /**
     * Autoscale info - return null to not affect autoscaling
     */
    autoscaleInfo() {
        // Return null so the labels don't affect the chart's autoscaling
        return null;
    }
}

/**
 * Price Level Labels Plugin
 * Manages the primitive and provides update methods
 */
export class PriceLevelLabelsPlugin {
    constructor(series, formatPrice, chart) {
        this._series = series;
        this._formatPrice = formatPrice;
        this._chart = chart;
        this._primitive = new PriceLevelLabelsPrimitive([], formatPrice);
        this._series.attachPrimitive(this._primitive);
    }

    /**
     * Update the price levels to display
     * @param {Array} levels - Array of {zScore, price, color}
     */
    updateLevels(levels) {
        this._primitive.updateLevels(levels);
        // Trigger a redraw by updating the series with a dummy point
        // This forces the primitive to redraw
        const lastData = this._series.data();
        if (lastData && lastData.length > 0) {
            const last = lastData[lastData.length - 1];
            this._series.update(last);
        }
    }

    /**
     * Remove the plugin
     */
    remove() {
        this._series.detachPrimitive(this._primitive);
    }
}

