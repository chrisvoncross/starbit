/**
 * Universal Fill Areas Plugin for Lightweight Charts
 * 
 * Renders smooth filled areas between two lines
 * Can be used for: Bollinger Bands, NWE Bands, Keltner Channels, Ichimoku Cloud, etc.
 * 
 * Based on TradingView's BandsIndicator plugin example
 */

class FillAreasPaneRenderer {
    constructor(viewData) {
        this._viewData = viewData;
    }

    draw() {}

    drawBackground(target) {
        const zones = this._viewData.zones;
        
        target.useBitmapCoordinateSpace(scope => {
            const ctx = scope.context;
            ctx.scale(scope.horizontalPixelRatio, scope.verticalPixelRatio);

            // Draw each filled zone
            for (const zone of zones) {
                this._drawZone(ctx, zone.upperPoints, zone.lowerPoints, zone.fillColor);
            }

            // Draw boundary lines if enabled
            if (this._viewData.options.showLines) {
                for (const zone of zones) {
                    if (zone.showUpperLine) {
                        this._drawLine(ctx, zone.upperPoints, zone.lineColor, zone.lineWidth);
                    }
                    if (zone.showLowerLine) {
                        this._drawLine(ctx, zone.lowerPoints, zone.lineColor, zone.lineWidth);
                    }
                }
            }
        });
    }

    _drawZone(ctx, upperPoints, lowerPoints, fillColor) {
        if (!upperPoints || !lowerPoints || upperPoints.length === 0 || lowerPoints.length === 0) {
            return;
        }

        const region = new Path2D();
        
        // Start at first point of upper line
        region.moveTo(upperPoints[0].x, upperPoints[0].y);
        
        // Draw upper line (left to right)
        for (const point of upperPoints) {
            region.lineTo(point.x, point.y);
        }
        
        // Connect to lower line at the end
        const lastIndex = lowerPoints.length - 1;
        region.lineTo(lowerPoints[lastIndex].x, lowerPoints[lastIndex].y);
        
        // Draw lower line (right to left)
        for (let i = lastIndex - 1; i >= 0; i--) {
            region.lineTo(lowerPoints[i].x, lowerPoints[i].y);
        }
        
        // Close the path
        region.closePath();
        
        // Fill the region
        ctx.fillStyle = fillColor;
        ctx.fill(region);
    }

    _drawLine(ctx, points, color, width) {
        if (!points || points.length === 0) return;

        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
        }
        
        ctx.stroke();
    }
}

class FillAreasPaneView {
    constructor(source) {
        this._source = source;
    }

    update() {
        const series = this._source._series;
        const timeScale = this._source._chart.timeScale();
        
        // Convert all zones to screen coordinates
        const zones = [];
        
        for (const zone of this._source._zones) {
            const upperPoints = [];
            const lowerPoints = [];
            
            // Convert upper line data to coordinates
            for (let i = 0; i < zone.upperData.length; i++) {
                const x = timeScale.timeToCoordinate(zone.upperData[i].time);
                if (x === null) continue;
                
                const y = series.priceToCoordinate(zone.upperData[i].value);
                if (y !== null) {
                    upperPoints.push({ x, y });
                }
            }
            
            // Convert lower line data to coordinates
            for (let i = 0; i < zone.lowerData.length; i++) {
                const x = timeScale.timeToCoordinate(zone.lowerData[i].time);
                if (x === null) continue;
                
                const y = series.priceToCoordinate(zone.lowerData[i].value);
                if (y !== null) {
                    lowerPoints.push({ x, y });
                }
            }
            
            zones.push({
                upperPoints,
                lowerPoints,
                fillColor: zone.fillColor,
                lineColor: zone.lineColor,
                lineWidth: zone.lineWidth,
                showUpperLine: zone.showUpperLine,
                showLowerLine: zone.showLowerLine,
            });
        }

        this._data = {
            zones,
            options: this._source._options
        };
    }

    renderer() {
        return new FillAreasPaneRenderer(this._data);
    }
}

export class FillAreasPlugin {
    constructor(options = {}) {
        this._options = {
            showLines: options.showLines !== undefined ? options.showLines : true,
        };

        this._zones = [];
        this._paneViews = [new FillAreasPaneView(this)];
    }

    updateAllViews() {
        this._paneViews.forEach(pv => pv.update());
    }

    paneViews() {
        return this._paneViews;
    }

    attached(param) {
        this._chart = param.chart;
        this._series = param.series;
        this._series.subscribeDataChanged(this._handleDataChanged);
        this.requestUpdate();
    }

    detached() {
        this._series.unsubscribeDataChanged(this._handleDataChanged);
    }

    _handleDataChanged = () => {
        this.requestUpdate();
    }

    requestUpdate() {
        if (this._chart) {
            this.updateAllViews();
        }
    }

    /**
     * Add a filled zone between two lines
     * @param {string} id - Unique identifier for this zone
     * @param {Array} upperData - Array of {time, value} for upper line
     * @param {Array} lowerData - Array of {time, value} for lower line
     * @param {string} fillColor - Fill color (rgba)
     * @param {Object} options - Optional: lineColor, lineWidth, showUpperLine, showLowerLine
     */
    addZone(id, upperData, lowerData, fillColor, options = {}) {
        // Remove existing zone with same ID
        this._zones = this._zones.filter(z => z.id !== id);

        // Add new zone
        this._zones.push({
            id,
            upperData,
            lowerData,
            fillColor,
            lineColor: options.lineColor || 'rgba(120, 123, 134, 0.3)',
            lineWidth: options.lineWidth || 1,
            showUpperLine: options.showUpperLine !== undefined ? options.showUpperLine : false,
            showLowerLine: options.showLowerLine !== undefined ? options.showLowerLine : false,
        });

        this.requestUpdate();
    }

    /**
     * Remove a zone by ID
     */
    removeZone(id) {
        this._zones = this._zones.filter(z => z.id !== id);
        this.requestUpdate();
    }

    /**
     * Clear all zones
     */
    clearZones() {
        this._zones = [];
        this.requestUpdate();
    }

    /**
     * Update plugin options
     */
    applyOptions(options) {
        this._options = { ...this._options, ...options };
        this.requestUpdate();
    }
}

