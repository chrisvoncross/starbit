/**
 * Horizontal Lines with Zones Plugin for Lightweight Charts
 * Renders horizontal lines and colored zones that only extend to the chart pane edge
 */

class HorizontalLinesRenderer {
    constructor(zones, lines, chart, lastBarX) {
        this._zones = zones;
        this._lines = lines;
        this._chart = chart;
        this._lastBarX = lastBarX;
    }

    draw(target) {
        target.useMediaCoordinateSpace(scope => {
            const ctx = scope.context;

            // If we don't have the last bar X position, draw to full width
            const endX = this._lastBarX !== null ? this._lastBarX : scope.mediaSize.width;

            // Draw zones first (background)
            this._zones.forEach(zone => {
                if (zone.y1 !== null && zone.y2 !== null) {
                    ctx.fillStyle = zone.color;
                    const y1 = Math.min(zone.y1, zone.y2);
                    const y2 = Math.max(zone.y1, zone.y2);
                    const height = y2 - y1;
                    // Draw from left edge (0) to the last bar position
                    ctx.fillRect(0, y1, endX, height);
                }
            });

            // Draw lines on top (same width as zones)
            this._lines.forEach(line => {
                if (line.y !== null) {
                    ctx.strokeStyle = line.color;
                    ctx.lineWidth = line.width;

                    // Set line dash style if specified
                    // lineStyle: 0 = solid, 1 = dotted, 2 = dashed, 3 = large dashed, 4 = sparse dotted
                    if (line.style === 1) {
                        ctx.setLineDash([1, 1]);  // Dotted
                    } else if (line.style === 2) {
                        ctx.setLineDash([5, 5]);  // Dashed
                    } else if (line.style === 3) {
                        ctx.setLineDash([10, 5]);  // Large dashed
                    } else if (line.style === 4) {
                        ctx.setLineDash([1, 4]);  // Sparse dotted
                    } else {
                        ctx.setLineDash([]);  // Solid (default)
                    }

                    ctx.beginPath();
                    ctx.moveTo(0, line.y);
                    ctx.lineTo(endX, line.y);
                    ctx.stroke();

                    // Reset line dash
                    ctx.setLineDash([]);
                }
            });
        });
    }
}

class HorizontalLinesPaneView {
    constructor(levels, zones, series, chart) {
        this._levels = levels;
        this._zoneConfigs = zones;
        this._series = series;
        this._chart = chart;
        this._renderer = null;
    }

    update() {
        const lines = [];
        const zones = [];

        this._levels.forEach(level => {
            const y = this._series.priceToCoordinate(level.price);
            if (y !== null) {
                lines.push({
                    y: y,
                    color: level.color,
                    width: level.width,
                    style: level.lineStyle || 0  // Default to solid if not specified
                });
            }
        });

        this._zoneConfigs.forEach(zone => {
            const y1 = this._series.priceToCoordinate(zone.topPrice);
            const y2 = this._series.priceToCoordinate(zone.bottomPrice);
            if (y1 !== null && y2 !== null) {
                zones.push({ y1: y1, y2: y2, color: zone.color });
            }
        });

        // Get the X coordinate of the LAST DATA POINT in the series (not just visible range)
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

        this._renderer = new HorizontalLinesRenderer(zones, lines, this._chart, lastBarX);
    }

    renderer() {
        if (!this._renderer) {
            this.update();
        }
        return this._renderer;
    }
}

class HorizontalLinesPrimitive {
    constructor(levels, zones) {
        this._levels = levels;
        this._zones = zones;
        this._series = null;
        this._chart = null;
        this._paneView = null;
    }

    updateLevelsAndZones(levels, zones) {
        this._levels = levels;
        this._zones = zones;
        if (this._paneView) {
            this._paneView._levels = levels;
            this._paneView._zoneConfigs = zones;
            this._paneView.update();
        }
    }

    paneViews() {
        if (!this._paneView) return [];
        return [this._paneView];
    }

    updateAllViews() {
        if (this._paneView) {
            this._paneView.update();
        }
    }

    attached(params) {
        this._series = params.series;
        this._chart = params.chart;
        this._paneView = new HorizontalLinesPaneView(this._levels, this._zones, this._series, this._chart);
    }

    detached() {
        this._series = null;
        this._chart = null;
        this._paneView = null;
    }

    autoscaleInfo() {
        return null;
    }
}

export class HorizontalLinesPlugin {
    constructor(series, chart, levels = [], zones = []) {
        this._series = series;
        this._chart = chart;
        this._primitive = new HorizontalLinesPrimitive(levels, zones);
        this._series.attachPrimitive(this._primitive);
    }

    updateLevelsAndZones(levels, zones) {
        this._primitive.updateLevelsAndZones(levels, zones);
    }

    remove() {
        this._series.detachPrimitive(this._primitive);
    }
}