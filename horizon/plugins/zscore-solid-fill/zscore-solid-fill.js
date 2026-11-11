/**
 * Z-Score Solid Fill Plugin
 * 
 * Renders 100% SOLID fill for Z-Score indicator
 * - Green fill above 0
 * - Red fill below 0
 * - White line for Z-Score values
 */

class ZScoreSolidFillPaneRenderer {
    constructor(viewData) {
        this._viewData = viewData;
    }

    draw(target) {
        const data = this._viewData.data;
        const options = this._viewData.options;

        if (!data || !data.segments || data.segments.length === 0 || options._hidden) return;

        target.useBitmapCoordinateSpace(scope => {
            const ctx = scope.context;
            ctx.scale(scope.horizontalPixelRatio, scope.verticalPixelRatio);

            // Draw each segment separately (no connection between segments)
            for (const segment of data.segments) {
                this._drawSegment(ctx, segment, options);
            }
        });
    }

    _drawSegment(ctx, segment, options) {
        if (!segment.points || segment.points.length === 0) return;

        const fillColor = segment.type === 'positive' ? options.positiveColor : options.negativeColor;

        const region = new Path2D();

        // Start at first point on z-score line
        region.moveTo(segment.points[0].x, segment.points[0].yValue);

        // Draw along z-score line to the right
        for (const point of segment.points) {
            region.lineTo(point.x, point.yValue);
        }

        // Draw back along the zero line (right to left)
        for (let i = segment.points.length - 1; i >= 0; i--) {
            region.lineTo(segment.points[i].x, segment.points[i].yZero);
        }

        // Close the path
        region.closePath();

        // Fill the segment with 100% SOLID color
        ctx.fillStyle = fillColor;
        ctx.fill(region);
    }
}

class ZScoreSolidFillPaneView {
    constructor(source) {
        this._source = source;
    }

    update() {
        const series = this._source._series;
        const chart = this._source._chart;
        const timeScale = chart.timeScale();

        const zscoreData = this._source._zscoreData;

        if (!zscoreData || zscoreData.length === 0) {
            this._data = { data: { segments: [] } };
            return;
        }

        // Create separate segments for positive and negative regions
        const segments = [];
        let currentSegment = null;
        let currentType = null;

        // Convert data to screen coordinates and create segments
        for (let i = 0; i < zscoreData.length; i++) {
            const x = timeScale.timeToCoordinate(zscoreData[i].time);
            if (x === null) continue;

            const value = zscoreData[i].value;
            const yZero = series.priceToCoordinate(0);
            const yValue = series.priceToCoordinate(value);

            if (yZero === null || yValue === null) continue;

            // Determine segment type (positive or negative)
            const segmentType = value > 0 ? 'positive' : (value < 0 ? 'negative' : null);

            // Check if we need to start a new segment (type changed or no current segment)
            if (currentType !== segmentType && segmentType !== null) {
                // If we have a previous segment and the type changed, we need to handle the crossing
                if (currentSegment && i > 0) {
                    // Get previous value to calculate intersection with zero line
                    const prevValue = zscoreData[i - 1].value;
                    const prevX = timeScale.timeToCoordinate(zscoreData[i - 1].time);

                    // Calculate intersection point with zero line (linear interpolation)
                    if (prevX !== null && ((prevValue > 0 && value < 0) || (prevValue < 0 && value > 0))) {
                        // Linear interpolation to find x coordinate where value crosses zero
                        const ratio = Math.abs(prevValue) / (Math.abs(prevValue) + Math.abs(value));
                        const intersectionX = prevX + (x - prevX) * ratio;

                        // Add intersection point to current segment (closes it at zero line)
                        currentSegment.points.push({
                            x: intersectionX,
                            yValue: yZero,
                            yZero: yZero
                        });
                    }

                    // Save previous segment
                    segments.push(currentSegment);
                }

                // Start new segment
                currentSegment = {
                    type: segmentType,
                    points: []
                };
                currentType = segmentType;

                // If this is a crossing point, add the intersection as first point of new segment
                if (i > 0) {
                    const prevValue = zscoreData[i - 1].value;
                    const prevX = timeScale.timeToCoordinate(zscoreData[i - 1].time);

                    if (prevX !== null && ((prevValue > 0 && value < 0) || (prevValue < 0 && value > 0))) {
                        const ratio = Math.abs(prevValue) / (Math.abs(prevValue) + Math.abs(value));
                        const intersectionX = prevX + (x - prevX) * ratio;

                        currentSegment.points.push({
                            x: intersectionX,
                            yValue: yZero,
                            yZero: yZero
                        });
                    }
                }
            }

            // Add point to current segment (skip zero values)
            if (segmentType !== null) {
                currentSegment.points.push({
                    x: x,
                    yValue: yValue,
                    yZero: yZero
                });
            }
        }

        // Don't forget the last segment
        if (currentSegment) {
            segments.push(currentSegment);
        }

        this._data = {
            data: { segments },
            options: { ...this._source._options, _hidden: !this._source.visible }
        };
    }

    renderer() {
        return new ZScoreSolidFillPaneRenderer(this._data);
    }
}

export class ZScoreSolidFillPlugin {
    constructor(zscoreData, options = {}) {
        this._zscoreData = zscoreData;
        this._options = {
            positiveColor: options.positiveColor || '#4CAF50',  // Green
            negativeColor: options.negativeColor || '#FF5252',  // Red
        };

        this._paneViews = [new ZScoreSolidFillPaneView(this)];
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

    updateData(zscoreData) {
        this._zscoreData = zscoreData;
        this.requestUpdate();
    }

    setVisible(visible) {
        this._visible = visible;
        this.requestUpdate();
    }

    get visible() {
        return this._visible !== false;
    }
}

