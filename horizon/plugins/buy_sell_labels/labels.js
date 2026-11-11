/**
 * Custom Labels Plugin for Lightweight Charts
 * 
 * Renders custom text labels (like TradingView's plotshape labels)
 * Used for Buy/Sell signals with styled boxes
 */

class LabelsPaneRenderer {
    constructor(viewData) {
        this._viewData = viewData;
    }

    draw() {}

    drawBackground(target) {
        const labels = this._viewData.labels;
        
        target.useBitmapCoordinateSpace(scope => {
            const ctx = scope.context;
            ctx.scale(scope.horizontalPixelRatio, scope.verticalPixelRatio);

            // Draw each label
            for (const label of labels) {
                this._drawLabel(ctx, label);
            }
        });
    }

    _drawLabel(ctx, label) {
        if (!label || !label.x || !label.y) return;

        const { x, y, text, bgColor, textColor, position, arrowSize } = label;

        // Set font
        ctx.font = 'bold 11px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Measure text
        const textMetrics = ctx.measureText(text);
        const textWidth = textMetrics.width;
        const padding = 6;
        const boxWidth = textWidth + padding * 2;
        const boxHeight = 26; // Increased height to match TradingView
        const arrowHeight = arrowSize || 6;
        const offset = 55; // Offset from bar (maximum spacing to match TradingView)

        // Draw based on position
        if (position === 'aboveBar') {
            // Sell label (blue speech bubble above bar with arrow pointing down)
            // Position: y - offset (center of box), arrow extends down from there
            const labelY = y - offset;
            this._drawSpeechBubble(ctx, x, labelY, boxWidth, boxHeight, arrowHeight, bgColor);
            // Draw text centered in the box
            ctx.fillStyle = textColor;
            ctx.fillText(text, x, labelY);
        } else {
            // Buy label (gray rectangle below bar with arrow pointing up)
            // Position: y + offset (center of box), arrow extends up from there
            const labelY = y + offset;
            this._drawRectangle(ctx, x, labelY, boxWidth, boxHeight, bgColor);
            // Draw text centered in the box
            ctx.fillStyle = textColor;
            ctx.fillText(text, x, labelY);
        }
    }

    _drawSpeechBubble(ctx, x, y, width, height, arrowHeight, color) {
        const radius = 3;
        const arrowWidth = 12; // Wider arrow
        const arrowTipRadius = 0.5; // Very subtle rounding

        ctx.fillStyle = color;
        ctx.beginPath();

        // Speech bubble pointing down (for SELL above bar)
        const top = y - height / 2;
        const bottom = y + height / 2;
        const left = x - width / 2;
        const right = x + width / 2;

        // Top left corner
        ctx.moveTo(left + radius, top);
        // Top right corner
        ctx.lineTo(right - radius, top);
        ctx.arcTo(right, top, right, top + radius, radius);
        // Right side
        ctx.lineTo(right, bottom - radius);
        ctx.arcTo(right, bottom, right - radius, bottom, radius);
        // Arrow (pointing down) - with very subtle rounded tip
        ctx.lineTo(x + arrowWidth / 2, bottom);
        // Very subtle curve to arrow tip
        ctx.quadraticCurveTo(x + arrowTipRadius, bottom + arrowHeight - arrowTipRadius,
                             x, bottom + arrowHeight);
        ctx.quadraticCurveTo(x - arrowTipRadius, bottom + arrowHeight - arrowTipRadius,
                             x - arrowWidth / 2, bottom);
        // Bottom left
        ctx.lineTo(left + radius, bottom);
        ctx.arcTo(left, bottom, left, bottom - radius, radius);
        // Left side
        ctx.lineTo(left, top + radius);
        ctx.arcTo(left, top, left + radius, top, radius);

        ctx.closePath();
        ctx.fill();
    }

    _drawRectangle(ctx, x, y, width, height, color) {
        const radius = 3;
        const arrowWidth = 12; // Wider arrow (matching sell)
        const arrowHeight = 6;
        const arrowTipRadius = 0.5; // Very subtle rounding
        const left = x - width / 2;
        const top = y - height / 2;

        ctx.fillStyle = color;
        ctx.beginPath();

        // Start at top-left corner
        ctx.moveTo(left + radius, top);

        // Top side to arrow start
        ctx.lineTo(x - arrowWidth / 2, top);

        // Arrow pointing up - with very subtle rounded tip
        // Very subtle curve to arrow tip
        ctx.quadraticCurveTo(x - arrowTipRadius, top - arrowHeight + arrowTipRadius,
                             x, top - arrowHeight);
        ctx.quadraticCurveTo(x + arrowTipRadius, top - arrowHeight + arrowTipRadius,
                             x + arrowWidth / 2, top);

        // Continue top side
        ctx.lineTo(left + width - radius, top);
        ctx.arcTo(left + width, top, left + width, top + radius, radius);

        // Right side
        ctx.lineTo(left + width, top + height - radius);
        ctx.arcTo(left + width, top + height, left + width - radius, top + height, radius);

        // Bottom side
        ctx.lineTo(left + radius, top + height);
        ctx.arcTo(left, top + height, left, top + height - radius, radius);

        // Left side
        ctx.lineTo(left, top + radius);
        ctx.arcTo(left, top, left + radius, top, radius);

        ctx.closePath();
        ctx.fill();
    }
}

class LabelsPaneView {
    constructor(source) {
        this._source = source;
    }

    update() {
        const series = this._source._series;
        const timeScale = this._source._chart.timeScale();
        
        // Convert all labels to screen coordinates
        const labels = [];
        
        for (const labelData of this._source._labels) {
            const x = timeScale.timeToCoordinate(labelData.time);
            if (x === null) continue;
            
            const y = series.priceToCoordinate(labelData.price);
            if (y === null) continue;
            
            labels.push({
                x,
                y,
                text: labelData.text,
                bgColor: labelData.bgColor,
                textColor: labelData.textColor,
                position: labelData.position,
                arrowSize: labelData.arrowSize,
            });
        }

        this._data = {
            labels
        };
    }

    renderer() {
        return new LabelsPaneRenderer(this._data);
    }
}

export class LabelsPlugin {
    constructor() {
        this._labels = [];
        this._paneViews = [new LabelsPaneView(this)];
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
     * Add a label
     * @param {number} time - Unix timestamp (seconds)
     * @param {number} price - Price level
     * @param {string} text - Label text
     * @param {string} position - 'aboveBar' or 'belowBar'
     * @param {string} bgColor - Background color
     * @param {string} textColor - Text color
     * @param {number} arrowSize - Arrow size (for speech bubbles)
     */
    addLabel(time, price, text, position, bgColor, textColor, arrowSize = 6) {
        this._labels.push({
            time,
            price,
            text,
            position,
            bgColor,
            textColor,
            arrowSize,
        });

        this.requestUpdate();
    }

    /**
     * Clear all labels
     */
    clearLabels() {
        this._labels = [];
        this.requestUpdate();
    }

    /**
     * Set multiple labels at once
     * @param {Array} labels - Array of label objects
     */
    setLabels(labels) {
        this._labels = labels;
        this.requestUpdate();
    }
}

