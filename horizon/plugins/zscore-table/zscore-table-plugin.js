/**
 * Z-Score Probability Table Plugin
 * 
 * Renders a probability table at the bottom of the Z-Score pane
 */

export class ZScoreTablePlugin {
    constructor() {
        this._currentZScore = 0;
        this._visible = true;
        
        // Table configuration
        this._config = {
            zScores: [-3, -2, -1, 0, 1, 2, 3],
            probabilities: ['0.0013', '0.0228', '0.1600', '0.5000', '0.8400', '0.9800', '0.9987'],
            colors: {
                red: '#ff5252',
                yellow: '#ffeb3b',
                green: '#4caf50',
                activeCell: '#9a9ca4',  // Gray for active Z-Score cell
                headerBg: '#ffffff',    // White background for Z-Score row
                textBlack: '#000000',
            },
            padding: 10,
            cellWidth: 80,
            cellHeight: 30,
            fontSize: 11,
            fontWeight: '500',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
            labelWidth: 100,  // Width for "Z-Score:" and "Probability:" labels
        };
    }

    attached(p) {
        this._series = p.series;
        this._requestUpdate = p.requestUpdate;
    }

    detached() {
        this._series = undefined;
        this._requestUpdate = undefined;
    }

    updateZScore(zscoreValue) {
        this._currentZScore = zscoreValue;
        if (this._requestUpdate) {
            this._requestUpdate();
        }
    }

    setVisible(visible) {
        this._visible = visible;
        if (this._requestUpdate) this._requestUpdate();
    }

    paneViews() {
        if (!this._visible) return [];
        return [new ZScoreTablePaneView(this._config, this._currentZScore)];
    }
}


class ZScoreTablePaneView {
    constructor(config, currentZScore) {
        this._config = config;
        this._currentZScore = currentZScore;
    }

    zOrder() {
        return 'top';
    }

    renderer() {
        return new ZScoreTableRenderer(this._config, this._currentZScore);
    }
}


class ZScoreTableRenderer {
    constructor(config, currentZScore) {
        this._config = config;
        this._currentZScore = currentZScore;
    }

    draw(target) {
        target.useMediaCoordinateSpace(scope => {
            const ctx = scope.context;
            const height = scope.mediaSize.height;
            const width = scope.mediaSize.width;

            const cfg = this._config;
            const tableWidth = cfg.labelWidth + cfg.zScores.length * cfg.cellWidth;
            const tableHeight = cfg.cellHeight * 2; // 2 rows

            // Position at bottom center
            const x = (width - tableWidth) / 2;
            const y = height - tableHeight - cfg.padding;

            // Draw outer border
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.2)';
            ctx.lineWidth = 1;
            ctx.strokeRect(x, y, tableWidth, tableHeight);

            // Set font
            ctx.font = `${cfg.fontWeight} ${cfg.fontSize}px ${cfg.fontFamily}`;
            ctx.textBaseline = 'middle';

            // Always highlight index 5 (Z-Score = 2) with gray background
            const activeIndex = 5;  // Index 5 = Z-Score value 2

            // Row 1: Z-Score header
            const headerY = y;

            // Draw "Z-Score:" label background (white)
            ctx.fillStyle = cfg.colors.headerBg;
            ctx.fillRect(x, headerY, cfg.labelWidth, cfg.cellHeight);

            // Draw "Z-Score:" label text (centered)
            ctx.fillStyle = cfg.colors.textBlack;
            ctx.textAlign = 'center';
            ctx.fillText('Z-Score:', x + cfg.labelWidth / 2, headerY + cfg.cellHeight / 2);

            // Draw Z-Score cells
            for (let i = 0; i < cfg.zScores.length; i++) {
                const cellX = x + cfg.labelWidth + i * cfg.cellWidth;

                // Background: white for all cells, gray for active
                if (i === activeIndex) {
                    ctx.fillStyle = cfg.colors.activeCell;
                } else {
                    ctx.fillStyle = cfg.colors.headerBg;
                }
                ctx.fillRect(cellX, headerY, cfg.cellWidth, cfg.cellHeight);

                // Draw Z-Score value (centered)
                ctx.fillStyle = cfg.colors.textBlack;
                ctx.textAlign = 'center';
                ctx.fillText(
                    cfg.zScores[i].toString(),
                    cellX + cfg.cellWidth / 2,
                    headerY + cfg.cellHeight / 2
                );
            }

            // Row 2: Probability
            const probY = y + cfg.cellHeight;

            // Draw "Probability:" label background (white)
            ctx.fillStyle = cfg.colors.headerBg;
            ctx.fillRect(x, probY, cfg.labelWidth, cfg.cellHeight);

            // Draw "Probability:" label text (centered)
            ctx.fillStyle = cfg.colors.textBlack;
            ctx.textAlign = 'center';
            ctx.fillText('Probability:', x + cfg.labelWidth / 2, probY + cfg.cellHeight / 2);

            // Draw Probability cells
            for (let i = 0; i < cfg.zScores.length; i++) {
                const cellX = x + cfg.labelWidth + i * cfg.cellWidth;

                // Color based on probability
                let bgColor = cfg.colors.green;
                if (cfg.probabilities[i] === '0.0013' || cfg.probabilities[i] === '0.9987') {
                    bgColor = cfg.colors.red;
                } else if (cfg.probabilities[i] === '0.0228' || cfg.probabilities[i] === '0.9800') {
                    bgColor = cfg.colors.yellow;
                }

                ctx.fillStyle = bgColor;
                ctx.fillRect(cellX, probY, cfg.cellWidth, cfg.cellHeight);

                // Draw probability value (centered)
                ctx.fillStyle = cfg.colors.textBlack;
                ctx.textAlign = 'center';
                ctx.fillText(
                    cfg.probabilities[i],
                    cellX + cfg.cellWidth / 2,
                    probY + cfg.cellHeight / 2
                );
            }
        });
    }
}

