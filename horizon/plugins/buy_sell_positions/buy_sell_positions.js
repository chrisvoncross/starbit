/**
 * Buy/Sell Positions Plugin
 *
 * Visualizes simulated trades on the chart (like TradingView Position Tool):
 * - White horizontal line at entry price
 * - Blue filled area between entry and stop loss (Entry <> SL)
 * - Gray filled area between entry and take profit (Entry <> TP)
 * - Entry/Exit markers
 */

export class BuySellPositionsPlugin {
    constructor(trades = []) {
        this._trades = trades;
        this._paneViews = [new BuySellPositionsPaneView(this)];
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
        if (this._requestUpdate) {
            this._requestUpdate();
        }
    }

    updateTrades(trades) {
        this._trades = trades;
        console.log('🎯 BuySellPositions: Updated with', trades.length, 'trades:', trades);
        this.requestUpdate();
    }

    getTrades() {
        return this._trades;
    }

    getChart() {
        return this._chart;
    }

    getSeries() {
        return this._series;
    }
}


class BuySellPositionsPaneView {
    constructor(plugin) {
        this._plugin = plugin;
    }

    zOrder() {
        return 'top';
    }

    renderer() {
        return new BuySellPositionsRenderer(this._plugin);
    }

    update() {
        // Trigger re-render
    }
}


class BuySellPositionsRenderer {
    constructor(plugin) {
        this._plugin = plugin;
    }

    draw(target) {
        const trades = this._plugin.getTrades();
        const chart = this._plugin.getChart();
        const series = this._plugin.getSeries();

        if (!trades || trades.length === 0 || !chart || !series) {
            console.log('🚫 BuySellPositions: Cannot draw - trades:', trades?.length, 'chart:', !!chart, 'series:', !!series);
            return;
        }

        console.log('🎨 BuySellPositions: Drawing', trades.length, 'trades');

        target.useMediaCoordinateSpace(scope => {
            const ctx = scope.context;
            const timeScale = chart.timeScale();

            trades.forEach(trade => {
                this._drawTrade(ctx, trade, timeScale, series, scope.mediaSize.width);
            });
        });
    }

    _drawTrade(ctx, trade, timeScale, series, canvasWidth) {
        const entryX = timeScale.timeToCoordinate(trade.entry_time);
        const exitX = timeScale.timeToCoordinate(trade.exit_time);
        const entryY = series.priceToCoordinate(trade.entry_price);
        const slY = trade.stop_loss ? series.priceToCoordinate(trade.stop_loss) : null;
        const tpY = trade.take_profit ? series.priceToCoordinate(trade.take_profit) : null;

        console.log('📍 Drawing trade:', {
            side: trade.side,
            entry_time: trade.entry_time,
            exit_time: trade.exit_time,
            entry_price: trade.entry_price,
            stop_loss: trade.stop_loss,
            take_profit: trade.take_profit,
            entryX, entryY, exitX, slY, tpY,
            canvasWidth
        });

        if (entryX === null || entryY === null) {
            console.log('⚠️ Skipping trade - entryX or entryY is null');
            return;
        }

        // 1. Draw Entry <> SL area (BLUE or RED depending on exit reason)
        if (slY !== null) {
            // If trade was stopped out, use RED. Otherwise use BLUE
            const wasStoppedOut = trade.exit_reason === 'STOP_LOSS';
            ctx.fillStyle = wasStoppedOut ? 'rgba(244, 67, 54, 0.35)' : 'rgba(33, 150, 243, 0.35)';  // Red if SL hit, Blue otherwise
            const topY = Math.min(entryY, slY);
            const bottomY = Math.max(entryY, slY);
            const width = exitX !== null ? (exitX - entryX) : (canvasWidth - entryX);
            ctx.fillRect(entryX, topY, width, bottomY - topY);
        }

        // 2. Draw Entry <> TP area (GRAY, more visible)
        if (tpY !== null) {
            ctx.fillStyle = 'rgba(128, 128, 128, 0.35)';  // Gray, more opaque
            const topY = Math.min(entryY, tpY);
            const bottomY = Math.max(entryY, tpY);
            const width = exitX !== null ? (exitX - entryX) : (canvasWidth - entryX);
            ctx.fillRect(entryX, topY, width, bottomY - topY);
        }

        // 3. Draw Entry line (WHITE, thinner)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = 1;  // Thinner line
        ctx.beginPath();
        ctx.moveTo(entryX, entryY);
        const lineEndX = exitX !== null ? exitX : canvasWidth;
        ctx.lineTo(lineEndX, entryY);
        ctx.stroke();
    }
}

