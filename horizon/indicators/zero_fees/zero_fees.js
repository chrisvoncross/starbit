// 0 Fees is Nordstern's max-cash path with the fee taken out.
// A ride counts only when the move is strictly above the measured BTC spread,
// 0.0131 bp. Among those rides the same one-position search keeps the path
// with the most cash. Stars, boxes and the table switch on their own.

import { MEASURED_SPREAD, paintCashOverlay } from '../nordstern/nordstern.js?v=zerofees6';

export function paintZeroFees({
    chart,
    series,
    candles,
    enabled,
    showStars = true,
    showPositions = true,
    showTable = true,
}) {
    paintCashOverlay('zero_fees', {
        chart,
        series,
        candles,
        enabled,
        showStars,
        showPositions,
        showTable,
        feeRate: 0,
        minMove: MEASURED_SPREAD,
        title: '0 Fees',
        feeLine: `Spread ${(MEASURED_SPREAD * 10000).toFixed(4).replace('.', ',')} bp. Nur Bewegungen darüber.`,
        anchor: 'right',
        riskReward: true,
        longColor: '#26a69a',
        shortColor: '#ef5350',
        markCandle: false,
        starFill: '#ffe14a',
        starStroke: '#c8960a',
    });
}
