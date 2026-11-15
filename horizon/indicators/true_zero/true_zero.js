// True Zero is the same one-position search as 0 Fees, with the spread floor removed.
// Fee is 0. A ride counts when the exit prints strictly beyond the entry.
// The smallest such step on the contract is one tick. No extra height is required.

import { paintCashOverlay } from '../nordstern/nordstern.js?v=truezero1';

export function paintTrueZero({
    chart,
    series,
    candles,
    enabled,
    showStars = true,
    showPositions = true,
    showTable = true,
}) {
    paintCashOverlay('true_zero', {
        chart,
        series,
        candles,
        enabled,
        showStars,
        showPositions,
        showTable,
        feeRate: 0,
        minMove: 0,
        title: 'True Zero',
        feeLine: 'Keine Mindesthöhe. Gebühr 0. Jede positive Bewegung.',
        anchor: 'left',
        riskReward: true,
        longColor: '#3ddc6a',
        shortColor: '#ff8a65',
        markCandle: false,
        starFill: '#f4fff8',
        starStroke: '#1f8f4e',
    });
}
