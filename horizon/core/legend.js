export function bindLegend(chart, getActiveSeries, legendEls) {
    const { legendOpen, legendHigh, legendLow, legendClose, legendChange } = legendEls;

    chart.subscribeCrosshairMove((param) => {
        const activeSeries = getActiveSeries();
        if (!param.time || !param.seriesData.get(activeSeries)) return;

        const data = param.seriesData.get(activeSeries);
        if (!data) return;

        const open = data.open;
        const high = data.high;
        const low = data.low;
        const close = data.close;
        const change = ((close - open) / open * 100).toFixed(2);
        const isUp = close >= open;
        const candleColor = isUp ? '#bdbfc6' : '#2c6ddd';

        legendOpen.textContent = open.toFixed(8);
        legendOpen.style.color = candleColor;
        legendHigh.textContent = high.toFixed(8);
        legendHigh.style.color = candleColor;
        legendLow.textContent = low.toFixed(8);
        legendLow.style.color = candleColor;
        legendClose.textContent = close.toFixed(8);
        legendClose.style.color = candleColor;
        legendChange.textContent = `${change >= 0 ? '+' : ''}${change}%`;
        legendChange.style.color = candleColor;
    });
}
