export function createChartContext({ chartContainer, fillAreas, labels, buySellPositions, OhlcBarsPlugin }) {
    const chart = LightweightCharts.createChart(chartContainer, {
        layout: {
            background: { color: '#000000' },
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: { visible: false },
            horzLines: { visible: false },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: { labelBackgroundColor: '#363a45' },
            horzLine: { labelBackgroundColor: '#363a45' },
        },
        handleScroll: {
            mouseWheel: true,
            pressedMouseMove: true,
            horzTouchDrag: true,
            vertTouchDrag: true,
        },
        handleScale: {
            axisPressedMouseMove: true,
            mouseWheel: true,
            pinch: true,
        },
        kineticScroll: {
            touch: true,
            mouse: false,
        },
        localization: {
            timeFormatter: (time) => {
                const date = new Date(time * 1000);
                const weekday = date.toLocaleDateString('en-US', { weekday: 'short', timeZone: 'UTC' });
                const day = String(date.getUTCDate()).padStart(2, '0');
                const month = date.toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' });
                const year = String(date.getUTCFullYear()).slice(-2);
                const hours = String(date.getUTCHours()).padStart(2, '0');
                const minutes = String(date.getUTCMinutes()).padStart(2, '0');
                return `${weekday} ${day} ${month} '${year}  ${hours}:${minutes}`;
            },
        },
        rightPriceScale: { borderColor: '#2a2a2a' },
        timeScale: {
            borderColor: '#2a2a2a',
            timeVisible: true,
            secondsVisible: false,
            tickMarkFormatter: (time, tickMarkType) => {
                const date = new Date(time * 1000);
                const hours = String(date.getUTCHours()).padStart(2, '0');
                const minutes = String(date.getUTCMinutes()).padStart(2, '0');
                if (tickMarkType === LightweightCharts.TickMarkType.Year) return date.getUTCFullYear().toString();
                if (tickMarkType === LightweightCharts.TickMarkType.Month) {
                    const month = date.toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' });
                    const year = String(date.getUTCFullYear()).slice(-2);
                    return `${month} ${date.getUTCDate()} '${year}`;
                }
                if (tickMarkType === LightweightCharts.TickMarkType.DayOfMonth) {
                    const weekday = date.toLocaleDateString('en-US', { weekday: 'short', timeZone: 'UTC' });
                    const day = date.getUTCDate();
                    const month = date.toLocaleDateString('en-US', { month: 'short', timeZone: 'UTC' });
                    const year = String(date.getUTCFullYear()).slice(-2);
                    return `${weekday} ${day} ${month} '${year}  ${hours}:${minutes}`;
                }
                return `${hours}:${minutes}`;
            },
        },
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight,
    });

    const PRICE_FORMAT = { type: 'price', precision: 8, minMove: 0.00000001 };

    function formatFor(price) {
        const abs = Math.abs(price);
        const needed = abs > 0 && abs < 1 ? Math.ceil(-Math.log10(abs)) + 4 : 8;
        const precision = Math.max(8, Math.min(12, needed));
        return { type: 'price', precision, minMove: 10 ** -precision };
    }

    function applyPrice(candles) {
        const sample = candles && candles.length ? candles[candles.length - 1].close : 0;
        if (!(sample > 0)) return;
        const format = formatFor(sample);
        haSeries.applyOptions({ priceFormat: format });
        rawSeries.applyOptions({ priceFormat: format });
        barSeries.applyOptions({ priceFormat: format });
    }
    const haSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
        upColor: '#FFFFFF',
        downColor: '#2962FF',
        borderVisible: false,
        wickUpColor: '#FFFFFF',
        wickDownColor: '#2962FF',
        priceFormat: PRICE_FORMAT,
        lastValueVisible: true,
        priceLineVisible: true,
    });
    const rawSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
        upColor: '#26A69A',
        downColor: '#EF5350',
        borderVisible: false,
        wickUpColor: '#26A69A',
        wickDownColor: '#EF5350',
        priceFormat: PRICE_FORMAT,
        lastValueVisible: false,
        priceLineVisible: false,
        visible: false,
    });
    const barSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
        upColor: 'transparent',
        downColor: 'transparent',
        wickUpColor: 'transparent',
        wickDownColor: 'transparent',
        borderVisible: false,
        priceFormat: PRICE_FORMAT,
        lastValueVisible: false,
        priceLineVisible: false,
        visible: false,
    });
    const ohlcBarsPlugin = new OhlcBarsPlugin({ color: '#CCCCCC' });
    barSeries.attachPrimitive(ohlcBarsPlugin);

    const overlayPlugins = [fillAreas, labels, buySellPositions];
    for (const p of overlayPlugins) haSeries.attachPrimitive(p);

    let activeSeries = haSeries;
    let chartStyle = 'candles';
    let rawOhlcData = [];

    function switchChartStyle(style) {
        if (style === chartStyle) return;
        const oldSeries = activeSeries;
        const hide = { visible: false, lastValueVisible: false, priceLineVisible: false };
        const show = { visible: true, lastValueVisible: true, priceLineVisible: true };
        haSeries.applyOptions(hide);
        rawSeries.applyOptions(hide);

        if (style === 'bars') {
            if (rawOhlcData.length > 0) barSeries.setData(rawOhlcData);
            barSeries.applyOptions(show);
            ohlcBarsPlugin._visible = true;
            activeSeries = barSeries;
        } else if (style === 'raw') {
            if (rawOhlcData.length > 0) rawSeries.setData(rawOhlcData);
            rawSeries.applyOptions(show);
            barSeries.applyOptions(hide);
            ohlcBarsPlugin._visible = false;
            activeSeries = rawSeries;
        } else {
            haSeries.applyOptions(show);
            barSeries.applyOptions(hide);
            ohlcBarsPlugin._visible = false;
            activeSeries = haSeries;
        }

        for (const p of overlayPlugins) {
            try {
                oldSeries.detachPrimitive(p);
            } catch (e) {
                // no-op
            }
            activeSeries.attachPrimitive(p);
        }
        chartStyle = style;
        console.log(`🎨 Chart style: ${style}`);
    }

    return {
        chart,
        chartContainer,
        series: { haSeries, rawSeries, barSeries },
        plugins: { fillAreas, labels, buySellPositions, overlayPlugins },
        getActiveSeries: () => activeSeries,
        getChartStyle: () => chartStyle,
        switchChartStyle,
        getRawOhlcData: () => rawOhlcData,
        applyPrice,
        setRawOhlcData: (data) => {
            rawOhlcData = data;
        },
        upsertRawCandle: (raw) => {
            if (rawOhlcData.length > 0 && rawOhlcData[rawOhlcData.length - 1].time === raw.time) {
                rawOhlcData[rawOhlcData.length - 1] = raw;
            } else {
                rawOhlcData.push(raw);
            }
        },
    };
}
