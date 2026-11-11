import { mountNordstern } from '/indicators/nordstern/nordstern.js?v=leiter1';

const grid = document.getElementById('grid');
const title = document.getElementById('title');
const sub = document.getElementById('sub');

function money(value) {
    const sign = value < 0 ? '-' : '';
    return `${sign}${Math.abs(value).toFixed(2).replace('.', ',')} $`;
}

function utcTick(time) {
    const date = new Date(Number(time) * 1000);
    const part = (n) => String(n).padStart(2, '0');
    return `${part(date.getUTCHours())}:${part(date.getUTCMinutes())}`;
}

function pane(frame) {
    const cell = document.createElement('section');
    cell.className = 'pane';
    const head = document.createElement('div');
    head.className = 'head';
    const chartHost = document.createElement('div');
    chartHost.className = 'chart';
    cell.appendChild(head);
    cell.appendChild(chartHost);
    grid.appendChild(cell);

    const chart = LightweightCharts.createChart(chartHost, {
        autoSize: true,
        layout: {
            background: { color: '#000000' },
            textColor: '#d1d4dc',
            attributionLogo: false,
        },
        grid: {
            vertLines: { color: '#111111' },
            horzLines: { color: '#111111' },
        },
        rightPriceScale: { borderColor: '#1c2430' },
        timeScale: { borderColor: '#1c2430', timeVisible: true, secondsVisible: false },
        localization: { timeFormatter: utcTick },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    });
    const series = chart.addSeries(LightweightCharts.CandlestickSeries, {
        upColor: '#FFFFFF',
        downColor: '#2962FF',
        borderVisible: false,
        wickUpColor: '#FFFFFF',
        wickDownColor: '#2962FF',
        priceFormat: { type: 'price', precision: 1, minMove: 0.1 },
        lastValueVisible: false,
        priceLineVisible: false,
    });
    series.setData(frame.candles);
    const book = mountNordstern({ chart, series, candles: frame.candles, showTable: false });
    chart.timeScale().fitContent();
    const from15 = frame.source === '15s' ? ' · aus 15s' : '';
    const drift = frame.matches_15s === false ? ' · weicht von 15s ab' : '';
    head.textContent = `${frame.label}${from15} · ${frame.candles.length} Kerzen · ${book.trades.length} Fahrten · ${money(book.total)}${drift}`;
}

const response = await fetch('/api/gegenueber', { cache: 'no-store' });
if (!response.ok) {
    sub.textContent = `Laden fehlgeschlagen (${response.status})`;
    throw new Error(await response.text());
}
const body = await response.json();
title.textContent = `${body.symbol}  ${body.from}  bis  ${body.to}`;
sub.textContent = 'Dieselben zwei Stunden. 2, 3 und 4 Minuten sind aus den 15-Sekunden-Kerzen gefaltet. Nordstern auf jeder Uhr.';
for (const frame of body.frames) pane(frame);
