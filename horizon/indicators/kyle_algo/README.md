# Kyle Algo Buy/Sell Signals

Displays buy/sell labels on the chart based on Kyle Algo signals from the database.

## What is Kyle Algo?

Kyle Algo is a trading indicator that uses a modified Supertrend with Keltner Channels to generate buy/sell signals. It combines:
- Modified Supertrend (using Keltner Channel instead of ATR)
- EMA Energy (15 EMAs from 9 to 51 periods)
- Crossover/Crossunder detection for signals

## Usage

```javascript
import { setupKyleSignals, KYLE_CONFIG } from './indicators/kyle_algo/kyle.js';

// Setup Kyle signals when data arrives
const kyleSignals = [
    {time: 1234567890, buy_signal: true, sell_signal: false, price: 100.5},
    {time: 1234567900, buy_signal: false, sell_signal: true, price: 101.2},
];

const markerCount = setupKyleSignals(candleSeries, kyleSignals);
console.log(`Added ${markerCount} Kyle signals`);
```

## Configuration

Default colors and styling can be customized via `KYLE_CONFIG`:

```javascript
import { KYLE_CONFIG } from './indicators/kyle_algo/kyle.js';

// Colors match original kyle algo
KYLE_CONFIG.buyColor = '#2BBC4D';   // Green
KYLE_CONFIG.sellColor = '#C51D0B';  // Red
```

## API

### `setupKyleSignals(candleSeries, kyleSignals)`

Setup Kyle Algo buy/sell markers on the chart.

**Parameters:**
- `candleSeries` (CandlestickSeries): The candlestick series
- `kyleSignals` (Array): Array of signals with `{time, buy_signal, sell_signal, price}`

**Returns:**
- `number`: Number of markers added

### `clearKyleSignals(candleSeries)`

Remove all Kyle Algo markers from the chart.

**Parameters:**
- `candleSeries` (CandlestickSeries): The candlestick series

### `toggleKyleSignals(candleSeries, visible, kyleSignals)`

Show or hide Kyle Algo signals.

**Parameters:**
- `candleSeries` (CandlestickSeries): The candlestick series
- `visible` (boolean): Show (true) or hide (false)
- `kyleSignals` (Array): Kyle signals data (needed if visible=true)

## Data Source

Kyle Algo signals are calculated in Python and stored in the PostgreSQL database in the `features` JSONB column:
- `kyle_buy_signal`: Boolean (true if buy signal)
- `kyle_sell_signal`: Boolean (true if sell signal)

The WebSocket server loads this data and sends it to the browser.

## Marker Styling

- **Buy Signal**: Green arrow up below bar with "BUY" text
- **Sell Signal**: Red arrow down above bar with "SELL" text

Matches the original kyle algo v1 indicator from TradingView.

