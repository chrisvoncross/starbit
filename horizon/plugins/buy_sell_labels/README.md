# Labels Plugin

Custom labels plugin for rendering text labels on the chart (like TradingView's plotshape labels).

## Features

- ✅ Custom text labels with styled backgrounds
- ✅ Speech bubble style (for sell signals above bar)
- ✅ Rectangle style (for buy signals below bar)
- ✅ Customizable colors and text
- ✅ Automatic positioning based on price

## Usage

```javascript
import { LabelsPlugin } from './plugins/labels/labels.js';

// Create plugin instance
const labels = new LabelsPlugin();

// Attach to series
candleSeries.attachPrimitive(labels);

// Add a sell label (blue speech bubble above bar)
labels.addLabel(
    1234567890,           // time (Unix timestamp in seconds)
    100.5,                // price
    'SELL',               // text
    'aboveBar',           // position
    '#2196F3',            // background color (blue)
    '#FFFFFF',            // text color (white)
    6                     // arrow size
);

// Add a buy label (gray rectangle below bar)
labels.addLabel(
    1234567900,           // time
    99.5,                 // price
    'BUY',                // text
    'belowBar',           // position
    '#666666',            // background color (gray)
    '#FFFFFF',            // text color (white)
);

// Clear all labels
labels.clearLabels();

// Set multiple labels at once
labels.setLabels([
    {time: 1234567890, price: 100.5, text: 'SELL', position: 'aboveBar', bgColor: '#2196F3', textColor: '#FFFFFF'},
    {time: 1234567900, price: 99.5, text: 'BUY', position: 'belowBar', bgColor: '#666666', textColor: '#FFFFFF'},
]);
```

## API

### `addLabel(time, price, text, position, bgColor, textColor, arrowSize)`

Add a single label to the chart.

**Parameters:**
- `time` (number): Unix timestamp in seconds
- `price` (number): Price level where label should appear
- `text` (string): Label text
- `position` (string): `'aboveBar'` or `'belowBar'`
- `bgColor` (string): Background color (hex or rgba)
- `textColor` (string): Text color (hex or rgba)
- `arrowSize` (number): Arrow size for speech bubbles (default: 6)

### `clearLabels()`

Remove all labels from the chart.

### `setLabels(labels)`

Set multiple labels at once.

**Parameters:**
- `labels` (Array): Array of label objects

## Label Styles

### Speech Bubble (aboveBar)
Used for sell signals. Renders as a rounded rectangle with a downward-pointing arrow.

### Rectangle (belowBar)
Used for buy signals. Renders as a simple rounded rectangle.

## Example: Buy/Sell Signals

```javascript
const labels = new LabelsPlugin();
candleSeries.attachPrimitive(labels);

// Sell signal (blue speech bubble)
labels.addLabel(time, price, 'Sell', 'aboveBar', '#2196F3', '#FFFFFF');

// Buy signal (gray rectangle)
labels.addLabel(time, price, 'Buy', 'belowBar', '#666666', '#FFFFFF');
```

## Based On

TradingView's `plotshape` with `shape.labelup` and `shape.labeldown` styles.

