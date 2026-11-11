# Fill Areas Plugin

Universal plugin for rendering smooth filled areas between two lines in Lightweight Charts.

## Features

- ✅ Smooth filled areas between any two lines
- ✅ Multiple zones support
- ✅ Optional boundary lines
- ✅ Customizable colors and styles
- ✅ Universal - works for any indicator (Bollinger Bands, NWE, Keltner, Ichimoku, etc.)

## Usage

```javascript
import { FillAreasPlugin } from './plugins/fill-areas/fill-areas.js';

// Create plugin instance
const fillAreas = new FillAreasPlugin({
    showLines: true,  // Show boundary lines (default: true)
});

// Attach to series
candleSeries.attachPrimitive(fillAreas);

// Add filled zones
fillAreas.addZone(
    'zone1',  // Unique ID
    upperLineData,  // [{time: 1234567890, value: 100.5}, ...]
    lowerLineData,  // [{time: 1234567890, value: 99.5}, ...]
    'rgba(239, 83, 80, 0.15)',  // Fill color
    {
        lineColor: 'rgba(120, 123, 134, 0.3)',  // Optional
        lineWidth: 1,  // Optional
        showUpperLine: true,  // Optional (default: false)
        showLowerLine: true,  // Optional (default: false)
    }
);

// Add more zones
fillAreas.addZone('zone2', upper2, lower2, 'rgba(38, 166, 154, 0.15)');

// Remove a zone
fillAreas.removeZone('zone1');

// Clear all zones
fillAreas.clearZones();
```

## Example: NWE Bands

```javascript
const fillAreas = new FillAreasPlugin();
candleSeries.attachPrimitive(fillAreas);

// Upper zones (red)
fillAreas.addZone('nwe_upper_far', nweData.upper_far, nweData.upper_avg, 'rgba(239, 83, 80, 0.15)');
fillAreas.addZone('nwe_upper_near', nweData.upper_avg, nweData.upper_near, 'rgba(239, 83, 80, 0.25)');

// Lower zones (green)
fillAreas.addZone('nwe_lower_near', nweData.lower_near, nweData.lower_avg, 'rgba(38, 166, 154, 0.25)');
fillAreas.addZone('nwe_lower_far', nweData.lower_avg, nweData.lower_far, 'rgba(38, 166, 154, 0.15)');
```

## Example: Bollinger Bands

```javascript
const fillAreas = new FillAreasPlugin();
candleSeries.attachPrimitive(fillAreas);

fillAreas.addZone('bb_upper', bbData.upper, bbData.middle, 'rgba(33, 150, 243, 0.1)');
fillAreas.addZone('bb_lower', bbData.middle, bbData.lower, 'rgba(33, 150, 243, 0.1)');
```

## API

### Constructor Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `showLines` | boolean | `true` | Show boundary lines |

### Methods

#### `addZone(id, upperData, lowerData, fillColor, options)`

Add a filled zone between two lines.

**Parameters:**
- `id` (string): Unique identifier for this zone
- `upperData` (Array): Array of `{time, value}` for upper line
- `lowerData` (Array): Array of `{time, value}` for lower line
- `fillColor` (string): Fill color (rgba recommended)
- `options` (Object): Optional settings
  - `lineColor` (string): Boundary line color (default: `'rgba(120, 123, 134, 0.3)'`)
  - `lineWidth` (number): Boundary line width (default: `1`)
  - `showUpperLine` (boolean): Show upper boundary line (default: `false`)
  - `showLowerLine` (boolean): Show lower boundary line (default: `false`)

#### `removeZone(id)`

Remove a zone by ID.

#### `clearZones()`

Remove all zones.

#### `applyOptions(options)`

Update plugin options.

## Based On

TradingView's official [BandsIndicator plugin example](https://github.com/tradingview/lightweight-charts/tree/master/plugin-examples/src/plugins/bands-indicator)

