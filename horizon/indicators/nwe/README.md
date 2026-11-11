# NWE (Nadaraya-Watson Envelope) Bands Indicator

Configuration and helper functions for NWE Bands visualization using the Fill Areas Plugin.

## What is NWE?

Nadaraya-Watson Envelope is a technical indicator that uses kernel regression to create adaptive price envelopes. It's similar to Bollinger Bands but uses a more sophisticated statistical approach.

The indicator consists of 6 bands:
- **Upper Far**: Outermost upper band
- **Upper Avg**: Middle upper band
- **Upper Near**: Innermost upper band
- **Lower Near**: Innermost lower band
- **Lower Avg**: Middle lower band
- **Lower Far**: Outermost lower band

## Usage

```javascript
import { FillAreasPlugin } from './plugins/fill-areas/fill-areas.js';
import { setupNWEBands, NWE_CONFIG } from './indicators/nwe/nwe.js';

// Create Fill Areas Plugin
const fillAreas = new FillAreasPlugin();
candleSeries.attachPrimitive(fillAreas);

// Setup NWE Bands when data arrives
const nweData = {
    upper_far: [{time: 1234567890, value: 100.5}, ...],
    upper_avg: [{time: 1234567890, value: 100.3}, ...],
    upper_near: [{time: 1234567890, value: 100.1}, ...],
    lower_near: [{time: 1234567890, value: 99.9}, ...],
    lower_avg: [{time: 1234567890, value: 99.7}, ...],
    lower_far: [{time: 1234567890, value: 99.5}, ...],
};

setupNWEBands(fillAreas, nweData);
```

## Configuration

Default colors can be customized via `NWE_CONFIG`:

```javascript
import { NWE_CONFIG, updateNWEColors } from './indicators/nwe/nwe.js';

// Update colors
updateNWEColors({
    upperFar: 'rgba(255, 0, 0, 0.1)',
    upperNear: 'rgba(255, 0, 0, 0.2)',
    lowerNear: 'rgba(0, 255, 0, 0.2)',
    lowerFar: 'rgba(0, 255, 0, 0.1)',
});
```

## API

### `setupNWEBands(fillAreasPlugin, nweData)`

Setup NWE Bands visualization.

**Parameters:**
- `fillAreasPlugin` (FillAreasPlugin): The Fill Areas Plugin instance
- `nweData` (Object): NWE data with 6 bands (upper_far, upper_avg, upper_near, lower_near, lower_avg, lower_far)

### `toggleNWEBands(fillAreasPlugin, visible)`

Show or hide NWE Bands.

**Parameters:**
- `fillAreasPlugin` (FillAreasPlugin): The Fill Areas Plugin instance
- `visible` (boolean): Show (true) or hide (false)

### `updateNWEColors(colors)`

Update NWE Bands colors.

**Parameters:**
- `colors` (Object): New colors (upperFar, upperNear, lowerNear, lowerFar)

## Data Source

NWE Bands are calculated in Python and stored in the PostgreSQL database in the `features` JSONB column of the `market_data_features` table.

The WebSocket server loads this data and sends it to the browser.

