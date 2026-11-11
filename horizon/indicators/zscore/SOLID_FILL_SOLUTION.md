# Z-Score Solid Fill Solution

## Problem

The Z-Score indicator needed **100% solid fill** (opaque) for positive (green) and negative (red) values, but:

1. **BaselineSeries** has hardcoded gradient transparency that cannot be disabled
2. **AreaSeries** also has gradient transparency
3. **Fill Areas Plugin** uses semi-transparent fills for probability zones

## Failed Attempts

### 1. ❌ Modifying BaselineSeries Colors
- Tried setting `topFillColor1` and `topFillColor2` to solid colors
- **Result:** Still had gradient transparency (hardcoded in LightweightCharts)

### 2. ❌ Using Fill Areas Plugin with Data Arrays
- Tried passing data arrays instead of Series objects to Fill Areas Plugin
- **Result:** Crashed Stochastic indicator, no improvement

### 3. ❌ Multiple Series Layering
- Tried creating multiple series and layering them
- **Result:** Rendering order issues, still transparent

## ✅ WORKING SOLUTION: Custom Primitive with `draw()`

### Key Insight

**Lightweight Charts Rendering Order:**
1. **`drawBackground()`** - renders BEHIND everything (background layer)
2. **`draw()`** - renders IN FRONT (foreground layer)

**The Fix:** Use `draw()` instead of `drawBackground()` in the custom primitive renderer!

### Implementation

#### Step 1: Create Custom Primitive Plugin

**File:** `lightweight_charts/plugins/zscore-solid-fill/zscore-solid-fill.js`

```javascript
class ZScoreSolidFillPaneRenderer {
    constructor(viewData) {
        this._viewData = viewData;
    }

    // ✅ USE draw() NOT drawBackground()!
    draw(target) {
        const data = this._viewData.data;
        const options = this._viewData.options;
        
        if (!data || data.length === 0) return;

        target.useBitmapCoordinateSpace(scope => {
            const ctx = scope.context;
            ctx.scale(scope.horizontalPixelRatio, scope.verticalPixelRatio);

            // Draw 100% SOLID fills
            this._drawFill(ctx, data.positivePoints, data.zeroPoints, options.positiveColor);
            this._drawFill(ctx, data.zeroPoints, data.negativePoints, options.negativeColor);
        });
    }

    _drawFill(ctx, upperPoints, lowerPoints, fillColor) {
        if (!upperPoints || !lowerPoints || upperPoints.length === 0) return;

        const region = new Path2D();
        region.moveTo(upperPoints[0].x, upperPoints[0].y);
        
        // Draw upper line
        for (const point of upperPoints) {
            region.lineTo(point.x, point.y);
        }
        
        // Connect to lower line
        const lastIndex = lowerPoints.length - 1;
        region.lineTo(lowerPoints[lastIndex].x, lowerPoints[lastIndex].y);
        
        // Draw lower line (reverse)
        for (let i = lastIndex - 1; i >= 0; i--) {
            region.lineTo(lowerPoints[i].x, lowerPoints[i].y);
        }
        
        region.closePath();
        
        // ✅ 100% SOLID COLOR - NO TRANSPARENCY!
        ctx.fillStyle = fillColor;
        ctx.fill(region);
    }
}
```

#### Step 2: Attach to Separate Series (for correct rendering order)

```javascript
// Create invisible series for solid fill (renders AFTER probability zones)
const solidFillBaseSeries = chart.addSeries(
    LightweightCharts.LineSeries,
    {
        color: 'transparent',
        lineWidth: 0,
    },
    1  // Same pane as Z-Score
);
solidFillBaseSeries.setData(constantData);

// Attach solid fill plugin
const zscoreSolidFill = new ZScoreSolidFillPlugin(zscoreData.zscore, {
    positiveColor: '#4CAF50',  // Green (100% solid)
    negativeColor: '#FF5252',  // Red (100% solid)
});
solidFillBaseSeries.attachPrimitive(zscoreSolidFill);
```

#### Step 3: Add White Line on Top

```javascript
const zscoreLineSeries = chart.addSeries(
    LightweightCharts.LineSeries,
    {
        color: '#FFFFFF',  // White line
        lineWidth: 2,
    },
    1
);
zscoreLineSeries.setData(zscoreData.zscore);
```

### Final Rendering Order

1. **Background:** Probability zones (green/yellow/red - transparent) via Fill Areas Plugin
2. **Foreground:** Solid fill (green/red - 100% opaque) via Custom Primitive with `draw()`
3. **Top:** White Z-Score line

## Why This Works

- **`draw()`** renders in the **foreground layer**, AFTER all `drawBackground()` calls
- **Separate series** ensures the solid fill is created AFTER probability zones
- **Custom primitive** gives full control over Canvas rendering (no hardcoded transparency)

## Server Configuration

Don't forget to add the plugin route in `websocket_server.py`:

```python
@app.get("/plugins/zscore-solid-fill/zscore-solid-fill.js")
async def serve_zscore_solid_fill_plugin():
    """Serve Z-Score Solid Fill Plugin"""
    with open("plugins/zscore-solid-fill/zscore-solid-fill.js", "r", encoding="utf-8") as f:
        js_content = f.read()
    from fastapi.responses import Response
    return Response(content=js_content, media_type="application/javascript")
```

## Result

✅ **100% SOLID fill** for Z-Score positive (green) and negative (red) values  
✅ **No transparency or gradients**  
✅ **Probability zones still visible in background**  
✅ **White line clearly visible on top**

## Lessons Learned

1. **`drawBackground()` vs `draw()`** - Critical for rendering order!
2. **Custom Primitives** - Full control over Canvas rendering
3. **Series order matters** - Later series render on top
4. **LightweightCharts limitations** - Some built-in series have hardcoded behavior

