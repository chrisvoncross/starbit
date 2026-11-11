# How to Connect a New Plugin to the Chart

When you create a new plugin in the `lightweight_charts/plugins/` directory, you need to register it with the FastAPI server so it can be served to the browser.

## Steps to Connect a New Plugin

### 1. Create Your Plugin

Create a new directory and JavaScript file for your plugin:

```
lightweight_charts/plugins/
  └── my-new-plugin/
      └── my-new-plugin.js
```

### 2. Add Server Route

Open `lightweight_charts/websocket_server.py` and add a new route for your plugin.

**Find this section** (near the end of the file, around line 490-500):

```python
@app.get("/plugins/zscore-table/zscore-table-plugin.js")
async def serve_zscore_table_plugin():
    """Serve Z-Score Table Plugin"""
    with open("plugins/zscore-table/zscore-table-plugin.js", "r", encoding="utf-8") as f:
        js_content = f.read()
    from fastapi.responses import Response
    return Response(content=js_content, media_type="application/javascript")
```

**Add your new route** (copy-paste and modify):

```python
@app.get("/plugins/my-new-plugin/my-new-plugin.js")
async def serve_my_new_plugin():
    """Serve My New Plugin"""
    with open("plugins/my-new-plugin/my-new-plugin.js", "r", encoding="utf-8") as f:
        js_content = f.read()
    from fastapi.responses import Response
    return Response(content=js_content, media_type="application/javascript")
```

### 3. Import in Your Indicator/Chart

In your indicator file (e.g., `indicators/zscore/zscore.js` or `chart_realtime.js`), import the plugin:

```javascript
import { MyNewPlugin } from '/plugins/my-new-plugin/my-new-plugin.js';
```

**Note:** The path starts with `/plugins/` (not `/lightweight_charts/plugins/`)

### 4. Restart the Server

**IMPORTANT:** You MUST restart the WebSocket server for the new route to take effect!

```bash
# Stop the server (Ctrl+C)
# Start it again
python lightweight_charts/websocket_server.py
```

### 5. Hard Refresh Browser

After restarting the server, do a **hard refresh** in your browser:
- **Windows/Linux:** `Ctrl + Shift + R` or `Ctrl + F5`
- **Mac:** `Cmd + Shift + R`

## Example: Existing Plugins

Here are the existing plugins and their routes:

| Plugin | File Path | Server Route |
|--------|-----------|--------------|
| Fill Areas | `plugins/fill-areas/fill-areas.js` | `/plugins/fill-areas/fill-areas.js` |
| Horizontal Lines | `plugins/horizontal-lines/horizontal-lines.js` | `/plugins/horizontal-lines/horizontal-lines.js` |
| Price Level Labels | `plugins/price-level-labels/price-level-labels.js` | `/plugins/price-level-labels/price-level-labels.js` |
| Z-Score Table | `plugins/zscore-table/zscore-table-plugin.js` | `/plugins/zscore-table/zscore-table-plugin.js` |
| Z-Score Solid Fill | `plugins/zscore-solid-fill/zscore-solid-fill.js` | `/plugins/zscore-solid-fill/zscore-solid-fill.js` |
| Buy/Sell Labels | `plugins/buy_sell_labels/labels.js` | `/plugins/buy_sell_labels/labels.js` |

## Troubleshooting

### 404 Error: Plugin Not Found

**Problem:** Browser shows `GET http://localhost:8000/plugins/my-plugin/my-plugin.js net::ERR_ABORTED 404 (Not Found)`

**Solution:**
1. Check that you added the route in `websocket_server.py`
2. **Restart the server** (this is the most common issue!)
3. Hard refresh the browser
4. Check that the file path in the route matches the actual file location

### Plugin Not Loading

**Problem:** No error, but plugin doesn't work

**Solution:**
1. Check browser console for JavaScript errors
2. Verify the plugin exports the correct class/function
3. Check that you're importing with the correct name
4. Verify the plugin is attached to the correct series

## Why Do We Need Server Routes?

The FastAPI server doesn't automatically serve all files like a simple HTTP server. Each file must have an explicit route defined. This gives us more control over what files are served and allows for future enhancements like:

- Minification
- Caching headers
- Authentication
- Dynamic content injection

## Future Improvement

Consider adding a wildcard route to automatically serve all plugin files:

```python
@app.get("/plugins/{plugin_name}/{file_name}")
async def serve_plugin(plugin_name: str, file_name: str):
    """Serve any plugin file"""
    file_path = f"plugins/{plugin_name}/{file_name}"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        from fastapi.responses import Response
        return Response(content=content, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Plugin not found")
```

This would eliminate the need to add a route for each new plugin.

