# Horizontal Lines Plugin

Ein Plugin für Lightweight Charts, das horizontale Linien und Zonen mit dynamischer Breite zeichnet.

## 🎯 Problem & Lösung

### Das Problem
Ursprünglich wurden die Zonen bis zur **Canvas-Breite** gezeichnet. Das führte dazu, dass:
- Die Zonen über die Preisskala hinausgingen
- Die Zonen nicht mit dem tatsächlichen Chart-Ende übereinstimmten
- Beim Scrollen die Zonen nicht korrekt positioniert waren

### Die Lösung
Die Zonen werden jetzt bis zur **X-Koordinate des letzten Datenpunkts** in der Serie gezeichnet:

```javascript
// Get the actual data from the series
const seriesData = this._series.data();

if (seriesData && seriesData.length > 0) {
    // Get the time of the last data point
    const lastDataPoint = seriesData[seriesData.length - 1];
    const lastTime = lastDataPoint.time;
    
    // Convert time to X coordinate
    lastBarX = timeScale.timeToCoordinate(lastTime);
}
```

**Warum funktioniert das?**
- `series.data()` gibt alle Datenpunkte der Serie zurück
- Der letzte Datenpunkt ist `seriesData[seriesData.length - 1]`
- `timeScale.timeToCoordinate(time)` konvertiert die Zeit zu einer X-Pixel-Koordinate
- Diese X-Koordinate ist **genau dort, wo der letzte Balken/Datenpunkt endet**

## 🔧 Verwendung

```javascript
const horizontalLines = new HorizontalLinesPlugin(
    [
        { price: 1.0, color: 'rgba(255, 0, 0, 0.5)', width: 2 },
        { price: 0.0, color: 'rgba(0, 255, 0, 0.5)', width: 2 }
    ],
    [
        { topPrice: 3.0, bottomPrice: 2.0, color: 'rgba(139, 0, 0, 0.2)' },
        { topPrice: 2.0, bottomPrice: 1.0, color: 'rgba(128, 128, 0, 0.2)' }
    ],
    series,
    chart
);
```

## 📊 Features

- ✅ Horizontale Linien mit konfigurierbarer Farbe und Breite
- ✅ Farbige Zonen zwischen zwei Preislevels
- ✅ Dynamische Breite basierend auf dem letzten Datenpunkt
- ✅ Automatisches Update bei neuen Daten
- ✅ Funktioniert beim Scrollen und Zoomen

## 🎨 Rendering-Reihenfolge

1. **Zonen** werden zuerst gezeichnet (Hintergrund)
2. **Linien** werden darüber gezeichnet (Vordergrund)

Dies stellt sicher, dass die Linien immer sichtbar sind, auch wenn sie auf einer Zonengrenze liegen.

## 🔍 Debugging

Das Plugin enthält Debug-Logs, die bei Bedarf aktiviert werden können:
- `lastBarX`: Die X-Koordinate des letzten Datenpunkts
- `endX`: Die tatsächlich verwendete Breite für das Zeichnen
- `mediaSize.width`: Die gesamte Canvas-Breite

## 📝 Wichtige Erkenntnisse

1. **Canvas-Koordinaten vs. Daten-Koordinaten**: 
   - Canvas-Breite ist NICHT gleich der Position des letzten Datenpunkts
   - Beim Scrollen ändert sich die Position des letzten Datenpunkts relativ zur Canvas

2. **TimeScale-Methoden**:
   - `timeToCoordinate(time)`: Zeit → X-Pixel-Koordinate
   - `logicalToCoordinate(logical)`: Logischer Index → X-Pixel-Koordinate
   - `getVisibleLogicalRange()`: Gibt den sichtbaren Bereich zurück (nicht den gesamten Datenbereich!)

3. **Series-Daten**:
   - `series.data()`: Gibt ALLE Datenpunkte zurück (nicht nur sichtbare)
   - Der letzte Datenpunkt kann außerhalb des sichtbaren Bereichs liegen

