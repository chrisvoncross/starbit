import { setupNWEBands, toggleNWEBands } from '../indicators/nwe/nwe.js';
import { setupKyleSignals } from '../indicators/kyle_algo/kyle.js';
import { createZScorePane, updateZScoreData } from '../indicators/zscore/zscore.js';
import { createStochObvPane, updateStochObv } from '../indicators/stoch_slow/stoch_slow.js';
import { paneManager } from '../indicators/pane_manager.js';
import { createIndicatorPanel, registerIndicator, isIndicatorVisible, isFeatureVisible } from '../indicators/indicator_panel.js?v=zerofees4';
import { paintSwingTurns } from '../indicators/turns/turns.js';
import { paintAtrZigZag } from '../indicators/atr_zigzag/atr_zigzag.js';
import { loadLuma, paintLuma } from '../indicators/luma/luma.js?v=luma099';
import { loadStarbit, paintStarbit } from '../indicators/starbit/starbit.js?v=starbit3';
import { loadLumaTrain, paintLumaTrain } from '../indicators/luma_train/luma_train.js?v=lageseq';
import { paintNordstern } from '../indicators/nordstern/nordstern.js?v=zerofees6';
import { paintZeroFees } from '../indicators/zero_fees/zero_fees.js?v=zerofees6';
import { paintEntropie } from '../indicators/entropie/entropie.js';
import { paintEntropiev2 } from '../indicators/entropiev2/entropiev2.js';
import { paintArcus } from '../indicators/arcus/arcus.js';
import { paintUmbra } from '../indicators/umbra/umbra.js';
import { loadPhase1Model, paintPhase1Model, phase1ArrowMarkers } from '../indicators/turns/phase1_model_exam.js';
import { loadCosmo, paintCosmo } from '../indicators/cosmo/cosmo.js';
import { loadNova, paintNova } from '../indicators/nova/nova.js';

export class RealtimeController {
    constructor({ chartCtx, statusEl, FillAreasPluginClass }) {
        this.chartCtx = chartCtx;
        this.statusEl = statusEl;
        this.FillAreasPluginClass = FillAreasPluginClass;
        this.ws = null;
        this.reconnectInterval = null;
        this.pingInterval = null;
        this.lazyLoadingBound = false;

        this.zscoreIndicator = null;
        this.stochObvIndicator = null;

        this.isLoadingMore = false;
        this.oldestLoadedTime = null;
        this.hasMoreData = true;
        this.chunkSize = 500;

        this.lastHaOpen = null;
        this.lastHaClose = null;

        this.allNweBands = { upper_band: [], mid_band: [], lower_band: [], width: [], position: [] };
        this.allKyleSignals = [];
        this.allZscoreData = { zscore: [], zscore_long: [], price_levels: [] };
        this.allStochData = { stoch_k: [] };

        this.panelSetupDone = false;
    }

    init() {
        createIndicatorPanel(this.chartCtx.chartContainer);
        paneManager.init(this.chartCtx.chart);

        window._switchChartStyle = (style) => {
            this.chartCtx.switchChartStyle(style);
            this.paintTurns();
        };
        window._getChartStyle = () => this.chartCtx.getChartStyle();

        window.addEventListener('resize', () => {
            this.chartCtx.chart.applyOptions({
                width: this.chartCtx.chartContainer.clientWidth,
                height: this.chartCtx.chartContainer.clientHeight,
            });
        });

        this.connectWebSocket();
        this.pingInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    }

    updateStatus(message, className) {
        this.statusEl.textContent = message;
        this.statusEl.className = `status ${className}`;
    }

    computeHeikinAshi(raw) {
        const haClose = (raw.open + raw.high + raw.low + raw.close) / 4;
        const haOpen =
            this.lastHaOpen !== null && this.lastHaClose !== null
                ? (this.lastHaOpen + this.lastHaClose) / 2
                : (raw.open + raw.close) / 2;
        const haHigh = Math.max(raw.high, haOpen, haClose);
        const haLow = Math.min(raw.low, haOpen, haClose);
        this.lastHaOpen = haOpen;
        this.lastHaClose = haClose;
        return { time: raw.time, open: haOpen, high: haHigh, low: haLow, close: haClose };
    }

    seedHeikinAshiFromHistory() {
        const active = this.chartCtx.getActiveSeries();
        const allData = active.data();
        if (!allData || allData.length === 0) return;
        const last = allData[allData.length - 1];
        this.lastHaOpen = last.open;
        this.lastHaClose = last.close;
        console.log(`🔗 HA state seeded from history: open=${this.lastHaOpen.toFixed(2)}, close=${this.lastHaClose.toFixed(2)}`);
    }

    async loadSimulatedTrades() {
        try {
            const response = await fetch('/api/trades');
            const data = await response.json();
            if (data.trades && data.trades.length > 0) {
                this.chartCtx.plugins.buySellPositions.updateTrades(data.trades);
            }
        } catch (error) {
            console.error('❌ Failed to load simulated trades:', error);
        }
    }

    setupLazyLoading() {
        if (this.lazyLoadingBound) return;
        this.lazyLoadingBound = true;

        const active = this.chartCtx.getActiveSeries();
        const allData = active.data();
        if (allData && allData.length > 0) this.oldestLoadedTime = allData[0].time;

        this.chartCtx.chart.timeScale().subscribeVisibleLogicalRangeChange((logicalRange) => {
            if (!logicalRange || this.isLoadingMore || !this.hasMoreData) return;
            if (logicalRange.from < 20) this.requestMoreData();
        });
    }

    requestMoreData() {
        if (
            this.isLoadingMore ||
            !this.hasMoreData ||
            !this.ws ||
            this.ws.readyState !== WebSocket.OPEN
        ) {
            return;
        }
        this.isLoadingMore = true;
        this.ws.send(
            JSON.stringify({
                type: 'load_more',
                before: this.oldestLoadedTime,
                limit: this.chunkSize,
            })
        );
    }

    connectWebSocket() {
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        const wsUrl = `${proto}://${location.host}/ws`;
        this.updateStatus('🔄 Connecting...', 'loading');
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.updateStatus('✅ Connected - Realtime', 'connected');
            if (this.reconnectInterval) {
                clearInterval(this.reconnectInterval);
                this.reconnectInterval = null;
            }
        };

        this.ws.onmessage = async (event) => {
            try {
                const message = JSON.parse(event.data);
                if (message.type === 'initial_data') {
                    await this.handleInitialData(message);
                    if (message.error) this.updateStatus(`❌ ${message.error}`, 'disconnected');
                } else if (message.type === 'more_data') {
                    this.handleMoreData(message);
                } else if (message.type === 'candle_update') {
                    this.handleCandleUpdate(message);
                } else if (message.type === 'indicator_update') {
                    this.handleIndicatorUpdate(message);
                }
            } catch (error) {
                console.error('❌ Error parsing message:', error);
                this.isLoadingMore = false;
            }
        };

        this.ws.onerror = (error) => {
            console.error('❌ WebSocket error:', error);
            this.updateStatus('❌ Connection error', 'disconnected');
        };

        this.ws.onclose = () => {
            this.updateStatus('🔄 Reconnecting...', 'loading');
            if (!this.reconnectInterval) {
                this.reconnectInterval = setInterval(() => this.connectWebSocket(), 3000);
            }
        };
    }

    async handleInitialData(message) {
        if (message.chunk_size) this.chunkSize = message.chunk_size;
        this.hasMoreData = message.has_more !== false;

        this.chartCtx.applyPrice(message.data);
        this.chartCtx.getActiveSeries().setData(message.data);
        if (message.raw_data && message.raw_data.length > 0) {
            this.chartCtx.setRawOhlcData(message.raw_data);
        }
        if (this.chartCtx.getActiveSeries() !== this.chartCtx.series.haSeries) {
            this.chartCtx.series.haSeries.setData(message.data);
        }

        if (message.nwe_bands) {
            this.allNweBands = message.nwe_bands;
            if (isIndicatorVisible('nwe')) {
                setupNWEBands(this.chartCtx.plugins.fillAreas, message.nwe_bands);
            }
        }
        if (message.kyle_signals) {
            this.allKyleSignals = message.kyle_signals;
            setupKyleSignals(this.chartCtx.plugins.labels, message.kyle_signals);
        }
        if (message.zscore_data) this.allZscoreData = message.zscore_data;
        if (message.stoch_obv_data) this.allStochData = message.stoch_obv_data;

        paneManager.register(
            'zscore',
            {
                height: 150,
                create: async (ch, paneIndex) => {
                    if (!this.allZscoreData?.zscore?.length) return [];
                    this.zscoreIndicator = await createZScorePane(ch, this.allZscoreData, this.FillAreasPluginClass, paneIndex);
                    return this.zscoreIndicator._allSeries || [];
                },
                destroy: (ch, seriesList) => {
                    for (const s of seriesList) {
                        try {
                            ch.removeSeries(s);
                        } catch (e) {
                            // no-op
                        }
                    }
                    this.zscoreIndicator = null;
                },
            },
            false
        );

        paneManager.register(
            'stoch_obv',
            {
                height: 150,
                create: (ch, paneIndex) => {
                    if (!this.allStochData?.stoch_k?.length) return [];
                    this.stochObvIndicator = createStochObvPane(ch, this.allStochData, paneIndex);
                    return [this.stochObvIndicator.stochKSeries].filter(Boolean);
                },
                destroy: (ch, seriesList) => {
                    for (const s of seriesList) {
                        try {
                            ch.removeSeries(s);
                        } catch (e) {
                            // no-op
                        }
                    }
                    this.stochObvIndicator = null;
                },
            },
            false
        );
        paneManager.rebuild();

        if (!this.panelSetupDone) {
            this.panelSetupDone = true;
            registerIndicator(
                'nwe',
                { fillAreas: this.chartCtx.plugins.fillAreas, nweData: this.allNweBands },
                (visible) => {
                    if (visible) setupNWEBands(this.chartCtx.plugins.fillAreas, this.allNweBands);
                    else toggleNWEBands(this.chartCtx.plugins.fillAreas, false);
                    this.chartCtx.chart.timeScale().applyOptions({});
                },
                {
                    outer_upper: (v) => {
                        if (v) this.chartCtx.plugins.fillAreas.addZone('nwe_outer_upper', this.allNweBands.upper_far, this.allNweBands.upper_avg, 'rgba(239, 83, 80, 0.20)', {});
                        else this.chartCtx.plugins.fillAreas.removeZone('nwe_outer_upper');
                        this.chartCtx.chart.timeScale().applyOptions({});
                    },
                    inner_upper: (v) => {
                        if (v) this.chartCtx.plugins.fillAreas.addZone('nwe_inner_upper', this.allNweBands.upper_avg, this.allNweBands.upper_near, 'rgba(239, 83, 80, 0.35)', {});
                        else this.chartCtx.plugins.fillAreas.removeZone('nwe_inner_upper');
                        this.chartCtx.chart.timeScale().applyOptions({});
                    },
                    inner_lower: (v) => {
                        if (v) this.chartCtx.plugins.fillAreas.addZone('nwe_inner_lower', this.allNweBands.lower_near, this.allNweBands.lower_avg, 'rgba(38, 166, 154, 0.35)', {});
                        else this.chartCtx.plugins.fillAreas.removeZone('nwe_inner_lower');
                        this.chartCtx.chart.timeScale().applyOptions({});
                    },
                    outer_lower: (v) => {
                        if (v) this.chartCtx.plugins.fillAreas.addZone('nwe_outer_lower', this.allNweBands.lower_avg, this.allNweBands.lower_far, 'rgba(38, 166, 154, 0.20)', {});
                        else this.chartCtx.plugins.fillAreas.removeZone('nwe_outer_lower');
                        this.chartCtx.chart.timeScale().applyOptions({});
                    },
                }
            );

            registerIndicator(
                'zscore',
                {},
                (visible) => paneManager.toggle('zscore', visible),
                {
                    solid_fill: (v) => {
                        if (this.zscoreIndicator?.zscoreSolidFill) this.zscoreIndicator.zscoreSolidFill.setVisible(v);
                    },
                    sma_line: (v) => {
                        if (this.zscoreIndicator?.zscoreSMASeries) this.zscoreIndicator.zscoreSMASeries.applyOptions({ visible: v });
                    },
                    prob_zones: (v) => {
                        if (v && this.zscoreIndicator?.zscoreFillAreas) this.zscoreIndicator.zscoreFillAreas.applyOptions({ showLines: false });
                    },
                    table: (v) => {
                        if (this.zscoreIndicator?.zscoreTablePlugin) this.zscoreIndicator.zscoreTablePlugin.setVisible(v);
                    },
                }
            );

            registerIndicator(
                'stoch_obv',
                {},
                (visible) => paneManager.toggle('stoch_obv', visible),
                {
                    stoch_k: (v) => {
                        if (this.stochObvIndicator?.stochKSeries) this.stochObvIndicator.stochKSeries.applyOptions({ visible: v });
                    },
                }
            );

            const repaintTurns = () => this.paintTurns();
            registerIndicator(
                'arcus',
                {},
                () => this.paintArcus()
            );

            registerIndicator(
                'umbra',
                {},
                () => this.paintUmbra()
            );

            registerIndicator(
                'atr_zigzag',
                {},
                () => this.paintAtrZigZag()
            );

            registerIndicator('luma', {}, () => this.paintLuma());
            registerIndicator('luma_train', {}, () => this.paintLumaTrain(), {
                marker: () => this.paintLumaTrain(),
                exploration: () => this.paintLumaTrain(),
                lage: () => this.paintLumaTrain(),
                imagination: () => this.paintLumaTrain(),
                strecken: () => this.paintLumaTrain(),
                liste: () => this.paintLumaTrain(),
            });
            registerIndicator('nordstern', {}, () => this.paintNordstern(), {
                sterne: () => this.paintNordstern(),
                positionen: () => this.paintNordstern(),
                tabelle: () => this.paintNordstern(),
            });
            registerIndicator('zero_fees', {}, () => this.paintZeroFees(), {
                sterne: () => this.paintZeroFees(),
                positionen: () => this.paintZeroFees(),
                tabelle: () => this.paintZeroFees(),
            });

            registerIndicator('entropie', {}, () => this.paintEntropie());
            registerIndicator('entropiev2', {}, () => this.paintEntropiev2());
            registerIndicator('cosmo', {}, () => this.paintCosmo());
            registerIndicator('nova', {}, () => this.paintNova());

            registerIndicator(
                'turns',
                {},
                repaintTurns,
                {
                    bp1: repaintTurns, bp2: repaintTurns, bp3: repaintTurns, bp4: repaintTurns,
                    bp5: repaintTurns, phase1: repaintTurns, bp10: repaintTurns, bp20: repaintTurns, bp50: repaintTurns,
                }
            );
        }

        this.paintTurns();
        this.paintUmbra();
        this.paintArcus();
        loadPhase1Model(() => this.paintTurns());
        loadCosmo(() => this.paintCosmo());
        loadNova(() => this.paintNova());
        loadLuma(() => this.paintLuma());
        loadLumaTrain(() => this.paintLumaTrain());

        this.updateStatus(`✅ Connected - ${message.data.length} candles (lazy)`, 'connected');
        this.seedHeikinAshiFromHistory();
        this.setupLazyLoading();
        await this.loadSimulatedTrades();

        const totalCandles = message.data.length;
        const visibleCandles = 240;
        const show = () => {
            const scale = this.chartCtx.chart.timeScale();
            if (totalCandles <= 2) {
                scale.fitContent();
                return;
            }
            const bars = Math.min(visibleCandles, totalCandles);
            const width = typeof scale.width === 'function' ? scale.width() : 0;
            if (width > 0) {
                scale.applyOptions({
                    barSpacing: Math.max(0.4, (width - 4) / bars),
                    minBarSpacing: 0.4,
                    rightOffset: 0,
                });
            }
            const end = message.has_more === false ? bars - 1 : totalCandles - 1;
            const start = message.has_more === false ? 0 : totalCandles - bars;
            scale.setVisibleLogicalRange({ from: start, to: end });
        };
        show();
        requestAnimationFrame(show);
    }

    paintArcus() {
        paintArcus({
            chart: this.chartCtx.chart,
            enabled: isIndicatorVisible('arcus'),
            onReady: () => this.paintTurns(),
        });
    }

    paintUmbra() {
        paintUmbra({
            chart: this.chartCtx.chart,
            series: this.chartCtx.getActiveSeries(),
            enabled: isIndicatorVisible('umbra'),
            onReady: () => this.paintTurns(),
        });
    }

    paintTurns() {
        const series = this.chartCtx.getActiveSeries();
        const candles = typeof series.data === 'function' ? series.data() : [];
        const phase1On = isIndicatorVisible('turns') && isFeatureVisible('turns', 'phase1');
        paintSwingTurns({
            chart: this.chartCtx.chart,
            series,
            candles,
            enabled: isIndicatorVisible('turns'),
            activeLevels: {
                bp1: isFeatureVisible('turns', 'bp1'),
                bp2: isFeatureVisible('turns', 'bp2'),
                bp3: isFeatureVisible('turns', 'bp3'),
                bp4: isFeatureVisible('turns', 'bp4'),
                bp5: isFeatureVisible('turns', 'bp5'),
                bp10: isFeatureVisible('turns', 'bp10'),
                bp20: isFeatureVisible('turns', 'bp20'),
                bp50: isFeatureVisible('turns', 'bp50'),
            },
            overlayMarkers: phase1ArrowMarkers(phase1On),
        });
        paintPhase1Model({
            chart: this.chartCtx.chart,
            enabled: phase1On,
        });
        this.paintAtrZigZag();
        this.paintLuma();
        this.paintLumaTrain();
        this.paintEntropie();
        this.paintEntropiev2();
        this.paintCosmo();
        this.paintNova();
        this.paintNordstern();
        this.paintZeroFees();
    }

    paintZeroFees() {
        const series = this.chartCtx.getActiveSeries();
        const raw = this.chartCtx.getRawOhlcData();
        const candles = raw && raw.length ? raw : (typeof series.data === 'function' ? series.data() : []);
        paintZeroFees({
            chart: this.chartCtx.chart,
            series,
            candles,
            enabled: isIndicatorVisible('zero_fees'),
            showStars: isFeatureVisible('zero_fees', 'sterne'),
            showPositions: isFeatureVisible('zero_fees', 'positionen'),
            showTable: isFeatureVisible('zero_fees', 'tabelle'),
        });
    }

    paintNordstern() {
        const series = this.chartCtx.getActiveSeries();
        const raw = this.chartCtx.getRawOhlcData();
        const candles = raw && raw.length ? raw : (typeof series.data === 'function' ? series.data() : []);
        paintNordstern({
            chart: this.chartCtx.chart,
            series,
            candles,
            enabled: isIndicatorVisible('nordstern'),
            showStars: isFeatureVisible('nordstern', 'sterne'),
            showPositions: isFeatureVisible('nordstern', 'positionen'),
            showTable: isFeatureVisible('nordstern', 'tabelle'),
        });
    }

    paintAtrZigZag() {
        const series = this.chartCtx.getActiveSeries();
        const candles = typeof series.data === 'function' ? series.data() : [];
        paintAtrZigZag({
            chart: this.chartCtx.chart,
            series,
            candles,
            enabled: isIndicatorVisible('atr_zigzag'),
        });
    }

    paintLuma() {
        const series = this.chartCtx.getActiveSeries();
        const candles = typeof series.data === 'function' ? series.data() : [];
        const raw = this.chartCtx.getRawOhlcData();
        paintLuma({
            chart: this.chartCtx.chart,
            series,
            candles: raw && raw.length ? raw : candles,
            enabled: isIndicatorVisible('luma'),
        });
    }

    paintLumaTrain() {
        const series = this.chartCtx.getActiveSeries();
        const candles = typeof series.data === 'function' ? series.data() : [];
        const raw = this.chartCtx.getRawOhlcData();
        const drawn = raw && raw.length ? raw : candles;
        loadLumaTrain(() => {
            paintLumaTrain({
                chart: this.chartCtx.chart,
                series,
                candles: drawn,
                enabled: isIndicatorVisible('luma_train'),
                showMarker: isFeatureVisible('luma_train', 'marker'),
                showExploration: isFeatureVisible('luma_train', 'exploration'),
                showLage: isFeatureVisible('luma_train', 'lage'),
                showImagination: isFeatureVisible('luma_train', 'imagination'),
                showStrecken: isFeatureVisible('luma_train', 'strecken'),
                showListe: isFeatureVisible('luma_train', 'liste'),
            });
        });
    }

    paintEntropie() {
        const series = this.chartCtx.getActiveSeries();
        const candles = typeof series.data === 'function' ? series.data() : [];
        paintEntropie({ chart: this.chartCtx.chart, candles, enabled: isIndicatorVisible('entropie') });
    }

    paintEntropiev2() {
        const series = this.chartCtx.getActiveSeries();
        const candles = typeof series.data === 'function' ? series.data() : [];
        paintEntropiev2({ chart: this.chartCtx.chart, candles, enabled: isIndicatorVisible('entropiev2') });
    }

    paintCosmo() {
        paintCosmo({ chart: this.chartCtx.chart, enabled: isIndicatorVisible('cosmo') });
    }

    paintNova() {
        const series = this.chartCtx.getActiveSeries();
        const candles = typeof series.data === 'function' ? series.data() : [];
        paintNova({ chart: this.chartCtx.chart, series, candles, enabled: isIndicatorVisible('nova') });
    }

    handleMoreData(message) {
        const moreCandles = message.data || [];
        this.hasMoreData = message.has_more !== false;
        if (moreCandles.length === 0) {
            this.hasMoreData = false;
            this.isLoadingMore = false;
            return;
        }

        const active = this.chartCtx.getActiveSeries();
        const currentData = active.data();
        const newData = [...moreCandles, ...currentData];
        active.setData(newData);

        const moreRaw = message.raw_data || [];
        if (moreRaw.length > 0) {
            this.chartCtx.setRawOhlcData([...moreRaw, ...this.chartCtx.getRawOhlcData()]);
        }
        if (active !== this.chartCtx.series.haSeries) this.chartCtx.series.haSeries.setData(newData);
        this.oldestLoadedTime = moreCandles[0].time;

        if (message.nwe_bands) {
            for (const key of Object.keys(this.allNweBands)) {
                const newBands = message.nwe_bands[key] || [];
                this.allNweBands[key] = [...newBands, ...this.allNweBands[key]];
            }
            if (isIndicatorVisible('nwe')) setupNWEBands(this.chartCtx.plugins.fillAreas, this.allNweBands);
        }

        if (message.zscore_data && this.zscoreIndicator) {
            for (const key of Object.keys(this.allZscoreData)) {
                const inc = message.zscore_data[key] || [];
                this.allZscoreData[key] = [...inc, ...this.allZscoreData[key]];
            }
            if (isIndicatorVisible('zscore')) updateZScoreData(this.zscoreIndicator, this.allZscoreData);
        }

        if (message.stoch_obv_data && this.stochObvIndicator) {
            this.allStochData.stoch_k = [...(message.stoch_obv_data.stoch_k || []), ...this.allStochData.stoch_k];
            if (isIndicatorVisible('stoch_obv')) updateStochObv(this.stochObvIndicator, this.allStochData);
        }

        this.updateStatus(`✅ ${newData.length} candles loaded`, 'connected');
        this.isLoadingMore = false;
    }

    handleCandleUpdate(message) {
        const raw = message.data;
        if (!raw || raw.time == null) return;
        raw.time = typeof raw.time === 'object' ? raw.time.timestamp || 0 : Number(raw.time);
        const allData = this.chartCtx.series.haSeries.data();
        const lastTime = allData.length > 0 ? allData[allData.length - 1].time : 0;
        const lastNum = typeof lastTime === 'object' ? lastTime.timestamp || 0 : Number(lastTime);
        if (raw.time < lastNum) return;

        const haCandle = this.computeHeikinAshi(raw);
        this.chartCtx.series.haSeries.update(haCandle);
        this.chartCtx.series.rawSeries.update(raw);
        this.chartCtx.series.barSeries.update(raw);
        this.chartCtx.upsertRawCandle(raw);
        this.updateStatus('✅ Connected - Realtime streaming', 'connected');
    }

    handleIndicatorUpdate(message) {
        const ind = message.data;
        if (!ind || !ind.time) return;
        const t = Number(ind.time);
        const allData = this.chartCtx.getActiveSeries().data();
        const lastCandleTime =
            allData.length > 0
                ? typeof allData[allData.length - 1].time === 'object'
                    ? allData[allData.length - 1].time.timestamp || 0
                    : Number(allData[allData.length - 1].time)
                : 0;
        if (t < lastCandleTime) return;

        const fillAreas = this.chartCtx.plugins.fillAreas;
        if (ind.nwe && fillAreas) {
            const nwePoint = (key) => ({ time: t, value: ind.nwe[key] });
            const keys = ['upper_far', 'upper_avg', 'upper_near', 'lower_near', 'lower_avg', 'lower_far', 'yhat'];
            for (const k of keys) {
                if (ind.nwe[k] != null) {
                    if (!this.allNweBands[k]) this.allNweBands[k] = [];
                    this.allNweBands[k].push(nwePoint(k));
                }
            }
            if (isIndicatorVisible('nwe')) {
                if (this.allNweBands.upper_far && this.allNweBands.upper_avg)
                    fillAreas.addZone('nwe_outer_upper', this.allNweBands.upper_far, this.allNweBands.upper_avg, 'rgba(239, 83, 80, 0.20)', {});
                if (this.allNweBands.upper_avg && this.allNweBands.upper_near)
                    fillAreas.addZone('nwe_inner_upper', this.allNweBands.upper_avg, this.allNweBands.upper_near, 'rgba(239, 83, 80, 0.35)', {});
                if (this.allNweBands.lower_near && this.allNweBands.lower_avg)
                    fillAreas.addZone('nwe_inner_lower', this.allNweBands.lower_near, this.allNweBands.lower_avg, 'rgba(38, 166, 154, 0.35)', {});
                if (this.allNweBands.lower_avg && this.allNweBands.lower_far)
                    fillAreas.addZone('nwe_outer_lower', this.allNweBands.lower_avg, this.allNweBands.lower_far, 'rgba(38, 166, 154, 0.20)', {});
            }
        }

        if (ind.zscore && this.zscoreIndicator) {
            const zVal = Number(ind.zscore.value);
            if (this.zscoreIndicator.zscoreAreaSeries) this.zscoreIndicator.zscoreAreaSeries.update({ time: t, value: zVal });
            if (this.zscoreIndicator.baseSeries) this.zscoreIndicator.baseSeries.update({ time: t, value: 0 });
            if (this.zscoreIndicator.zscoreSolidFill) {
                const zData = this.zscoreIndicator.zscoreSolidFill._zscoreData;
                const lastIdx = zData.length > 0 ? zData.length - 1 : -1;
                if (lastIdx >= 0 && zData[lastIdx].time === t) zData[lastIdx].value = zVal;
                else zData.push({ time: t, value: zVal });
                this.zscoreIndicator.zscoreSolidFill.requestUpdate();
            }
        }
        if (ind.zscore_long && this.zscoreIndicator?.zscoreSMASeries) {
            this.zscoreIndicator.zscoreSMASeries.update({ time: t, value: Number(ind.zscore_long.value) });
        }
        if (ind.stoch_k && this.stochObvIndicator?.stochKSeries) {
            this.stochObvIndicator.stochKSeries.update({ time: t, value: Number(ind.stoch_k.value) });
        }
    }
}
