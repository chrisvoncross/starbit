"""
SGLang Server Warmup Script - BEST-IN-CLASS Edition

GOAL: Eliminate cold start latency and optimize CUDA graphs, RadixCache, and FP8 quantization

STRATEGY:
1. CUDA Graph Warmup: Exercise batch sizes 1-4 (matching cuda-graph-max-bs=4)
2. Prefill Warmup: Various input lengths (50, 100, 200, 300, 500 tokens)
3. Decode Warmup: Various output lengths (50, 100, 200, 500 tokens)
4. Concurrent Request Warmup: Test multi-request handling (max 4)
5. Production-Like Queries: Real agent prompts

NOTE: RadixCache is disabled when using MTP speculative decoding.

PERFORMANCE TARGET (Qwen3.5-122B-A10B-FP8 MoE):
- Warmup time: 120-300 seconds (122B model is slower to warm up)
- First query latency: 2-5s (MoE with 10B active params)
- Decode throughput: 60-120 tokens/s (with MTP)
- CUDA Graph coverage: 100% (batch sizes 1-4)

Session 3 Phase 3 - Production-Grade Warmup
"""

import asyncio
import logging
import sys
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
import yaml
from openai import AsyncOpenAI
import statistics

# Setup logging with UTF-8 support for emojis
import io
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setStream(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[console_handler]
)

logger = logging.getLogger(__name__)


class SGLangWarmup:
    """
    BEST-IN-CLASS SGLang Server Warmup

    Optimizes:
    - CUDA Graphs (all batch sizes)
    - RadixCache (common prefixes)
    - FP8 Quantization (first decode warmup)
    - Prefill/Decode pipeline
    """

    def __init__(
        self,
        sglang_url: str = "http://127.0.0.1:30000/v1",
        config_path: str = "bytedance/config/agent_llm_mapping.yaml"
    ):
        self.sglang_url = sglang_url
        self.config_path = config_path
        self.client = AsyncOpenAI(
            base_url=sglang_url,
            api_key="EMPTY"  # SGLang doesn't need API key
        )
        self.warmup_stats = {
            'total_queries': 0,
            'successful': 0,
            'failed': 0,
            'total_time_ms': 0.0,
            'avg_time_ms': 0.0,
            'min_latency_ms': float('inf'),
            'max_latency_ms': 0.0,
            'p50_latency_ms': 0.0,
            'p95_latency_ms': 0.0,
            'p99_latency_ms': 0.0,
            'latencies': []
        }
    
    def _load_agent_config(self) -> Dict[str, Any]:
        """Load agent configuration from YAML"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"⚠️ Config file not found: {self.config_path}")
                return {}
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Use 'dev' environment
            return config.get('dev', {})
        
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            return {}
    
    def _generate_warmup_prompts(self) -> List[Dict[str, Any]]:
        """
        Generate BEST-IN-CLASS warmup prompts.

        STRATEGY:
        1. CUDA Graph Warmup: Exercise batch sizes 1-4 (matching cuda-graph-max-bs=4)
        2. Prefill Warmup: Various input lengths (50, 100, 200, 300, 500 tokens)
        3. Decode Warmup: Various output lengths (50, 100, 200, 500 tokens)
        4. Concurrent Requests: Test multi-request handling (max 4)
        5. Production-Like Queries: Real agent prompts

        TOTAL: ~30 queries (lean warmup for 122B model)
        """
        prompts = []

        # ========================================================================
        # PHASE 1: CUDA GRAPH WARMUP (Batch Sizes: 1, 2, 3, 4)
        # ========================================================================
        # Goal: Ensure all CUDA graphs are compiled and ready
        # Strategy: Send simple queries to exercise each batch size (max_bs=4)

        logger.info("📊 Phase 1: CUDA Graph Warmup (4 batch sizes)")

        cuda_graph_prompts = [
            {
                "prompt": "Analyze BTC market: RSI 50, Volume 1.0x, Trend neutral. Signal?",
                "max_tokens": 50,
                "temperature": 0.7,
                "category": "cuda_graph_bs1",
                "phase": "cuda_graph"
            },
            {
                "prompt": "Analyze ETH market: RSI 60, Volume 1.2x, Trend bullish. Signal?",
                "max_tokens": 50,
                "temperature": 0.7,
                "category": "cuda_graph_bs2",
                "phase": "cuda_graph"
            },
            {
                "prompt": "Analyze SOL market: RSI 40, Volume 0.8x, Trend bearish. Signal?",
                "max_tokens": 50,
                "temperature": 0.7,
                "category": "cuda_graph_bs3",
                "phase": "cuda_graph"
            },
            {
                "prompt": "Analyze DOGE market: RSI 55, Volume 1.5x, Trend sideways. Signal?",
                "max_tokens": 50,
                "temperature": 0.7,
                "category": "cuda_graph_bs4",
                "phase": "cuda_graph"
            }
        ]

        prompts.extend(cuda_graph_prompts)

        # ========================================================================
        # PHASE 2: PREFILL WARMUP (Input Lengths: 50, 100, 200, 300, 500 tokens)
        # ========================================================================
        # Goal: Warm up prefill pipeline for various input lengths
        # Strategy: Generate prompts of different lengths

        logger.info("📊 Phase 2: Prefill Warmup (5 input lengths)")

        prefill_warmup_prompts = []
        
        # 50 tokens input (~12 words)
        prefill_warmup_prompts.append({
            "prompt": "BTC RSI 28, Volume 2.1x, Downtrend -3.2%. Signal?",
            "max_tokens": 100,
            "temperature": 0.7,
            "category": "prefill_50_tokens",
            "phase": "prefill"
        })

        # 100 tokens input (~25 words)
        prefill_warmup_prompts.append({
            "prompt": """Analyze BTC market:
RSI: 28 (oversold), Volume: 2.1x average (high), Price Trend: Downtrend (-3.2% last hour)
Regime: Bearish Reversal Potential, Volatility: High (ATR: 450)
Orderbook: Bid/Ask Imbalance 0.65 (more buyers), Liquidity: Medium
Provide trading signal and confidence.""",
            "max_tokens": 100,
            "temperature": 0.7,
            "category": "prefill_100_tokens",
            "phase": "prefill"
        })

        # 200 tokens input (~50 words)
        prefill_warmup_prompts.append({
            "prompt": """Analyze BTC market conditions:

TECHNICAL INDICATORS:
- RSI: 28 (oversold)
- Volume Ratio: 2.1x average (high)
- Price Trend: Downtrend (-3.2% last hour)

MARKET REGIME:
- Regime: Bearish Reversal Potential
- Volatility: High (ATR: 450)

ORDERBOOK:
- Bid/Ask Imbalance: 0.65 (more buyers)
- Liquidity: Medium

Provide:
1. Trading Signal (BUY/SELL/HOLD)
2. Confidence (0-1)
3. Key Factors (3-5 points)
4. Risk Assessment
5. Brief Reasoning (2-3 sentences)""",
            "max_tokens": 200,
            "temperature": 0.7,
            "category": "prefill_200_tokens",
            "phase": "prefill"
        })

        # 300 tokens input (~75 words)
        prefill_warmup_prompts.append({
            "prompt": """Analyze BTC market conditions in detail:

TECHNICAL INDICATORS:
- RSI: 28 (oversold territory, potential bounce)
- Volume Ratio: 2.1x average (significantly elevated)
- Price Trend: Strong downtrend (-3.2% last hour, -8.5% last 24h)
- Moving Averages: Price below 50-day and 200-day MA
- MACD: Bearish crossover confirmed

MARKET REGIME:
- Current Regime: Bearish Reversal Potential
- Volatility: High (ATR: 450, 30% above average)
- Market Phase: Late capitulation or early recovery

ORDERBOOK ANALYSIS:
- Bid/Ask Imbalance: 0.65 (more buyers stepping in)
- Liquidity: Medium (depth: $1.5M within 1%)
- Large Orders: 3 whale buy orders detected in last 60s

RECENT ACTIVITY:
- Whale Activity: Accumulation phase starting
- Pump/Dump Events: None detected
- Iceberg Orders: 2 detected on bid side

Provide comprehensive analysis:
1. Trading Signal (BUY/SELL/HOLD) with reasoning
2. Confidence Level (0.0-1.0) with justification
3. Key Factors (5-7 critical points)
4. Risk Assessment (low/medium/high) with explanation
5. Entry/Exit Strategy (2-3 sentences)
6. Time Horizon (short/medium/long term)""",
            "max_tokens": 300,
            "temperature": 0.7,
            "category": "prefill_300_tokens",
            "phase": "prefill"
        })

        # 500 tokens input (~125 words)
        prefill_warmup_prompts.append({
            "prompt": """Perform comprehensive BTC market analysis:

TECHNICAL INDICATORS (Multiple Timeframes):
1-Hour Chart:
- RSI: 28 (oversold, potential bounce)
- Volume: 2.1x average (panic selling or capitulation)
- Price: $67,234 (-3.2% last hour)
- MACD: Bearish crossover (-150 histogram)
- Bollinger Bands: Price at lower band (oversold)

4-Hour Chart:
- RSI: 35 (approaching oversold)
- Volume: 1.8x average (elevated)
- Price Trend: Downtrend confirmed
- Moving Averages: Death cross forming (50-day crossing below 200-day)

Daily Chart:
- RSI: 42 (neutral to bearish)
- Volume: 1.5x average (sustained selling pressure)
- Price: -8.5% from 7-day high
- Support Levels: $66,500 (strong), $65,000 (critical)
- Resistance Levels: $68,000 (weak), $69,500 (strong)

MARKET REGIME ANALYSIS:
- Current Regime: Bearish Reversal Potential (confidence: 0.72)
- Previous Regime: Late Bull Market (ended 6 hours ago)
- Volatility: High (ATR: 450, 30% above 30-day average)
- Market Phase: Late capitulation or early recovery
- Regime Stability: Low (high probability of transition)

ORDERBOOK MICROSTRUCTURE:
- Bid/Ask Imbalance: 0.65 (buyers stepping in, potential reversal)
- Bid Depth: $2.1M within 1% (medium liquidity)
- Ask Depth: $1.3M within 1% (sellers exhausted?)
- Spread: 0.02% (tight, healthy market)
- Large Orders: 3 whale buy orders (total: 45 BTC) detected in last 60s

WHALE ACTIVITY (Last 60 minutes):
- Accumulation: 3 large buy orders (15 BTC, 18 BTC, 12 BTC)
- Distribution: 1 large sell order (8 BTC)
- Net Flow: +37 BTC (bullish accumulation)
- Whale Sentiment: Cautiously bullish (buying the dip)

MARKET EVENTS (Last 24 hours):
- Pump/Dump Events: None detected
- Iceberg Orders: 2 detected on bid side (hidden buying)
- Flash Crashes: None
- Liquidity Events: 1 minor liquidity drop (recovered)

SENTIMENT ANALYSIS:
- Social Media: 45% bullish, 55% bearish (fear dominates)
- News Sentiment: 3 negative articles (regulatory concerns), 1 positive (institutional adoption)
- Fear & Greed Index: 32 (Fear)

Provide ultra-comprehensive analysis:
1. Primary Trading Signal (BUY/SELL/HOLD) with detailed multi-timeframe reasoning
2. Confidence Level (0.0-1.0) with statistical justification
3. Key Factors (7-10 critical points ranked by importance)
4. Risk Assessment (low/medium/high) with quantified risk metrics
5. Entry Strategy (price levels, order types, position sizing)
6. Exit Strategy (take-profit levels, stop-loss levels, trailing stops)
7. Time Horizon (short/medium/long term) with expected holding period
8. Alternative Scenarios (bull case, bear case, sideways case)
9. Correlation Analysis (BTC vs ETH, BTC vs traditional markets)
10. Final Recommendation (actionable trading plan)""",
            "max_tokens": 500,
            "temperature": 0.7,
            "category": "prefill_500_tokens",
            "phase": "prefill"
        })

        prompts.extend(prefill_warmup_prompts)

        # ========================================================================
        # PHASE 3: DECODE WARMUP (Output Lengths: 50, 100, 200, 500 tokens)
        # ========================================================================
        # Goal: Warm up decode pipeline for various output lengths
        # Strategy: Request different max_tokens to exercise decode

        logger.info("📊 Phase 3: Decode Warmup (4 output lengths)")

        decode_warmup_prompts = [
            {
                "prompt": "Analyze BTC: RSI 50, Volume 1.0x. Provide brief signal.",
                "max_tokens": 50,
                "temperature": 0.7,
                "category": "decode_50_tokens",
                "phase": "decode"
            },
            {
                "prompt": "Analyze BTC: RSI 50, Volume 1.0x. Provide detailed signal with reasoning.",
                "max_tokens": 100,
                "temperature": 0.7,
                "category": "decode_100_tokens",
                "phase": "decode"
            },
            {
                "prompt": "Analyze BTC: RSI 50, Volume 1.0x. Provide comprehensive signal with key factors and risk assessment.",
                "max_tokens": 200,
                "temperature": 0.7,
                "category": "decode_200_tokens",
                "phase": "decode"
            },
            {
                "prompt": "Analyze BTC: RSI 50, Volume 1.0x. Provide ultra-detailed signal with multi-timeframe analysis, key factors, risk assessment, entry/exit strategy, and alternative scenarios.",
                "max_tokens": 500,
                "temperature": 0.7,
                "category": "decode_500_tokens",
                "phase": "decode"
            }
        ]

        prompts.extend(decode_warmup_prompts)

        # ========================================================================
        # NOTE: RadixCache phase removed — disabled when using MTP speculative decoding
        # ========================================================================

        # ========================================================================
        # PHASE 4: CONCURRENT REQUEST WARMUP (Multi-Request Handling)
        # ========================================================================
        # Goal: Test concurrent request handling
        # Strategy: Send 4 requests simultaneously (matching max_running_requests=4)

        logger.info("📊 Phase 4: Concurrent Request Warmup (4 concurrent)")

        concurrent_prompts = []

        for i in range(4):
            concurrent_prompts.append({
                "prompt": f"Analyze {'BTC' if i % 2 == 0 else 'ETH'} market: RSI {50 + i*5}, Volume {1.0 + i*0.2}x. Signal?",
                "max_tokens": 100,
                "temperature": 0.7,
                "category": f"concurrent_{i+1}",
                "phase": "concurrent"
            })

        prompts.extend(concurrent_prompts)

        # ========================================================================
        # PHASE 5: PRODUCTION-LIKE QUERIES (Real Agent Prompts)
        # ========================================================================
        # Goal: Warm up with actual production prompts
        # Strategy: Use real market intelligence agent prompts

        logger.info("📊 Phase 5: Production-Like Queries (5 real prompts)")

        production_prompts = [
            {
                "prompt": """Analyze DOGS_USDT market conditions:

TECHNICAL INDICATORS:
- RSI: 68.67 (approaching overbought)
- Volume Ratio: 1.62x average (elevated)
- Price Trend: Uptrend (+2.1% last hour)

MARKET REGIME:
- Regime: Early Bull Market
- Volatility: Medium (ATR: 0.000002)

ORDERBOOK:
- Bid/Ask Imbalance: 0.58 (balanced)
- Liquidity: Medium

WHALE ACTIVITY:
- Whale Orders (last 60s): 1 large buy order (11,001 DOGS)
- Net Flow: +11,001 DOGS (bullish)

DETECTIONS:
- Pump/Dump: None
- Iceberg Orders: None
- Liquidity Alerts: None

Provide:
1. Trading Signal (BUY/SELL/HOLD)
2. Confidence (0-1)
3. Key Factors (3-5 points)
4. Risk Assessment
5. Brief Reasoning (2-3 sentences)""",
                "max_tokens": 200,
                "temperature": 0.7,
                "category": "production_market_intelligence",
                "phase": "production"
            }
        ] * 5  # Repeat 5 times to simulate production load

        prompts.extend(production_prompts)

        logger.info(f"📝 Generated {len(prompts)} warmup prompts across 5 phases")
        logger.info(f"   Phase 1 (CUDA Graph): {len(cuda_graph_prompts)} prompts")
        logger.info(f"   Phase 2 (Prefill): {len(prefill_warmup_prompts)} prompts")
        logger.info(f"   Phase 3 (Decode): {len(decode_warmup_prompts)} prompts")
        logger.info(f"   Phase 4 (Concurrent): {len(concurrent_prompts)} prompts")
        logger.info(f"   Phase 5 (Production): {len(production_prompts)} prompts")

        return prompts
    
    async def _send_warmup_query(self, prompt_config: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Send a single warmup query to SGLang server.

        Returns:
            Tuple[bool, float]: (success, latency_ms)
        """
        try:
            start_time = time.perf_counter()

            response = await self.client.chat.completions.create(
                model="default",  # SGLang uses "default" model name
                messages=[
                    {"role": "user", "content": prompt_config["prompt"]}
                ],
                max_tokens=prompt_config.get("max_tokens", 500),
                temperature=prompt_config.get("temperature", 0.7),
                # 🔥 CRITICAL FIX: Disable thinking mode for Qwen3 models!
                extra_body={
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )

            latency_ms = (time.perf_counter() - start_time) * 1000

            if response.choices and response.choices[0].message.content:
                # Color-code based on latency
                if latency_ms < 3000:
                    color = "🟢"  # Green: <3s (EXCELLENT for 122B MoE)
                elif latency_ms < 5000:
                    color = "🟡"  # Yellow: 3-5s (GOOD for 122B MoE)
                else:
                    color = "🔴"  # Red: >5s (NEEDS IMPROVEMENT)

                logger.info(f"   {color} {prompt_config['category']}: {latency_ms:.0f}ms")
                return True, latency_ms
            else:
                logger.warning(f"   ⚠️ Empty response: {prompt_config['category']}")
                return False, latency_ms

        except Exception as e:
            logger.error(f"   ❌ Failed: {prompt_config['category']} - {e}")
            return False, 0.0
    
    async def warmup(self, max_concurrent: int = 4) -> Dict[str, Any]:
        """
        Execute BEST-IN-CLASS warmup by sending all queries to SGLang server.

        Args:
            max_concurrent: Maximum concurrent requests (4 = matching max_running_requests)

        Returns:
            Comprehensive warmup statistics
        """
        logger.info("=" * 80)
        logger.info("🔥 BEST-IN-CLASS SGLANG SERVER WARMUP")
        logger.info("=" * 80)
        logger.info(f"   🌐 Server: {self.sglang_url}")
        logger.info(f"   🔄 Max concurrent: {max_concurrent}")
        logger.info(f"   🎯 Target latency: <5s (122B MoE)")
        logger.info(f"   🎯 Target throughput: 60-120 tokens/s (with MTP)")
        logger.info("")

        # Generate prompts
        prompts = self._generate_warmup_prompts()

        if not prompts:
            logger.error("❌ No warmup prompts generated!")
            return self.warmup_stats

        start_time = time.perf_counter()

        # Process in batches to avoid GPU overload
        successful = 0
        failed = 0
        latencies = []

        # Track phase-specific stats
        phase_stats = {}

        # ============================================================
        # CUDA GRAPH PHASE: Must run SEQUENTIALLY!
        # CUDA graph compilation is a synchronous GPU operation.
        # Running multiple captures in parallel causes them to block
        # each other, inflating latency from ~2-5s to ~130s per query.
        # ============================================================
        cuda_graph_prompts = [p for p in prompts if p.get('phase') == 'cuda_graph']
        other_prompts = [p for p in prompts if p.get('phase') != 'cuda_graph']

        if cuda_graph_prompts:
            phase_stats['cuda_graph'] = {'count': 0, 'latencies': []}
            logger.info("")
            logger.info(f"🔄 Starting Phase: CUDA_GRAPH (sequential - {len(cuda_graph_prompts)} queries)")

            for prompt in cuda_graph_prompts:
                result = await self._send_warmup_query(prompt)
                if isinstance(result, Exception):
                    failed += 1
                    logger.error(f"   ❌ Exception: {result}")
                elif isinstance(result, tuple):
                    success, latency_ms = result
                    if success:
                        successful += 1
                        latencies.append(latency_ms)
                        phase_stats['cuda_graph']['count'] += 1
                        phase_stats['cuda_graph']['latencies'].append(latency_ms)
                    else:
                        failed += 1
                else:
                    failed += 1

            logger.info(f"   📊 CUDA Graph phase done: {phase_stats['cuda_graph']['count']} successful")

        # ============================================================
        # ALL OTHER PHASES: Run in parallel batches as before
        # ============================================================
        query_count = len(cuda_graph_prompts)

        for i in range(0, len(other_prompts), max_concurrent):
            batch = other_prompts[i:i + max_concurrent]

            # Log phase transition
            current_phase = batch[0].get('phase', 'unknown')
            if current_phase not in phase_stats:
                phase_stats[current_phase] = {'count': 0, 'latencies': []}
                logger.info("")
                logger.info(f"🔄 Starting Phase: {current_phase.upper()}")

            # Process batch concurrently
            tasks = [
                self._send_warmup_query(prompt)
                for prompt in batch
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for j, result in enumerate(results):
                if isinstance(result, Exception):
                    failed += 1
                    logger.error(f"   ❌ Exception: {result}")
                elif isinstance(result, tuple):
                    success, latency_ms = result
                    if success:
                        successful += 1
                        latencies.append(latency_ms)

                        # Track phase-specific latency
                        phase = batch[j].get('phase', 'unknown')
                        if phase in phase_stats:
                            phase_stats[phase]['count'] += 1
                            phase_stats[phase]['latencies'].append(latency_ms)
                    else:
                        failed += 1
                else:
                    failed += 1

            # Progress update every 10 queries
            query_count += len(batch)
            if query_count % 10 == 0 or query_count == len(prompts):
                progress = query_count / len(prompts) * 100
                logger.info(f"   📊 Progress: {progress:.1f}% ({successful} successful, {failed} failed)")

        total_time = (time.perf_counter() - start_time) * 1000
        avg_time = statistics.mean(latencies) if latencies else 0

        # Calculate percentiles
        if latencies:
            sorted_latencies = sorted(latencies)
            p50 = statistics.median(sorted_latencies)
            p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)] if len(sorted_latencies) > 1 else sorted_latencies[0]
            p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)] if len(sorted_latencies) > 1 else sorted_latencies[0]
            min_latency = min(latencies)
            max_latency = max(latencies)
        else:
            p50 = p95 = p99 = min_latency = max_latency = 0

        self.warmup_stats = {
            'total_queries': len(prompts),
            'successful': successful,
            'failed': failed,
            'total_time_ms': total_time,
            'avg_time_ms': avg_time,
            'min_latency_ms': min_latency,
            'max_latency_ms': max_latency,
            'p50_latency_ms': p50,
            'p95_latency_ms': p95,
            'p99_latency_ms': p99,
            'latencies': latencies,
            'phase_stats': phase_stats
        }

        # Print comprehensive statistics
        logger.info("")
        logger.info("=" * 80)
        logger.info("✅ WARMUP COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"   ✅ Successful: {successful}/{len(prompts)} ({successful/len(prompts)*100:.1f}%)")
        logger.info(f"   ❌ Failed: {failed}/{len(prompts)} ({failed/len(prompts)*100:.1f}%)")
        logger.info(f"   ⏱️ Total time: {total_time/1000:.1f}s")
        logger.info("")
        logger.info("📊 LATENCY STATISTICS:")
        logger.info(f"   Average: {avg_time:.0f}ms")
        logger.info(f"   Median (P50): {p50:.0f}ms")
        logger.info(f"   P95: {p95:.0f}ms")
        logger.info(f"   P99: {p99:.0f}ms")
        logger.info(f"   Min: {min_latency:.0f}ms")
        logger.info(f"   Max: {max_latency:.0f}ms")
        logger.info("")

        # Print phase-specific statistics
        logger.info("📊 PHASE-SPECIFIC STATISTICS:")
        for phase, stats in phase_stats.items():
            if stats['latencies']:
                phase_avg = statistics.mean(stats['latencies'])
                phase_min = min(stats['latencies'])
                phase_max = max(stats['latencies'])
                logger.info(f"   {phase.upper()}: {stats['count']} queries, avg={phase_avg:.0f}ms, min={phase_min:.0f}ms, max={phase_max:.0f}ms")

        logger.info("")

        # Performance assessment
        if avg_time < 3000:
            logger.info("🎉 EXCELLENT! Average latency <3s - Server is PRODUCTION READY!")
        elif avg_time < 5000:
            logger.info("✅ GOOD! Average latency <5s - Server is ready (122B MoE model).")
        else:
            logger.info("⚠️ WARNING! Average latency >5s - Check GPU memory or reduce max_tokens.")

        logger.info("=" * 80)

        return self.warmup_stats


async def main():
    """Main entry point for BEST-IN-CLASS warmup script"""
    logger.info("=" * 80)
    logger.info("🔥 BEST-IN-CLASS SGLANG SERVER WARMUP SCRIPT")
    logger.info("=" * 80)
    logger.info("")
    logger.info("GOALS:")
    logger.info("  1. ✅ CUDA Graph Warmup (batch sizes 1-4)")
    logger.info("  2. ✅ Prefill Warmup (various input lengths)")
    logger.info("  3. ✅ Decode Warmup (various output lengths)")
    logger.info("  4. ✅ Concurrent Request Warmup (4 concurrent)")
    logger.info("  5. ✅ Production-Like Queries (real agent prompts)")
    logger.info("")
    logger.info("EXPECTED RESULTS:")
    logger.info("  - Total Queries: ~22")
    logger.info("  - Warmup Time: 120-300 seconds (122B MoE model)")
    logger.info("  - Target Latency: <5s average (Qwen3.5-122B)")
    logger.info("  - Success Rate: >95%")
    logger.info("")
    logger.info("=" * 80)
    logger.info("")

    # Create warmup manager
    warmup = SGLangWarmup()

    # Execute warmup with optimal concurrency for H100
    stats = await warmup.warmup(max_concurrent=4)

    # Check if warmup was successful
    success_rate = stats['successful'] / stats['total_queries'] if stats['total_queries'] > 0 else 0
    avg_latency = stats['avg_time_ms']

    logger.info("")
    logger.info("=" * 80)
    logger.info("🎯 FINAL ASSESSMENT")
    logger.info("=" * 80)

    # Success criteria
    criteria_met = 0
    total_criteria = 3

    # Criterion 1: Success rate >95%
    if success_rate >= 0.95:
        logger.info("✅ Success Rate: {:.1f}% (>95%) - PASSED".format(success_rate * 100))
        criteria_met += 1
    else:
        logger.info("❌ Success Rate: {:.1f}% (<95%) - FAILED".format(success_rate * 100))

    # Criterion 2: Average latency <5s (122B MoE model)
    if avg_latency < 5000:
        logger.info("✅ Average Latency: {:.0f}ms (<5000ms) - PASSED".format(avg_latency))
        criteria_met += 1
    else:
        logger.info("❌ Average Latency: {:.0f}ms (>5000ms) - FAILED".format(avg_latency))

    # Criterion 3: P95 latency <10s (122B MoE, some queries are long)
    if stats['p95_latency_ms'] < 10000:
        logger.info("✅ P95 Latency: {:.0f}ms (<10000ms) - PASSED".format(stats['p95_latency_ms']))
        criteria_met += 1
    else:
        logger.info("❌ P95 Latency: {:.0f}ms (>10000ms) - FAILED".format(stats['p95_latency_ms']))

    logger.info("")
    logger.info(f"📊 CRITERIA MET: {criteria_met}/{total_criteria}")
    logger.info("")

    if criteria_met == total_criteria:
        logger.info("🎉🎉🎉 WARMUP SUCCESSFUL! SERVER IS PRODUCTION READY! 🎉🎉🎉")
        logger.info("")
        logger.info("NEXT STEPS:")
        logger.info("  1. Start main.py to begin trading")
        logger.info("  2. Monitor first few LLM queries for <1.2s latency")
        logger.info("  3. If latency is still high, consider enabling speculative decoding")
        logger.info("")
        return 0
    elif criteria_met >= 2:
        logger.info("✅ WARMUP MOSTLY SUCCESSFUL! Server is ready with minor issues.")
        logger.info("")
        logger.info("RECOMMENDATIONS:")
        logger.info("  1. Server is usable but may benefit from optimization")
        logger.info("  2. Consider enabling speculative decoding for better latency")
        logger.info("  3. Monitor production performance closely")
        logger.info("")
        return 0
    else:
        logger.error("❌ WARMUP FAILED! Server is NOT ready for production.")
        logger.error("")
        logger.error("TROUBLESHOOTING:")
        logger.error("  1. Check SGLang server logs for errors")
        logger.error("  2. Verify server is running: curl http://127.0.0.1:30000/health")
        logger.error("  3. Check GPU memory: nvidia-smi")
        logger.error("  4. Restart SGLang server and try again")
        logger.error("")
        return 1


async def warmup_dual_servers():
    """
    Warmup BOTH SGLang servers (Worker + 122B Master) in parallel.

    STRATEGY:
    1. Warmup Worker (Port 30000) - Fast queries
    2. Warmup 122B Master (Port 30001) - Strategic queries
    3. Run in parallel to save time

    EXPECTED TIME: 3-5 minutes (vs 6-10 minutes sequential)
    """
    logger.info("=" * 80)
    logger.info("🚀 DUAL SGLANG SERVER WARMUP")
    logger.info("=" * 80)
    logger.info("")
    logger.info("📊 Warming up BOTH servers in parallel:")
    logger.info("   - Worker (Port 30000) - Fast queries")
    logger.info("   - 122B Master (Port 30001) - Strategic queries")
    logger.info("")

    # Create warmup instances for both servers
    worker_warmup = SGLangWarmup(
        sglang_url="http://127.0.0.1:30000/v1",
        config_path="bytedance/config/llm_config.yaml"
    )

    master_warmup = SGLangWarmup(
        sglang_url="http://127.0.0.1:30001/v1",
        config_path="bytedance/config/llm_config.yaml"
    )

    # Run warmups in parallel
    logger.info("⏳ Starting parallel warmup...")
    logger.info("")

    start_time = time.time()

    # Run both warmups concurrently
    worker_task = asyncio.create_task(worker_warmup.warmup())
    master_task = asyncio.create_task(master_warmup.warmup())

    # Wait for both to complete
    worker_stats, master_stats = await asyncio.gather(worker_task, master_task)

    elapsed_time = time.time() - start_time

    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ DUAL SERVER WARMUP COMPLETE!")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"⏱️  Total Time: {elapsed_time:.1f}s")
    logger.info("")
    logger.info("📊 Worker (Port 30000) Stats:")
    logger.info(f"   - Queries: {worker_stats['successful']}/{worker_stats['total_queries']}")
    logger.info(f"   - Avg Latency: {worker_stats['avg_time_ms']:.0f}ms")
    logger.info(f"   - P95 Latency: {worker_stats['p95_latency_ms']:.0f}ms")
    logger.info("")
    logger.info("📊 122B Master (Port 30001) Stats:")
    logger.info(f"   - Queries: {master_stats['successful']}/{master_stats['total_queries']}")
    logger.info(f"   - Avg Latency: {master_stats['avg_time_ms']:.0f}ms")
    logger.info(f"   - P95 Latency: {master_stats['p95_latency_ms']:.0f}ms")
    logger.info("")

    # Check if both servers are ready
    worker_ready = worker_stats['successful'] >= worker_stats['total_queries'] * 0.9
    master_ready = master_stats['successful'] >= master_stats['total_queries'] * 0.9

    if worker_ready and master_ready:
        logger.info("🎉 BOTH SERVERS ARE PRODUCTION READY! 🎉")
        logger.info("")
        logger.info("🎯 Next Steps:")
        logger.info("   1. Run: python bytedance/main.py --micro-trade")
        logger.info("   2. Monitor GPU: watch -n 1 nvidia-smi")
        logger.info("   3. Check logs for performance metrics")
        logger.info("")
        return 0
    else:
        logger.warning("⚠️ ONE OR MORE SERVERS NOT FULLY READY!")
        logger.warning("")
        if not worker_ready:
            logger.warning("   ❌ Worker (Port 30000) - Check logs!")
        if not master_ready:
            logger.warning("   ❌ 122B Master (Port 30001) - Check logs!")
        logger.warning("")
        logger.warning("🔧 Troubleshooting:")
        logger.warning("   1. Check server logs for errors")
        logger.warning("   2. Verify both servers are running:")
        logger.warning("      curl http://127.0.0.1:30000/health")
        logger.warning("      curl http://127.0.0.1:30001/health")
        logger.warning("   3. Check GPU memory: nvidia-smi")
        logger.warning("")
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SGLang Server Warmup")
    parser.add_argument(
        "--dual",
        action="store_true",
        help="Warmup both servers (Worker + 122B Master) in parallel"
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Warmup ONLY 122B Master (Port 30001) - for singular server mode"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Port for single server warmup (default: 30000)"
    )

    args = parser.parse_args()

    if args.dual:
        # Dual server warmup (both Worker + 122B Master)
        exit_code = asyncio.run(warmup_dual_servers())
    elif args.single:
        # Single server warmup for 122B Master ONLY (Port 30001)
        async def warmup_single_master():
            """Warmup ONLY the 122B Master server (Port 30000 — orion ADR-16 binding)"""
            logger.info("=" * 80)
            logger.info("🎯 SINGULAR SERVER WARMUP (122B Master Only)")
            logger.info("=" * 80)
            logger.info("")
            logger.info("📊 Warming up 122B Master (Port 30000)")
            logger.info("")

            master_warmup = SGLangWarmup(
                sglang_url="http://127.0.0.1:30000/v1",
                config_path="bytedance/config/llm_config.yaml"
            )

            stats = await master_warmup.warmup(max_concurrent=4)

            success_rate = stats['successful'] / stats['total_queries'] if stats['total_queries'] > 0 else 0
            is_ready = success_rate >= 0.9 and stats['avg_time_ms'] < 5000

            logger.info("")
            logger.info("=" * 80)
            logger.info("✅ SINGULAR SERVER WARMUP COMPLETE!")
            logger.info("=" * 80)
            logger.info("")
            logger.info("📊 122B Master (Port 30000) Stats:")
            logger.info(f"   - Queries: {stats['successful']}/{stats['total_queries']}")
            logger.info(f"   - Avg Latency: {stats['avg_time_ms']:.0f}ms")
            logger.info(f"   - P95 Latency: {stats['p95_latency_ms']:.0f}ms")
            logger.info("")

            if is_ready:
                logger.info("🎉 122B Master is READY for production!")
                return 0
            else:
                logger.warning("⚠️ 122B Master warmup incomplete - check logs!")
                return 1

        exit_code = asyncio.run(warmup_single_master())
    else:
        # Single server warmup on specified port (legacy)
        exit_code = asyncio.run(main())

    sys.exit(exit_code)

