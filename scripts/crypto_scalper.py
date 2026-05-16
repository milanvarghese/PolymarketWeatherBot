"""Pro-grade crypto scalping bot v3.5 — research-backed, multi-signal confluence.

Architecture:
1. DYNAMIC PAIR DISCOVERY — scans Coinbase for all liquid USD pairs, refreshes hourly
2. REGIME DETECTION — ADX classifies trending vs ranging, adapts strategy
   MACRO REGIME SYSTEM — composite scoring (F&G, vol, ADX, breadth, WR) with 5 profiles
3. MULTI-TIMEFRAME — 1H trend direction, 5M entry timing
4. CONFLUENCE SCORING — 0-17 scale with signal performance weighting
5. ORDER BOOK IMBALANCE — confirms direction from Coinbase L2 book
6. FUNDING RATE — extreme rates predict reversals (Binance -> dYdX -> OKX fallback)
7. FAIR VALUE GAPS — institutional footprints in price action
8. VOLUME PROFILE — POC/VAH/VAL as targets and support/resistance
9. DYNAMIC ATR EXITS — volatility-scaled TP/SL, not fixed percentages
10. FEAR & GREED FILTER — sentiment regime from alternative.me API
11. CUSUM ENTRY FILTER — only trade on meaningful price moves (Lopez de Prado)
12. PUMP DETECTION — skip coins with suspicious volume spikes
13. CROSS-ASSET VOL TRACKER — BTC vol spillover early warning
14. SIGNAL PERFORMANCE BANDIT — auto-weight signals by recent success
15. RSI DIVERGENCE — proven mean reversion signal (price vs RSI disagreement)
16. ENGULFING PATTERNS — strong reversal candlestick patterns
17. SWING LEVEL S/R — entry confirmation at key support/resistance
18. ATR EXPANSION FILTER — only trade during active vol (avoid fee-eating chop)
19. PROGRESSIVE STOP TIGHTENING — gradually tighten SL in second half of hold
20. PAIR WHITELIST — restrict to backtest-validated profitable pairs
21. BIDIRECTIONAL TRADING — both long (buy) and short (sell) signals
22. ADAPTIVE MACRO REGIME — 5 market profiles with auto-switching and side bias

v3.5 changes:
- Adaptive macro regime system: 5 profiles (TRENDING_BULL, TRENDING_BEAR, HIGH_VOL_FEAR,
  LOW_VOL_CHOP, RECOVERY) with composite scoring from F&G, BTC vol, ADX, breadth, WR
- Each profile overrides TP/SL/confluence/sizing/hold time/position limits
- Side bias: +1-2 confluence bonus for preferred direction per regime
- Anti-whipsaw: hysteresis bonus, 3-scan confirmation, 10-scan lockout
- Regime-aware auto-tuning: tracks performance per macro regime

v3.4 changes:
- Funding rate fallback chain: Binance -> dYdX -> OKX (fixes geo-blocking on US servers)
- Confirmed short signals already working: 112 of 166 trades (67%) are shorts
- Backtest: LONG 54t 50% WR $+0.05 | SHORT 112t 65% WR $+7.23 (shorts carry the edge)

v3.3 changes (backtest-validated, 166 trades, 90 days, 6 pairs):
- Raised confluence to 7 (higher selectivity: 59.6% WR vs 52% at score=6)
- Added progressive stop tightening (50%/0.3x): tighten SL from 3.5x to 1.05x ATR
  over second half of hold. Converts losing time_exits into earlier SL exits.
- Added pair whitelist: only trade ATOM/LINK/DOGE/ETH/ADA/BTC (top 6 from backtest)
- Results @ Binance 0.1%: PF=1.77, +$7.60/90d (+15.2% ROI), DD=2.3%
- Results @ CB Standard 0.6%: PF=0.52 (not profitable — need sub-0.25% fees)

Paper trades by default. Uses real market data from Coinbase + dYdX/OKX.
"""

import argparse
import copy
import csv
import json
import logging
import math
import os
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# --- Configuration ---
SCALPER_CONFIG = {
    # Dynamic pair discovery: scans Coinbase for all liquid USD pairs
    # These are fallback seeds — replaced at startup by live scan
    "pairs": [
        "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD",
        "AVAX-USD", "LINK-USD", "XRP-USD", "SUI-USD",
    ],
    # Dynamic pair discovery settings
    "dynamic_pairs": True,               # Enable auto-discovery (disable with --pairs flag)
    "min_24h_volume_usd": 500_000,       # Min 24h USD volume to be tradeable
    "max_spread_pct": 0.15,              # Max bid-ask spread % (skip illiquid pairs)
    "pair_refresh_interval_sec": 3600,   # Re-scan pair universe every hour
    "max_pairs": 80,                     # Cap to avoid API rate limits
    # Bankroll
    "bankroll": 50.00,
    "max_position_usd": 4.00,       # Smaller per-trade, spread across more pairs
    "max_open_positions": 3,         # 2026-05-15: tightened from 8 after -34% live drawdown
    "max_exposure_pct": 0.75,        # 75% of bankroll at risk
    # Confluence requirements (v3.3: raised from 6 to 7 — backtest: 59.6% WR vs 52% at score=6)
    "min_confluence_score": 7,       # 2026-05-15: hard-coded 7 after live data showed 41% WR with score=5/6
    "min_signal_quality": "C",       # Accept C-grade+ signals
    "min_ranging_score": 7,          # 2026-05-15: matched to min_confluence_score
    # Pair whitelist: data-validated winners only (live 50-day perf 2026-05-15)
    # Kept: pairs with WR >=47% AND n>=15 AND PnL>=$0 on live trading
    # Removed: BTC (50% WR n=30 but -$0.09), ETH/DOGE/LINK/SOL/NEAR/APT/XRP/etc (all losing)
    "pair_whitelist": ["AVAX", "SEI", "INJ", "ATOM", "SUI", "ADA"],
    # Regime-adaptive thresholds
    "adx_trending": 25,              # ADX > 25 = trending
    "adx_ranging": 20,               # ADX < 20 = ranging
    "rsi_oversold_trending": 35,     # More conservative in trends
    "rsi_overbought_trending": 65,
    "rsi_oversold_ranging": 28,      # Slightly relaxed for more signals (was 25)
    "rsi_overbought_ranging": 72,    # Slightly relaxed for more signals (was 75)
    "volume_spike_mult": 1.8,        # HFT: catch smaller volume moves (was 2.0)
    "bb_squeeze_threshold": 0.012,   # HFT: tighter squeeze detection (was 0.015)
    # Dynamic exits (1H ATR-based) — v3.2 validated via backtest (90d, 14 pairs, 25K+ candles)
    # Key insight: wider SL lets winners develop. PF improved from 0.23 to 0.82.
    # At Binance 0.1%/side: PF=1.18, +$3.56/90d. At CB Advanced 0.25%: PF=1.00 (break-even).
    "tp_atr_mult": 4.0,             # 4.0x 1H ATR — wider target catches bigger moves (was 2.5)
    "sl_atr_mult": 3.5,             # 3.5x 1H ATR — wide stop avoids premature exits (was 2.0)
    "trailing_atr_mult": 999.0,     # Disabled — trailing stop was cutting winners short (was 1.5)
    "min_trail_activation_atr": 0,   # Min ATR profit before trailing activates (0 = immediate)
    "max_hold_hours": 36,            # 36h holds for 1H ATR moves to develop (was 12)
    "breakeven_at_1r": False,        # Disabled — was hurting PF by cutting winners (was True)
    # Progressive stop tightening (v3.3): after 50% of max_hold, tighten SL from 3.5x→1.05x ATR
    # Backtest: time_exit PnL went from -$4.24 to -$0.38 (converts losers to earlier SL exits)
    "progressive_stop": True,        # Enable progressive stop tightening
    "progressive_stop_start_pct": 0.5,  # Start tightening at 50% of max_hold (18h)
    "progressive_stop_end_mult": 0.3,   # Final SL = 0.3x original distance (1.05x ATR)
    # Time exit filter: only close via time_exit if trade is profitable
    "time_exit_require_profit": False,   # If True, extend hold for losing time exits
    "time_exit_extension_hours": 6,      # Extra hours for losing time exits (then force close)
    # Risk management
    "kelly_fraction": 0.10,          # Slightly conservative per-trade
    "max_daily_loss": -5.00,
    "max_daily_trades": 30,          # HFT: many more trades per day (was 10)
    "cooldown_after_loss_sec": 180,  # HFT: 3 min cooldown (was 10 min)
    "max_consecutive_losses": 4,     # HFT: tolerate more losses before pausing (was 3)
    # Scan frequency — 1H ATR targets need minutes not seconds
    "scan_interval_sec": 60,         # 1-min scans (was 10s HFT — wasted API calls)
    "candle_granularity_entry": "FIVE_MINUTE",   # 5M for entry signals
    "candle_granularity_trend": "ONE_HOUR",      # 1H for trend context
    "candle_lookback": 50,
    # Fees (Coinbase taker — worst case for paper trading)
    "taker_fee_pct": 0.001,            # Binance taker fee (target exchange for live)
    "maker_fee_pct": 0.001,            # Binance maker fee
    # Order book imbalance thresholds — HFT: more sensitive
    "ob_strong_buy": 0.20,           # HFT: lower threshold (was 0.25)
    "ob_strong_sell": -0.20,         # HFT: lower threshold (was -0.25)
    # Funding rate thresholds (from Binance perps)
    "funding_extreme_high": 0.0005,  # > 0.05% = overheated longs
    "funding_extreme_low": -0.0003,  # < -0.03% = oversold
    # --- v3.0: Profit-maximizing filters ---
    # Fear & Greed sentiment filter (alternative.me free API)
    "fear_greed_extreme_fear": 25,   # < 25 = extreme fear -> mean reversion strongest
    "fear_greed_extreme_greed": 80,  # > 80 = extreme greed -> momentum/caution
    "fear_greed_no_trade_zone": 10,  # < 10 = panic, skip all entries
    # CUSUM entry filter (Lopez de Prado): only trade on meaningful moves
    "cusum_threshold": 0.002,        # HFT: 0.2% threshold, more sensitive (was 0.3%)
    # Pump detection: skip suspicious volume spikes
    "pump_volume_mult": 5.0,         # Volume > 5x 1H average = pump risk
    "pump_price_change_pct": 0.03,   # + 3% in 1 hour = pump risk
    # Cross-asset vol: BTC vol spillover warning
    "vol_lookback_hours": 24,        # Hours of data for realized vol
    "vol_spike_mult": 2.0,           # BTC vol > 2x average = widen stops / cautious
    # Signal performance bandit: auto-weight signals
    "bandit_lookback": 50,           # Rolling window for signal performance
    "bandit_decay": 0.95,            # Exponential decay for older trades
    # v3.2: Volatility expansion filter
    "atr_contraction_threshold": 0.8,  # ATR ratio < 0.8 = contracted, skip entries
    "atr_expansion_threshold": 1.3,    # ATR ratio > 1.3 = expanding, favor entries
}

# Coinbase Exchange API (free, no auth)
COINBASE_EXCHANGE_API = "https://api.exchange.coinbase.com"

# Binance Futures API (free, no auth for public endpoints)
BINANCE_FAPI = "https://fapi.binance.com"
# Fallback funding rate APIs (when Binance is geo-blocked, returns 451)
BYBIT_API = "https://api.bybit.com"
DYDX_API = "https://indexer.dydx.trade"
OKX_API = "https://www.okx.com"

GRANULARITY_SECONDS = {
    "ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900,
    "THIRTY_MINUTE": 1800, "ONE_HOUR": 3600, "TWO_HOUR": 7200,
    "SIX_HOUR": 21600, "ONE_DAY": 86400,
}

# Coinbase pair → Binance symbol mapping
def _to_binance_symbol(pair: str) -> str:
    """BTC-USD → BTCUSDT"""
    base = pair.replace("-USD", "").replace("-USDT", "")
    return f"{base}USDT"

# --- Data directory ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "crypto_trading")
STATE_FILE = os.path.join(DATA_DIR, "state.json")
TRADE_LOG = os.path.join(DATA_DIR, "trades.csv")
ANALYTICS_FILE = os.path.join(DATA_DIR, "analytics.jsonl")  # Rich trade analytics (JSONL: 1 record/line)
TUNING_FILE = os.path.join(DATA_DIR, "tuning.json")         # Auto-tuned parameters


# ================================================================
# TECHNICAL INDICATORS
# ================================================================

def calc_rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """Relative Strength Index (Wilder's smoothing)."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(0, delta))
        losses.append(max(0, -delta))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def calc_ema(values: list[float], period: int) -> Optional[float]:
    """Exponential Moving Average."""
    if len(values) < period:
        return None
    mult = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for val in values[period:]:
        ema = (val - ema) * mult + ema
    return ema


def calc_ema_series(values: list[float], period: int) -> list[float]:
    """Full EMA series for ADX calculation."""
    if len(values) < period:
        return []
    mult = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    series = [ema]
    for val in values[period:]:
        ema = (val - ema) * mult + ema
        series.append(ema)
    return series


def calc_adx(highs: list[float], lows: list[float], closes: list[float],
             period: int = 14) -> Optional[float]:
    """Average Directional Index — measures trend strength (0-100)."""
    if len(closes) < period * 2:
        return None

    plus_dm, minus_dm, tr_list = [], [], []
    for i in range(1, len(closes)):
        high_diff = highs[i] - highs[i - 1]
        low_diff = lows[i - 1] - lows[i]

        pdm = high_diff if (high_diff > low_diff and high_diff > 0) else 0
        mdm = low_diff if (low_diff > high_diff and low_diff > 0) else 0
        plus_dm.append(pdm)
        minus_dm.append(mdm)

        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        tr_list.append(tr)

    if len(tr_list) < period:
        return None

    # Wilder's smoothing
    atr = sum(tr_list[:period]) / period
    plus_di_smooth = sum(plus_dm[:period]) / period
    minus_di_smooth = sum(minus_dm[:period]) / period

    dx_values = []
    for i in range(period, len(tr_list)):
        atr = (atr * (period - 1) + tr_list[i]) / period
        plus_di_smooth = (plus_di_smooth * (period - 1) + plus_dm[i]) / period
        minus_di_smooth = (minus_di_smooth * (period - 1) + minus_dm[i]) / period

        if atr == 0:
            continue
        plus_di = 100 * plus_di_smooth / atr
        minus_di = 100 * minus_di_smooth / atr

        di_sum = plus_di + minus_di
        if di_sum == 0:
            continue
        dx = 100 * abs(plus_di - minus_di) / di_sum
        dx_values.append(dx)

    if len(dx_values) < period:
        return None

    adx = sum(dx_values[:period]) / period
    for dx in dx_values[period:]:
        adx = (adx * (period - 1) + dx) / period

    return adx


def calc_macd(closes: list[float]) -> Optional[tuple[float, float, float]]:
    """MACD (12,26,9). Returns (macd_line, signal_line, histogram).

    Uses proper running EMA series (not re-seeded subsets).
    """
    if len(closes) < 35:
        return None

    # Build full EMA12 and EMA26 series
    ema12_series = calc_ema_series(closes, 12)
    ema26_series = calc_ema_series(closes, 26)
    if not ema12_series or not ema26_series:
        return None

    # MACD line = EMA12 - EMA26 (aligned from the point both exist)
    # EMA12 starts at index 12, EMA26 starts at index 26
    # So MACD starts at index 26 (the later one)
    offset = 26 - 12  # EMA12 has 14 more values than EMA26
    macd_series = []
    for i in range(len(ema26_series)):
        macd_series.append(ema12_series[i + offset] - ema26_series[i])

    if len(macd_series) < 9:
        return None

    # Signal line = 9-period EMA of MACD line
    signal_series = calc_ema_series(macd_series, 9)
    if not signal_series:
        return None

    macd_line = macd_series[-1]
    signal = signal_series[-1]
    return (macd_line, signal, macd_line - signal)


def calc_macd_last_two_histograms(closes: list[float]) -> Optional[tuple[float, float]]:
    """Return (prev_histogram, current_histogram) from a single continuous MACD run.

    This avoids the alignment error of computing MACD on closes[:-1] vs closes
    with independently seeded EMA series.
    """
    if len(closes) < 36:
        return None

    ema12_series = calc_ema_series(closes, 12)
    ema26_series = calc_ema_series(closes, 26)
    if not ema12_series or not ema26_series:
        return None

    offset = 26 - 12
    macd_series = []
    for i in range(len(ema26_series)):
        macd_series.append(ema12_series[i + offset] - ema26_series[i])

    if len(macd_series) < 10:
        return None

    signal_series = calc_ema_series(macd_series, 9)
    if not signal_series or len(signal_series) < 2:
        return None

    prev_hist = macd_series[-2] - signal_series[-2]
    curr_hist = macd_series[-1] - signal_series[-1]
    return (prev_hist, curr_hist)


def calc_bollinger(closes: list[float], period: int = 20, num_std: float = 2.0
                   ) -> Optional[tuple[float, float, float, float]]:
    """Bollinger Bands → (upper, middle, lower, bandwidth_pct)."""
    if len(closes) < period:
        return None
    window = closes[-period:]
    middle = sum(window) / period
    std = math.sqrt(sum((x - middle) ** 2 for x in window) / period)
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle if middle > 0 else 0
    return (upper, middle, lower, bandwidth)


def calc_atr(highs: list[float], lows: list[float], closes: list[float],
             period: int = 14) -> Optional[float]:
    """Average True Range."""
    if len(closes) < period + 1:
        return None
    true_ranges = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i],
                 abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        true_ranges.append(tr)
    if len(true_ranges) < period:
        return None
    # Wilder's smoothing
    atr = sum(true_ranges[:period]) / period
    for i in range(period, len(true_ranges)):
        atr = (atr * (period - 1) + true_ranges[i]) / period
    return atr


def calc_vwap(closes: list[float], volumes: list[float], highs: list[float],
              lows: list[float]) -> Optional[float]:
    """Volume-Weighted Average Price."""
    if not closes or not volumes:
        return None
    typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    total_tp_vol = sum(tp * v for tp, v in zip(typical_prices, volumes))
    total_vol = sum(volumes)
    return total_tp_vol / total_vol if total_vol > 0 else None


def detect_fvg(candles: list[dict]) -> list[dict]:
    """Detect Fair Value Gaps (institutional footprints).

    Bullish FVG: candle[i] low > candle[i-2] high (gap up)
    Bearish FVG: candle[i] high < candle[i-2] low (gap down)
    """
    fvgs = []
    for i in range(2, len(candles)):
        # Bullish FVG
        if candles[i]["low"] > candles[i - 2]["high"]:
            fvgs.append({
                "type": "bullish",
                "top": candles[i]["low"],
                "bottom": candles[i - 2]["high"],
                "midpoint": (candles[i]["low"] + candles[i - 2]["high"]) / 2,
                "size_pct": (candles[i]["low"] - candles[i - 2]["high"]) / candles[i - 2]["high"],
                "candle_idx": i - 1,
            })
        # Bearish FVG
        if candles[i]["high"] < candles[i - 2]["low"]:
            fvgs.append({
                "type": "bearish",
                "top": candles[i - 2]["low"],
                "bottom": candles[i]["high"],
                "midpoint": (candles[i - 2]["low"] + candles[i]["high"]) / 2,
                "size_pct": (candles[i - 2]["low"] - candles[i]["high"]) / candles[i - 2]["low"],
                "candle_idx": i - 1,
            })
    return fvgs


def calc_volume_profile(candles: list[dict], num_bins: int = 30
                        ) -> Optional[dict]:
    """Volume Profile: POC, Value Area High/Low.

    Builds a histogram of volume at each price level,
    finds the Point of Control (highest volume price) and
    Value Area (70% of volume).
    """
    if len(candles) < 10:
        return None

    price_min = min(c["low"] for c in candles)
    price_max = max(c["high"] for c in candles)
    if price_max == price_min:
        return None

    bin_size = (price_max - price_min) / num_bins
    bins = [0.0] * num_bins
    bin_prices = [price_min + (i + 0.5) * bin_size for i in range(num_bins)]

    for c in candles:
        # Distribute volume across the candle's range
        low_bin = int((c["low"] - price_min) / bin_size)
        high_bin = int((c["high"] - price_min) / bin_size)
        low_bin = max(0, min(low_bin, num_bins - 1))
        high_bin = max(0, min(high_bin, num_bins - 1))
        bins_in_range = high_bin - low_bin + 1
        vol_per_bin = c["volume"] / bins_in_range if bins_in_range > 0 else 0
        for b in range(low_bin, high_bin + 1):
            bins[b] += vol_per_bin

    # POC = bin with highest volume
    poc_idx = max(range(num_bins), key=lambda i: bins[i])
    poc = bin_prices[poc_idx]

    # Value Area = 70% of total volume, expanding from POC
    total_vol = sum(bins)
    if total_vol == 0:
        return None
    target_vol = total_vol * 0.70
    va_vol = bins[poc_idx]
    lo, hi = poc_idx, poc_idx

    while va_vol < target_vol and (lo > 0 or hi < num_bins - 1):
        vol_above = bins[hi + 1] if hi + 1 < num_bins else 0
        vol_below = bins[lo - 1] if lo - 1 >= 0 else 0
        if vol_above >= vol_below and hi + 1 < num_bins:
            hi += 1
            va_vol += bins[hi]
        elif lo - 1 >= 0:
            lo -= 1
            va_vol += bins[lo]
        else:
            break

    return {
        "poc": poc,
        "vah": bin_prices[hi],
        "val": bin_prices[lo],
        "total_volume": total_vol,
    }


def detect_rsi_divergence(closes: list[float], period: int = 14,
                          lookback: int = 20) -> Optional[str]:
    """Detect RSI divergence — proven mean reversion signal in crypto.

    Bullish divergence: price makes lower low, RSI makes higher low
    Bearish divergence: price makes higher high, RSI makes lower high

    Returns: "bullish", "bearish", or None
    """
    if len(closes) < period + lookback + 1:
        return None

    # Compute RSI at each point in the lookback window
    rsi_values = []
    for i in range(lookback + 1):
        end = len(closes) - lookback + i
        rsi = calc_rsi(closes[:end], period)
        if rsi is None:
            return None
        rsi_values.append(rsi)

    # Find swing lows and highs in the lookback window
    price_window = closes[-(lookback + 1):]

    # Find local minima (swing lows) for bullish divergence
    swing_lows = []
    for i in range(2, len(price_window) - 1):
        if price_window[i] < price_window[i - 1] and price_window[i] < price_window[i - 2]:
            if price_window[i] <= price_window[i + 1]:
                swing_lows.append((i, price_window[i], rsi_values[i]))

    # Bullish divergence: latest swing low has lower price but higher RSI
    if len(swing_lows) >= 2:
        prev_low = swing_lows[-2]
        curr_low = swing_lows[-1]
        if (curr_low[1] < prev_low[1] * 0.999  # Price lower (with tolerance)
            and curr_low[2] > prev_low[2] + 2.0):  # RSI higher by at least 2 pts
            return "bullish"

    # Find local maxima (swing highs) for bearish divergence
    swing_highs = []
    for i in range(2, len(price_window) - 1):
        if price_window[i] > price_window[i - 1] and price_window[i] > price_window[i - 2]:
            if price_window[i] >= price_window[i + 1]:
                swing_highs.append((i, price_window[i], rsi_values[i]))

    # Bearish divergence: latest swing high has higher price but lower RSI
    if len(swing_highs) >= 2:
        prev_high = swing_highs[-2]
        curr_high = swing_highs[-1]
        if (curr_high[1] > prev_high[1] * 1.001  # Price higher
            and curr_high[2] < prev_high[2] - 2.0):  # RSI lower by at least 2 pts
            return "bearish"

    return None


def detect_engulfing(candles: list[dict]) -> Optional[str]:
    """Detect bullish/bearish engulfing candle patterns.

    Bullish engulfing: bearish candle followed by larger bullish candle that
    completely engulfs the previous body.
    """
    if len(candles) < 2:
        return None

    prev = candles[-2]
    curr = candles[-1]
    prev_body = abs(prev["close"] - prev["open"])
    curr_body = abs(curr["close"] - curr["open"])

    if prev_body == 0 or curr_body == 0:
        return None

    # Require current candle body to be at least 1.2x previous
    if curr_body < prev_body * 1.2:
        return None

    # Bullish engulfing
    if (prev["close"] < prev["open"]  # Previous bearish
        and curr["close"] > curr["open"]  # Current bullish
        and curr["open"] <= prev["close"]  # Opens at or below prev close
        and curr["close"] >= prev["open"]):  # Closes at or above prev open
        return "bullish"

    # Bearish engulfing
    if (prev["close"] > prev["open"]  # Previous bullish
        and curr["close"] < curr["open"]  # Current bearish
        and curr["open"] >= prev["close"]  # Opens at or above prev close
        and curr["close"] <= prev["open"]):  # Closes at or below prev open
        return "bearish"

    return None


def find_swing_levels(candles: list[dict], num_levels: int = 3) -> dict:
    """Find recent swing high/low support and resistance levels from 1H candles.

    Returns dict with 'support' and 'resistance' lists of price levels.
    """
    if len(candles) < 10:
        return {"support": [], "resistance": []}

    swing_highs = []
    swing_lows = []

    for i in range(2, len(candles) - 2):
        # Swing high: higher than 2 bars on each side
        if (candles[i]["high"] > candles[i-1]["high"]
            and candles[i]["high"] > candles[i-2]["high"]
            and candles[i]["high"] > candles[i+1]["high"]
            and candles[i]["high"] > candles[i+2]["high"]):
            swing_highs.append(candles[i]["high"])

        # Swing low: lower than 2 bars on each side
        if (candles[i]["low"] < candles[i-1]["low"]
            and candles[i]["low"] < candles[i-2]["low"]
            and candles[i]["low"] < candles[i+1]["low"]
            and candles[i]["low"] < candles[i+2]["low"]):
            swing_lows.append(candles[i]["low"])

    return {
        "support": sorted(swing_lows)[-num_levels:] if swing_lows else [],
        "resistance": sorted(swing_highs)[-num_levels:] if swing_highs else [],
    }


def calc_atr_ratio(highs: list[float], lows: list[float], closes: list[float],
                   fast_period: int = 5, slow_period: int = 20) -> Optional[float]:
    """ATR expansion ratio: current ATR vs longer-term average.

    > 1.2 means volatility is expanding (good for trend entries).
    < 0.8 means volatility is contracting (chop, skip entries).
    """
    fast_atr = calc_atr(highs, lows, closes, fast_period)
    slow_atr = calc_atr(highs, lows, closes, slow_period)
    if fast_atr is None or slow_atr is None or slow_atr == 0:
        return None
    return fast_atr / slow_atr


# ================================================================
# MARKET DATA CLIENTS
# ================================================================

class CoinbaseDataClient:
    """Fetches real-time market data from Coinbase public API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "CryptoScalper/2.0",
        })
        self._candle_cache = {}
        self._book_cache = {}

    def get_ticker(self, product_id: str) -> Optional[dict]:
        """Get current price from Coinbase Exchange."""
        try:
            resp = self.session.get(
                f"{COINBASE_EXCHANGE_API}/products/{product_id}/ticker",
                timeout=10,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return {
                "price": float(data.get("price", 0)),
                "volume_24h": float(data.get("volume", 0)),
                "bid": float(data.get("bid", 0)),
                "ask": float(data.get("ask", 0)),
            }
        except Exception as e:
            logger.warning(f"Ticker error {product_id}: {e}")
            return None

    def get_candles(self, product_id: str, granularity: str = "FIVE_MINUTE",
                    num_candles: int = 50) -> Optional[list[dict]]:
        """Get OHLCV candles. Returns oldest-first."""
        cache_key = f"{product_id}_{granularity}_{num_candles}"
        cached = self._candle_cache.get(cache_key)
        if cached and time.time() - cached["time"] < 15:  # HFT: 15s cache (was 25s)
            return cached["data"]

        try:
            gran_sec = GRANULARITY_SECONDS.get(granularity, 300)
            end = datetime.now(timezone.utc)
            start = end - timedelta(seconds=gran_sec * num_candles)

            resp = self.session.get(
                f"{COINBASE_EXCHANGE_API}/products/{product_id}/candles",
                params={
                    "granularity": gran_sec,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
                timeout=10,
            )
            if resp.status_code != 200:
                return None

            raw = resp.json()
            if not raw:
                return None

            # Coinbase returns: [time, low, high, open, close, volume] newest-first
            candles = []
            for c in reversed(raw):
                candles.append({
                    "time": c[0],
                    "open": float(c[3]),
                    "high": float(c[2]),
                    "low": float(c[1]),
                    "close": float(c[4]),
                    "volume": float(c[5]),
                })

            self._candle_cache[cache_key] = {"data": candles, "time": time.time()}
            return candles
        except Exception as e:
            logger.warning(f"Candle error {product_id}: {e}")
            return None

    def get_order_book(self, product_id: str, depth_pct: float = 0.025) -> Optional[dict]:
        """Get L2 order book for imbalance calculation.

        Uses price-based depth (default 2.5% from mid) per hftbacktest research,
        not a fixed level count, so imbalance is comparable across all pairs.
        """
        cache_key = product_id
        cached = self._book_cache.get(cache_key)
        if cached and time.time() - cached["time"] < 5:
            return cached["data"]

        try:
            resp = self.session.get(
                f"{COINBASE_EXCHANGE_API}/products/{product_id}/book",
                params={"level": 2},
                timeout=8,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])

            if not bids or not asks:
                return None

            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2

            # Filter to within depth_pct of mid-price (research: 2.5%)
            price_floor = mid_price * (1 - depth_pct)
            price_ceil = mid_price * (1 + depth_pct)

            bid_vol = sum(float(b[1]) for b in bids if float(b[0]) >= price_floor)
            ask_vol = sum(float(a[1]) for a in asks if float(a[0]) <= price_ceil)

            total = bid_vol + ask_vol
            imbalance = (bid_vol - ask_vol) / total if total > 0 else 0

            result = {
                "bid_volume": bid_vol,
                "ask_volume": ask_vol,
                "imbalance": round(imbalance, 4),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": mid_price,
                "spread_pct": round((best_ask - best_bid) / mid_price * 100, 4),
            }

            self._book_cache[cache_key] = {"data": result, "time": time.time()}
            return result
        except Exception as e:
            logger.warning(f"Order book error {product_id}: {e}")
            return None


class BinanceDataClient:
    """Fetches funding rate and open interest from Binance/Bybit Futures (free, no auth)."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._funding_cache = {}
        self._oi_cache = {}
        self._binance_blocked = False  # Track if Binance returns 451

    def _get_funding_dydx(self, pair: str) -> Optional[dict]:
        """Fallback: get funding rate from dYdX v4 (decentralized, no geo-blocking)."""
        try:
            # dYdX uses BTC-USD format (same as Coinbase)
            base = pair.replace("-USD", "").replace("-USDT", "")
            dydx_symbol = f"{base}-USD"
            resp = self.session.get(
                f"{DYDX_API}/v4/perpetualMarkets",
                timeout=8,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            market = data.get("markets", {}).get(dydx_symbol)
            if not market:
                return None

            rate = float(market.get("nextFundingRate", 0))
            result = {
                "rate": rate,
                "time": int(time.time() * 1000),
                "rates_history": [rate],  # dYdX only gives next rate
            }
            return result
        except Exception as e:
            logger.debug(f"dYdX funding rate error {pair}: {e}")
            return None

    def _get_funding_okx(self, pair: str) -> Optional[dict]:
        """Fallback: get funding rate from OKX (usually not geo-blocked for read)."""
        try:
            base = pair.replace("-USD", "").replace("-USDT", "")
            okx_symbol = f"{base}-USDT-SWAP"
            resp = self.session.get(
                f"{OKX_API}/api/v5/public/funding-rate",
                params={"instId": okx_symbol},
                timeout=8,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            rows = data.get("data", [])
            if not rows:
                return None

            current = rows[0]
            rate = float(current.get("fundingRate", 0))
            result = {
                "rate": rate,
                "time": int(current.get("fundingTime", 0)),
                "rates_history": [rate],
            }
            return result
        except Exception as e:
            logger.debug(f"OKX funding rate error {pair}: {e}")
            return None

    def get_funding_rate(self, pair: str) -> Optional[dict]:
        """Get funding rate with fallback chain: Binance -> dYdX -> OKX."""
        symbol = _to_binance_symbol(pair)
        cached = self._funding_cache.get(symbol)
        if cached and time.time() - cached["time"] < 60:  # 1-min cache
            return cached["data"]

        result = None

        # Try Binance first (unless previously blocked)
        if not self._binance_blocked:
            try:
                resp = self.session.get(
                    f"{BINANCE_FAPI}/fapi/v1/fundingRate",
                    params={"symbol": symbol, "limit": 3},
                    timeout=8,
                )
                if resp.status_code in (451, 403):
                    logger.info(f"Binance geo-blocked ({resp.status_code}), switching to dYdX/OKX for funding rates")
                    self._binance_blocked = True
                elif resp.status_code == 200:
                    data = resp.json()
                    if data:
                        current = data[-1]
                        result = {
                            "rate": float(current.get("fundingRate", 0)),
                            "time": int(current.get("fundingTime", 0)),
                            "rates_history": [float(d.get("fundingRate", 0)) for d in data],
                        }
            except Exception as e:
                logger.debug(f"Binance funding rate error {pair}: {e}")

        # Fallback to dYdX (decentralized, most reliable)
        if result is None:
            result = self._get_funding_dydx(pair)

        # Fallback to OKX
        if result is None:
            result = self._get_funding_okx(pair)

        if result:
            self._funding_cache[symbol] = {"data": result, "time": time.time()}
            return result
        return None

    def get_open_interest(self, pair: str) -> Optional[dict]:
        """Get open interest for a pair."""
        symbol = _to_binance_symbol(pair)
        cached = self._oi_cache.get(symbol)
        if cached and time.time() - cached["time"] < 60:
            return cached["data"]

        try:
            resp = self.session.get(
                f"{BINANCE_FAPI}/fapi/v1/openInterest",
                params={"symbol": symbol},
                timeout=8,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            result = {
                "open_interest": float(data.get("openInterest", 0)),
                "symbol": data.get("symbol", symbol),
            }
            self._oi_cache[symbol] = {"data": result, "time": time.time()}
            return result
        except Exception as e:
            logger.warning(f"OI error {pair}: {e}")
            return None


# ================================================================
# DYNAMIC PAIR DISCOVERY (v3.1)
# ================================================================

class PairScanner:
    """Discovers tradeable pairs from Coinbase by scanning the full product catalog.

    Filters by: USD quote currency, 'online' status, 24h volume, and bid-ask spread.
    Refreshes the pair list periodically (default: every hour).
    """

    # Stablecoins and wrapped tokens to exclude (not useful for scalping)
    EXCLUDED_BASES = {
        "USDT", "USDC", "DAI", "BUSD", "TUSD", "USDP", "GUSD", "FRAX",
        "PYUSD", "EURC", "CBETH",  # wrapped/staking tokens
        "WBTC", "WETH",  # wrapped (trade the underlying instead)
    }

    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "CryptoScalper/3.1",
        })
        self._last_scan_time = 0
        self._cached_pairs = []

    def scan(self) -> list[str]:
        """Scan Coinbase for all liquid USD trading pairs.

        Returns a list of pair IDs sorted by 24h volume (highest first).
        Uses cached result if within refresh interval.
        """
        now = time.time()
        refresh = self.config.get("pair_refresh_interval_sec", 3600)
        if self._cached_pairs and (now - self._last_scan_time) < refresh:
            return self._cached_pairs

        logger.info("=== PAIR SCANNER: Discovering tradeable pairs from Coinbase ===")

        # Step 1: Get all products from Coinbase
        try:
            resp = self.session.get(
                f"{COINBASE_EXCHANGE_API}/products",
                timeout=15,
            )
            if resp.status_code != 200:
                logger.warning(f"Products API returned {resp.status_code}")
                return self._cached_pairs or self.config.get("pairs", [])
            products = resp.json()
        except Exception as e:
            logger.warning(f"Products API error: {e}")
            return self._cached_pairs or self.config.get("pairs", [])

        # Step 2: Filter to USD pairs that are online and not excluded
        usd_pairs = []
        for p in products:
            pair_id = p.get("id", "")
            quote = p.get("quote_currency", "")
            base = p.get("base_currency", "")
            status = p.get("status", "")

            if quote != "USD":
                continue
            if status != "online":
                continue
            if base in self.EXCLUDED_BASES:
                continue
            # Skip pairs with "auction" in trading_disabled or similar flags
            if p.get("trading_disabled", False):
                continue
            if p.get("cancel_only", False):
                continue
            if p.get("limit_only", False):
                continue

            usd_pairs.append(pair_id)

        logger.info(f"  Found {len(usd_pairs)} USD pairs on Coinbase (online, not excluded)")

        # Step 3: Fetch 24h stats for volume + spread filtering
        # Coinbase has /products/{id}/stats for 24h volume
        min_vol = self.config.get("min_24h_volume_usd", 500_000)
        max_spread = self.config.get("max_spread_pct", 0.15)
        qualified = []

        # Batch: fetch stats for all pairs (with rate limiting)
        for pair_id in usd_pairs:
            try:
                resp = self.session.get(
                    f"{COINBASE_EXCHANGE_API}/products/{pair_id}/stats",
                    timeout=8,
                )
                if resp.status_code != 200:
                    continue
                stats = resp.json()

                volume_24h = float(stats.get("volume", 0))
                last_price = float(stats.get("last", 0))
                open_price = float(stats.get("open", 0))

                # Volume in USD (Coinbase stats volume is in base currency)
                price_for_vol = last_price if last_price > 0 else open_price
                vol_usd = volume_24h * price_for_vol

                if vol_usd < min_vol:
                    continue

                # Quick spread check via ticker
                ticker_resp = self.session.get(
                    f"{COINBASE_EXCHANGE_API}/products/{pair_id}/ticker",
                    timeout=8,
                )
                spread_pct = 999
                if ticker_resp.status_code == 200:
                    ticker = ticker_resp.json()
                    bid = float(ticker.get("bid", 0))
                    ask = float(ticker.get("ask", 0))
                    mid = (bid + ask) / 2
                    if mid > 0 and bid > 0 and ask > 0:
                        spread_pct = (ask - bid) / mid * 100

                if spread_pct > max_spread:
                    continue

                qualified.append({
                    "pair": pair_id,
                    "volume_usd": vol_usd,
                    "spread_pct": spread_pct,
                    "price": price_for_vol,
                })

            except Exception as e:
                logger.debug(f"  Stats error {pair_id}: {e}")
                continue

            time.sleep(0.05)  # Rate limit: ~20 req/sec

        # Step 4: Sort by volume (most liquid first) and cap
        qualified.sort(key=lambda x: x["volume_usd"], reverse=True)
        max_pairs = self.config.get("max_pairs", 80)
        qualified = qualified[:max_pairs]

        result = [q["pair"] for q in qualified]

        # v3.3: Apply pair whitelist if configured
        whitelist = self.config.get("pair_whitelist")
        if whitelist:
            whitelist_set = {w.upper() for w in whitelist}
            before = len(result)
            result = [p for p in result if p.replace("-USD", "") in whitelist_set]
            logger.info(f"  Pair whitelist active: {before} -> {len(result)} pairs "
                       f"(whitelist: {', '.join(sorted(whitelist_set))})")

        self._cached_pairs = result
        self._last_scan_time = now

        # Log results
        logger.info(f"  Qualified: {len(result)} pairs (vol >= ${min_vol:,.0f}, spread <= {max_spread}%)")
        for i, q in enumerate(qualified[:10]):
            logger.info(f"    {i+1}. {q['pair']}: ${q['volume_usd']:,.0f} vol, {q['spread_pct']:.3f}% spread")
        if len(qualified) > 10:
            logger.info(f"    ... and {len(qualified) - 10} more")

        return result


# ================================================================
# GLOBAL MARKET FILTERS (v3.0)
# ================================================================

class FearGreedIndex:
    """Fetch crypto Fear & Greed Index from alternative.me (free, no key)."""

    def __init__(self):
        self._cache = None
        self._cache_time = 0
        self._cache_ttl = 300  # 5 min cache (updates daily anyway)

    def get(self) -> Optional[dict]:
        """Returns {"value": 0-100, "classification": "Extreme Fear"|..., "timestamp": ...}"""
        if self._cache and time.time() - self._cache_time < self._cache_ttl:
            return self._cache
        try:
            resp = requests.get(
                "https://api.alternative.me/fng/?limit=1",
                timeout=8,
            )
            if resp.status_code != 200:
                return self._cache
            data = resp.json().get("data", [{}])[0]
            result = {
                "value": int(data.get("value", 50)),
                "classification": data.get("value_classification", "Neutral"),
            }
            self._cache = result
            self._cache_time = time.time()
            return result
        except Exception as e:
            logger.warning(f"Fear & Greed API error: {e}")
            return self._cache


class CUSUMFilter:
    """Cumulative Sum filter (Lopez de Prado, AFML Ch. 2).

    Only triggers when cumulative positive or negative returns exceed
    a threshold. Filters out noise trades and only acts on meaningful moves.
    """

    def __init__(self, threshold: float = 0.003):
        self.threshold = threshold
        self._pos = {}  # pair -> cumulative positive sum
        self._neg = {}  # pair -> cumulative negative sum

    def update(self, pair: str, ret: float) -> Optional[str]:
        """Feed a return. Returns 'up'/'down' when threshold breached, else None."""
        pos = self._pos.get(pair, 0)
        neg = self._neg.get(pair, 0)

        pos = max(0, pos + ret)
        neg = min(0, neg + ret)

        self._pos[pair] = pos
        self._neg[pair] = neg

        if pos > self.threshold:
            self._pos[pair] = 0  # Reset
            return "up"
        if neg < -self.threshold:
            self._neg[pair] = 0  # Reset
            return "down"
        return None


class CrossAssetVolTracker:
    """Track BTC realized volatility to detect vol spillover to altcoins.

    Research (SSRN 5048674): BTC vol spikes predict altcoin vol with a lag.
    When BTC vol is elevated, widen stops or reduce position sizes.
    """

    def __init__(self):
        self._btc_returns = deque(maxlen=288)  # 24h of 5-min returns
        self._last_btc_price = None
        self._cache_time = 0

    def update_btc_price(self, price: float):
        if self._last_btc_price and self._last_btc_price > 0:
            ret = (price - self._last_btc_price) / self._last_btc_price
            self._btc_returns.append(ret)
        self._last_btc_price = price

    @property
    def btc_realized_vol(self) -> float:
        """Annualized realized vol from 5-min returns."""
        if len(self._btc_returns) < 20:
            return 0
        returns = list(self._btc_returns)
        mean = sum(returns) / len(returns)
        var = sum((r - mean) ** 2 for r in returns) / len(returns)
        return math.sqrt(var) * math.sqrt(288 * 365)  # Annualize from 5-min

    @property
    def btc_vol_ratio(self) -> float:
        """Current vol vs rolling average. >2.0 = elevated."""
        if len(self._btc_returns) < 100:
            return 1.0
        # Compare recent (last 2h = 24 bars) vs full window
        recent = list(self._btc_returns)[-24:]
        full = list(self._btc_returns)
        recent_var = sum(r**2 for r in recent) / len(recent)
        full_var = sum(r**2 for r in full) / len(full)
        if full_var == 0:
            return 1.0
        return math.sqrt(recent_var / full_var)


class SignalBandit:
    """Multi-armed bandit that tracks signal component performance.

    Research (Taylor & Francis 2025): DQN strategy selection achieved 120x growth.
    This is a simpler Thompson Sampling approach — weights signals by recent success.
    """

    def __init__(self, decay: float = 0.95):
        self.decay = decay
        self._wins = {}   # component -> decayed win count
        self._total = {}  # component -> decayed total count

    def record(self, components: list, won: bool):
        """Record trade outcome for each signal component that fired."""
        # Only decay components that are being updated (preserve history for infrequent signals)
        for comp in components:
            self._wins[comp] = self._wins.get(comp, 0) * self.decay
            self._total[comp] = self._total.get(comp, 0) * self.decay
            self._total[comp] += 1
            if won:
                self._wins[comp] = self._wins.get(comp, 0) + 1

    def get_weight(self, component: str) -> float:
        """Get performance weight for a signal component (0.5 - 1.5).

        Returns 1.0 (neutral) if insufficient data. Winning signals get
        boosted up to 1.5x, consistently losing signals get penalized to 0.5x.
        """
        total = self._total.get(component, 0)
        if total < 3:
            return 1.0  # Not enough data
        wins = self._wins.get(component, 0)
        win_rate = wins / total
        # Map win_rate [0, 1] to weight [0.5, 1.5]
        return 0.5 + win_rate

    def get_all_weights(self) -> dict:
        """Return all component weights for logging."""
        result = {}
        for comp in self._total:
            result[comp] = round(self.get_weight(comp), 2)
        return result


# ================================================================
# ADAPTIVE MACRO REGIME SYSTEM (v3.5)
# ================================================================

class MacroRegime(Enum):
    """Five macro market regimes with distinct trading profiles."""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    HIGH_VOL_FEAR = "high_vol_fear"
    LOW_VOL_CHOP = "low_vol_chop"
    RECOVERY = "recovery"


# Parameter overrides per regime. Keys must match SCALPER_CONFIG keys.
# 'side_bias' and 'side_bias_bonus' are new keys consumed by SignalDetector.
REGIME_PROFILES = {
    MacroRegime.TRENDING_BULL: {
        "tp_atr_mult": 5.0,
        "sl_atr_mult": 3.5,
        "min_confluence_score": 5,
        "min_ranging_score": 6,
        "kelly_fraction": 0.12,
        "max_position_usd": 5.00,
        "max_open_positions": 8,
        "max_hold_hours": 48,
        "cooldown_after_loss_sec": 120,
        "progressive_stop_start_pct": 0.6,
        "progressive_stop_end_mult": 0.4,
        "side_bias": "long",
        "side_bias_bonus": 1,
    },
    MacroRegime.TRENDING_BEAR: {
        "tp_atr_mult": 3.0,
        "sl_atr_mult": 2.5,
        "min_confluence_score": 6,
        "min_ranging_score": 7,
        "kelly_fraction": 0.07,
        "max_position_usd": 3.00,
        "max_open_positions": 5,
        "max_hold_hours": 24,
        "cooldown_after_loss_sec": 300,
        "progressive_stop_start_pct": 0.4,
        "progressive_stop_end_mult": 0.2,
        "side_bias": "short",
        "side_bias_bonus": 2,
    },
    MacroRegime.HIGH_VOL_FEAR: {
        "tp_atr_mult": 2.5,
        "sl_atr_mult": 2.0,
        "min_confluence_score": 7,
        "min_ranging_score": 8,
        "kelly_fraction": 0.05,
        "max_position_usd": 2.00,
        "max_open_positions": 3,
        "max_hold_hours": 12,
        "cooldown_after_loss_sec": 600,
        "progressive_stop_start_pct": 0.3,
        "progressive_stop_end_mult": 0.15,
        "side_bias": "short",
        "side_bias_bonus": 2,
    },
    MacroRegime.LOW_VOL_CHOP: {
        "tp_atr_mult": 3.0,
        "sl_atr_mult": 2.5,
        "min_confluence_score": 7,
        "min_ranging_score": 8,
        "kelly_fraction": 0.04,
        "max_position_usd": 2.00,
        "max_open_positions": 2,
        "max_hold_hours": 12,
        "cooldown_after_loss_sec": 900,
        "progressive_stop_start_pct": 0.3,
        "progressive_stop_end_mult": 0.2,
        "side_bias": None,
        "side_bias_bonus": 0,
    },
    MacroRegime.RECOVERY: {
        "tp_atr_mult": 4.0,
        "sl_atr_mult": 3.0,
        "min_confluence_score": 5,
        "min_ranging_score": 6,
        "kelly_fraction": 0.08,
        "max_position_usd": 3.50,
        "max_open_positions": 6,
        "max_hold_hours": 30,
        "cooldown_after_loss_sec": 240,
        "progressive_stop_start_pct": 0.5,
        "progressive_stop_end_mult": 0.3,
        "side_bias": "long",
        "side_bias_bonus": 1,
    },
}


class RegimeDetector:
    """Composite macro regime detector with hysteresis.

    Combines Fear & Greed, BTC volatility, average ADX, rolling win rate,
    and market breadth into a regime classification. Uses sticky scoring
    and confirmation to prevent whipsawing between regimes.
    """

    HYSTERESIS_BONUS = 0.15
    CONFIRM_SCANS = 3      # Consecutive scans before switching
    LOCKOUT_SCANS = 10     # Minimum scans between switches

    def __init__(self):
        self._current = MacroRegime.TRENDING_BEAR  # Safe default for bear markets
        self._candidate = None
        self._candidate_count = 0
        self._locked_until = 0
        self._cycle = 0
        self._fg_history = deque(maxlen=24)  # ~24 scans of F&G history

    @property
    def current_regime(self) -> MacroRegime:
        return self._current

    @property
    def regime_age(self) -> int:
        """Number of scans since last regime switch."""
        return self._cycle - max(0, self._locked_until - self.LOCKOUT_SCANS)

    def detect(self, fear_greed: int, btc_vol_ratio: float,
               avg_adx: float, rolling_wr: float,
               market_breadth: float) -> MacroRegime:
        """Detect current macro regime from composite inputs."""
        self._cycle += 1
        self._fg_history.append(fear_greed)

        scores = self._score_all(fear_greed, btc_vol_ratio, avg_adx,
                                 rolling_wr, market_breadth)

        # Hysteresis: current regime gets a bonus (must be clearly beaten)
        scores[self._current] += self.HYSTERESIS_BONUS

        best = max(scores, key=scores.get)

        # If current regime still wins, reset candidate
        if best == self._current:
            self._candidate = None
            self._candidate_count = 0
            return self._current

        # Lockout check
        if self._cycle < self._locked_until:
            return self._current

        # Confirmation: need CONFIRM_SCANS consecutive cycles
        if best == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = best
            self._candidate_count = 1

        if self._candidate_count >= self.CONFIRM_SCANS:
            old = self._current
            self._current = best
            self._candidate = None
            self._candidate_count = 0
            self._locked_until = self._cycle + self.LOCKOUT_SCANS
            logger.info(f"MACRO REGIME CHANGE: {old.value} -> {best.value}")
            return best

        return self._current

    def _score_all(self, fg: int, vol: float, adx: float,
                   wr: float, breadth: float) -> dict:
        """Compute regime scores (0-1 scale) for all five regimes."""
        # Normalize inputs to 0-1
        fg_n = max(0, min(100, fg)) / 100.0            # 0=fear, 1=greed
        vol_n = max(0, min(1, (vol - 0.5) / 2.5))      # 0=low vol, 1=high vol
        adx_n = max(0, min(1, (adx - 10) / 40.0))      # 0=no trend, 1=strong trend
        # breadth already 0-1 (fraction of bullish pairs)
        # wr already 0-1

        # F&G trend: is fear_greed rising? (recovery signal)
        fg_rising = 0.0
        if len(self._fg_history) >= 6:
            recent_avg = sum(list(self._fg_history)[-6:]) / 6
            older_avg = sum(list(self._fg_history)[:max(1, len(self._fg_history) - 6)]) / max(1, len(self._fg_history) - 6)
            if recent_avg > older_avg + 3:
                fg_rising = min(1.0, (recent_avg - older_avg) / 15.0)

        scores = {}

        # TRENDING_BULL: high F&G, strong trend, most pairs bullish, winning
        scores[MacroRegime.TRENDING_BULL] = (
            0.30 * fg_n +
            0.20 * adx_n +
            0.25 * breadth +
            0.15 * wr +
            0.10 * (1.0 - vol_n)  # Calm bull preferred
        )

        # TRENDING_BEAR: low F&G, strong trend, most pairs bearish, losing
        scores[MacroRegime.TRENDING_BEAR] = (
            0.30 * (1.0 - fg_n) +
            0.20 * adx_n +
            0.25 * (1.0 - breadth) +
            0.15 * (1.0 - wr) +
            0.10 * min(0.7, vol_n)  # Moderate vol is bearish, extreme is fear
        )

        # HIGH_VOL_FEAR: very low F&G + high vol
        fear_intensity = max(0, (25 - fg) / 25.0) if fg < 25 else 0.0
        scores[MacroRegime.HIGH_VOL_FEAR] = (
            0.35 * fear_intensity +
            0.25 * vol_n +
            0.15 * (1.0 - breadth) +
            0.15 * (1.0 - wr) +
            0.10 * adx_n  # Can be trending or not
        )

        # LOW_VOL_CHOP: low ADX, low vol, mid F&G
        mid_fg = 1.0 - abs(fg_n - 0.5) * 2  # Peaks at F&G=50
        scores[MacroRegime.LOW_VOL_CHOP] = (
            0.30 * (1.0 - adx_n) +
            0.25 * (1.0 - vol_n) +
            0.20 * mid_fg +
            0.15 * (0.5 - abs(breadth - 0.5)) * 2 +  # Mixed breadth
            0.10 * (1.0 - abs(wr - 0.5) * 2)  # Middling WR
        )

        # RECOVERY: F&G rising from fear, improving WR
        was_fearful = 1.0 if fg < 45 and len(self._fg_history) >= 3 else 0.0
        scores[MacroRegime.RECOVERY] = (
            0.30 * fg_rising +
            0.25 * was_fearful * fg_rising +  # Must be rising FROM fear
            0.20 * wr +
            0.15 * breadth +
            0.10 * (1.0 - vol_n)
        )

        return scores


class RegimeAdapter:
    """Applies regime-specific parameter overrides to the config dict.

    Stores a snapshot of the original config so profiles can be cleanly
    re-applied without accumulating drift from multiple switches.
    """

    # Keys that regime profiles can override
    PROFILE_KEYS = {
        "tp_atr_mult", "sl_atr_mult", "min_confluence_score", "min_ranging_score",
        "kelly_fraction", "max_position_usd", "max_open_positions",
        "max_hold_hours", "cooldown_after_loss_sec",
        "progressive_stop_start_pct", "progressive_stop_end_mult",
        "side_bias", "side_bias_bonus",
    }

    def __init__(self, profiles: dict, base_config: dict):
        self._profiles = profiles
        self._base = {k: base_config.get(k) for k in self.PROFILE_KEYS if k in base_config}
        # side_bias/side_bias_bonus don't exist in base config — default to None/0
        self._base.setdefault("side_bias", None)
        self._base.setdefault("side_bias_bonus", 0)
        self._applied = None

    def apply(self, regime: MacroRegime, config: dict):
        """Apply regime profile to config. No-op if regime unchanged."""
        if regime == self._applied:
            return

        profile = self._profiles.get(regime, {})

        # Reset all profile keys to base values first
        for key in self.PROFILE_KEYS:
            if key in self._base:
                config[key] = self._base[key]

        # Apply regime overrides
        for key, value in profile.items():
            config[key] = value

        self._applied = regime
        logger.info(
            f"REGIME ADAPTER: {regime.value} — "
            f"confluence={config.get('min_confluence_score')}, "
            f"TP={config.get('tp_atr_mult')}x, SL={config.get('sl_atr_mult')}x, "
            f"kelly={config.get('kelly_fraction')}, max_pos={config.get('max_open_positions')}, "
            f"hold={config.get('max_hold_hours')}h, bias={config.get('side_bias')}"
        )

    @property
    def applied_regime(self) -> MacroRegime:
        return self._applied


# ================================================================
# MARKET CONTEXT (regime + external signals)
# ================================================================

@dataclass
class MarketContext:
    """Aggregated market context for a single pair."""
    pair: str
    # Regime
    regime: str = "unknown"         # "trending_up", "trending_down", "ranging", "volatile"
    macro_regime: str = "unknown"   # Macro regime (v3.5): "trending_bull", "high_vol_fear", etc.
    adx: float = 0.0
    trend_direction: str = "neutral"  # "up", "down", "neutral"
    # Hourly trend
    hourly_ema_fast: float = 0.0    # 9 EMA on 1H
    hourly_ema_slow: float = 0.0    # 20 EMA on 1H
    hourly_rsi: float = 50.0
    hourly_trend: str = "neutral"   # "bullish", "bearish", "neutral"
    # Order book
    ob_imbalance: float = 0.0       # -1 to +1
    spread_pct: float = 0.0
    # Funding
    funding_rate: float = 0.0
    funding_signal: str = "neutral"  # "overbought", "oversold", "neutral"
    # Volume profile
    poc: float = 0.0
    vah: float = 0.0
    val: float = 0.0
    # Current price
    price: float = 0.0
    atr_5m: float = 0.0
    atr_1h: float = 0.0             # 1H ATR for exit calculations (overcomes fees)
    atr_pct: float = 0.0            # ATR as % of price
    # v3.0 additions
    fear_greed: int = 50            # 0-100
    cusum_signal: str = ""          # "up", "down", or ""
    btc_vol_ratio: float = 1.0     # BTC vol vs average (>2 = elevated)
    is_pump: bool = False           # Pump detection flag
    # v3.2 additions (signal quality improvements)
    rsi_divergence: str = ""        # "bullish", "bearish", or ""
    engulfing: str = ""             # "bullish", "bearish", or ""
    atr_ratio: float = 1.0         # ATR expansion ratio (>1.2 = vol expanding)
    swing_supports: list = field(default_factory=list)
    swing_resistances: list = field(default_factory=list)
    near_support: bool = False      # Price within 0.5% of a swing support
    near_resistance: bool = False   # Price within 0.5% of a swing resistance


# ================================================================
# SIGNAL DETECTION (confluence-based)
# ================================================================

GRADE_MAP = {"A": 4, "B": 3, "C": 2, "D": 1}
MIN_GRADE = {"A": 4, "B": 3, "C": 2, "D": 1}


@dataclass
class Signal:
    """A trading signal with confluence scoring."""
    pair: str
    side: str
    signal_type: str
    confluence_score: int    # 0-10
    quality_grade: str       # A, B, C, D
    price: float
    rsi: float
    atr: float
    take_profit: float
    stop_loss: float
    regime: str
    reasoning: str
    components: list         # Which signals fired
    timestamp: str


class SignalDetector:
    """Multi-signal confluence detector. Thinks like a pro trader."""

    def __init__(self, config: dict = None, bandit: SignalBandit = None):
        self.config = config or SCALPER_CONFIG
        self.bandit = bandit

    def analyze(self, pair: str, candles_5m: list[dict],
                ctx: MarketContext) -> Optional[Signal]:
        """Analyze a pair using all available data. Returns best signal or None."""
        if not candles_5m or len(candles_5m) < 35:
            logger.debug(f"  {pair}: insufficient candles ({len(candles_5m) if candles_5m else 0}/35)")
            return None

        closes = [c["close"] for c in candles_5m]
        highs = [c["high"] for c in candles_5m]
        lows = [c["low"] for c in candles_5m]
        volumes = [c["volume"] for c in candles_5m]
        price = closes[-1]

        # Calculate 5M indicators
        rsi = calc_rsi(closes)
        macd = calc_macd(closes)
        bb = calc_bollinger(closes)
        atr = calc_atr(highs, lows, closes)
        vwap = calc_vwap(closes[-20:], volumes[-20:], highs[-20:], lows[-20:])
        fvgs = detect_fvg(candles_5m[-10:])  # Recent FVGs only

        if rsi is None or atr is None:
            logger.debug(f"  {pair}: indicator calc failed (rsi={rsi}, atr={atr})")
            return None

        # Store ATR in context
        ctx.atr_5m = atr
        ctx.atr_pct = atr / price if price > 0 else 0
        ctx.price = price

        # === CONFLUENCE SCORING ===
        score = 0
        components = []
        reasons = []

        # Determine regime-adaptive RSI thresholds
        if ctx.regime.startswith("trending"):
            rsi_os = self.config["rsi_oversold_trending"]
            rsi_ob = self.config["rsi_overbought_trending"]
        else:
            rsi_os = self.config["rsi_oversold_ranging"]
            rsi_ob = self.config["rsi_overbought_ranging"]

        # --- 1. HOURLY TREND ALIGNMENT (worth 2 points) ---
        if ctx.hourly_trend == "bullish":
            score += 2
            components.append("1H_TREND_UP")
            reasons.append(f"1H trend bullish (EMA9>{ctx.hourly_ema_fast:.0f} > EMA20>{ctx.hourly_ema_slow:.0f})")
        elif ctx.hourly_trend == "bearish":
            score -= 2  # Penalty for counter-trend
            components.append("1H_TREND_DOWN")

        # --- 2. RSI SIGNAL (worth 1-2 points) ---
        if rsi < rsi_os:
            pts = 2 if rsi < rsi_os - 5 else 1
            score += pts
            components.append("RSI_OVERSOLD")
            reasons.append(f"RSI={rsi:.0f} (oversold<{rsi_os})")
        elif rsi > rsi_ob:
            score -= 1  # Overbought = don't buy
            components.append("RSI_OVERBOUGHT")

        # --- 3. MACD MOMENTUM (worth 1 point) ---
        if macd:
            macd_line, signal_line, histogram = macd
            hist_pair = calc_macd_last_two_histograms(closes)
            if hist_pair:
                prev_hist, curr_hist = hist_pair
                if prev_hist < 0 and curr_hist > 0:
                    score += 1
                    components.append("MACD_BULL_CROSS")
                    reasons.append(f"MACD bullish cross ({prev_hist:.4f}→{curr_hist:.4f})")
                elif prev_hist > 0 and curr_hist < 0:
                    score -= 1
                    components.append("MACD_BEAR_CROSS")

        # --- 4. ORDER BOOK IMBALANCE (worth 1-2 points) ---
        if ctx.ob_imbalance > self.config["ob_strong_buy"]:
            pts = 2 if ctx.ob_imbalance > 0.4 else 1
            score += pts
            components.append("OB_BID_HEAVY")
            reasons.append(f"Order book bid-heavy ({ctx.ob_imbalance:+.2f})")
        elif ctx.ob_imbalance < self.config["ob_strong_sell"]:
            score -= 1
            components.append("OB_ASK_HEAVY")

        # --- 5. VOLUME SPIKE (worth 1 point) ---
        avg_vol = sum(volumes[-20:-1]) / 19 if len(volumes) >= 20 else sum(volumes) / len(volumes)
        curr_vol = volumes[-1]
        if avg_vol > 0 and curr_vol > avg_vol * self.config["volume_spike_mult"]:
            price_chg = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] > 0 else 0
            if price_chg > 0.001:  # Bullish volume spike
                score += 1
                components.append("VOL_SPIKE_BULL")
                reasons.append(f"Volume {curr_vol / avg_vol:.1f}x avg, price +{price_chg:.2%}")
            elif price_chg < -0.001:
                score -= 1
                components.append("VOL_SPIKE_BEAR")

        # --- 6. BOLLINGER BAND POSITION (worth 1 point) ---
        if bb:
            upper, middle, lower, bandwidth = bb
            if price <= lower:
                score += 1
                components.append("BB_LOWER_TOUCH")
                reasons.append(f"Price at BB lower ({lower:.2f})")
            elif price >= upper and bandwidth > self.config["bb_squeeze_threshold"]:
                score -= 1  # At upper band in expanded market = risky buy
                components.append("BB_UPPER_TOUCH")

        # --- 7. FUNDING RATE CONTRARIAN (worth 1 point) ---
        if ctx.funding_signal == "oversold":  # Negative funding = good for longs
            score += 1
            components.append("FUNDING_OVERSOLD")
            reasons.append(f"Funding negative ({ctx.funding_rate:.4%}) = shorts paying")
        elif ctx.funding_signal == "overbought":
            score -= 1
            components.append("FUNDING_OVERHEATED")

        # --- 8. VWAP POSITION (worth 1 point) ---
        if vwap:
            if price > vwap and ctx.hourly_trend == "bullish":
                score += 1
                components.append("ABOVE_VWAP")
                reasons.append(f"Price above VWAP ({vwap:.2f})")
            elif price < vwap and ctx.hourly_trend == "bearish":
                score -= 1
                components.append("BELOW_VWAP_BEARISH")

        # --- 9. FAIR VALUE GAP (worth 1 point) ---
        bullish_fvgs = [f for f in fvgs if f["type"] == "bullish"
                        and f["bottom"] <= price <= f["top"]]
        if bullish_fvgs:
            # Price is filling a bullish FVG — institutional support
            score += 1
            components.append("FVG_FILL")
            fvg = bullish_fvgs[-1]
            reasons.append(f"Filling bullish FVG ({fvg['bottom']:.2f}-{fvg['top']:.2f})")

        # --- 10. VOLUME PROFILE (worth 1 point) ---
        if ctx.val > 0 and price <= ctx.val * 1.005:
            score += 1
            components.append("AT_VAL")
            reasons.append(f"Price near Value Area Low ({ctx.val:.2f})")
        elif ctx.poc > 0 and abs(price - ctx.poc) / ctx.poc < 0.003:
            score += 0  # Neutral at POC

        # --- 11. FEAR & GREED SENTIMENT (worth 1 point) ---
        if ctx.fear_greed > 0:
            if ctx.fear_greed <= self.config.get("fear_greed_extreme_fear", 20):
                # Extreme fear = contrarian buy signal (mean reversion strongest)
                score += 1
                components.append("SENTIMENT_FEAR")
                reasons.append(f"Fear & Greed={ctx.fear_greed} (extreme fear, contrarian buy)")
            elif ctx.fear_greed >= self.config.get("fear_greed_extreme_greed", 80):
                # Extreme greed = caution, not a buy zone
                score -= 1
                components.append("SENTIMENT_GREED")

        # --- 12. CUSUM CONFIRMATION (worth 1 point) ---
        if ctx.cusum_signal == "up":
            score += 1
            components.append("CUSUM_UP")
            reasons.append("CUSUM filter: meaningful upward move detected")
        elif ctx.cusum_signal == "down":
            score -= 1
            components.append("CUSUM_DOWN")

        # --- 13. BTC VOL SPILLOVER (penalty only) ---
        if ctx.btc_vol_ratio > self.config.get("vol_spike_mult", 2.0):
            score -= 1
            components.append("BTC_VOL_ELEVATED")
            reasons.append(f"BTC vol {ctx.btc_vol_ratio:.1f}x elevated — caution")

        # --- 14. RSI DIVERGENCE (worth 2 points — strong reversal signal) ---
        if ctx.rsi_divergence == "bullish":
            score += 2
            components.append("RSI_DIV_BULL")
            reasons.append("Bullish RSI divergence (price lower low, RSI higher low)")
        elif ctx.rsi_divergence == "bearish":
            score -= 1  # Against buy
            components.append("RSI_DIV_BEAR")

        # --- 15. ENGULFING CANDLE PATTERN (worth 1 point) ---
        if ctx.engulfing == "bullish":
            score += 1
            components.append("ENGULF_BULL")
            reasons.append("Bullish engulfing candle pattern")
        elif ctx.engulfing == "bearish":
            score -= 1
            components.append("ENGULF_BEAR")

        # --- 16. SWING LEVEL SUPPORT (worth 1-2 points) ---
        if ctx.near_support and ctx.hourly_trend in ("bullish", "neutral"):
            pts = 2 if ctx.hourly_trend == "bullish" else 1
            score += pts
            components.append("SWING_SUPPORT")
            reasons.append("Price near swing support level")

        # --- 17. ATR EXPANSION FILTER (penalty for low vol) ---
        if ctx.atr_ratio < self.config.get("atr_contraction_threshold", 0.8):
            score -= 2  # Strong penalty: low vol = chop/fees eat everything
            components.append("ATR_CONTRACTED")
            reasons.append(f"ATR ratio {ctx.atr_ratio:.2f} — volatility contracted, skip")
        elif ctx.atr_ratio > self.config.get("atr_expansion_threshold", 1.3):
            score += 1  # Expanding vol = bigger moves, can overcome fees
            components.append("ATR_EXPANDING")
            reasons.append(f"ATR ratio {ctx.atr_ratio:.2f} — volatility expanding")

        # === BANDIT-WEIGHTED SCORE (auto-learned signal weighting) ===
        if self.bandit and components:
            weighted_bonus = 0
            for comp in components:
                w = self.bandit.get_weight(comp)
                if w != 1.0:
                    # Adjust: signals with >60% win rate get +0.5, <40% get -0.5
                    weighted_bonus += (w - 1.0)
            score += round(weighted_bonus)

        # === SELL SIGNAL SCORING ===
        # Mirror: bearish components contribute positively to sell score
        sell_score = 0
        sell_components = []
        sell_reasons = []

        # 1. Hourly trend DOWN
        if ctx.hourly_trend == "bearish":
            sell_score += 2
            sell_components.append("1H_TREND_DOWN")
            sell_reasons.append(f"1H trend bearish (EMA9 < EMA20)")

        # 2. RSI Overbought
        if rsi > rsi_ob:
            pts = 2 if rsi > rsi_ob + 5 else 1
            sell_score += pts
            sell_components.append("RSI_OVERBOUGHT")
            sell_reasons.append(f"RSI={rsi:.0f} (overbought>{rsi_ob})")
        elif rsi < rsi_os:
            sell_score -= 1  # Oversold = don't sell

        # 3. MACD Bear Cross
        if macd:
            hist_pair = calc_macd_last_two_histograms(closes)
            if hist_pair:
                prev_hist, curr_hist = hist_pair
                if prev_hist > 0 and curr_hist < 0:
                    sell_score += 1
                    sell_components.append("MACD_BEAR_CROSS")
                    sell_reasons.append(f"MACD bearish cross ({prev_hist:.4f}->{curr_hist:.4f})")

        # 4. Order Book Ask Heavy
        if ctx.ob_imbalance < self.config["ob_strong_sell"]:
            pts = 2 if ctx.ob_imbalance < -0.4 else 1
            sell_score += pts
            sell_components.append("OB_ASK_HEAVY")
            sell_reasons.append(f"Order book ask-heavy ({ctx.ob_imbalance:+.2f})")

        # 5. Bearish Volume Spike
        if avg_vol > 0 and curr_vol > avg_vol * self.config["volume_spike_mult"]:
            price_chg = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] > 0 else 0
            if price_chg < -0.001:
                sell_score += 1
                sell_components.append("VOL_SPIKE_BEAR")
                sell_reasons.append(f"Volume {curr_vol / avg_vol:.1f}x avg, price {price_chg:.2%}")

        # 6. Bollinger Upper Touch
        if bb:
            upper, middle, lower, bandwidth = bb
            if price >= upper and bandwidth > self.config["bb_squeeze_threshold"]:
                sell_score += 1
                sell_components.append("BB_UPPER_TOUCH")
                sell_reasons.append(f"Price at BB upper ({upper:.2f})")

        # 7. Funding Overheated (contrarian: extreme longs = short opportunity)
        if ctx.funding_signal == "overbought":
            sell_score += 1
            sell_components.append("FUNDING_OVERHEATED")
            sell_reasons.append(f"Funding positive ({ctx.funding_rate:.4%}) = longs paying")

        # 8. Below VWAP in downtrend
        if vwap and price < vwap and ctx.hourly_trend == "bearish":
            sell_score += 1
            sell_components.append("BELOW_VWAP_BEARISH")
            sell_reasons.append(f"Price below VWAP ({vwap:.2f}) in downtrend")

        # 9. Bearish FVG fill
        bearish_fvgs = [f for f in fvgs if f["type"] == "bearish"
                        and f["bottom"] <= price <= f["top"]]
        if bearish_fvgs:
            sell_score += 1
            sell_components.append("FVG_FILL_BEAR")
            fvg = bearish_fvgs[-1]
            sell_reasons.append(f"Filling bearish FVG ({fvg['bottom']:.2f}-{fvg['top']:.2f})")

        # 10. Volume profile: at VAH = resistance
        if ctx.vah > 0 and price >= ctx.vah * 0.995:
            sell_score += 1
            sell_components.append("AT_VAH")
            sell_reasons.append(f"Price near Value Area High ({ctx.vah:.2f})")

        # 11. Extreme greed sentiment (contrarian)
        if ctx.fear_greed > 0 and ctx.fear_greed >= self.config.get("fear_greed_extreme_greed", 80):
            sell_score += 1
            sell_components.append("SENTIMENT_GREED")
            sell_reasons.append(f"Fear & Greed={ctx.fear_greed} (extreme greed, contrarian sell)")

        # 12. CUSUM down signal
        if ctx.cusum_signal == "down":
            sell_score += 1
            sell_components.append("CUSUM_DOWN")
            sell_reasons.append("CUSUM filter: meaningful downward move detected")

        # 13. RSI bearish divergence
        if ctx.rsi_divergence == "bearish":
            sell_score += 2
            sell_components.append("RSI_DIV_BEAR")
            sell_reasons.append("Bearish RSI divergence (price higher high, RSI lower high)")

        # 14. Bearish engulfing
        if ctx.engulfing == "bearish":
            sell_score += 1
            sell_components.append("ENGULF_BEAR")
            sell_reasons.append("Bearish engulfing candle pattern")

        # 15. Swing resistance
        if ctx.near_resistance and ctx.hourly_trend in ("bearish", "neutral"):
            pts = 2 if ctx.hourly_trend == "bearish" else 1
            sell_score += pts
            sell_components.append("SWING_RESISTANCE")
            sell_reasons.append("Price near swing resistance level")

        # 16. ATR expansion/contraction for sell
        # Note: no contraction penalty for sells — in sustained bear markets,
        # low vol doesn't invalidate the short thesis (unlike longs in chop)
        if ctx.atr_ratio > self.config.get("atr_expansion_threshold", 1.3):
            sell_score += 1
            sell_components.append("ATR_EXPANDING")

        # Bandit weighting for sell signals
        if self.bandit and sell_components:
            weighted_bonus = 0
            for comp in sell_components:
                w = self.bandit.get_weight(comp)
                if w != 1.0:
                    weighted_bonus += (w - 1.0)
            sell_score += round(weighted_bonus)

        # === MACRO REGIME SIDE BIAS (v3.5) ===
        # Boost preferred direction's score based on current regime profile
        side_bias = self.config.get("side_bias")
        bias_bonus = self.config.get("side_bias_bonus", 0)
        if bias_bonus and side_bias == "long":
            score += bias_bonus
        elif bias_bonus and side_bias == "short":
            sell_score += bias_bonus

        # === QUALITY GRADING & SIGNAL SELECTION ===
        # Pick the stronger direction
        min_score = self.config["min_confluence_score"]
        min_grade = self.config["min_signal_quality"]

        best_side = None
        best_score = 0
        best_components = []
        best_reasons = []

        # Check buy signal
        if score >= min_score:
            buy_grade = "A" if score >= 7 and "1H_TREND_UP" in components else "B" if score >= 5 else "C" if score >= 4 else "D"
            if GRADE_MAP.get(buy_grade, 0) >= GRADE_MAP.get(min_grade, 0):
                best_side = "buy"
                best_score = score
                best_components = components
                best_reasons = reasons

        # Check sell signal
        if sell_score >= min_score and sell_score > best_score:
            sell_grade = "A" if sell_score >= 7 and "1H_TREND_DOWN" in sell_components else "B" if sell_score >= 5 else "C" if sell_score >= 4 else "D"
            if GRADE_MAP.get(sell_grade, 0) >= GRADE_MAP.get(min_grade, 0):
                best_side = "sell"
                best_score = sell_score
                best_components = sell_components
                best_reasons = sell_reasons

        if best_side is None:
            logger.info(
                f"  {pair}: buy={score} sell={sell_score} "
                f"(need {min_score}) | b:[{','.join(components[:4])}] "
                f"s:[{','.join(sell_components[:4])}]"
            )
            return None

        grade = "A" if best_score >= 7 else "B" if best_score >= 5 else "C" if best_score >= 4 else "D"

        # === HARD FILTERS: PUMP, PANIC, LOW-ADX RANGING ===
        if ctx.is_pump:
            logger.info(f"  {pair}: PUMP detected — skipping entry (exit liquidity risk)")
            return None
        if ctx.fear_greed > 0 and ctx.fear_greed < self.config.get("fear_greed_no_trade_zone", 10):
            logger.info(f"  {pair}: Fear & Greed={ctx.fear_greed} PANIC — no entries")
            return None
        # Backtest finding: ranging regime has ~11% WR — require higher confluence
        if ctx.regime == "ranging" and best_score < self.config.get("min_ranging_score", 5):
            logger.debug(f"  {pair}: ranging regime, score {best_score} < {self.config.get('min_ranging_score', 5)} — skipping")
            return None

        # === DYNAMIC ATR-BASED EXITS ===
        # Use 1H ATR for exit levels (5m ATR is too small to overcome fees)
        # Fall back to 5m ATR * 12 if 1H not available (approximate)
        exit_atr = ctx.atr_1h if ctx.atr_1h > 0 else atr * 12
        # Widen stops when BTC vol is elevated (research: vol spillover)
        vol_adj = min(1.5, max(1.0, ctx.btc_vol_ratio)) if ctx.btc_vol_ratio > 1.5 else 1.0

        if best_side == "buy":
            tp_price = price + exit_atr * self.config["tp_atr_mult"] * vol_adj
            sl_price = price - exit_atr * self.config["sl_atr_mult"] * vol_adj
        else:  # sell
            tp_price = price - exit_atr * self.config["tp_atr_mult"] * vol_adj
            sl_price = price + exit_atr * self.config["sl_atr_mult"] * vol_adj

        now = datetime.now(timezone.utc).isoformat()

        return Signal(
            pair=pair,
            side=best_side,
            signal_type=best_components[0] if best_components else "confluence",
            confluence_score=best_score,
            quality_grade=grade,
            price=price,
            rsi=rsi,
            atr=exit_atr,  # Store exit ATR (1H) for position management
            take_profit=round(tp_price, 6),
            stop_loss=round(sl_price, 6),
            regime=ctx.regime,
            reasoning=" | ".join(best_reasons),
            components=best_components,
            timestamp=now,
        )


# ================================================================
# PAPER TRADING ENGINE
# ================================================================

@dataclass
class ScalperState:
    """Paper trading state."""
    bankroll: float = 50.00
    starting_bankroll: float = 50.00
    peak_bankroll: float = 50.00
    positions: list = field(default_factory=list)
    closed_trades: list = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    daily_trade_count: int = 0
    last_loss_time: float = 0.0
    consecutive_losses: int = 0
    last_reset_date: str = ""       # YYYY-MM-DD, for daily counter reset
    last_scan: str = ""
    removed_pairs: list = field(default_factory=list)  # Pairs removed by auto-tune
    tuned_overrides: dict = field(default_factory=dict)  # Persisted auto-tune config overrides

    @property
    def open_exposure(self) -> float:
        return sum(p.get("cost_usd", 0) for p in self.positions)

    @property
    def win_rate(self) -> float:
        closed = len(self.closed_trades)
        return self.winning_trades / closed if closed > 0 else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.total_trades if self.total_trades > 0 else 0.0


class CryptoScalper:
    """Pro-grade paper trading crypto scalper."""

    def __init__(self, config: dict = None):
        self.config = config or SCALPER_CONFIG
        os.makedirs(DATA_DIR, exist_ok=True)
        self.state = self._load_state()
        # Re-apply auto-tune pair removals from previous runs
        if self.state.removed_pairs:
            for pair in self.state.removed_pairs:
                if pair in self.config["pairs"]:
                    self.config["pairs"].remove(pair)
                    logger.info(f"Re-applying auto-tune: {pair} excluded")
        # Re-apply tuned thresholds from previous sessions
        if self.state.tuned_overrides:
            for key, val in self.state.tuned_overrides.items():
                if key in self.config:
                    logger.info(f"Re-applying tuned: {key}={val}")
                    self.config[key] = val
        self.coinbase = CoinbaseDataClient()
        self.binance = BinanceDataClient()
        # v3.1: Dynamic pair discovery
        self.pair_scanner = PairScanner(self.config) if self.config.get("dynamic_pairs") else None
        # v3.0: Profit-maximizing filters
        self.fear_greed = FearGreedIndex()
        self.cusum = CUSUMFilter(threshold=self.config.get("cusum_threshold", 0.003))
        self._last_cusum_ts = {}  # pair -> last candle timestamp fed to CUSUM
        self.vol_tracker = CrossAssetVolTracker()
        self.bandit = SignalBandit(decay=self.config.get("bandit_decay", 0.95))
        self.detector = SignalDetector(self.config, bandit=self.bandit)
        # v3.5: Adaptive macro regime system
        self._base_config = dict(self.config)  # Snapshot before regime overrides
        self.regime_detector = RegimeDetector()
        self._regime_profiles = copy.deepcopy(REGIME_PROFILES)
        self.regime_adapter = RegimeAdapter(self._regime_profiles, self._base_config)
        self._last_avg_adx = 20.0      # Warm-up default (transitional)
        self._last_market_breadth = 0.5  # Warm-up default (neutral)

    def _load_state(self) -> ScalperState:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    content = f.read().strip()
                if not content:
                    logger.warning("State file empty, starting fresh")
                    return ScalperState(
                        bankroll=self.config["bankroll"],
                        starting_bankroll=self.config["bankroll"],
                        peak_bankroll=self.config["bankroll"],
                    )
                data = json.loads(content)
                known = {f.name for f in ScalperState.__dataclass_fields__.values()}
                return ScalperState(**{k: v for k, v in data.items() if k in known})
            except json.JSONDecodeError as e:
                logger.error(f"State file corrupt: {e}. Remove {STATE_FILE} to reset.")
                raise SystemExit(f"FATAL: corrupt state file: {STATE_FILE}")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
        return ScalperState(
            bankroll=self.config["bankroll"],
            starting_bankroll=self.config["bankroll"],
            peak_bankroll=self.config["bankroll"],
        )

    def _save_state(self):
        # Atomic write: temp file + rename (prevents corruption on crash/kill)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(asdict(self.state), f, indent=2)
        os.replace(tmp, STATE_FILE)

    def _log_trade(self, action: str, data: dict):
        write_header = not os.path.exists(TRADE_LOG) or os.path.getsize(TRADE_LOG) == 0
        with open(TRADE_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "timestamp", "action", "pair", "side", "price",
                    "shares", "cost_usd", "pnl", "signal_type",
                    "confluence_score", "quality_grade", "regime",
                    "rsi", "reasoning",
                ])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                action, data.get("pair", ""), data.get("side", ""),
                data.get("price", 0), data.get("shares", 0),
                data.get("cost_usd", 0), data.get("pnl", 0),
                data.get("signal_type", ""),
                data.get("confluence_score", 0),
                data.get("quality_grade", ""),
                data.get("regime", ""),
                data.get("rsi", 0),
                data.get("reasoning", "")[:120],
            ])

    def _build_market_context(self, pair: str) -> Optional[MarketContext]:
        """Build full market context for a pair: regime, trend, book, funding.

        Returns None if critical data (1H candles) is unavailable — skips pair.
        """
        ctx = MarketContext(pair=pair)

        # 1. Fetch 1H candles for trend/regime (REQUIRED — skip pair if missing)
        candles_1h = self.coinbase.get_candles(
            pair, granularity="ONE_HOUR", num_candles=50
        )
        if not candles_1h or len(candles_1h) < 30:
            logger.debug(f"  {pair}: skipping — insufficient 1H data")
            return None

        if candles_1h and len(candles_1h) >= 30:
            closes_1h = [c["close"] for c in candles_1h]
            highs_1h = [c["high"] for c in candles_1h]
            lows_1h = [c["low"] for c in candles_1h]

            # Hourly EMAs
            ema9 = calc_ema(closes_1h, 9)
            ema20 = calc_ema(closes_1h, 20)
            rsi_1h = calc_rsi(closes_1h)

            if ema9 and ema20:
                ctx.hourly_ema_fast = ema9
                ctx.hourly_ema_slow = ema20
                if ema9 > ema20 and closes_1h[-1] > ema9:
                    ctx.hourly_trend = "bullish"
                elif ema9 < ema20 and closes_1h[-1] < ema9:
                    ctx.hourly_trend = "bearish"
                else:
                    ctx.hourly_trend = "neutral"

            if rsi_1h:
                ctx.hourly_rsi = rsi_1h

            # ADX for regime
            adx = calc_adx(highs_1h, lows_1h, closes_1h)
            if adx is not None:
                ctx.adx = adx
                if adx > self.config["adx_trending"]:
                    if ctx.hourly_trend == "bullish":
                        ctx.regime = "trending_up"
                    elif ctx.hourly_trend == "bearish":
                        ctx.regime = "trending_down"
                    else:
                        ctx.regime = "trending"
                    ctx.trend_direction = "up" if ctx.hourly_trend == "bullish" else "down"
                elif adx < self.config["adx_ranging"]:
                    ctx.regime = "ranging"
                else:
                    ctx.regime = "transitional"

            # 1H ATR for exit calculations (overcomes fee structure)
            atr_1h = calc_atr(highs_1h, lows_1h, closes_1h)
            if atr_1h:
                ctx.atr_1h = atr_1h

            # Volume profile from 1H candles
            vp = calc_volume_profile(candles_1h)
            if vp:
                ctx.poc = vp["poc"]
                ctx.vah = vp["vah"]
                ctx.val = vp["val"]

        time.sleep(0.1)

        # 2. Order book
        book = self.coinbase.get_order_book(pair)
        if book:
            ctx.ob_imbalance = book["imbalance"]
            ctx.spread_pct = book["spread_pct"]

        time.sleep(0.05)

        # 3. Funding rate from Binance
        funding = self.binance.get_funding_rate(pair)
        if funding:
            ctx.funding_rate = funding["rate"]
            if funding["rate"] > self.config["funding_extreme_high"]:
                ctx.funding_signal = "overbought"
            elif funding["rate"] < self.config["funding_extreme_low"]:
                ctx.funding_signal = "oversold"

        # 4. v3.0: Fear & Greed Index (global, not per-pair)
        fg = self.fear_greed.get()
        if fg:
            ctx.fear_greed = fg["value"]

        # 5. v3.0: CUSUM filter — feed NEW 5M candle returns only (AFML: use HF returns)
        candles_5m_brief = self.coinbase.get_candles(pair, "FIVE_MINUTE", num_candles=10)
        if candles_5m_brief and len(candles_5m_brief) >= 2:
            last_ts = self._last_cusum_ts.get(pair, 0)
            for i in range(1, len(candles_5m_brief)):
                candle = candles_5m_brief[i]
                ts = candle.get("start", candle.get("time", 0))
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                    except (ValueError, TypeError):
                        ts = 0
                if ts > last_ts and candles_5m_brief[i-1]["close"] > 0:
                    ret = (candle["close"] - candles_5m_brief[i-1]["close"]) / candles_5m_brief[i-1]["close"]
                    cusum_sig = self.cusum.update(pair, ret)
                    if cusum_sig:
                        ctx.cusum_signal = cusum_sig
                    self._last_cusum_ts[pair] = ts

        # 6. v3.0: BTC vol tracker (update BTC price, compute vol ratio)
        if pair == "BTC-USD" and candles_1h:
            self.vol_tracker.update_btc_price(candles_1h[-1]["close"])
        ctx.btc_vol_ratio = self.vol_tracker.btc_vol_ratio

        # 7. v3.0: Pump detection — volume spike + sharp price move in 1H
        if candles_1h and len(candles_1h) >= 5:
            avg_vol_1h = sum(c["volume"] for c in candles_1h[-5:-1]) / 4
            curr_vol_1h = candles_1h[-1]["volume"]
            price_chg_1h = (candles_1h[-1]["close"] - candles_1h[-2]["close"]) / candles_1h[-2]["close"]
            if (avg_vol_1h > 0
                and curr_vol_1h > avg_vol_1h * self.config.get("pump_volume_mult", 5.0)
                and abs(price_chg_1h) > self.config.get("pump_price_change_pct", 0.03)):
                ctx.is_pump = True
                logger.info(f"  {pair}: PUMP DETECTED — vol {curr_vol_1h/avg_vol_1h:.1f}x, "
                           f"price {price_chg_1h:+.2%}")

        # 8. v3.2: RSI divergence from 5m candles
        candles_5m = self.coinbase.get_candles(pair, "FIVE_MINUTE", num_candles=50)
        if candles_5m and len(candles_5m) >= 35:
            closes_5m = [c["close"] for c in candles_5m]
            div = detect_rsi_divergence(closes_5m, period=14, lookback=15)
            if div:
                ctx.rsi_divergence = div

        # 9. v3.2: Engulfing candle pattern
        if candles_5m and len(candles_5m) >= 2:
            eng = detect_engulfing(candles_5m)
            if eng:
                ctx.engulfing = eng

        # 10. v3.2: Swing levels from 1H candles
        if candles_1h and len(candles_1h) >= 10:
            levels = find_swing_levels(candles_1h)
            ctx.swing_supports = levels["support"]
            ctx.swing_resistances = levels["resistance"]
            price = candles_1h[-1]["close"]
            if price > 0:
                for s in ctx.swing_supports:
                    if abs(price - s) / price < 0.005:
                        ctx.near_support = True
                        break
                for r in ctx.swing_resistances:
                    if abs(price - r) / price < 0.005:
                        ctx.near_resistance = True
                        break

        # 11. v3.2: ATR expansion ratio
        if candles_5m and len(candles_5m) >= 25:
            highs_5m = [c["high"] for c in candles_5m]
            lows_5m = [c["low"] for c in candles_5m]
            closes_5m = [c["close"] for c in candles_5m]
            atr_r = calc_atr_ratio(highs_5m, lows_5m, closes_5m, fast_period=5, slow_period=20)
            if atr_r is not None:
                ctx.atr_ratio = atr_r

        return ctx

    def open_position(self, signal: Signal, ctx: MarketContext = None) -> bool:
        """Open a paper position from a signal."""
        # Risk checks
        if len(self.state.positions) >= self.config["max_open_positions"]:
            return False
        if self.state.open_exposure >= self.state.bankroll * self.config["max_exposure_pct"]:
            return False
        if self.state.daily_pnl <= self.config["max_daily_loss"]:
            logger.info("Daily loss limit hit")
            return False
        if self.state.daily_trade_count >= self.config["max_daily_trades"]:
            return False
        if time.time() - self.state.last_loss_time < self.config["cooldown_after_loss_sec"]:
            return False
        if self.state.consecutive_losses >= self.config["max_consecutive_losses"]:
            # Allow recovery after 4 hours (not permanent halt)
            hours_since_loss = (time.time() - self.state.last_loss_time) / 3600
            if hours_since_loss < 4:
                logger.info(f"Paused: {self.state.consecutive_losses} consecutive losses "
                           f"({4 - hours_since_loss:.1f}h until resume)")
                return False
            else:
                logger.info(f"Resuming after {hours_since_loss:.0f}h cooldown from {self.state.consecutive_losses} losses")
                self.state.consecutive_losses = 0
                self._save_state()

        # No duplicate pairs
        for pos in self.state.positions:
            if pos["pair"] == signal.pair:
                return False

        # Position sizing: volatility-adjusted Kelly
        atr_pct = signal.atr / signal.price if signal.price > 0 else 0.03
        risk_per_trade = atr_pct * self.config["sl_atr_mult"]

        # Kelly: f* = (p*b - q) / b where b = reward/risk, p = win probability
        # Conservative: use fixed kelly fraction scaled by ATR
        kelly_size = self.state.bankroll * self.config["kelly_fraction"]

        # Scale down by volatility (riskier = smaller position)
        vol_adj = min(1.0, 0.02 / max(atr_pct, 0.005))  # Normalize to 2% ATR

        # Scale up by signal quality
        quality_mult = {"A": 1.3, "B": 1.0, "C": 0.7, "D": 0.5}
        q_mult = quality_mult.get(signal.quality_grade, 0.7)

        position_size = min(
            self.config["max_position_usd"],
            kelly_size * vol_adj * q_mult,
        )
        position_size = max(1.0, round(position_size, 2))

        shares = position_size / signal.price
        pos_id = f"S{self.state.total_trades + 1:04d}"

        position = {
            "id": pos_id,
            "pair": signal.pair,
            "side": signal.side,
            "entry_price": signal.price,
            "shares": round(shares, 8),
            "cost_usd": position_size,
            "signal_type": signal.signal_type,
            "confluence_score": signal.confluence_score,
            "quality_grade": signal.quality_grade,
            "regime": signal.regime,
            "macro_regime": self.regime_detector.current_regime.value,
            "rsi_at_entry": signal.rsi,
            "reasoning": signal.reasoning,
            "take_profit": signal.take_profit,
            "stop_loss": signal.stop_loss,
            "atr_at_entry": signal.atr,
            "peak_price": signal.price,
            "trough_price": signal.price,
            "current_price": signal.price,
            "unrealized_pnl": 0.0,
            "breakeven_moved": False,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "components": signal.components,
            # Rich context snapshot for analytics & auto-tuning
            "entry_ob_imbalance": ctx.ob_imbalance if ctx else 0,
            "entry_funding_rate": ctx.funding_rate if ctx else 0,
            "entry_adx": ctx.adx if ctx else 0,
            "entry_hourly_trend": ctx.hourly_trend if ctx else "unknown",
            "entry_hourly_rsi": ctx.hourly_rsi if ctx else 50,
            "entry_poc": ctx.poc if ctx else 0,
            "entry_spread_pct": ctx.spread_pct if ctx else 0,
            # v3.0 context
            "entry_fear_greed": ctx.fear_greed if ctx else 50,
            "entry_cusum_signal": ctx.cusum_signal if ctx else "",
            "entry_btc_vol_ratio": ctx.btc_vol_ratio if ctx else 1.0,
            "bandit_weights": self.bandit.get_all_weights(),
        }

        self.state.positions.append(position)
        self.state.bankroll -= position_size
        self.state.total_trades += 1
        self.state.daily_trade_count += 1
        self._save_state()
        self._log_trade("OPEN", {**position, "price": signal.price, "pnl": 0})

        def _fmt(p):
            if p >= 1: return f"${p:,.2f}"
            elif p >= 0.01: return f"${p:.4f}"
            else: return f"${p:.8f}"

        logger.info(
            f"OPEN [{pos_id}] {signal.side.upper()} {signal.pair} @ {_fmt(signal.price)} | "
            f"${position_size:.2f} | Grade:{signal.quality_grade} Score:{signal.confluence_score} | "
            f"TP={_fmt(signal.take_profit)} SL={_fmt(signal.stop_loss)} | "
            f"Regime:{signal.regime} | {signal.signal_type}"
        )
        return True

    def update_prices(self):
        """Update current prices for all open positions."""
        for pos in self.state.positions:
            ticker = self.coinbase.get_ticker(pos["pair"])
            if ticker and ticker.get("price", 0) > 0:
                new_price = ticker["price"]
                pos["current_price"] = new_price
                side = pos.get("side", "buy")
                if side == "buy":
                    pos["unrealized_pnl"] = round(
                        (new_price - pos["entry_price"]) * pos["shares"], 4
                    )
                else:  # sell
                    pos["unrealized_pnl"] = round(
                        (pos["entry_price"] - new_price) * pos["shares"], 4
                    )
                if new_price > pos.get("peak_price", pos["entry_price"]):
                    pos["peak_price"] = new_price
                if new_price < pos.get("trough_price", pos["entry_price"]):
                    pos["trough_price"] = new_price
            time.sleep(0.1)
        self._save_state()

    def check_exits(self):
        """Check exit conditions with pro-grade logic."""
        to_close = []

        for pos in self.state.positions:
            current = pos.get("current_price", pos["entry_price"])
            entry = pos["entry_price"]
            atr = pos.get("atr_at_entry", entry * 0.02)
            side = pos.get("side", "buy")
            is_long = (side == "buy")

            # 1. TAKE PROFIT (ATR-based, direction-aware)
            if is_long and current >= pos["take_profit"]:
                to_close.append((pos, "take_profit", current))
                continue
            elif not is_long and current <= pos["take_profit"]:
                to_close.append((pos, "take_profit", current))
                continue

            # 2. STOP LOSS (ATR-based, direction-aware)
            if is_long and current <= pos["stop_loss"]:
                to_close.append((pos, "stop_loss", current))
                continue
            elif not is_long and current >= pos["stop_loss"]:
                to_close.append((pos, "stop_loss", current))
                continue

            # 3. BREAKEVEN STOP: move stop to entry after 1R profit
            if self.config["breakeven_at_1r"] and not pos.get("breakeven_moved"):
                one_r = atr * self.config["sl_atr_mult"]
                if is_long and current >= entry + one_r:
                    pos["stop_loss"] = round(entry + atr * 0.2, 6)  # Slightly above entry
                    pos["breakeven_moved"] = True
                    logger.info(f"[{pos['id']}] Moved stop to breakeven+")
                elif not is_long and current <= entry - one_r:
                    pos["stop_loss"] = round(entry - atr * 0.2, 6)  # Slightly below entry
                    pos["breakeven_moved"] = True
                    logger.info(f"[{pos['id']}] Moved stop to breakeven+")

            # 4. TRAILING STOP: direction-aware ratcheting
            trail_distance = atr * self.config["trailing_atr_mult"]
            if is_long:
                peak = pos.get("peak_price", entry)
                trailing_stop = peak - trail_distance
                if trailing_stop > pos["stop_loss"]:
                    pos["stop_loss"] = round(trailing_stop, 6)
                    logger.debug(f"[{pos['id']}] Trail ratcheted stop to {trailing_stop:.6f}")
                if current < pos["stop_loss"] and pos["stop_loss"] > entry:
                    to_close.append((pos, "trailing_stop", current))
                    continue
            else:
                trough = pos.get("trough_price", entry)
                trailing_stop = trough + trail_distance
                if trailing_stop < pos["stop_loss"]:
                    pos["stop_loss"] = round(trailing_stop, 6)
                    logger.debug(f"[{pos['id']}] Trail ratcheted stop to {trailing_stop:.6f}")
                if current > pos["stop_loss"] and pos["stop_loss"] < entry:
                    to_close.append((pos, "trailing_stop", current))
                    continue

            # 5. PROGRESSIVE STOP TIGHTENING
            # After X% of max_hold, gradually tighten SL toward entry
            # Backtest-validated: converts losing time_exits into earlier SL exits
            try:
                opened = datetime.fromisoformat(pos["opened_at"])
                elapsed_h = (datetime.now(timezone.utc) - opened).total_seconds() / 3600
            except (ValueError, TypeError):
                elapsed_h = 0

            if self.config.get("progressive_stop", False) and elapsed_h > 0:
                max_hold = self.config["max_hold_hours"]
                start_pct = self.config.get("progressive_stop_start_pct", 0.5)
                end_mult = self.config.get("progressive_stop_end_mult", 0.3)
                hold_pct = elapsed_h / max_hold if max_hold > 0 else 0

                if hold_pct >= start_pct:
                    progress = min(1.0, (hold_pct - start_pct) / (1.0 - start_pct))
                    original_sl_dist = atr * self.config["sl_atr_mult"]
                    tight_sl_dist = original_sl_dist * end_mult
                    current_sl_dist = original_sl_dist - (original_sl_dist - tight_sl_dist) * progress

                    if is_long:
                        new_sl = round(entry - current_sl_dist, 6)
                        if new_sl > pos["stop_loss"]:
                            pos["stop_loss"] = new_sl
                            logger.debug(f"[{pos['id']}] Progressive stop tightened to {new_sl:.6f} "
                                       f"({hold_pct:.0%} of max hold)")
                    else:
                        new_sl = round(entry + current_sl_dist, 6)
                        if new_sl < pos["stop_loss"]:
                            pos["stop_loss"] = new_sl
                            logger.debug(f"[{pos['id']}] Progressive stop tightened to {new_sl:.6f} "
                                       f"({hold_pct:.0%} of max hold)")

            # 6. TIME EXIT
            if elapsed_h > self.config["max_hold_hours"]:
                to_close.append((pos, "time_exit", current))
                continue

        for pos, reason, price in to_close:
            self.close_position(pos, price, reason)

    def close_position(self, pos: dict, exit_price: float, reason: str):
        """Close a position and record P&L."""
        side = pos.get("side", "buy")
        if side == "buy":
            gross_pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        else:  # sell
            gross_pnl = (pos["entry_price"] - exit_price) * pos["shares"]
        fee_pct = self.config["taker_fee_pct"]
        entry_fee = pos["cost_usd"] * fee_pct
        exit_value = exit_price * pos["shares"]
        exit_fee = exit_value * fee_pct
        total_fees = entry_fee + exit_fee
        pnl = round(gross_pnl - total_fees, 4)

        self.state.bankroll += pos["cost_usd"] + pnl
        self.state.total_pnl += pnl
        self.state.daily_pnl += pnl

        if pnl > 0:
            self.state.winning_trades += 1
            self.state.consecutive_losses = 0
        else:
            self.state.last_loss_time = time.time()
            self.state.consecutive_losses += 1

        if self.state.bankroll > self.state.peak_bankroll:
            self.state.peak_bankroll = self.state.bankroll

        pnl_pct = (pnl / pos["cost_usd"]) * 100 if pos["cost_usd"] > 0 else 0
        hold_time = ""
        try:
            opened = datetime.fromisoformat(pos["opened_at"])
            mins = (datetime.now(timezone.utc) - opened).total_seconds() / 60
            if mins >= 60:
                hold_time = f" ({mins / 60:.1f}h)"
            else:
                hold_time = f" ({mins:.0f}m)"
        except (ValueError, TypeError):
            pass

        closed = {
            **pos,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": round(pnl_pct, 2),
            "fees": round(total_fees, 4),
            "reason": reason,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.state.closed_trades.append(closed)
        # Cap closed trades in memory/state to prevent unbounded growth on 24/7 server
        MAX_CLOSED_IN_STATE = 500
        if len(self.state.closed_trades) > MAX_CLOSED_IN_STATE:
            self.state.closed_trades = self.state.closed_trades[-MAX_CLOSED_IN_STATE:]
        self.state.positions = [p for p in self.state.positions if p["id"] != pos["id"]]
        self._save_state()
        self._log_trade("CLOSE", {**pos, "price": exit_price, "pnl": pnl})

        def _fmt(p):
            if p >= 1: return f"${p:,.2f}"
            elif p >= 0.01: return f"${p:.4f}"
            else: return f"${p:.8f}"

        icon = "+" if pnl >= 0 else ""
        logger.info(
            f"CLOSE [{pos['id']}] {pos['pair']} @ {_fmt(exit_price)} | "
            f"PnL: {icon}${pnl:.4f} ({icon}{pnl_pct:.1f}%) | "
            f"fees: ${total_fees:.4f} | {reason}{hold_time}"
        )

        # Record rich analytics for auto-tuning
        self._record_analytics(closed)

        # v3.0: Update signal performance bandit
        self.bandit.record(
            components=pos.get("components", []),
            won=(pnl > 0),
        )

        # Auto-tune after every 20 closed trades
        if len(self.state.closed_trades) > 0 and len(self.state.closed_trades) % 20 == 0:
            self._auto_tune()

    def _record_analytics(self, trade: dict):
        """Record detailed trade analytics for learning.

        Uses JSONL (one JSON object per line) for O(1) append instead of
        reading and rewriting the entire file on every trade close.
        """
        record = {
            "id": trade.get("id"),
            "pair": trade.get("pair"),
            "opened_at": trade.get("opened_at"),
            "closed_at": trade.get("closed_at"),
            "entry_price": trade.get("entry_price"),
            "exit_price": trade.get("exit_price"),
            "pnl": trade.get("pnl"),
            "pnl_pct": trade.get("pnl_pct"),
            "fees": trade.get("fees"),
            "reason": trade.get("reason"),
            "won": trade.get("pnl", 0) > 0,
            "signal_type": trade.get("signal_type"),
            "confluence_score": trade.get("confluence_score"),
            "quality_grade": trade.get("quality_grade"),
            "components": trade.get("components", []),
            "regime": trade.get("regime"),
            "entry_ob_imbalance": trade.get("entry_ob_imbalance", 0),
            "entry_funding_rate": trade.get("entry_funding_rate", 0),
            "entry_adx": trade.get("entry_adx", 0),
            "entry_hourly_trend": trade.get("entry_hourly_trend", ""),
            "entry_hourly_rsi": trade.get("entry_hourly_rsi", 50),
            "entry_rsi_5m": trade.get("rsi_at_entry", 0),
            "entry_atr": trade.get("atr_at_entry", 0),
            "entry_spread_pct": trade.get("entry_spread_pct", 0),
            # v3.0 context
            "entry_fear_greed": trade.get("entry_fear_greed", 50),
            "entry_cusum_signal": trade.get("entry_cusum_signal", ""),
            "entry_btc_vol_ratio": trade.get("entry_btc_vol_ratio", 1.0),
            "bandit_weights": trade.get("bandit_weights", {}),
        }
        with open(ANALYTICS_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _auto_tune(self):
        """Auto-tune parameters based on closed trade performance.

        Analyzes win rate by: signal component, regime, quality grade, pair.
        Adjusts confluence threshold and component weights based on what works.
        """
        trades = self.state.closed_trades
        if len(trades) < 20:
            return

        logger.info(f"=== AUTO-TUNE: Analyzing {len(trades)} trades ===")

        # --- Win rate by confluence component ---
        component_stats = {}
        for t in trades:
            won = t.get("pnl", 0) > 0
            for comp in t.get("components", []):
                if comp not in component_stats:
                    component_stats[comp] = {"wins": 0, "total": 0, "total_pnl": 0}
                component_stats[comp]["total"] += 1
                if won:
                    component_stats[comp]["wins"] += 1
                component_stats[comp]["total_pnl"] += t.get("pnl", 0)

        # --- Win rate by regime ---
        regime_stats = {}
        for t in trades:
            regime = t.get("regime", "unknown")
            won = t.get("pnl", 0) > 0
            if regime not in regime_stats:
                regime_stats[regime] = {"wins": 0, "total": 0, "total_pnl": 0}
            regime_stats[regime]["total"] += 1
            if won:
                regime_stats[regime]["wins"] += 1
            regime_stats[regime]["total_pnl"] += t.get("pnl", 0)

        # --- Win rate by grade ---
        grade_stats = {}
        for t in trades:
            grade = t.get("quality_grade", "?")
            won = t.get("pnl", 0) > 0
            if grade not in grade_stats:
                grade_stats[grade] = {"wins": 0, "total": 0, "total_pnl": 0}
            grade_stats[grade]["total"] += 1
            if won:
                grade_stats[grade]["wins"] += 1
            grade_stats[grade]["total_pnl"] += t.get("pnl", 0)

        # --- Win rate by pair ---
        pair_stats = {}
        for t in trades:
            pair = t.get("pair", "?")
            won = t.get("pnl", 0) > 0
            if pair not in pair_stats:
                pair_stats[pair] = {"wins": 0, "total": 0, "total_pnl": 0}
            pair_stats[pair]["total"] += 1
            if won:
                pair_stats[pair]["wins"] += 1
            pair_stats[pair]["total_pnl"] += t.get("pnl", 0)

        # --- Win rate by exit reason ---
        exit_stats = {}
        for t in trades:
            reason = t.get("reason", "?")
            won = t.get("pnl", 0) > 0
            if reason not in exit_stats:
                exit_stats[reason] = {"wins": 0, "total": 0, "total_pnl": 0}
            exit_stats[reason]["total"] += 1
            if won:
                exit_stats[reason]["wins"] += 1
            exit_stats[reason]["total_pnl"] += t.get("pnl", 0)

        # --- v3.5: Win rate by macro regime ---
        macro_regime_stats = {}
        for t in trades:
            mr = t.get("macro_regime", "unknown")
            won = t.get("pnl", 0) > 0
            if mr not in macro_regime_stats:
                macro_regime_stats[mr] = {"wins": 0, "total": 0, "total_pnl": 0}
            macro_regime_stats[mr]["total"] += 1
            if won:
                macro_regime_stats[mr]["wins"] += 1
            macro_regime_stats[mr]["total_pnl"] += t.get("pnl", 0)

        # --- Average OB imbalance on wins vs losses ---
        win_obs = [t.get("entry_ob_imbalance", 0) for t in trades if t.get("pnl", 0) > 0]
        loss_obs = [t.get("entry_ob_imbalance", 0) for t in trades if t.get("pnl", 0) <= 0]
        avg_win_ob = sum(win_obs) / len(win_obs) if win_obs else 0
        avg_loss_ob = sum(loss_obs) / len(loss_obs) if loss_obs else 0

        # --- Adaptive tuning decisions ---
        tuning = {
            "analyzed_at": datetime.now(timezone.utc).isoformat(),
            "total_trades": len(trades),
            "overall_win_rate": sum(1 for t in trades if t.get("pnl", 0) > 0) / len(trades),
            "overall_pnl": sum(t.get("pnl", 0) for t in trades),
            "component_stats": {k: {**v, "win_rate": v["wins"] / v["total"] if v["total"] > 0 else 0}
                               for k, v in component_stats.items()},
            "regime_stats": {k: {**v, "win_rate": v["wins"] / v["total"] if v["total"] > 0 else 0}
                           for k, v in regime_stats.items()},
            "grade_stats": {k: {**v, "win_rate": v["wins"] / v["total"] if v["total"] > 0 else 0}
                          for k, v in grade_stats.items()},
            "pair_stats": {k: {**v, "win_rate": v["wins"] / v["total"] if v["total"] > 0 else 0}
                         for k, v in pair_stats.items()},
            "exit_stats": {k: {**v, "win_rate": v["wins"] / v["total"] if v["total"] > 0 else 0}
                         for k, v in exit_stats.items()},
            "macro_regime_stats": {k: {**v, "win_rate": v["wins"] / v["total"] if v["total"] > 0 else 0}
                                  for k, v in macro_regime_stats.items()},
            "avg_ob_imbalance_wins": round(avg_win_ob, 4),
            "avg_ob_imbalance_losses": round(avg_loss_ob, 4),
            "adjustments": [],
        }

        # --- Apply adjustments ---
        overall_wr = tuning["overall_win_rate"]

        # 1. If win rate < 40%, raise confluence threshold (be more selective)
        if overall_wr < 0.40 and len(trades) >= 20:
            old = self.config["min_confluence_score"]
            self.config["min_confluence_score"] = min(old + 1, 7)
            self.state.tuned_overrides["min_confluence_score"] = self.config["min_confluence_score"]
            tuning["adjustments"].append(
                f"Win rate {overall_wr:.0%} < 40%: raised min_confluence {old} → {self.config['min_confluence_score']}")

        # 2. If win rate > 60%, can lower threshold to take more trades
        if overall_wr > 0.60 and len(trades) >= 30:
            old = self.config["min_confluence_score"]
            self.config["min_confluence_score"] = max(old - 1, 3)
            self.state.tuned_overrides["min_confluence_score"] = self.config["min_confluence_score"]
            tuning["adjustments"].append(
                f"Win rate {overall_wr:.0%} > 60%: lowered min_confluence {old} → {self.config['min_confluence_score']}")

        # 3. If a regime has <30% win rate with 5+ trades, avoid it
        for regime, stats in regime_stats.items():
            wr = stats["wins"] / stats["total"] if stats["total"] > 0 else 0
            if wr < 0.30 and stats["total"] >= 5:
                tuning["adjustments"].append(
                    f"Regime '{regime}' has {wr:.0%} win rate over {stats['total']} trades — POOR")

        # 4. If OB imbalance is higher on wins, raise the threshold
        if avg_win_ob > avg_loss_ob + 0.1 and len(win_obs) >= 5:
            new_thresh = round((avg_win_ob + avg_loss_ob) / 2, 2)
            if new_thresh > self.config["ob_strong_buy"]:
                old = self.config["ob_strong_buy"]
                self.config["ob_strong_buy"] = new_thresh
                self.config["ob_strong_sell"] = -new_thresh
                self.state.tuned_overrides["ob_strong_buy"] = new_thresh
                self.state.tuned_overrides["ob_strong_sell"] = -new_thresh
                tuning["adjustments"].append(
                    f"OB imbalance: wins avg {avg_win_ob:.2f} vs losses {avg_loss_ob:.2f}. "
                    f"Raised threshold {old} → {new_thresh}")

        # 5. If a pair is consistently losing (>5 trades, <25% WR), remove it
        for pair, stats in pair_stats.items():
            wr = stats["wins"] / stats["total"] if stats["total"] > 0 else 0
            if wr < 0.25 and stats["total"] >= 5:
                if pair in self.config["pairs"]:
                    self.config["pairs"].remove(pair)
                    if pair not in self.state.removed_pairs:
                        self.state.removed_pairs.append(pair)
                    tuning["adjustments"].append(
                        f"Removed {pair}: {wr:.0%} win rate over {stats['total']} trades")

        # 6. v3.5: If a macro regime has poor WR over 10+ trades, tighten its profile
        for mr_name, stats in macro_regime_stats.items():
            if mr_name == "unknown" or stats["total"] < 10:
                continue
            mr_wr = stats["wins"] / stats["total"]
            # Find the matching regime enum
            mr_enum = None
            for r in MacroRegime:
                if r.value == mr_name:
                    mr_enum = r
                    break
            if mr_enum and mr_enum in self._regime_profiles and mr_wr < 0.30:
                profile = self._regime_profiles[mr_enum]
                old_conf = profile.get("min_confluence_score", 5)
                new_conf = min(old_conf + 1, 9)
                if new_conf != old_conf:
                    profile["min_confluence_score"] = new_conf
                    tuning["adjustments"].append(
                        f"Macro regime '{mr_name}' WR={mr_wr:.0%} over {stats['total']}t: "
                        f"raised confluence {old_conf} -> {new_conf}")
                old_kelly = profile.get("kelly_fraction", 0.10)
                new_kelly = max(0.03, round(old_kelly - 0.01, 2))
                if new_kelly != old_kelly:
                    profile["kelly_fraction"] = new_kelly
                    tuning["adjustments"].append(
                        f"Macro regime '{mr_name}': reduced kelly {old_kelly} -> {new_kelly}")

        # Log results
        for adj in tuning["adjustments"]:
            logger.info(f"  AUTO-TUNE: {adj}")
        if not tuning["adjustments"]:
            logger.info(f"  AUTO-TUNE: No adjustments needed (WR={overall_wr:.0%})")

        # Log macro regime performance
        for mr, stats in macro_regime_stats.items():
            wr = stats["wins"] / stats["total"] if stats["total"] > 0 else 0
            logger.info(f"  Macro regime {mr}: {wr:.0%} WR ({stats['total']} trades) "
                        f"PnL: ${stats['total_pnl']:+.4f}")

        # Log component performance
        for comp, stats in sorted(component_stats.items(),
                                   key=lambda x: x[1].get("total_pnl", 0), reverse=True):
            wr = stats["wins"] / stats["total"] if stats["total"] > 0 else 0
            logger.info(f"  Component {comp}: {wr:.0%} WR ({stats['total']} trades) "
                        f"PnL: ${stats['total_pnl']:+.4f}")

        # Save tuning report
        tmp = TUNING_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(tuning, f, indent=2)
        os.replace(tmp, TUNING_FILE)

    def _check_daily_reset(self):
        """Reset daily counters at midnight UTC."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.last_reset_date != today:
            if self.state.last_reset_date:  # Not first run
                logger.info(f"New day ({today}): resetting daily P&L and trade count")
            self.state.daily_pnl = 0.0
            self.state.daily_trade_count = 0
            self.state.consecutive_losses = 0  # Fresh start each day
            self.state.last_reset_date = today
            self._save_state()

    def scan_and_trade(self):
        """Full scan cycle: context → exits → signals → trade."""
        self._check_daily_reset()

        # v3.0: Fetch global context once per scan
        fg = self.fear_greed.get()
        fg_val = fg["value"] if fg else "?"
        vol_r = self.vol_tracker.btc_vol_ratio

        # v3.5: Adaptive macro regime detection
        fg_int = fg_val if isinstance(fg_val, int) else 50
        recent_trades = self.state.closed_trades[-20:]
        rolling_wr = sum(1 for t in recent_trades if t.get("pnl", 0) > 0) / max(len(recent_trades), 1)
        macro_regime = self.regime_detector.detect(
            fear_greed=fg_int,
            btc_vol_ratio=vol_r,
            avg_adx=self._last_avg_adx,
            rolling_wr=rolling_wr,
            market_breadth=self._last_market_breadth,
        )
        self.regime_adapter.apply(macro_regime, self.config)

        logger.info(
            f"--- SCAN {datetime.now(timezone.utc).strftime('%H:%M:%S')} | "
            f"${self.state.bankroll:.2f} | "
            f"{len(self.state.positions)} open | "
            f"PnL: ${self.state.total_pnl:+.2f} | "
            f"W/L: {self.state.winning_trades}/{self.state.total_trades - self.state.winning_trades} | "
            f"Pairs:{len(self.config['pairs'])} | "
            f"F&G:{fg_val} | BTC-vol:{vol_r:.1f}x | "
            f"Regime:{macro_regime.value} ---"
        )

        # Update existing positions first
        if self.state.positions:
            self.update_prices()
            self.check_exits()

        # v3.1: Refresh pair universe dynamically
        if self.pair_scanner:
            scanned = self.pair_scanner.scan()
            if scanned:
                # Filter out auto-tuned removed pairs
                active_pairs = [p for p in scanned if p not in self.state.removed_pairs]
                self.config["pairs"] = active_pairs

        btc_ticker = self.coinbase.get_ticker("BTC-USD")
        if btc_ticker and btc_ticker.get("price", 0) > 0:
            self.vol_tracker.update_btc_price(btc_ticker["price"])

        # Scan for new signals with full context
        all_signals = []
        scan_contexts = []  # v3.5: collect for regime breadth/ADX caching
        for pair in self.config["pairs"]:
            try:
                # Build full market context (1H trend, order book, funding)
                ctx = self._build_market_context(pair)
                if ctx is None:
                    continue  # Skip pair if critical data missing
                ctx.macro_regime = macro_regime.value
                scan_contexts.append(ctx)

                # Get 5M candles for entry timing
                candles_5m = self.coinbase.get_candles(
                    pair,
                    granularity=self.config["candle_granularity_entry"],
                    num_candles=self.config["candle_lookback"],
                )

                if candles_5m:
                    signal = self.detector.analyze(pair, candles_5m, ctx)
                    if signal:
                        all_signals.append((signal, ctx))
                        logger.info(
                            f"  SIGNAL: {pair} | Grade:{signal.quality_grade} "
                            f"Score:{signal.confluence_score} | "
                            f"Regime:{ctx.regime} | 1H:{ctx.hourly_trend} | "
                            f"OB:{ctx.ob_imbalance:+.2f} | "
                            f"Fund:{ctx.funding_rate:.4%} | "
                            f"{signal.reasoning[:80]}"
                        )
                    else:
                        logger.debug(
                            f"  {pair}: regime={ctx.regime} 1H={ctx.hourly_trend} "
                            f"OB={ctx.ob_imbalance:+.2f} fund={ctx.funding_rate:.4%} — no signal"
                        )
            except Exception as e:
                logger.debug(f"  Error scanning {pair}: {e}")

            time.sleep(0.08)  # HFT: faster inter-pair delay (was 0.15)

        # v3.5: Cache market breadth and avg ADX for next cycle's regime detection
        if scan_contexts:
            adx_vals = [c.adx for c in scan_contexts if c.adx > 0]
            self._last_avg_adx = sum(adx_vals) / len(adx_vals) if adx_vals else 20.0
            bullish_count = sum(1 for c in scan_contexts if c.hourly_trend == "bullish")
            self._last_market_breadth = bullish_count / len(scan_contexts)

        # Sort by confluence score (highest first), then by grade
        all_signals.sort(
            key=lambda s: (s[0].confluence_score, GRADE_MAP.get(s[0].quality_grade, 0)),
            reverse=True
        )

        opened = 0
        for signal, ctx in all_signals:
            if self.open_position(signal, ctx):
                opened += 1
            if opened >= 2:  # HFT: up to 2 new positions per scan (was 1)
                break

        if all_signals:
            logger.info(f"Signals: {len(all_signals)} qualified | Opened: {opened}")

        self.state.last_scan = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def get_summary(self) -> str:
        """Human-readable summary."""
        s = self.state
        roi = (s.total_pnl / s.starting_bankroll * 100) if s.starting_bankroll > 0 else 0
        drawdown = ((s.peak_bankroll - s.bankroll) / s.peak_bankroll * 100) if s.peak_bankroll > 0 else 0

        pair_mode = "dynamic" if self.pair_scanner else "static"
        regime = self.regime_detector.current_regime.value
        lines = [
            f"{'=' * 65}",
            f"  CRYPTO SCALPER v3.5 ({pair_mode}: {len(self.config['pairs'])} pairs)",
            f"  Macro Regime: {regime} | Confluence>={self.config['min_confluence_score']} "
            f"TP={self.config['tp_atr_mult']}x SL={self.config['sl_atr_mult']}x "
            f"Bias={self.config.get('side_bias', 'none')}",
            f"{'=' * 65}",
            f"  Bankroll:   ${s.bankroll:.2f} (started ${s.starting_bankroll:.2f})",
            f"  Total PnL:  ${s.total_pnl:+.2f} ({roi:+.1f}% ROI)",
            f"  Daily PnL:  ${s.daily_pnl:+.2f}",
            f"  Drawdown:   {drawdown:.1f}% from peak ${s.peak_bankroll:.2f}",
            f"  Trades:     {s.total_trades} ({s.win_rate:.0%} win rate)",
            f"  Avg PnL:    ${s.avg_pnl:+.4f}/trade",
            f"  Open:       {len(s.positions)} (${s.open_exposure:.2f} exposed)",
            f"  Streak:     {s.consecutive_losses} consecutive losses",
            f"{'=' * 65}",
        ]

        if s.positions:
            lines.append("  OPEN POSITIONS:")
            for pos in s.positions:
                upnl = pos.get("unrealized_pnl", 0)
                pct = ((pos.get("current_price", 0) - pos["entry_price"]) / pos["entry_price"] * 100) if pos["entry_price"] > 0 else 0
                lines.append(
                    f"  [{pos['id']}] {pos['pair']} @ ${pos['entry_price']:,.4f} → "
                    f"${pos.get('current_price', 0):,.4f} ({pct:+.1f}%) | "
                    f"${upnl:+.4f} | {pos.get('quality_grade', '?')}{pos.get('confluence_score', 0)} "
                    f"| {pos.get('regime', '?')}"
                )

        if s.closed_trades:
            recent = s.closed_trades[-5:]
            lines.append(f"\n  RECENT CLOSED ({len(s.closed_trades)} total):")
            for t in reversed(recent):
                lines.append(
                    f"  [{t['id']}] {t['pair']} ${t['entry_price']:,.4f}→${t['exit_price']:,.4f} "
                    f"${t['pnl']:+.4f} ({t.get('pnl_pct', 0):+.1f}%) {t['reason']} "
                    f"| {t.get('quality_grade', '?')}{t.get('confluence_score', 0)}"
                )

        lines.append(f"{'=' * 65}")
        return "\n".join(lines)

    def run_loop(self):
        """Main trading loop."""
        interval = self.config["scan_interval_sec"]
        logger.info(f"Starting crypto scalper v3.1 (scan every {interval}s)")
        if self.pair_scanner:
            # Do initial pair discovery before first scan
            scanned = self.pair_scanner.scan()
            if scanned:
                active_pairs = [p for p in scanned if p not in self.state.removed_pairs]
                self.config["pairs"] = active_pairs
            logger.info(f"Dynamic pairs: {len(self.config['pairs'])} discovered")
        else:
            logger.info(f"Static pairs: {', '.join(self.config['pairs'])}")
        logger.info(f"Min confluence: {self.config['min_confluence_score']}, "
                     f"Min grade: {self.config['min_signal_quality']}")
        logger.info(self.get_summary())

        self.scan_and_trade()

        try:
            while True:
                time.sleep(interval)
                self.scan_and_trade()
        except KeyboardInterrupt:
            logger.info("Scalper stopped")
            print(self.get_summary())


def setup_logging(verbose: bool = False):
    os.makedirs(DATA_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(DATA_DIR, "scalper.log"), mode="a",
            ),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Crypto Scalping Bot v3.5")
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--scan-once", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Reset state (fresh start)")
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--pairs", type=str, default=None)
    parser.add_argument("--min-score", type=int, default=None,
                       help="Min confluence score (default: 4)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--regime", type=str, default=None,
                       choices=[r.value for r in MacroRegime],
                       help="Force a specific macro regime (for testing)")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    setup_logging(verbose=args.verbose)

    if args.bankroll:
        SCALPER_CONFIG["bankroll"] = args.bankroll
    if args.interval:
        SCALPER_CONFIG["scan_interval_sec"] = args.interval
    if args.pairs:
        pairs = []
        for p in args.pairs.split(","):
            p = p.strip().upper()
            if "-" not in p:
                p = f"{p}-USD"
            pairs.append(p)
        SCALPER_CONFIG["pairs"] = pairs
        SCALPER_CONFIG["dynamic_pairs"] = False  # Explicit pairs override dynamic discovery
    if args.min_score:
        SCALPER_CONFIG["min_confluence_score"] = args.min_score

    if args.reset:
        for f in [STATE_FILE, TRADE_LOG]:
            if os.path.exists(f):
                os.remove(f)
        print("Scalper v3.5 reset.")
        return

    scalper = CryptoScalper(SCALPER_CONFIG)

    # v3.5: Force macro regime override for testing
    if args.regime:
        forced = MacroRegime(args.regime)
        scalper.regime_detector._current = forced
        scalper.regime_detector._locked_until = 999999  # Lock permanently
        scalper.regime_adapter.apply(forced, scalper.config)
        logger.info(f"FORCED REGIME: {forced.value}")

    if args.status:
        scalper._check_daily_reset()
        if scalper.state.positions:
            scalper.update_prices()
        print(scalper.get_summary())
        return

    if args.scan_once:
        scalper.scan_and_trade()
        print(scalper.get_summary())
        return

    scalper.run_loop()


if __name__ == "__main__":
    main()
