"""Stock trading bot v1.0 — Alpaca-based, multi-signal confluence.

Architecture:
1. HARDCODED UNIVERSE — 110 liquid stocks + ETFs, no discovery needed
2. MARKET REGIME — SPY/VIX-based regime detection (Bull/Cautious/Bear/HighVol/Choppy)
3. MULTI-TIMEFRAME — Daily trend, hourly entry timing
4. CONFLUENCE SCORING — 0-12 scale with Thompson Sampling bandit weights
5. RELATIVE STRENGTH — vs SPY for stock/sector selection
6. VWAP — intraday trend confirmation
7. SECTOR MOMENTUM — ETF-based sector tailwind detection
8. EARNINGS SHIELD — avoid binary event risk via calendar API
9. DYNAMIC ATR EXITS — volatility-scaled TP/SL, progressive stop tightening
10. MARKET HOURS — only trades 9:30-16:00 ET, sleeps off-hours
11. CIRCUIT BREAKERS — daily loss limit, consecutive losses, drawdown pause/kill
12. AUTO-TUNING — Thompson Sampling bandit, per-sector/regime win rate tracking

Paper trades by default. Uses Alpaca paper trading API.
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "stock_trading")
STATE_FILE = os.path.join(DATA_DIR, "state.json")
TRADE_LOG = os.path.join(DATA_DIR, "trades.csv")
ANALYTICS_LOG = os.path.join(DATA_DIR, "analytics.jsonl")
TUNING_LOG = os.path.join(DATA_DIR, "tuning.json")
LOG_FILE = os.path.join(DATA_DIR, "stock_trader.log")

# --- Configuration ---
STOCK_CONFIG = {
    # Bankroll
    "bankroll": 1_000.00,             # Paper trading starts with $1k
    "max_position_usd": 20.00,      # Max per-trade (for real money: min($20, 10% bankroll))
    "max_open_positions": 15,
    "max_exposure_pct": 0.60,        # 60% of bankroll at risk
    "kelly_fraction": 0.08,          # 8% of bankroll
    # Confluence
    "min_confluence_score": 5,
    "min_signal_quality": "C",
    # Exits (ATR-based)
    "tp_atr_mult": 3.0,             # Take profit at 3x daily ATR
    "sl_atr_mult": 1.5,             # Stop loss at 1.5x daily ATR
    "trailing_atr_mult": 1.5,       # Trailing stop at 1.5x ATR
    "trailing_activation_atr": 2.0,  # Activate trailing after 2x ATR profit
    "breakeven_at_1r": True,         # Move stop to breakeven after 1R profit
    "max_hold_days": 10,             # Max 10 trading days
    # Progressive stop tightening
    "progressive_stop": True,
    "progressive_stop_start_pct": 0.5,   # Start tightening at 50% of max hold
    "progressive_stop_end_mult": 0.33,   # Tighten SL from 1.5x to 0.5x ATR
    # Circuit breakers
    "max_daily_loss_pct": -0.03,     # -3% of bankroll
    "max_consecutive_losses": 3,
    "cooldown_after_loss_sec": 3600, # 1 hour
    "max_daily_trades": 10,
    "drawdown_pause_pct": -0.15,     # -15% from peak
    "drawdown_kill_pct": -0.25,      # -25% from peak
    # Scan interval
    "scan_interval_sec": 600,        # 10 minutes
    # Signal thresholds
    "volume_spike_mult": 1.5,        # Volume > 1.5x 20-day avg
    "bb_period": 20,
    "bb_std": 2.0,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_bounce_upper": 45,
    # EMA periods
    "daily_ema_fast": 9,
    "daily_ema_slow": 21,
    "weekly_ema_fast": 10,
    "weekly_ema_slow": 40,
    # Regime
    "spy_ema_200_period": 200,
    "vix_high_threshold": 30,
    "vix_cautious_threshold": 20,
    "spy_adx_choppy_threshold": 20,
    "regime_confirmation_scans": 3,
    # Relative strength
    "relative_strength_period": 20,
    # 52-week position
    "near_52w_high_pct": 0.10,      # Within 10% of 52-week high
    # Earnings shield
    "earnings_entry_shield_days": 5,
    "earnings_exit_shield_days": 2,
    # Auto-tune
    "auto_tune_every_n_trades": 20,
    # Bandit
    "bandit_decay": 0.95,
}

# --- Trading Universe ---
UNIVERSE = {
    "mega_cap": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK.B",
        "JPM", "V", "UNH", "MA", "HD", "PG", "JNJ", "LLY", "AVGO", "XOM",
        "COST", "WMT",
    ],
    "growth_tech": [
        "AMD", "NFLX", "ADBE", "INTC", "QCOM", "MU",
        "AMAT", "PANW", "SHOP", "SQ", "COIN", "PLTR", "SNOW",
    ],
    "sector_leaders": [
        "BA", "CAT", "DE", "GE", "LMT", "PFE", "MRK", "ABBV", "CVX",
        "COP", "GS", "MS", "DIS", "SBUX", "NKE",
    ],
    "high_beta": [
        "SMCI", "MSTR", "RIVN", "LCID", "SOFI", "HOOD", "RBLX", "SNAP",
        "ROKU", "UPST",
    ],
    "financials": [
        "BX", "SCHW", "C", "BAC", "WFC", "AXP", "ICE", "CME", "SPGI", "MCO",
    ],
    "healthcare_biotech": [
        "TMO", "ISRG", "VRTX", "REGN", "GILD", "AMGN", "BMY", "ZTS", "SYK", "MDT",
    ],
    "industrials": [
        "HON", "RTX", "UNP", "FDX", "UPS", "WM", "ETN", "ITW", "EMR", "CSX",
    ],
    "consumer_retail": [
        "MCD", "LOW", "TJX", "TGT", "ORLY", "AZO", "ROST", "DG", "YUM", "CMG",
    ],
    "semiconductors": [
        "MRVL", "KLAC", "LRCX", "ASML", "ADI", "NXPI", "ON", "MCHP", "TXN", "SNPS",
    ],
    "software_cloud": [
        "CRM", "ORCL", "NOW", "WDAY", "DDOG", "NET", "CRWD", "ZS", "TEAM", "FTNT",
    ],
    "energy_materials": [
        "SLB", "EOG", "OXY", "FCX", "NEM", "LIN", "APD", "ECL", "DD", "DOW",
    ],
    "media_comm": [
        "CMCSA", "T", "VZ", "CHTR", "TMUS", "ABNB", "BKNG", "UBER", "LYFT", "DASH",
    ],
    "etfs": [
        "SPY", "QQQ", "IWM", "XLF", "XLE", "XLK", "XLV", "XLI", "XLC",
        "XLY", "XLP", "SMH", "GLD", "TLT", "UVXY",
    ],
}

# Map stock -> sector for sector momentum checks
SECTOR_MAP = {}
SECTOR_ETF_MAP = {
    "mega_cap": "SPY",
    "growth_tech": "XLK",
    "sector_leaders": "SPY",
    "high_beta": "QQQ",
    "financials": "XLF",
    "healthcare_biotech": "XLV",
    "industrials": "XLI",
    "consumer_retail": "XLY",
    "semiconductors": "SMH",
    "software_cloud": "XLK",
    "energy_materials": "XLE",
    "media_comm": "XLC",
    "etfs": None,
    "dynamic": "SPY",
}
for sector, symbols in UNIVERSE.items():
    for sym in symbols:
        SECTOR_MAP[sym] = sector

ALL_SYMBOLS = []
for symbols in UNIVERSE.values():
    ALL_SYMBOLS.extend(symbols)
ALL_SYMBOLS_SET = set(ALL_SYMBOLS)


# --- Technical Analysis Functions ---

def calc_ema(closes: list[float], period: int) -> Optional[float]:
    """Calculate EMA using Wilder's smoothing."""
    if len(closes) < period:
        return None
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
    return ema


def calc_ema_series(closes: list[float], period: int) -> list[float]:
    """Calculate full EMA series for MACD."""
    if len(closes) < period:
        return []
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period
    series = [ema]
    for price in closes[period:]:
        ema = price * k + ema * (1 - k)
        series.append(ema)
    return series


def calc_rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """Calculate RSI using Wilder's smoothing."""
    if len(closes) < period + 1:
        return None
    gains, losses = 0.0, 0.0
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        if delta > 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
        if delta > 0:
            avg_gain = (avg_gain * (period - 1) + delta) / period
            avg_loss = (avg_loss * (period - 1)) / period
        else:
            avg_gain = (avg_gain * (period - 1)) / period
            avg_loss = (avg_loss * (period - 1) - delta) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def calc_macd(closes: list[float]) -> Optional[tuple]:
    """Calculate MACD (12, 26, 9) using running EMA series."""
    if len(closes) < 35:
        return None
    ema12 = calc_ema_series(closes, 12)
    ema26 = calc_ema_series(closes, 26)
    if not ema12 or not ema26:
        return None
    # Align series
    offset = len(ema12) - len(ema26)
    macd_line = [ema12[i + offset] - ema26[i] for i in range(len(ema26))]
    if len(macd_line) < 9:
        return None
    signal = calc_ema_series(macd_line, 9)
    if not signal:
        return None
    offset2 = len(macd_line) - len(signal)
    histogram = macd_line[-1] - signal[-1]
    return macd_line[-1], signal[-1], histogram


def calc_macd_last_two_histograms(closes: list[float]) -> Optional[tuple]:
    """Get last two MACD histogram values for crossover detection."""
    if len(closes) < 36:
        return None
    macd_now = calc_macd(closes)
    macd_prev = calc_macd(closes[:-1])
    if macd_now and macd_prev:
        return macd_prev[2], macd_now[2]
    return None


def calc_bollinger(closes: list[float], period: int = 20, std_mult: float = 2.0):
    """Calculate Bollinger Bands."""
    if len(closes) < period:
        return None
    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    std = math.sqrt(variance)
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    bandwidth = (upper - lower) / middle if middle > 0 else 0
    return upper, middle, lower, bandwidth


def calc_atr(highs: list[float], lows: list[float], closes: list[float],
             period: int = 14) -> Optional[float]:
    """Calculate ATR using Wilder's smoothing."""
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def calc_adx(highs: list[float], lows: list[float], closes: list[float],
             period: int = 14) -> Optional[float]:
    """Calculate ADX using Wilder's smoothing."""
    if len(closes) < period * 2 + 1:
        return None
    plus_dm, minus_dm, trs = [], [], []
    for i in range(1, len(closes)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0)
        minus_dm.append(down if down > up and down > 0 else 0)
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]),
                 abs(lows[i] - closes[i - 1]))
        trs.append(tr)
    # Wilder's smoothing
    atr_s = sum(trs[:period]) / period
    plus_s = sum(plus_dm[:period]) / period
    minus_s = sum(minus_dm[:period]) / period
    dx_vals = []
    for i in range(period, len(trs)):
        atr_s = (atr_s * (period - 1) + trs[i]) / period
        plus_s = (plus_s * (period - 1) + plus_dm[i]) / period
        minus_s = (minus_s * (period - 1) + minus_dm[i]) / period
        if atr_s > 0:
            plus_di = 100 * plus_s / atr_s
            minus_di = 100 * minus_s / atr_s
            di_sum = plus_di + minus_di
            dx = 100 * abs(plus_di - minus_di) / di_sum if di_sum > 0 else 0
            dx_vals.append(dx)
    if len(dx_vals) < period:
        return None
    adx = sum(dx_vals[:period]) / period
    for dx in dx_vals[period:]:
        adx = (adx * (period - 1) + dx) / period
    return adx


def calc_vwap(closes: list[float], volumes: list[float],
              highs: list[float], lows: list[float]) -> Optional[float]:
    """Calculate VWAP (Volume Weighted Average Price)."""
    if not closes or not volumes or len(closes) != len(volumes):
        return None
    total_vp = 0.0
    total_vol = 0.0
    for i in range(len(closes)):
        typical = (highs[i] + lows[i] + closes[i]) / 3
        total_vp += typical * volumes[i]
        total_vol += volumes[i]
    return total_vp / total_vol if total_vol > 0 else None


# --- Market Regime ---

class MarketRegime(Enum):
    BULL = "bull"
    CAUTIOUS = "cautious"
    BEAR = "bear"
    HIGH_VOL = "high_vol"
    CHOPPY = "choppy"


REGIME_PARAMS = {
    MarketRegime.BULL: {
        "min_confluence": 5,
        "max_positions": 8,
        "position_scale": 1.0,
        "description": "SPY > 200 EMA, VIX < 20",
    },
    MarketRegime.CAUTIOUS: {
        "min_confluence": 6,
        "max_positions": 6,
        "position_scale": 0.85,
        "description": "SPY > 200 EMA, VIX 20-25",
    },
    MarketRegime.BEAR: {
        "min_confluence": 7,
        "max_positions": 3,
        "position_scale": 0.5,
        "description": "SPY < 200 EMA",
    },
    MarketRegime.HIGH_VOL: {
        "min_confluence": 7,
        "max_positions": 2,
        "position_scale": 0.3,
        "description": "VIX > 30",
    },
    MarketRegime.CHOPPY: {
        "min_confluence": 6,
        "max_positions": 5,
        "position_scale": 0.7,
        "description": "SPY ADX < 20",
    },
}


class RegimeDetector:
    """Detects market regime from SPY and VIX data."""

    def __init__(self, config: dict):
        self.config = config
        self._current = MarketRegime.BULL
        self._pending = None
        self._pending_count = 0
        self._confirmation_needed = config.get("regime_confirmation_scans", 3)
        self._first_update = True

    @property
    def current(self) -> MarketRegime:
        return self._current

    def update(self, spy_data: dict, vix_price: float) -> MarketRegime:
        """Update regime based on SPY and VIX data.

        spy_data should contain:
          - ema_200: SPY 200-day EMA
          - price: current SPY price
          - adx: SPY ADX value
        """
        spy_price = spy_data.get("price", 0)
        spy_ema_200 = spy_data.get("ema_200", 0)
        spy_adx = spy_data.get("adx", 25)

        # Determine raw regime
        if vix_price > self.config.get("vix_high_threshold", 30):
            raw = MarketRegime.HIGH_VOL
        elif spy_price < spy_ema_200 and spy_ema_200 > 0:
            raw = MarketRegime.BEAR
        elif spy_adx < self.config.get("spy_adx_choppy_threshold", 20):
            raw = MarketRegime.CHOPPY
        elif vix_price > self.config.get("vix_cautious_threshold", 20):
            raw = MarketRegime.CAUTIOUS
        else:
            raw = MarketRegime.BULL

        # On first update, set regime immediately (no anti-whipsaw delay)
        if self._first_update:
            self._first_update = False
            if raw != self._current:
                logger.info(f"REGIME INIT: {self._current.value} -> {raw.value}")
                self._current = raw
            return self._current

        # Anti-whipsaw: require N confirmations
        if raw != self._current:
            if raw == self._pending:
                self._pending_count += 1
            else:
                self._pending = raw
                self._pending_count = 1

            if self._pending_count >= self._confirmation_needed:
                old = self._current
                self._current = raw
                self._pending = None
                self._pending_count = 0
                logger.info(f"REGIME CHANGE: {old.value} -> {raw.value}")
        else:
            self._pending = None
            self._pending_count = 0

        return self._current


# --- Thompson Sampling Signal Bandit ---

class SignalBandit:
    """Thompson Sampling bandit for signal weighting."""

    SIGNAL_NAMES = [
        "DAILY_TREND", "WEEKLY_TREND", "RSI_BOUNCE", "MACD_CROSS",
        "VOLUME_SPIKE", "REL_STRENGTH", "BB_LOWER", "VWAP_RECLAIM",
        "NEAR_52W_HIGH", "SECTOR_MOMENTUM", "EARNINGS_SHIELD", "MARKET_REGIME",
    ]

    def __init__(self, decay: float = 0.95):
        self.decay = decay
        self.alpha = {s: 1.0 for s in self.SIGNAL_NAMES}
        self.beta = {s: 1.0 for s in self.SIGNAL_NAMES}

    def get_weight(self, signal_name: str) -> float:
        a = self.alpha.get(signal_name, 1.0)
        b = self.beta.get(signal_name, 1.0)
        return random.betavariate(a, b)

    def get_all_weights(self) -> dict:
        return {s: round(self.alpha[s] / (self.alpha[s] + self.beta[s]), 3)
                for s in self.SIGNAL_NAMES}

    def update(self, signal_name: str, won: bool):
        if signal_name not in self.alpha:
            return
        # Decay old observations
        self.alpha[signal_name] *= self.decay
        self.beta[signal_name] *= self.decay
        # Add new observation
        if won:
            self.alpha[signal_name] += 1
        else:
            self.beta[signal_name] += 1

    def to_dict(self) -> dict:
        return {"alpha": dict(self.alpha), "beta": dict(self.beta)}

    @classmethod
    def from_dict(cls, data: dict, decay: float = 0.95) -> "SignalBandit":
        b = cls(decay=decay)
        if data:
            b.alpha.update(data.get("alpha", {}))
            b.beta.update(data.get("beta", {}))
        return b


# --- Data Structures ---

@dataclass
class MarketContext:
    """Full market context for a stock."""
    symbol: str = ""
    price: float = 0.0
    sector: str = ""
    # Daily data
    daily_ema_fast: float = 0.0
    daily_ema_slow: float = 0.0
    daily_trend: str = "neutral"     # bullish / bearish / neutral
    # Weekly data
    weekly_ema_fast: float = 0.0
    weekly_ema_slow: float = 0.0
    weekly_trend: str = "neutral"
    # Intraday
    vwap: float = 0.0
    # ATR
    atr_daily: float = 0.0
    atr_pct: float = 0.0
    # Indicators
    rsi: float = 50.0
    macd_histogram: float = 0.0
    macd_cross: str = ""             # bullish / bearish / none
    bb_position: str = ""            # lower / upper / middle
    volume_ratio: float = 1.0
    # Relative strength
    rel_strength_vs_spy: float = 0.0
    # 52-week
    high_52w: float = 0.0
    low_52w: float = 0.0
    pct_from_52w_high: float = 0.0
    # Sector
    sector_trend: str = "neutral"
    # Regime
    regime: MarketRegime = MarketRegime.BULL
    # ADX
    adx: float = 25.0
    # Earnings
    earnings_within_entry_shield: bool = False
    earnings_within_exit_shield: bool = False
    next_earnings_date: str = ""


@dataclass
class Signal:
    """Trading signal from confluence engine."""
    symbol: str = ""
    price: float = 0.0
    confluence_score: int = 0
    quality_grade: str = "D"
    components: list = field(default_factory=list)
    reasoning: str = ""
    atr: float = 0.0
    rsi: float = 50.0
    regime: str = "bull"
    take_profit: float = 0.0
    stop_loss: float = 0.0
    signal_type: str = "confluence"
    sector: str = ""


# --- Alpaca API Client ---

class AlpacaClient:
    """Lightweight Alpaca API client for market data and paper trading."""

    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url.rstrip("/")
        self.data_url = "https://data.alpaca.markets"
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        })
        self._rate_limit_remaining = 200
        self._rate_limit_reset = 0

    def _request(self, method: str, url: str, **kwargs) -> Optional[dict]:
        """Make API request with rate limiting and error handling."""
        try:
            resp = self.session.request(method, url, timeout=15, **kwargs)
            self._rate_limit_remaining = int(resp.headers.get("x-ratelimit-remaining", 200))
            if self._rate_limit_remaining < 10:
                logger.warning(f"Rate limit low: {self._rate_limit_remaining} remaining")
                time.sleep(1)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited, waiting {wait}s")
                time.sleep(wait)
                resp = self.session.request(method, url, timeout=15, **kwargs)
            if resp.status_code >= 400:
                body = resp.text[:500]
                logger.error(f"Alpaca API {method} {url.split('/')[-1]} → {resp.status_code}: {body}")
                return None
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Alpaca API request failed: {e}")
            return None

    def get_account(self) -> Optional[dict]:
        """Get account info."""
        return self._request("GET", f"{self.base_url}/v2/account")

    def get_clock(self) -> Optional[dict]:
        """Get market clock (is_open, next_open, next_close)."""
        return self._request("GET", f"{self.base_url}/v2/clock")

    def get_calendar(self, start: str = None, end: str = None) -> Optional[list]:
        """Get market calendar."""
        params = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return self._request("GET", f"{self.base_url}/v2/calendar", params=params)

    def get_bars(self, symbol: str, timeframe: str = "1Day",
                 limit: int = 100, start: str = None, end: str = None) -> Optional[dict]:
        """Get historical bars.

        timeframe: 1Min, 5Min, 15Min, 30Min, 1Hour, 1Day, 1Week
        """
        params = {"timeframe": timeframe, "limit": limit, "feed": "iex"}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return self._request(
            "GET", f"{self.data_url}/v2/stocks/{symbol}/bars", params=params
        )

    def get_bars_multi(self, symbols: list[str], timeframe: str = "1Day",
                       limit: int = 100, start: str = None) -> Optional[dict]:
        """Get bars for multiple symbols in one request."""
        params = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "limit": limit,
            "feed": "iex",
        }
        if start:
            params["start"] = start
        return self._request(
            "GET", f"{self.data_url}/v2/stocks/bars", params=params
        )

    def get_latest_quote(self, symbol: str) -> Optional[dict]:
        """Get latest quote for a symbol."""
        return self._request(
            "GET", f"{self.data_url}/v2/stocks/{symbol}/quotes/latest",
            params={"feed": "iex"},
        )

    def get_latest_quotes_multi(self, symbols: list[str]) -> Optional[dict]:
        """Get latest quotes for multiple symbols."""
        return self._request(
            "GET", f"{self.data_url}/v2/stocks/quotes/latest",
            params={"symbols": ",".join(symbols), "feed": "iex"},
        )

    def get_snapshot(self, symbol: str) -> Optional[dict]:
        """Get snapshot (latest trade, quote, minute bar, daily bar, prev daily bar)."""
        return self._request(
            "GET", f"{self.data_url}/v2/stocks/{symbol}/snapshot",
            params={"feed": "iex"},
        )

    def get_snapshots_multi(self, symbols: list[str]) -> Optional[dict]:
        """Get snapshots for multiple symbols."""
        return self._request(
            "GET", f"{self.data_url}/v2/stocks/snapshots",
            params={"symbols": ",".join(symbols), "feed": "iex"},
        )

    def get_positions(self) -> Optional[list]:
        """Get all open positions from Alpaca."""
        return self._request("GET", f"{self.base_url}/v2/positions")

    def place_order(self, symbol: str, qty: float, side: str,
                    order_type: str = "market", time_in_force: str = "day",
                    notional: float = None) -> Optional[dict]:
        """Place an order. Use notional (dollar amount) for fractional shares."""
        payload = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if notional is not None:
            payload["notional"] = str(round(notional, 2))
        else:
            payload["qty"] = str(round(qty, 2))
        return self._request("POST", f"{self.base_url}/v2/orders", json=payload)

    def close_position(self, symbol: str) -> Optional[dict]:
        """Close a position."""
        return self._request("DELETE", f"{self.base_url}/v2/positions/{symbol}")


# --- Dynamic Stock Screener ---

class DynamicScreener:
    """Discovers tradeable stocks outside the static universe using Alpaca screener APIs."""

    def __init__(self, alpaca: 'AlpacaClient', config: dict):
        self.alpaca = alpaca
        self.config = config
        self._cache = {"symbols": [], "ts": 0}
        self._cache_ttl = 900  # 15 minutes

    def screen(self) -> list[str]:
        """Return dynamic symbols to add to scan. Cached 15 min."""
        if time.time() - self._cache["ts"] < self._cache_ttl:
            return self._cache["symbols"]

        candidates = set()

        # 1. Most active by volume (1 API call)
        actives = self.alpaca._request(
            "GET", f"{self.alpaca.data_url}/v1beta1/screener/stocks/most-actives",
            params={"by": "volume", "top": 50},
        )
        if actives and "most_actives" in actives:
            for item in actives["most_actives"]:
                sym = item.get("symbol", "")
                price = float(item.get("price", 0) or 0)
                vol = int(item.get("volume", 0) or 0)
                if (sym and 5.0 <= price <= 1000.0 and vol >= 500_000
                        and "." not in sym and len(sym) <= 5
                        and sym not in ALL_SYMBOLS_SET):
                    candidates.add(sym)

        # 2. Top movers by % change (1 API call)
        movers = self.alpaca._request(
            "GET", f"{self.alpaca.data_url}/v1beta1/screener/stocks/movers",
            params={"top": 25},
        )
        if movers:
            for item in movers.get("gainers", []) + movers.get("losers", []):
                sym = item.get("symbol", "")
                price = float(item.get("price", 0) or 0)
                change = abs(float(item.get("percent_change", 0) or 0))
                if (sym and 5.0 <= price <= 1000.0 and change >= 3.0
                        and "." not in sym and len(sym) <= 5
                        and sym not in ALL_SYMBOLS_SET):
                    candidates.add(sym)

        dynamic = list(candidates)[:30]

        # Assign sector for dynamic symbols
        for sym in dynamic:
            if sym not in SECTOR_MAP:
                SECTOR_MAP[sym] = "dynamic"

        self._cache = {"symbols": dynamic, "ts": time.time()}
        if dynamic:
            logger.info(f"Dynamic screener: {len(dynamic)} new symbols: {', '.join(dynamic[:10])}")
        return dynamic


# --- VIX Data (from CBOE / Yahoo proxy) ---

def get_vix_price() -> Optional[float]:
    """Get current VIX price from Yahoo Finance."""
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
            return float(price)
    except Exception as e:
        logger.debug(f"VIX fetch failed: {e}")

    # Fallback: use UVXY as VIX proxy (always in our universe)
    try:
        url = "https://data.alpaca.markets/v2/stocks/UVXY/snapshot"
        resp = requests.get(url, headers={
            "APCA-API-KEY-ID": os.getenv("ALPACA_API_KEY", ""),
            "APCA-API-SECRET-KEY": os.getenv("ALPACA_SECRET_KEY", ""),
        }, params={"feed": "iex"}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            # UVXY is ~1.5x VIX, rough approximation
            uvxy_price = float(data.get("latestTrade", {}).get("p", 0))
            if uvxy_price > 0:
                return uvxy_price * 0.7  # Rough VIX estimate
    except Exception:
        pass
    return None


# --- Earnings Calendar ---

class EarningsCalendar:
    """Track earnings dates to avoid binary events."""

    def __init__(self):
        self._cache = {}  # symbol -> next earnings date string or None
        self._last_fetch = {}  # symbol -> timestamp of last fetch
        self._fetch_interval = 86400  # Refresh daily

    def _fetch_earnings(self, symbol: str) -> Optional[str]:
        """Fetch next earnings date from Yahoo Finance."""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                events = data.get("chart", {}).get("result", [{}])[0].get("events", {})
                earnings = events.get("earnings", {})
                if earnings:
                    # Find next future earnings
                    now = datetime.now(timezone.utc)
                    for ts_str, info in sorted(earnings.items()):
                        try:
                            ts = int(ts_str)
                            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                            if dt > now:
                                return dt.strftime("%Y-%m-%d")
                        except (ValueError, TypeError):
                            continue
        except Exception as e:
            logger.debug(f"Earnings fetch failed for {symbol}: {e}")
        return None

    def get_next_earnings(self, symbol: str) -> Optional[str]:
        """Get next earnings date for a symbol (cached daily)."""
        now = time.time()
        if symbol in self._cache and now - self._last_fetch.get(symbol, 0) < self._fetch_interval:
            return self._cache[symbol]

        date_str = self._fetch_earnings(symbol)
        self._cache[symbol] = date_str
        self._last_fetch[symbol] = now
        return date_str

    def is_within_days(self, symbol: str, days: int) -> bool:
        """Check if next earnings is within N trading days."""
        date_str = self.get_next_earnings(symbol)
        if not date_str:
            return False
        try:
            earnings_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = (earnings_dt - now).days
            # Approximate trading days as calendar days * 5/7
            trading_days = delta * 5 / 7
            return 0 <= trading_days <= days
        except (ValueError, TypeError):
            return False


# --- Signal Engine ---

class SignalDetector:
    """12-signal confluence detector for stocks."""

    def __init__(self, config: dict, bandit: SignalBandit = None):
        self.config = config
        self.bandit = bandit

    def analyze(self, symbol: str, ctx: MarketContext) -> Optional[Signal]:
        """Analyze a stock using all available context. Returns signal or None."""
        if ctx.price <= 0 or ctx.atr_daily <= 0:
            return None

        score = 0
        components = []
        reasons = []

        # --- 1. DAILY TREND (EMA 9/21) ---
        if ctx.daily_trend == "bullish":
            score += 1
            components.append("DAILY_TREND")
            reasons.append(f"Daily EMA9 {ctx.daily_ema_fast:.2f} > EMA21 {ctx.daily_ema_slow:.2f}")

        # --- 2. WEEKLY TREND (EMA 10/40) ---
        if ctx.weekly_trend == "bullish":
            score += 1
            components.append("WEEKLY_TREND")
            reasons.append(f"Weekly trend bullish")

        # --- 3. RSI OVERSOLD BOUNCE (30-45) ---
        if self.config["rsi_oversold"] <= ctx.rsi <= self.config["rsi_bounce_upper"]:
            score += 1
            components.append("RSI_BOUNCE")
            reasons.append(f"RSI={ctx.rsi:.0f} (oversold bounce zone)")

        # --- 4. MACD BULLISH CROSSOVER ---
        if ctx.macd_cross == "bullish":
            score += 1
            components.append("MACD_CROSS")
            reasons.append(f"MACD bullish crossover (hist={ctx.macd_histogram:.4f})")

        # --- 5. VOLUME SPIKE ---
        if ctx.volume_ratio > self.config["volume_spike_mult"]:
            score += 1
            components.append("VOLUME_SPIKE")
            reasons.append(f"Volume {ctx.volume_ratio:.1f}x avg")

        # --- 6. RELATIVE STRENGTH VS SPY ---
        if ctx.rel_strength_vs_spy > 0:
            score += 1
            components.append("REL_STRENGTH")
            reasons.append(f"Outperforming SPY by {ctx.rel_strength_vs_spy:.1%}")

        # --- 7. BOLLINGER BAND LOWER TOUCH + REVERSAL ---
        if ctx.bb_position == "lower":
            score += 1
            components.append("BB_LOWER")
            reasons.append("Price at Bollinger lower band")

        # --- 8. VWAP RECLAIM ---
        if ctx.vwap > 0 and ctx.price > ctx.vwap:
            score += 1
            components.append("VWAP_RECLAIM")
            reasons.append(f"Price ${ctx.price:.2f} > VWAP ${ctx.vwap:.2f}")

        # --- 9. 52-WEEK HIGH PROXIMITY ---
        if ctx.high_52w > 0 and ctx.pct_from_52w_high <= self.config["near_52w_high_pct"]:
            score += 1
            components.append("NEAR_52W_HIGH")
            reasons.append(f"Within {ctx.pct_from_52w_high:.1%} of 52w high")

        # --- 10. SECTOR MOMENTUM ---
        if ctx.sector_trend == "bullish":
            score += 1
            components.append("SECTOR_MOMENTUM")
            reasons.append(f"Sector ETF in uptrend")

        # --- 11. EARNINGS SHIELD ---
        if not ctx.earnings_within_entry_shield:
            score += 1
            components.append("EARNINGS_SHIELD")
            reasons.append("No earnings within 5 days")

        # --- 12. MARKET REGIME (BULL) ---
        if ctx.regime == MarketRegime.BULL:
            score += 1
            components.append("MARKET_REGIME")
            reasons.append(f"Bull market (SPY > 200 EMA)")

        # Quality grading
        if score >= 7:
            grade = "A"
        elif score >= 6:
            grade = "B"
        elif score >= 5:
            grade = "C"
        else:
            grade = "D"

        # Check minimum confluence
        regime_params = REGIME_PARAMS.get(ctx.regime, REGIME_PARAMS[MarketRegime.BULL])
        min_score = max(self.config["min_confluence_score"],
                       regime_params["min_confluence"])

        # In choppy regime, only allow mean-reversion signals (RSI + BB)
        if ctx.regime == MarketRegime.CHOPPY:
            has_mean_reversion = ("RSI_BOUNCE" in components or "BB_LOWER" in components)
            if not has_mean_reversion:
                logger.debug(f"  {symbol}: Choppy regime — skipping non-mean-reversion signal")
                return None

        if score < min_score:
            logger.debug(f"  {symbol}: score {score} < min {min_score}")
            return None

        GRADE_RANK = {"A": 4, "B": 3, "C": 2, "D": 1}
        if GRADE_RANK.get(grade, 0) < GRADE_RANK.get(self.config["min_signal_quality"], 2):
            return None

        # Calculate ATR-based exits
        atr = ctx.atr_daily
        tp = round(ctx.price + atr * self.config["tp_atr_mult"], 2)
        sl = round(ctx.price - atr * self.config["sl_atr_mult"], 2)

        return Signal(
            symbol=symbol,
            price=ctx.price,
            confluence_score=score,
            quality_grade=grade,
            components=components,
            reasoning=" | ".join(reasons),
            atr=atr,
            rsi=ctx.rsi,
            regime=ctx.regime.value,
            take_profit=tp,
            stop_loss=sl,
            signal_type="confluence",
            sector=ctx.sector,
        )


# --- State Management ---

@dataclass
class StockTraderState:
    """Paper trading state — persisted to disk."""
    bankroll: float = 1_000.00
    starting_bankroll: float = 1_000.00
    peak_bankroll: float = 1_000.00
    positions: list = field(default_factory=list)
    closed_trades: list = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    daily_trade_count: int = 0
    last_loss_time: float = 0.0
    consecutive_losses: int = 0
    last_reset_date: str = ""
    last_scan: str = ""
    removed_symbols: list = field(default_factory=list)
    tuned_overrides: dict = field(default_factory=dict)
    regime: str = "bull"
    bandit_state: dict = field(default_factory=dict)
    symbol_cooldowns: dict = field(default_factory=dict)  # {symbol: expiry_timestamp}

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

    @property
    def drawdown_pct(self) -> float:
        if self.peak_bankroll <= 0:
            return 0.0
        return (self.bankroll - self.peak_bankroll) / self.peak_bankroll


# --- Main Bot ---

class StockTrader:
    """Automated stock trading bot using Alpaca API."""

    def __init__(self, config: dict = None):
        self.config = config or STOCK_CONFIG
        os.makedirs(DATA_DIR, exist_ok=True)

        # Alpaca client
        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        if not api_key or not secret_key:
            logger.error("ALPACA_API_KEY and ALPACA_SECRET_KEY required in .env")
            raise SystemExit("Missing Alpaca credentials")
        self.alpaca = AlpacaClient(api_key, secret_key, base_url)

        # Load state
        self.state = self._load_state()

        # Components
        self.bandit = SignalBandit.from_dict(
            self.state.bandit_state,
            decay=self.config.get("bandit_decay", 0.95),
        )
        self.detector = SignalDetector(self.config, bandit=self.bandit)
        self.regime_detector = RegimeDetector(self.config)
        self.earnings = EarningsCalendar()
        self.screener = DynamicScreener(self.alpaca, self.config)

        # Caches
        self._spy_data_cache = {}
        self._vix_cache = {"price": 20.0, "ts": 0}
        self._daily_bars_cache = {}    # symbol -> {bars, ts}
        self._weekly_bars_cache = {}   # symbol -> {bars, ts}
        self._sector_trend_cache = {}  # sector -> {trend, ts}

        # Re-apply tuned overrides
        if self.state.tuned_overrides:
            for key, val in self.state.tuned_overrides.items():
                if key in self.config:
                    logger.info(f"Re-applying tuned: {key}={val}")
                    self.config[key] = val

    def _load_state(self) -> StockTraderState:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    content = f.read().strip()
                if not content:
                    logger.warning("State file empty, starting fresh")
                    return StockTraderState(
                        bankroll=self.config["bankroll"],
                        starting_bankroll=self.config["bankroll"],
                        peak_bankroll=self.config["bankroll"],
                    )
                data = json.loads(content)
                known = {f.name for f in StockTraderState.__dataclass_fields__.values()}
                return StockTraderState(**{k: v for k, v in data.items() if k in known})
            except json.JSONDecodeError as e:
                logger.error(f"State file corrupt: {e}. Remove {STATE_FILE} to reset.")
                raise SystemExit(f"FATAL: corrupt state file: {STATE_FILE}")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
        return StockTraderState(
            bankroll=self.config["bankroll"],
            starting_bankroll=self.config["bankroll"],
            peak_bankroll=self.config["bankroll"],
        )

    def _save_state(self):
        """Atomic write: temp file + rename."""
        self.state.bandit_state = self.bandit.to_dict()
        self.state.regime = self.regime_detector.current.value
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(asdict(self.state), f, indent=2)
        os.replace(tmp, STATE_FILE)

    def _log_trade(self, action: str, data: dict):
        """Append trade to CSV log."""
        with open(TRADE_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow([
                    "timestamp", "action", "symbol", "price", "shares",
                    "cost_usd", "pnl", "signal_type", "confluence_score",
                    "quality_grade", "regime", "rsi", "sector", "reasoning",
                ])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                action, data.get("symbol", ""), data.get("price", 0),
                data.get("shares", 0), data.get("cost_usd", 0),
                data.get("pnl", 0), data.get("signal_type", ""),
                data.get("confluence_score", 0), data.get("quality_grade", ""),
                data.get("regime", ""), data.get("rsi", 0),
                data.get("sector", ""), data.get("reasoning", "")[:120],
            ])

    def _log_analytics(self, data: dict):
        """Append rich trade context to JSONL."""
        with open(ANALYTICS_LOG, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def _check_daily_reset(self):
        """Reset daily counters at midnight UTC."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.last_reset_date != today:
            self.state.daily_pnl = 0.0
            self.state.daily_trade_count = 0
            self.state.last_reset_date = today
            # Purge expired symbol cooldowns
            now = time.time()
            self.state.symbol_cooldowns = {
                sym: exp for sym, exp in self.state.symbol_cooldowns.items()
                if exp > now
            }
            self._save_state()

    def is_market_open(self) -> bool:
        """Check if US stock market is currently open."""
        clock = self.alpaca.get_clock()
        if clock:
            return clock.get("is_open", False)
        return False

    def time_until_market_event(self) -> dict:
        """Get time until next market open/close."""
        clock = self.alpaca.get_clock()
        if not clock:
            return {"is_open": False, "next_event": "unknown", "minutes": 0}

        is_open = clock.get("is_open", False)
        try:
            if is_open:
                close_str = clock.get("next_close", "")
                close_dt = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                delta = (close_dt - datetime.now(timezone.utc)).total_seconds() / 60
                return {"is_open": True, "next_event": "close", "minutes": max(0, delta)}
            else:
                open_str = clock.get("next_open", "")
                open_dt = datetime.fromisoformat(open_str.replace("Z", "+00:00"))
                delta = (open_dt - datetime.now(timezone.utc)).total_seconds() / 60
                return {"is_open": False, "next_event": "open", "minutes": max(0, delta)}
        except (ValueError, TypeError):
            return {"is_open": is_open, "next_event": "unknown", "minutes": 0}

    # --- Data Fetching ---

    def _get_daily_bars(self, symbol: str, limit: int = 250) -> list[dict]:
        """Fetch daily bars, cached for 15 minutes."""
        cache = self._daily_bars_cache.get(symbol)
        if cache and time.time() - cache["ts"] < 900:
            return cache["bars"]

        # Calculate start date for enough history
        start_dt = datetime.now(timezone.utc) - timedelta(days=int(limit * 1.5))
        start_str = start_dt.strftime("%Y-%m-%dT00:00:00Z")

        data = self.alpaca.get_bars(symbol, "1Day", limit=limit, start=start_str)
        if not data or "bars" not in data:
            return []

        bars = data["bars"]
        self._daily_bars_cache[symbol] = {"bars": bars, "ts": time.time()}
        return bars

    def _get_weekly_bars(self, symbol: str, limit: int = 52) -> list[dict]:
        """Fetch weekly bars, cached for 1 hour."""
        cache = self._weekly_bars_cache.get(symbol)
        if cache and time.time() - cache["ts"] < 3600:
            return cache["bars"]

        start_dt = datetime.now(timezone.utc) - timedelta(days=limit * 7 + 30)
        start_str = start_dt.strftime("%Y-%m-%dT00:00:00Z")

        data = self.alpaca.get_bars(symbol, "1Week", limit=limit, start=start_str)
        if not data or "bars" not in data:
            return []

        bars = data["bars"]
        self._weekly_bars_cache[symbol] = {"bars": bars, "ts": time.time()}
        return bars

    def _get_intraday_bars(self, symbol: str, timeframe: str = "5Min",
                           limit: int = 50) -> list[dict]:
        """Fetch intraday bars (not cached — always fresh)."""
        start_dt = datetime.now(timezone.utc) - timedelta(hours=6)
        start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        data = self.alpaca.get_bars(symbol, timeframe, limit=limit, start=start_str)
        if not data or "bars" not in data:
            return []
        return data["bars"]

    def _get_spy_regime_data(self) -> dict:
        """Get SPY data for regime detection."""
        cache = self._spy_data_cache
        if cache and time.time() - cache.get("ts", 0) < 900:
            return cache

        bars = self._get_daily_bars("SPY", limit=250)
        if not bars or len(bars) < 50:
            return {"price": 0, "ema_200": 0, "adx": 25, "ts": time.time()}

        closes = [b["c"] for b in bars]
        highs = [b["h"] for b in bars]
        lows = [b["l"] for b in bars]
        price = closes[-1]

        ema_200 = calc_ema(closes, min(200, len(closes))) if len(closes) >= 50 else price
        adx = calc_adx(highs, lows, closes) or 25

        result = {"price": price, "ema_200": ema_200, "adx": adx, "ts": time.time()}
        self._spy_data_cache = result
        return result

    def _get_vix(self) -> float:
        """Get current VIX, cached for 15 min."""
        if time.time() - self._vix_cache["ts"] < 900:
            return self._vix_cache["price"]

        vix = get_vix_price()
        if vix and vix > 0:
            self._vix_cache = {"price": vix, "ts": time.time()}
            return vix
        return self._vix_cache["price"]  # Return stale if fetch fails

    def _get_sector_trend(self, sector: str) -> str:
        """Check if sector ETF is in uptrend."""
        etf = SECTOR_ETF_MAP.get(sector)
        if not etf:
            return "neutral"

        cache = self._sector_trend_cache.get(sector)
        if cache and time.time() - cache["ts"] < 900:
            return cache["trend"]

        bars = self._get_daily_bars(etf, limit=30)
        if not bars or len(bars) < 21:
            return "neutral"

        closes = [b["c"] for b in bars]
        ema9 = calc_ema(closes, 9)
        ema21 = calc_ema(closes, 21)

        if ema9 and ema21:
            trend = "bullish" if ema9 > ema21 else "bearish"
        else:
            trend = "neutral"

        self._sector_trend_cache[sector] = {"trend": trend, "ts": time.time()}
        return trend

    # --- Market Context Building ---

    def _pre_filter_snapshots(self, symbols: list[str], top_n: int = 60) -> list[str]:
        """Use snapshots to quickly filter to the hottest symbols by volume/movement."""
        if len(symbols) <= top_n:
            return symbols
        hot = []
        for i in range(0, len(symbols), 100):
            batch = symbols[i:i + 100]
            data = self.alpaca.get_snapshots_multi(batch)
            if not data:
                hot.extend(batch[:10])
                continue
            scored = []
            for sym in batch:
                snap = data.get(sym)
                if not snap:
                    continue
                daily = snap.get("dailyBar", {})
                prev = snap.get("prevDailyBar", {})
                trade = snap.get("latestTrade", {})
                price = float(trade.get("p", 0) or 0)
                today_vol = int(daily.get("v", 0) or 0)
                prev_vol = int(prev.get("v", 1) or 1)
                prev_close = float(prev.get("c", 0) or 0)
                if price <= 0 or prev_close <= 0:
                    continue
                vol_ratio = today_vol / max(prev_vol, 1)
                change_pct = abs((price - prev_close) / prev_close)
                heat = vol_ratio * 0.6 + change_pct * 100 * 0.4
                scored.append((sym, heat))
            scored.sort(key=lambda x: x[1], reverse=True)
            hot.extend([s for s, _ in scored])
        result = hot[:top_n]
        logger.info(f"Pre-filter: {len(result)} hot symbols from {len(symbols)} candidates")
        return result

    def _build_market_context(self, symbol: str) -> Optional[MarketContext]:
        """Build full market context for a stock."""
        ctx = MarketContext(symbol=symbol, sector=SECTOR_MAP.get(symbol, "unknown"))
        ctx.regime = self.regime_detector.current

        # 1. Daily bars for trend + ATR
        daily_bars = self._get_daily_bars(symbol, limit=60)
        if not daily_bars or len(daily_bars) < 30:
            logger.debug(f"  {symbol}: insufficient daily data ({len(daily_bars) if daily_bars else 0}/30)")
            return None

        closes = [b["c"] for b in daily_bars]
        highs = [b["h"] for b in daily_bars]
        lows = [b["l"] for b in daily_bars]
        volumes = [b["v"] for b in daily_bars]

        # Use live quote price during market hours (daily bar close is stale)
        quote = self.alpaca.get_latest_quote(symbol)
        if quote and "quote" in quote:
            q = quote["quote"]
            mid = (q.get("ap", 0) + q.get("bp", 0)) / 2
            if mid > 0:
                ctx.price = mid
            else:
                ctx.price = closes[-1]
        else:
            ctx.price = closes[-1]

        # Daily EMAs
        ema_fast = calc_ema(closes, self.config["daily_ema_fast"])
        ema_slow = calc_ema(closes, self.config["daily_ema_slow"])
        if ema_fast and ema_slow:
            ctx.daily_ema_fast = ema_fast
            ctx.daily_ema_slow = ema_slow
            if ctx.price > ema_fast > ema_slow:
                ctx.daily_trend = "bullish"
            elif ctx.price < ema_fast < ema_slow:
                ctx.daily_trend = "bearish"

        # Daily ATR
        atr = calc_atr(highs, lows, closes)
        if atr:
            ctx.atr_daily = atr
            ctx.atr_pct = atr / ctx.price if ctx.price > 0 else 0

        # RSI
        rsi = calc_rsi(closes, self.config["rsi_period"])
        if rsi is not None:
            ctx.rsi = rsi

        # MACD
        macd_hist = calc_macd_last_two_histograms(closes)
        if macd_hist:
            prev_h, curr_h = macd_hist
            ctx.macd_histogram = curr_h
            if prev_h < 0 and curr_h > 0:
                ctx.macd_cross = "bullish"
            elif prev_h > 0 and curr_h < 0:
                ctx.macd_cross = "bearish"

        # Bollinger
        bb = calc_bollinger(closes, self.config["bb_period"], self.config["bb_std"])
        if bb:
            upper, middle, lower, bw = bb
            if ctx.price <= lower:
                ctx.bb_position = "lower"
            elif ctx.price >= upper:
                ctx.bb_position = "upper"
            else:
                ctx.bb_position = "middle"

        # Volume ratio
        if len(volumes) >= 21:
            avg_vol = sum(volumes[-21:-1]) / 20
            if avg_vol > 0:
                ctx.volume_ratio = volumes[-1] / avg_vol

        # ADX
        adx = calc_adx(highs, lows, closes)
        if adx is not None:
            ctx.adx = adx

        # 52-week high/low
        if len(closes) >= 50:
            # Use available data (up to 250 bars)
            period = min(len(closes), 252)
            ctx.high_52w = max(highs[-period:])
            ctx.low_52w = min(lows[-period:])
            if ctx.high_52w > 0:
                ctx.pct_from_52w_high = (ctx.high_52w - ctx.price) / ctx.high_52w

        # 2. Weekly bars for weekly trend
        weekly_bars = self._get_weekly_bars(symbol, limit=52)
        if weekly_bars and len(weekly_bars) >= 10:
            w_closes = [b["c"] for b in weekly_bars]
            w_ema_fast = calc_ema(w_closes, self.config["weekly_ema_fast"])
            w_ema_slow = calc_ema(w_closes, min(self.config["weekly_ema_slow"], len(w_closes)))
            if w_ema_fast and w_ema_slow:
                ctx.weekly_ema_fast = w_ema_fast
                ctx.weekly_ema_slow = w_ema_slow
                if w_ema_fast > w_ema_slow:
                    ctx.weekly_trend = "bullish"
                elif w_ema_fast < w_ema_slow:
                    ctx.weekly_trend = "bearish"

        # 3. Intraday VWAP
        intraday = self._get_intraday_bars(symbol, "5Min", limit=50)
        if intraday and len(intraday) >= 5:
            i_closes = [b["c"] for b in intraday]
            i_volumes = [b["v"] for b in intraday]
            i_highs = [b["h"] for b in intraday]
            i_lows = [b["l"] for b in intraday]
            vwap = calc_vwap(i_closes, i_volumes, i_highs, i_lows)
            if vwap:
                ctx.vwap = vwap

        # 4. Relative strength vs SPY
        spy_bars = self._get_daily_bars("SPY", limit=30)
        if spy_bars and len(spy_bars) >= 20 and len(daily_bars) >= 20:
            spy_closes = [b["c"] for b in spy_bars]
            rs_period = self.config["relative_strength_period"]
            if len(spy_closes) >= rs_period and len(closes) >= rs_period:
                stock_return = (closes[-1] / closes[-rs_period]) - 1
                spy_return = (spy_closes[-1] / spy_closes[-rs_period]) - 1
                ctx.rel_strength_vs_spy = stock_return - spy_return

        # 5. Sector momentum
        ctx.sector_trend = self._get_sector_trend(ctx.sector)

        # 6. Earnings shield
        ctx.earnings_within_entry_shield = self.earnings.is_within_days(
            symbol, self.config["earnings_entry_shield_days"]
        )
        ctx.earnings_within_exit_shield = self.earnings.is_within_days(
            symbol, self.config["earnings_exit_shield_days"]
        )
        next_earn = self.earnings.get_next_earnings(symbol)
        if next_earn:
            ctx.next_earnings_date = next_earn

        return ctx

    # --- Position Management ---

    def open_position(self, signal: Signal) -> bool:
        """Open a paper position from a signal."""
        # Risk checks
        if len(self.state.positions) >= self.config["max_open_positions"]:
            return False

        regime_params = REGIME_PARAMS.get(
            self.regime_detector.current, REGIME_PARAMS[MarketRegime.BULL]
        )
        if len(self.state.positions) >= regime_params["max_positions"]:
            return False

        if self.state.open_exposure >= self.state.bankroll * self.config["max_exposure_pct"]:
            return False

        daily_loss_limit = self.state.bankroll * self.config["max_daily_loss_pct"]
        if self.state.daily_pnl <= daily_loss_limit:
            logger.info("Daily loss limit hit")
            return False

        if self.state.daily_trade_count >= self.config["max_daily_trades"]:
            return False

        # Consecutive loss cooldown
        if self.state.consecutive_losses >= self.config["max_consecutive_losses"]:
            hours_since = (time.time() - self.state.last_loss_time) / 3600
            cooldown_hours = self.config["cooldown_after_loss_sec"] / 3600
            if hours_since < cooldown_hours:
                logger.info(f"Paused: {self.state.consecutive_losses} consecutive losses "
                           f"({cooldown_hours - hours_since:.1f}h until resume)")
                return False
            else:
                self.state.consecutive_losses = 0
                self._save_state()

        # Drawdown checks
        dd = self.state.drawdown_pct
        if dd <= self.config["drawdown_kill_pct"]:
            logger.warning(f"DRAWDOWN KILL: {dd:.1%} — all trading paused")
            return False
        if dd <= self.config["drawdown_pause_pct"]:
            logger.warning(f"Drawdown pause: {dd:.1%} — reduced trading")
            # In drawdown pause, require higher confluence
            if signal.confluence_score < 7:
                return False

        # No duplicate symbols
        for pos in self.state.positions:
            if pos["symbol"] == signal.symbol:
                return False

        # Per-symbol cooldown after stop loss (avoid re-entering a losing trade)
        cooldown_expiry = self.state.symbol_cooldowns.get(signal.symbol, 0)
        if time.time() < cooldown_expiry:
            remaining_h = (cooldown_expiry - time.time()) / 3600
            logger.debug(f"  {signal.symbol}: symbol cooldown ({remaining_h:.1f}h remaining)")
            return False

        # Position sizing
        atr_pct = signal.atr / signal.price if signal.price > 0 else 0.02
        kelly_size = self.state.bankroll * self.config["kelly_fraction"]

        # Volatility adjustment
        vol_adj = min(1.0, 0.02 / max(atr_pct, 0.005))

        # Quality multiplier
        quality_mult = {"A": 1.3, "B": 1.0, "C": 0.7, "D": 0.5}
        q_mult = quality_mult.get(signal.quality_grade, 0.7)

        # Regime scaling
        regime_scale = regime_params["position_scale"]

        # Drawdown scaling
        dd_scale = 1.0
        if dd <= self.config["drawdown_pause_pct"]:
            dd_scale = 0.5

        position_size = min(
            self.config["max_position_usd"],
            kelly_size * vol_adj * q_mult * regime_scale * dd_scale,
        )
        position_size = max(1.0, round(position_size, 2))

        # Ensure we can afford it
        if position_size > self.state.bankroll:
            position_size = max(1.0, round(self.state.bankroll * 0.9, 2))

        shares = position_size / signal.price
        pos_id = f"STK{self.state.total_trades + 1:04d}"

        position = {
            "id": pos_id,
            "symbol": signal.symbol,
            "entry_price": signal.price,
            "shares": round(shares, 6),
            "cost_usd": position_size,
            "signal_type": signal.signal_type,
            "confluence_score": signal.confluence_score,
            "quality_grade": signal.quality_grade,
            "regime": signal.regime,
            "rsi_at_entry": signal.rsi,
            "reasoning": signal.reasoning,
            "take_profit": signal.take_profit,
            "stop_loss": signal.stop_loss,
            "atr_at_entry": signal.atr,
            "peak_price": signal.price,
            "current_price": signal.price,
            "unrealized_pnl": 0.0,
            "breakeven_moved": False,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "components": signal.components,
            "sector": signal.sector,
            "bandit_weights": self.bandit.get_all_weights(),
        }

        self.state.positions.append(position)
        self.state.bankroll -= position_size
        self.state.total_trades += 1
        self.state.daily_trade_count += 1

        try:
            order = self.alpaca.place_order(
                symbol=signal.symbol,
                qty=0,
                side="buy",
                notional=position_size,
            )
            if order is None:
                raise RuntimeError(f"Alpaca order returned None for {signal.symbol}")
            position["order_id"] = order.get("id")

            # Update entry price from fill to avoid stale daily-bar prices
            fill_price = float(order.get("filled_avg_price") or 0)
            if fill_price <= 0:
                # Market orders fill fast — wait briefly then check
                time.sleep(1)
                order_id = order.get("id")
                if order_id:
                    check = self.alpaca._request(
                        "GET", f"{self.alpaca.base_url}/v2/orders/{order_id}"
                    )
                    if check:
                        fill_price = float(check.get("filled_avg_price") or 0)

            if fill_price > 0 and abs(fill_price - signal.price) / signal.price > 0.005:
                logger.info(
                    f"[{pos_id}] Fill price ${fill_price:.2f} vs signal ${signal.price:.2f} "
                    f"— updating entry/exits"
                )
                atr = signal.atr
                position["entry_price"] = fill_price
                position["peak_price"] = fill_price
                position["current_price"] = fill_price
                position["shares"] = position_size / fill_price
                position["take_profit"] = round(fill_price + atr * self.config["tp_atr_mult"], 2)
                position["stop_loss"] = round(fill_price - atr * self.config["sl_atr_mult"], 2)
        except Exception as e:
            logger.error(f"Alpaca order failed for {signal.symbol}: {e} — rolling back")
            self.state.positions.remove(position)
            self.state.bankroll += position_size
            self.state.total_trades -= 1
            self.state.daily_trade_count -= 1
            self._save_state()
            return False

        self._save_state()
        self._log_trade("OPEN", {**position, "price": position["entry_price"], "pnl": 0})

        logger.info(
            f"OPEN [{pos_id}] {signal.symbol} @ ${position['entry_price']:.2f} | "
            f"${position_size:.2f} | Grade:{signal.quality_grade} "
            f"Score:{signal.confluence_score} | "
            f"TP=${position['take_profit']:.2f} SL=${position['stop_loss']:.2f} | "
            f"Regime:{signal.regime} | {signal.sector}"
        )
        return True

    def update_prices(self):
        """Update current prices for all open positions."""
        if not self.state.positions:
            return

        symbols = [p["symbol"] for p in self.state.positions]

        # Batch fetch quotes
        quotes = self.alpaca.get_latest_quotes_multi(symbols)
        if not quotes or "quotes" not in quotes:
            # Fallback: individual fetches
            for pos in self.state.positions:
                quote = self.alpaca.get_latest_quote(pos["symbol"])
                if quote and "quote" in quote:
                    q = quote["quote"]
                    mid = (q.get("ap", 0) + q.get("bp", 0)) / 2
                    if mid > 0:
                        self._update_position_price(pos, mid)
                time.sleep(0.1)
        else:
            for pos in self.state.positions:
                q = quotes["quotes"].get(pos["symbol"])
                if q:
                    mid = (q.get("ap", 0) + q.get("bp", 0)) / 2
                    if mid > 0:
                        self._update_position_price(pos, mid)

        self._save_state()

    def _update_position_price(self, pos: dict, new_price: float):
        """Update a single position's price and P&L."""
        pos["current_price"] = new_price
        pos["unrealized_pnl"] = round(
            (new_price - pos["entry_price"]) * pos["shares"], 4
        )
        if new_price > pos.get("peak_price", pos["entry_price"]):
            pos["peak_price"] = new_price

    def check_exits(self):
        """Check all exit conditions for open positions."""
        to_close = []

        for pos in self.state.positions:
            current = pos.get("current_price", pos["entry_price"])
            entry = pos["entry_price"]
            atr = pos.get("atr_at_entry", entry * 0.02)

            # Calculate hold time
            try:
                opened = datetime.fromisoformat(pos["opened_at"])
                elapsed_h = (datetime.now(timezone.utc) - opened).total_seconds() / 3600
                hold_days = elapsed_h / 6.5  # ~6.5 trading hours/day
            except (ValueError, TypeError):
                elapsed_h = 0
                hold_days = 0

            # 1. TAKE PROFIT
            if current >= pos["take_profit"]:
                to_close.append((pos, "take_profit", current))
                continue

            # 2. STOP LOSS
            if current <= pos["stop_loss"]:
                to_close.append((pos, "stop_loss", current))
                continue

            # 3. BREAKEVEN STOP: move stop to entry after 1R profit
            if self.config["breakeven_at_1r"] and not pos.get("breakeven_moved"):
                one_r = atr * self.config["sl_atr_mult"]
                if current >= entry + one_r:
                    pos["stop_loss"] = round(entry + atr * 0.2, 2)
                    pos["breakeven_moved"] = True
                    logger.info(f"[{pos['id']}] Moved stop to breakeven+ ${pos['stop_loss']:.2f}")

            # 4. TRAILING STOP
            activation = atr * self.config["trailing_activation_atr"]
            if current >= entry + activation:
                trail_distance = atr * self.config["trailing_atr_mult"]
                peak = pos.get("peak_price", entry)
                trailing_stop = round(peak - trail_distance, 2)
                if trailing_stop > pos["stop_loss"]:
                    pos["stop_loss"] = trailing_stop
                    logger.debug(f"[{pos['id']}] Trail ratcheted to ${trailing_stop:.2f}")

            # 5. PROGRESSIVE STOP TIGHTENING
            if self.config.get("progressive_stop", False) and hold_days > 0:
                max_hold = self.config["max_hold_days"]
                start_pct = self.config.get("progressive_stop_start_pct", 0.5)
                end_mult = self.config.get("progressive_stop_end_mult", 0.33)
                hold_pct = hold_days / max_hold if max_hold > 0 else 0

                if hold_pct >= start_pct:
                    progress = min(1.0, (hold_pct - start_pct) / (1.0 - start_pct))
                    original_sl_dist = atr * self.config["sl_atr_mult"]
                    tight_sl_dist = original_sl_dist * end_mult
                    current_sl_dist = original_sl_dist - (original_sl_dist - tight_sl_dist) * progress
                    new_sl = round(entry - current_sl_dist, 2)
                    if new_sl > pos["stop_loss"]:
                        pos["stop_loss"] = new_sl
                        logger.debug(f"[{pos['id']}] Progressive stop -> ${new_sl:.2f} "
                                   f"({hold_pct:.0%} of max hold)")

            # 6. TIME EXIT: max hold days
            if hold_days >= self.config["max_hold_days"]:
                to_close.append((pos, "time_exit", current))
                continue

            # 7. EARNINGS EXIT: close if earnings imminent
            if self.earnings.is_within_days(
                pos["symbol"], self.config["earnings_exit_shield_days"]
            ):
                to_close.append((pos, "earnings_exit", current))
                continue

        # Close positions
        for pos, reason, price in to_close:
            self._close_position(pos, reason, price)

    def _close_position(self, pos: dict, reason: str, price: float):
        """Close a position and update state."""
        try:
            result = self.alpaca.close_position(pos["symbol"])
            if result is None:
                logger.error(f"Alpaca close failed for {pos['symbol']} — skipping state update")
                return
        except Exception as e:
            logger.error(f"Alpaca close error for {pos['symbol']}: {e} — skipping state update")
            return

        pnl = round((price - pos["entry_price"]) * pos["shares"], 4)
        pos["close_price"] = price
        pos["pnl"] = pnl
        pos["close_reason"] = reason
        pos["closed_at"] = datetime.now(timezone.utc).isoformat()

        # Calculate hold time
        try:
            opened = datetime.fromisoformat(pos["opened_at"])
            hold_h = (datetime.now(timezone.utc) - opened).total_seconds() / 3600
            pos["hold_hours"] = round(hold_h, 1)
        except (ValueError, TypeError):
            pos["hold_hours"] = 0

        # Update state
        self.state.bankroll += pos["cost_usd"] + pnl
        self.state.total_pnl = round(self.state.total_pnl + pnl, 4)
        self.state.daily_pnl = round(self.state.daily_pnl + pnl, 4)

        if pnl > 0:
            self.state.winning_trades += 1
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1
            self.state.last_loss_time = time.time()
            # Per-symbol cooldown: 4 hours after stop loss to avoid re-entering losers
            if reason == "stop_loss":
                self.state.symbol_cooldowns[pos["symbol"]] = time.time() + 4 * 3600

        if self.state.bankroll > self.state.peak_bankroll:
            self.state.peak_bankroll = self.state.bankroll

        # Update bandit
        won = pnl > 0
        for comp in pos.get("components", []):
            self.bandit.update(comp, won)

        # Move to closed trades (keep last 100 in state)
        self.state.closed_trades.append(pos)
        if len(self.state.closed_trades) > 100:
            self.state.closed_trades = self.state.closed_trades[-100:]

        # Remove from positions
        self.state.positions = [p for p in self.state.positions if p["id"] != pos["id"]]

        self._save_state()
        # 2026-05-15: write the exit reason into signal_type so closed-trade CSV is diagnosable
        self._log_trade("CLOSE", {**pos, "price": price, "signal_type": reason})
        self._log_analytics(pos)

        logger.info(
            f"CLOSE [{pos['id']}] {pos['symbol']} @ ${price:.2f} | "
            f"PnL: ${pnl:+.4f} | Reason: {reason} | "
            f"Hold: {pos.get('hold_hours', 0):.1f}h | "
            f"Bankroll: ${self.state.bankroll:.2f}"
        )

    # --- Auto-Tuning ---

    def _auto_tune(self):
        """Auto-tune parameters every N closed trades."""
        n = self.config["auto_tune_every_n_trades"]
        if len(self.state.closed_trades) < n:
            return
        if len(self.state.closed_trades) % n != 0:
            return

        recent = self.state.closed_trades[-n:]
        adjustments = []

        # 1. Per-sector win rates
        sector_stats = {}
        for t in recent:
            sector = t.get("sector", "unknown")
            won = t.get("pnl", 0) > 0
            if sector not in sector_stats:
                sector_stats[sector] = {"wins": 0, "total": 0}
            sector_stats[sector]["total"] += 1
            if won:
                sector_stats[sector]["wins"] += 1

        for sector, stats in sector_stats.items():
            if stats["total"] >= 5:
                wr = stats["wins"] / stats["total"]
                if wr < 0.30:
                    logger.info(f"AUTO-TUNE: Sector {sector} WR={wr:.0%} — consider reducing exposure")
                    adjustments.append(f"sector_{sector}_low_wr={wr:.0%}")

        # 2. Adjust confluence threshold based on rolling WR
        total_wr = sum(1 for t in recent if t.get("pnl", 0) > 0) / len(recent)
        if total_wr < 0.40:
            new_min = min(self.config["min_confluence_score"] + 1, 8)
            if new_min != self.config["min_confluence_score"]:
                self.config["min_confluence_score"] = new_min
                self.state.tuned_overrides["min_confluence_score"] = new_min
                adjustments.append(f"min_confluence={new_min}")
                logger.info(f"AUTO-TUNE: WR={total_wr:.0%}, raised confluence to {new_min}")
        elif total_wr > 0.65:
            new_min = max(self.config["min_confluence_score"] - 1, 4)
            if new_min != self.config["min_confluence_score"]:
                self.config["min_confluence_score"] = new_min
                self.state.tuned_overrides["min_confluence_score"] = new_min
                adjustments.append(f"min_confluence={new_min}")
                logger.info(f"AUTO-TUNE: WR={total_wr:.0%}, lowered confluence to {new_min}")

        # 3. Log tuning
        if adjustments:
            tuning_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trades_analyzed": n,
                "win_rate": round(total_wr, 3),
                "adjustments": adjustments,
                "sector_stats": sector_stats,
            }
            try:
                existing = []
                if os.path.exists(TUNING_LOG):
                    with open(TUNING_LOG) as f:
                        existing = json.load(f)
                existing.append(tuning_entry)
                with open(TUNING_LOG, "w") as f:
                    json.dump(existing, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not write tuning log: {e}")

        self._save_state()

    # --- Scan & Trade ---

    def scan_and_trade(self):
        """Main scan cycle: update regime, scan universe, trade signals."""
        self._check_daily_reset()

        # 1. Update market regime
        spy_data = self._get_spy_regime_data()
        vix = self._get_vix()
        regime = self.regime_detector.update(spy_data, vix)
        logger.info(f"Regime: {regime.value} | SPY ${spy_data.get('price', 0):.2f} | VIX {vix:.1f}")

        # 2. Update prices for open positions
        self.update_prices()

        # 3. Check exits
        self.check_exits()

        # 4. Drawdown kill check
        if self.state.drawdown_pct <= self.config["drawdown_kill_pct"]:
            logger.warning(f"DRAWDOWN KILL active ({self.state.drawdown_pct:.1%}) — closing all")
            for pos in list(self.state.positions):
                self._close_position(pos, "drawdown_kill", pos.get("current_price", pos["entry_price"]))
            return

        # 4b. Session-start cooldown (2026-05-15): live data showed 77% of losses
        # occurred in the first 90 min of the session (open volatility). Skip new
        # entries for the first 30 min while still managing exits.
        try:
            evt = self.time_until_market_event()
            if evt.get("is_open") and evt.get("next_event") == "close":
                minutes_left = evt.get("minutes", 0)
                # 6.5h session = 390 min; >360 min remaining means <30 min since open
                if minutes_left > 360:
                    logger.info(f"Session-start cooldown: {390 - minutes_left:.0f}min into session, skipping new entries")
                    return
        except Exception as e:
            logger.debug(f"Session cooldown check failed: {e}")

        # 5. Scan universe for new entries
        signals = []
        held = {p["symbol"] for p in self.state.positions}
        removed = set(self.state.removed_symbols)
        skip = held | removed | {"SPY", "UVXY"}

        # Static universe
        static_symbols = [s for s in ALL_SYMBOLS if s not in skip]

        # Dynamic screener DISABLED 2026-05-15: live data shows screened small caps
        # (RLYB, BAND, GHRS, PRIM, VELO, POET, MXL, HUT, LCID, MRDN, ATAI...) drove
        # most of the -$48 loss. Rolling back to the validated 110-symbol universe.
        dynamic_symbols = []
        all_scan = static_symbols

        # Pre-filter via snapshots: pick the ~60 hottest symbols
        scan_symbols = self._pre_filter_snapshots(all_scan)

        # Rate-limit friendly: process in batches
        batch_size = 10
        for i in range(0, len(scan_symbols), batch_size):
            batch = scan_symbols[i:i + batch_size]
            for symbol in batch:
                try:
                    ctx = self._build_market_context(symbol)
                    if not ctx:
                        continue
                    signal = self.detector.analyze(symbol, ctx)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.debug(f"Error analyzing {symbol}: {e}")
                time.sleep(0.15)  # Rate limiting

            # Check if we've hit position limits
            if len(self.state.positions) >= self.config["max_open_positions"]:
                break

        # 6. Sort by confluence score (highest first) and open positions
        signals.sort(key=lambda s: (s.confluence_score, s.quality_grade), reverse=True)

        opened = 0
        for signal in signals:
            if self.open_position(signal):
                opened += 1
            if len(self.state.positions) >= self.config["max_open_positions"]:
                break

        # 7. Auto-tune check
        self._auto_tune()

        self.state.last_scan = datetime.now(timezone.utc).isoformat()
        self._save_state()

        logger.info(
            f"Scan complete: {len(signals)} signals, {opened} opened | "
            f"Positions: {len(self.state.positions)} | "
            f"Bankroll: ${self.state.bankroll:.2f} | "
            f"P&L: ${self.state.total_pnl:+.2f} | "
            f"WR: {self.state.win_rate:.0%}"
        )

    # --- Summary ---

    def get_summary(self) -> str:
        """Get formatted status summary."""
        lines = [
            "=" * 60,
            "STOCK TRADER v1.0 STATUS",
            "=" * 60,
            f"Regime:        {self.regime_detector.current.value.upper()}",
            f"Bankroll:      ${self.state.bankroll:,.2f}",
            f"Starting:      ${self.state.starting_bankroll:,.2f}",
            f"Total P&L:     ${self.state.total_pnl:+,.2f} "
            f"({self.state.total_pnl / self.state.starting_bankroll * 100:+.1f}%)"
            if self.state.starting_bankroll > 0 else f"Total P&L:     ${self.state.total_pnl:+,.2f}",
            f"Daily P&L:     ${self.state.daily_pnl:+,.2f}",
            f"Win Rate:      {self.state.win_rate:.0%} ({self.state.winning_trades}/{len(self.state.closed_trades)})",
            f"Drawdown:      {self.state.drawdown_pct:.1%}",
            f"Peak:          ${self.state.peak_bankroll:,.2f}",
            f"Open Positions: {len(self.state.positions)}/{self.config['max_open_positions']}",
            f"Daily Trades:  {self.state.daily_trade_count}/{self.config['max_daily_trades']}",
            f"Total Trades:  {self.state.total_trades}",
            f"Last Scan:     {self.state.last_scan[:19] if self.state.last_scan else 'Never'}",
        ]

        if self.state.positions:
            lines.append("")
            lines.append("OPEN POSITIONS:")
            lines.append(f"{'Symbol':8} {'Entry':>10} {'Current':>10} {'P&L':>10} {'Score':>6} {'Grade':>6}")
            lines.append("-" * 52)
            for p in self.state.positions:
                lines.append(
                    f"{p['symbol']:8} "
                    f"${p['entry_price']:>9.2f} "
                    f"${p.get('current_price', p['entry_price']):>9.2f} "
                    f"${p.get('unrealized_pnl', 0):>+9.4f} "
                    f"{p['confluence_score']:>5} "
                    f"{p['quality_grade']:>5}"
                )

        market = self.time_until_market_event()
        if market["next_event"] != "unknown":
            status = "OPEN" if market["is_open"] else "CLOSED"
            mins = market["minutes"]
            hours = int(mins // 60)
            mins_rem = int(mins % 60)
            lines.append("")
            lines.append(f"Market: {status} | {market['next_event'].upper()} in {hours}h {mins_rem}m")

        lines.append("=" * 60)
        return "\n".join(lines)

    # --- Main Loop ---

    def run_loop(self):
        """Main trading loop — runs during market hours, sleeps off-hours."""
        logger.info("Stock Trader v1.0 starting")
        logger.info(f"Bankroll: ${self.state.bankroll:,.2f} | "
                    f"Universe: {len(ALL_SYMBOLS)} symbols | "
                    f"Interval: {self.config['scan_interval_sec']}s")

        while True:
            try:
                market = self.time_until_market_event()

                if market["is_open"]:
                    self.scan_and_trade()
                    sleep_sec = self.config["scan_interval_sec"]
                    logger.info(f"Next scan in {sleep_sec}s")
                    time.sleep(sleep_sec)
                else:
                    # Market closed — sleep until open
                    mins_to_open = market["minutes"]
                    if mins_to_open > 0:
                        # Log daily stats at close
                        if self.state.daily_trade_count > 0:
                            logger.info(
                                f"DAILY CLOSE | P&L: ${self.state.daily_pnl:+.2f} | "
                                f"Trades: {self.state.daily_trade_count} | "
                                f"Bankroll: ${self.state.bankroll:,.2f}"
                            )
                        # Sleep, but wake up every 30 min to check
                        sleep_sec = min(mins_to_open * 60, 1800)
                        hours = int(mins_to_open // 60)
                        mins_rem = int(mins_to_open % 60)
                        logger.info(f"Market closed. Opens in {hours}h {mins_rem}m. "
                                   f"Sleeping {sleep_sec // 60}m...")
                        time.sleep(sleep_sec)
                    else:
                        # Clock API issue — try again in 5 min
                        logger.warning("Could not determine market hours, retrying in 5m")
                        time.sleep(300)

            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self._save_state()
                break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(60)


# --- Setup & CLI ---

def setup_logging(verbose: bool = False):
    os.makedirs(DATA_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode="a"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Stock Trading Bot v1.0")
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--scan-once", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Reset state (fresh start)")
    parser.add_argument("--interval", type=int, default=None,
                       help="Scan interval in seconds (default: 900)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    setup_logging(verbose=args.verbose)

    # Load .env
    env_path = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    os.environ.setdefault(key.strip(), val.strip())

    if args.bankroll:
        STOCK_CONFIG["bankroll"] = args.bankroll
    if args.interval:
        STOCK_CONFIG["scan_interval_sec"] = args.interval

    if args.reset:
        for f in [STATE_FILE, TRADE_LOG, ANALYTICS_LOG]:
            if os.path.exists(f):
                os.remove(f)
        print("Stock Trader v1.0 reset.")
        return

    trader = StockTrader(STOCK_CONFIG)

    if args.status:
        trader._check_daily_reset()
        if trader.state.positions:
            if trader.is_market_open():
                trader.update_prices()
        print(trader.get_summary())
        return

    if args.scan_once:
        if not trader.is_market_open():
            logger.warning("Market is closed. Running scan anyway for testing...")
        trader.scan_and_trade()
        print(trader.get_summary())
        return

    trader.run_loop()


if __name__ == "__main__":
    main()
