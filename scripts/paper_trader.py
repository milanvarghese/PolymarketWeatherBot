"""Paper trading engine: simulates trades with fake money.

Tracks positions, monitors price changes, calculates P&L,
and builds a track record before going live.
"""

import csv
import json
import logging
import os
import statistics
import sys
import time
import argparse
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import requests
import schedule

from config import CONFIG, GAMMA_API_URL, PROXIES
from opportunity_scanner import MarketAnalyzer, Opportunity
from risk_manager import RiskManager

logger = logging.getLogger(__name__)

PAPER_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper_trading")
STATE_FILE = os.path.join(PAPER_DIR, "state.json")
TRADE_LOG = os.path.join(PAPER_DIR, "trades.csv")
PNL_LOG = os.path.join(PAPER_DIR, "daily_pnl.csv")


@dataclass
class PaperPosition:
    """A simulated open position."""
    id: str
    market_question: str
    market_id: str
    token_id: str
    category: str
    side: str               # YES or NO
    entry_price: float
    shares: float
    cost_usd: float
    estimated_prob: float
    edge_at_entry: float
    confidence: str
    reasoning: str
    end_date: str
    opened_at: str


@dataclass
class PaperState:
    """Paper trading account state."""
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
    last_reset_date: str = ""
    last_scan: str = ""

    @property
    def open_exposure(self) -> float:
        return sum(p["cost_usd"] for p in self.positions)

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.get("unrealized_pnl", 0) for p in self.positions)

    @property
    def win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        return sum(1 for t in self.closed_trades if t.get("pnl", 0) > 0) / len(self.closed_trades)

    @property
    def roi(self) -> float:
        if self.starting_bankroll <= 0:
            return 0.0
        return self.total_pnl / self.starting_bankroll


class PaperTrader:
    """Simulates trading on Polymarket with fake money."""

    def __init__(self, bankroll: float = None):
        os.makedirs(PAPER_DIR, exist_ok=True)
        had_state = os.path.exists(STATE_FILE)
        self.state = self._load_state()
        # Only set bankroll on fresh start (no existing state file)
        if bankroll is not None and not had_state:
            self.state.bankroll = bankroll
            self.state.starting_bankroll = bankroll
            self.state.peak_bankroll = bankroll
        self.analyzer = MarketAnalyzer()
        self.risk_manager = RiskManager()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        })
        if PROXIES:
            self.session.proxies.update(PROXIES)

        # Cooldown: market_id -> earliest UTC timestamp we can reopen
        self._reopen_cooldown: dict[str, float] = {}

    def _load_state(self) -> PaperState:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    content = f.read().strip()
                if not content:
                    logger.warning(f"State file empty, starting fresh")
                    return PaperState()
                data = json.loads(content)
                known = {f.name for f in PaperState.__dataclass_fields__.values()}
                return PaperState(**{k: v for k, v in data.items() if k in known})
            except json.JSONDecodeError as e:
                logger.error(f"State file corrupt: {e}. Remove {STATE_FILE} to reset.")
                raise SystemExit(f"FATAL: corrupt state file: {STATE_FILE}")
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
        return PaperState()

    def _save_state(self):
        # Atomic write: write to temp file then rename (prevents corruption on crash)
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
                    "timestamp", "action", "market_question", "category",
                    "side", "price", "shares", "cost_usd", "pnl",
                    "edge", "confidence", "market_id", "actual_outcome",
                ])
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                action,
                data.get("market_question", "")[:80],
                data.get("category", ""),
                data.get("side", ""),
                data.get("price", 0),
                data.get("shares", 0),
                data.get("cost_usd", 0),
                data.get("pnl", 0),
                data.get("edge", 0),
                data.get("confidence", ""),
                data.get("market_id", ""),
                data.get("actual_outcome", ""),
            ])

    def _get_drawdown_multiplier(self) -> float:
        """Graduated drawdown heat system (from dylanpersonguy bot).

        Instead of binary pause, reduce position size as drawdown increases.
        Normal -> Warning -> Critical -> Max -> Pause
        """
        open_exposure = sum(p.get("cost_usd", 0) for p in self.state.positions)
        current_value = self.state.bankroll + open_exposure
        dd = self.state.peak_bankroll - current_value
        dd_pct = dd / self.state.peak_bankroll if self.state.peak_bankroll > 0 else 0

        if dd_pct < CONFIG.get("drawdown_normal", 0.10):
            return 1.0    # Full sizing
        elif dd_pct < CONFIG.get("drawdown_warning", 0.15):
            logger.info(f"Drawdown WARNING ({dd_pct:.0%}): reducing to 75% sizing")
            return 0.75
        elif dd_pct < CONFIG.get("drawdown_critical", 0.20):
            logger.info(f"Drawdown CRITICAL ({dd_pct:.0%}): reducing to 50% sizing")
            return 0.50
        elif dd_pct < CONFIG.get("max_drawdown_pct", 0.25):
            logger.info(f"Drawdown MAX ({dd_pct:.0%}): reducing to 25% sizing")
            return 0.25
        else:
            logger.warning(f"Drawdown KILL ({dd_pct:.0%}): trading paused")
            return 0.0  # Don't trade

    def _get_dynamic_kelly(self) -> float:
        """Dynamic Kelly fraction based on rolling CLV of closed trades.

        Increase aggression only when edge is proven by CLV data.
        Research: Uhrin et al. (2021) - adaptive fractional Kelly.
        """
        closed = self.state.closed_trades
        if len(closed) < 10:
            return CONFIG.get("dynamic_kelly_default", 0.10)

        # Use last 50 trades (or all if fewer)
        recent = closed[-50:]
        clvs = [t.get("clv", 0) for t in recent if "clv" in t]

        if not clvs:
            return CONFIG.get("dynamic_kelly_default", 0.10)

        avg_clv = sum(clvs) / len(clvs)

        if avg_clv > 0.10:
            return CONFIG.get("dynamic_kelly_strong", 0.25)
        elif avg_clv > 0.05:
            return CONFIG.get("dynamic_kelly_moderate", 0.20)
        elif avg_clv > 0.02:
            return CONFIG.get("dynamic_kelly_marginal", 0.15)
        else:
            return CONFIG.get("dynamic_kelly_default", 0.10)

    def open_position(self, opp: Opportunity) -> bool:
        """Open a paper position from an opportunity signal."""
        # Daily circuit breakers
        if self.state.daily_pnl <= CONFIG.get("daily_loss_limit", -5.0):
            logger.warning(f"Skip: daily loss limit hit (${self.state.daily_pnl:.2f})")
            return False

        if self.state.daily_trade_count >= CONFIG.get("max_daily_trades", 30):
            logger.info(f"Skip: daily trade limit reached ({self.state.daily_trade_count})")
            return False

        # Graduated drawdown check
        dd_mult = self._get_drawdown_multiplier()
        if dd_mult <= 0:
            logger.info("Skip: drawdown kill switch active")
            return False

        # Risk checks
        if self.state.open_exposure + opp.kelly_size > self.state.bankroll * CONFIG["max_exposure_pct"]:
            logger.info(f"Skip: would exceed exposure limit")
            return False

        if len(self.state.positions) >= CONFIG["max_open_positions"]:
            logger.info(f"Skip: max positions reached")
            return False

        # Check for duplicate
        for pos in self.state.positions:
            if pos["market_id"] == opp.market_id:
                logger.info(f"Skip: already have position in this market")
                return False

        # Reopen cooldown: don't re-enter a market we just exited
        cooldown_until = self._reopen_cooldown.get(opp.market_id, 0)
        if time.time() < cooldown_until:
            mins_left = (cooldown_until - time.time()) / 60
            logger.info(f"Skip: reopen cooldown ({mins_left:.0f}m remaining) for {opp.market_question[:40]}")
            return False

        # 2026-05-15: time-to-resolution filter. Live data showed bot was buying
        # Dec-2026 crypto markets in March, locking capital for 9 months with
        # never-resolved closes at pnl=0. Cap "crypto" and "event" categories at
        # 30 days; "near_expiry" is already capped at 2 days; "dutch_book" exempt
        # (those have their own time horizon based on the arb structure).
        if opp.category in ("crypto", "event") and opp.end_date:
            try:
                end_dt = datetime.strptime(opp.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                days_left = (end_dt - datetime.now(timezone.utc)).total_seconds() / 86400
                max_days = CONFIG.get("max_days_to_resolution", 30)
                if days_left > max_days:
                    logger.info(f"Skip: {days_left:.0f}d to resolution > {max_days}d cap for {opp.market_question[:40]}")
                    return False
            except (ValueError, TypeError):
                pass

        # Dynamic Kelly + Drawdown multiplier
        dynamic_kelly = self._get_dynamic_kelly()
        adjusted_size = opp.kelly_size * dd_mult * (dynamic_kelly / CONFIG["kelly_fraction"]) if CONFIG["kelly_fraction"] > 0 else opp.kelly_size

        # Minimum order size on Polymarket is $1 (research says no minimum, but practical min)
        size_usd = max(1.0, min(adjusted_size, CONFIG["max_position_usd"], self.state.bankroll * 0.10))

        # Calculate shares
        shares = size_usd / opp.market_price

        pos_id = f"P{self.state.total_trades + 1:04d}"

        position = {
            "id": pos_id,
            "market_question": opp.market_question,
            "market_id": opp.market_id,
            "token_id": opp.token_id,
            "category": opp.category,
            "side": opp.side,
            "entry_price": opp.market_price,
            "shares": round(shares, 2),
            "cost_usd": round(size_usd, 2),
            "estimated_prob": opp.estimated_prob,
            "edge_at_entry": opp.edge,
            "confidence": opp.confidence,
            "reasoning": opp.reasoning,
            "end_date": opp.end_date,
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "current_price": opp.market_price,
            "unrealized_pnl": 0.0,
        }

        self.state.positions.append(position)
        self.state.bankroll -= size_usd
        self.state.total_trades += 1
        self.state.daily_trade_count += 1
        self._save_state()

        self._log_trade("OPEN", {
            **position,
            "price": opp.market_price,
            "pnl": 0,
            "edge": opp.edge,
        })

        logger.info(
            f"PAPER OPEN [{pos_id}]: {opp.side} {opp.market_question[:50]} "
            f"@ {opp.market_price:.1%} | ${size_usd:.2f} ({shares:.1f} shares) | "
            f"edge={opp.edge:.1%}"
        )
        return True

    def update_prices(self):
        """Fetch current prices for all open positions.

        BUG FIX: Uses query param ?condition_id= instead of path param /markets/{id}
        which was returning 404s. Gamma API uses conditionId as query param.
        """
        for pos in self.state.positions:
            try:
                # FIX: Use query param, not path param (was causing 404s)
                resp = self.session.get(
                    f"{GAMMA_API_URL}/markets",
                    params={"condition_id": pos["market_id"]},
                    timeout=15,
                )
                if resp.status_code != 200:
                    logger.debug(f"Price update HTTP {resp.status_code} for {pos['id']}")
                    continue

                data = resp.json()
                # API returns a list when using query params
                market = data[0] if isinstance(data, list) and data else data
                if not market:
                    continue

                tokens = market.get("tokens", [])

                # FIX: Match by outcome name, don't assume ordering
                for token in tokens:
                    outcome = token.get("outcome", "").lower()
                    if (pos["side"] == "YES" and outcome == "yes") or \
                       (pos["side"] == "NO" and outcome == "no"):
                        new_price = float(token.get("price", pos["entry_price"]))
                        old_price = pos.get("current_price", pos["entry_price"])
                        pos["current_price"] = new_price
                        pos["unrealized_pnl"] = round(
                            (new_price - pos["entry_price"]) * pos["shares"], 2
                        )
                        # Track peak price for trailing stop
                        peak = pos.get("peak_price", pos["entry_price"])
                        if new_price > peak:
                            pos["peak_price"] = new_price
                        break

                time.sleep(0.3)
            except Exception as e:
                logger.debug(f"Price update failed for {pos['id']}: {e}")

        self._save_state()

    def check_exits(self):
        """Smart multi-strategy exit logic (inspired by dylanpersonguy bot).

        6 exit strategies ranked by priority:
        1. Kill switch: price crashed to near-zero
        2. Edge reversal: current market price > our estimated probability
        3. Trailing stop: price dropped 10c from peak
        4. Time-based: within 48h of resolution (market gets efficient)
        5. Target hit: price >= 0.95 (near-certain resolution in our favor)
        6. Market expired: past end date
        """
        to_close = []

        for pos in self.state.positions:
            current_price = pos.get("current_price", pos["entry_price"])
            entry_price = pos["entry_price"]
            peak_price = pos.get("peak_price", entry_price)
            estimated_prob = pos.get("estimated_prob", 0.5)

            # 1. KILL SWITCH: price crashed (position is almost worthless)
            if current_price < 0.03:
                to_close.append((pos, "kill_switch", current_price))
                continue

            # 2. EDGE REVERSAL: market now prices meaningfully ABOVE our estimate
            #    Our edge has evaporated or reversed — get out
            #    Requires price to exceed estimate by at least 3c to avoid noise-driven loops
            #    (e.g. estimated_prob=0.85, price=0.86 is noise, price=0.89 is real reversal)
            if CONFIG.get("exit_edge_reversal") and current_price > estimated_prob + 0.03:
                to_close.append((pos, "edge_reversal", current_price))
                continue

            # 3. TRAILING STOP: price dropped from peak
            trailing_stop = CONFIG.get("exit_trailing_stop", 0.10)
            if peak_price - current_price >= trailing_stop and current_price < entry_price:
                to_close.append((pos, "trailing_stop", current_price))
                continue

            # 4. STOP LOSS: hard stop at 15c drop from entry
            if current_price < entry_price - 0.15:
                to_close.append((pos, "stop_loss", current_price))
                continue

            # 5. TIME-BASED EXIT: too close to resolution
            if pos.get("end_date"):
                try:
                    end_dt = datetime.strptime(pos["end_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    hours_left = (end_dt - datetime.now(timezone.utc)).total_seconds() / 3600
                    exit_hours = CONFIG.get("exit_time_hours", 48)
                    if 0 < hours_left < exit_hours and current_price < 0.85 and current_price <= pos["entry_price"] + 0.05:
                        to_close.append((pos, "time_exit", current_price))
                        continue
                except (ValueError, TypeError):
                    pass

            # 6. TARGET HIT: price near $1 (virtually certain win)
            if current_price >= 0.95:
                to_close.append((pos, "target_hit", current_price))
                continue

            # 7. EXPIRED: market past end date (use end-of-day to match scanner filter)
            if pos.get("end_date"):
                try:
                    end = datetime.strptime(pos["end_date"], "%Y-%m-%d")
                    end = end.replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
                    if datetime.now(timezone.utc) > end:
                        to_close.append((pos, "expired", current_price))
                except (ValueError, TypeError):
                    pass

        for pos, reason, exit_price in to_close:
            self.close_position(pos, exit_price, reason)

    def close_position(self, pos: dict, exit_price: float, reason: str = "manual"):
        """Close a paper position, record P&L, and track CLV.

        CLV (Closing Line Value) = exit_price - entry_price
        The gold standard for measuring genuine trading edge.
        Consistent positive CLV = real edge. Profit alone can be luck.
        """
        # Fee-aware P&L: Polymarket charges 2% on exit proceeds when winning
        raw_pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        fee_rate = CONFIG.get("fee_rate", 0.02)
        if raw_pnl > 0:
            # Fee = 2% of net winnings (profit only), not total exit proceeds
            fee = raw_pnl * fee_rate
            pnl = round(raw_pnl - fee, 2)
        else:
            pnl = round(raw_pnl, 2)

        # Track CLV for edge validation
        clv = round(exit_price - pos["entry_price"], 4)

        # 2026-05-15: only set actual_outcome on resolution-based exits.
        # Early exits (edge_reversal/trailing_stop/stop_loss/manual) are NOT
        # ground truth — recording pnl=0 closes as actual_outcome=0.0 was
        # corrupting the Brier score and dynamic-Kelly calibration.
        if reason in ("expired", "target_hit", "kill_switch"):
            actual_outcome = 1.0 if pnl > 0 else 0.0
        else:
            actual_outcome = None  # Position closed before resolution; outcome unknown

        self.state.bankroll += pos["cost_usd"] + pnl
        self.state.total_pnl += pnl
        self.state.daily_pnl += pnl

        if pnl > 0:
            self.state.winning_trades += 1

        if self.state.bankroll > self.state.peak_bankroll:
            self.state.peak_bankroll = self.state.bankroll

        # Move to closed trades with CLV tracking
        closed = {
            **pos,
            "exit_price": exit_price,
            "pnl": pnl,
            "clv": clv,
            "actual_outcome": actual_outcome,
            "reason": reason,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
        self.state.closed_trades.append(closed)
        if len(self.state.closed_trades) > 500:
            self.state.closed_trades = self.state.closed_trades[-500:]
        self.state.positions = [p for p in self.state.positions if p["id"] != pos["id"]]

        # Set 2-hour reopen cooldown to prevent open/close loops
        self._reopen_cooldown[pos["market_id"]] = time.time() + 7200

        self._save_state()
        self._log_trade("CLOSE", {
            **pos,
            "price": exit_price,
            "pnl": pnl,
            "actual_outcome": actual_outcome,
        })

        self.risk_manager.record_trade(pnl)

        logger.info(
            f"PAPER CLOSE [{pos['id']}]: {pos['side']} {pos['market_question'][:50]} "
            f"@ {exit_price:.1%} | PnL: ${pnl:+.2f} ({reason})"
        )

    def _check_daily_reset(self):
        """Reset daily counters at midnight UTC."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.state.last_reset_date != today:
            if self.state.last_reset_date:
                logger.info(f"New day ({today}): resetting daily P&L and trade count")
            self.state.daily_pnl = 0.0
            self.state.daily_trade_count = 0
            self.state.last_reset_date = today
            self._save_state()

    def scan_and_trade(self):
        """Run a full scan and open positions on good opportunities."""
        self._check_daily_reset()

        allowed, reason = self.risk_manager.can_trade(0)
        if not allowed:
            logger.warning(f"RiskManager blocked trading: {reason}")
            if self.state.positions:
                self.update_prices()
                self.check_exits()
            return

        logger.info("=" * 60)
        logger.info(f"PAPER SCAN | {datetime.now(timezone.utc).isoformat()}")
        logger.info(self.get_summary())

        # Update existing positions
        if self.state.positions:
            logger.info(f"Updating {len(self.state.positions)} open positions...")
            self.update_prices()
            self.check_exits()

        # Scan for new opportunities
        opportunities = self.analyzer.scan_all()

        opened = 0
        for opp in opportunities:
            if opp.edge >= CONFIG["entry_threshold"] and opp.confidence in ("high", "medium"):
                # Skip penny markets (< 5c) — longshot bias means these are overpriced
                # Skip overpriced markets (> 97c) — fees eat all margin
                if opp.market_price < 0.05 or opp.market_price > 0.97:
                    logger.debug(f"Skip: price {opp.market_price:.1%} outside tradeable range (5c-97c)")
                    continue
                if self.open_position(opp):
                    opened += 1
                if opened >= 10:
                    break

        self.state.last_scan = datetime.now(timezone.utc).isoformat()
        self._save_state()

        logger.info(f"SCAN COMPLETE | Opened {opened} new positions | "
                    f"{len(self.state.positions)} open | "
                    f"Bankroll: ${self.state.bankroll:.2f}")

    def get_summary(self) -> str:
        """Human-readable summary of paper trading state."""
        s = self.state
        lines = [
            f"{'='*60}",
            f"  PAPER TRADING DASHBOARD",
            f"{'='*60}",
            f"  Bankroll:     ${s.bankroll:.2f} (started ${s.starting_bankroll:.2f})",
            f"  Total P&L:    ${s.total_pnl:+.2f} ({s.roi:+.1%} ROI)",
            f"  Unrealized:   ${s.unrealized_pnl:+.2f}",
            f"  Peak:         ${s.peak_bankroll:.2f}",
            f"  Trades:       {s.total_trades} ({s.win_rate:.0%} win rate)",
            f"  Open:         {len(s.positions)} positions (${s.open_exposure:.2f} exposed)",
            f"  Last scan:    {s.last_scan[:19] if s.last_scan else 'never'}",
            f"{'='*60}",
        ]

        if s.positions:
            lines.append("")
            lines.append("  OPEN POSITIONS:")
            for pos in s.positions:
                upnl = pos.get("unrealized_pnl", 0)
                lines.append(
                    f"  [{pos['id']}] {pos['side']} {pos['market_question'][:45]}"
                )
                lines.append(
                    f"         Entry: {pos['entry_price']:.1%} -> "
                    f"Now: {pos.get('current_price', pos['entry_price']):.1%} | "
                    f"P&L: ${upnl:+.2f} | ${pos['cost_usd']:.2f}"
                )

        if s.closed_trades:
            lines.append("")
            recent = s.closed_trades[-5:]  # Last 5
            lines.append(f"  RECENT CLOSED ({len(s.closed_trades)} total):")
            for trade in reversed(recent):
                lines.append(
                    f"  [{trade['id']}] {trade['side']} {trade['market_question'][:45]}"
                )
                lines.append(
                    f"         {trade['entry_price']:.1%} -> {trade['exit_price']:.1%} | "
                    f"P&L: ${trade['pnl']:+.2f} ({trade['reason']})"
                )

        # Performance analytics (only when enough data)
        analytics = self.get_analytics()
        if analytics.get("status") == "ok":
            lines.append("")
            lines.append("  PERFORMANCE ANALYTICS (N={} trades):".format(analytics["n_trades"]))
            lines.append("  Brier Score:    {:.4f} (target <0.15, random=0.25)".format(
                analytics["brier_score"]))
            lines.append("  Avg CLV:        {:+.1%} ({:.0%} positive)".format(
                analytics["avg_clv"], analytics["pct_positive_clv"]))
            lines.append("  Profit Factor:  {:.1f}x".format(analytics["profit_factor"]))
            lines.append("  Edge Accuracy:  predicted {:.1%} vs actual {:.1%}".format(
                analytics["avg_predicted_edge"], analytics["avg_actual_return"]))

            cat_parts = []
            for cat, wr in sorted(analytics["category_win_rates"].items()):
                cat_parts.append("{} {:.0%} W".format(cat, wr))
            if cat_parts:
                lines.append("  By Category:    {}".format(" | ".join(cat_parts)))

        lines.append(f"{'='*60}")
        return "\n".join(lines)

    def get_analytics(self) -> dict:
        """Compute performance analytics from closed trades.

        Returns dict with Brier score, CLV stats, category breakdown,
        edge accuracy, and profit factor. Requires >= 5 closed trades.
        """
        closed = self.state.closed_trades
        if len(closed) < 5:
            return {"status": "insufficient_data", "n_trades": len(closed)}

        # Brier score: mean((estimated_prob - actual_outcome)^2)
        # Lower = better. 0.25 = random coin flip. Target < 0.15.
        brier_scores = []
        for t in closed:
            prob = t.get("estimated_prob", 0.5)
            outcome = t.get("actual_outcome", 1.0 if t.get("pnl", 0) > 0 else 0.0)
            brier_scores.append((prob - outcome) ** 2)
        brier = sum(brier_scores) / len(brier_scores)

        # CLV stats
        clvs = [t.get("clv", 0) for t in closed if "clv" in t]
        avg_clv = sum(clvs) / len(clvs) if clvs else 0
        pct_positive_clv = sum(1 for c in clvs if c > 0) / len(clvs) if clvs else 0
        median_clv = statistics.median(clvs) if clvs else 0

        # Win rate by category
        cat_stats = {}
        for t in closed:
            cat = t.get("category", "unknown")
            if cat not in cat_stats:
                cat_stats[cat] = {"wins": 0, "losses": 0}
            if t.get("pnl", 0) > 0:
                cat_stats[cat]["wins"] += 1
            else:
                cat_stats[cat]["losses"] += 1

        cat_win_rates = {}
        for cat, stats in cat_stats.items():
            total = stats["wins"] + stats["losses"]
            cat_win_rates[cat] = stats["wins"] / total if total > 0 else 0

        # Edge accuracy: avg predicted edge vs avg actual return
        predicted_edges = [t.get("edge_at_entry", 0) for t in closed]
        actual_returns = [t.get("pnl", 0) / t.get("cost_usd", 1) for t in closed if t.get("cost_usd", 0) > 0]
        avg_predicted_edge = sum(predicted_edges) / len(predicted_edges) if predicted_edges else 0
        avg_actual_return = sum(actual_returns) / len(actual_returns) if actual_returns else 0

        # Profit factor: gross wins / gross losses
        gross_wins = sum(t["pnl"] for t in closed if t.get("pnl", 0) > 0)
        gross_losses = abs(sum(t["pnl"] for t in closed if t.get("pnl", 0) < 0))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

        return {
            "status": "ok",
            "n_trades": len(closed),
            "brier_score": round(brier, 4),
            "avg_clv": round(avg_clv, 4),
            "median_clv": round(median_clv, 4),
            "pct_positive_clv": round(pct_positive_clv, 4),
            "avg_predicted_edge": round(avg_predicted_edge, 4),
            "avg_actual_return": round(avg_actual_return, 4),
            "profit_factor": round(profit_factor, 2),
            "category_win_rates": cat_win_rates,
        }

    def quick_update(self):
        """Fast price update + exit check for open positions only.

        Runs between full scans to catch exit signals faster.
        No market scanning, no new positions — just monitor existing.
        """
        if not self.state.positions:
            return
        logger.debug(f"Quick update: {len(self.state.positions)} positions")
        self.update_prices()
        self.check_exits()

    def backtest_historical(self):
        """Backtest against historical opportunity scans.

        Reads reports/opportunities.csv, fetches current price for each market,
        compares our estimated_prob at scan time vs where the market moved.
        Computes Brier score, edge accuracy, and category breakdown.
        """
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "opportunities.csv")
        if not os.path.exists(csv_path):
            print("No historical data found at reports/opportunities.csv")
            return

        # Read and deduplicate opportunities (same market_id = keep first scan)
        seen = {}
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = row.get("market_id", "")
                if mid and mid not in seen:
                    seen[mid] = row

        opps = list(seen.values())
        if not opps:
            print("No opportunities in CSV to backtest.")
            return

        print(f"Backtesting {len(opps)} unique opportunities...")
        print(f"{'='*70}")

        results = []
        for opp in opps:
            market_id = opp["market_id"]
            side = opp.get("side", "YES")
            est_prob = float(opp.get("estimated_prob", 0.5))
            entry_price = float(opp.get("market_price", 0))
            category = opp.get("category", "unknown")
            question = opp.get("question", "")[:60]

            # Fetch current market price
            try:
                resp = self.session.get(
                    f"{GAMMA_API_URL}/markets",
                    params={"condition_id": market_id},
                    timeout=15,
                )
                if resp.status_code != 200:
                    continue

                data = resp.json()
                market = data[0] if isinstance(data, list) and data else data
                if not market:
                    continue

                # Check if market resolved
                resolved = market.get("resolved", False)
                resolution = market.get("resolution", "")

                tokens = market.get("tokens", [])
                current_price = entry_price  # fallback
                for token in tokens:
                    outcome = token.get("outcome", "").upper()
                    if outcome == side:
                        current_price = float(token.get("price", entry_price))
                        break

                # Determine actual outcome
                if resolved:
                    if resolution.upper() == side:
                        actual_outcome = 1.0
                    elif resolution:
                        actual_outcome = 0.0
                    else:
                        actual_outcome = current_price
                else:
                    # Not resolved: use current market price as proxy
                    actual_outcome = current_price

                clv = current_price - entry_price
                brier = (est_prob - actual_outcome) ** 2

                results.append({
                    "question": question,
                    "category": category,
                    "side": side,
                    "est_prob": est_prob,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "actual_outcome": actual_outcome,
                    "resolved": resolved,
                    "clv": clv,
                    "brier": brier,
                })

                status = "RESOLVED" if resolved else f"now {current_price:.1%}"
                print(f"  [{category:>12}] {question}")
                print(f"    est={est_prob:.1%} entry={entry_price:.1%} {status} "
                      f"| CLV={clv:+.1%} | Brier={brier:.4f}")

                time.sleep(0.3)  # Rate limit

            except Exception as e:
                logger.debug(f"Backtest fetch failed for {market_id}: {e}")
                continue

        if not results:
            print("\nNo results -- could not fetch any market data.")
            return

        # Aggregate stats
        n = len(results)
        avg_brier = sum(r["brier"] for r in results) / n
        avg_clv = sum(r["clv"] for r in results) / n
        pct_pos_clv = sum(1 for r in results if r["clv"] > 0) / n
        resolved_count = sum(1 for r in results if r["resolved"])

        # Category breakdown
        cat_stats = {}
        for r in results:
            cat = r["category"]
            if cat not in cat_stats:
                cat_stats[cat] = {"briers": [], "clvs": [], "n": 0}
            cat_stats[cat]["briers"].append(r["brier"])
            cat_stats[cat]["clvs"].append(r["clv"])
            cat_stats[cat]["n"] += 1

        print(f"\n{'='*70}")
        print(f"  BACKTEST RESULTS ({n} opportunities, {resolved_count} resolved)")
        print(f"{'='*70}")
        print(f"  Brier Score:    {avg_brier:.4f} (target <0.15, random=0.25)")
        print(f"  Avg CLV:        {avg_clv:+.1%} ({pct_pos_clv:.0%} positive)")
        print(f"")
        print(f"  BY CATEGORY:")
        for cat, stats in sorted(cat_stats.items()):
            cat_brier = sum(stats["briers"]) / len(stats["briers"])
            cat_clv = sum(stats["clvs"]) / len(stats["clvs"])
            print(f"    {cat:>12}: Brier={cat_brier:.4f} | CLV={cat_clv:+.1%} | N={stats['n']}")
        print(f"{'='*70}")

    def run_loop(self, interval_min: int = 30, update_interval_min: int = 1):
        """Run paper trading on a schedule.

        Two loops:
        - Full scan every `interval_min` minutes (find new opportunities)
        - Quick price update every `update_interval_min` minutes (catch exits fast)
        """
        logger.info(f"Starting paper trading loop (scan every {interval_min}m, "
                     f"price updates every {update_interval_min}m)")
        logger.info(self.get_summary())

        # Run immediately
        self.scan_and_trade()

        # Schedule full scan + quick updates
        schedule.every(interval_min).minutes.do(self.scan_and_trade)
        if update_interval_min < interval_min:
            schedule.every(update_interval_min).minutes.do(self.quick_update)

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Paper trading stopped")
            print(self.get_summary())


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(PAPER_DIR, "paper_trading.log"),
                mode="a",
            ),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Paper Trading Simulator")
    parser.add_argument("--bankroll", type=float, default=None,
                       help="Starting bankroll (default: $50)")
    parser.add_argument("--scan-once", action="store_true",
                       help="Run one scan and exit")
    parser.add_argument("--status", action="store_true",
                       help="Show current status")
    parser.add_argument("--update", action="store_true",
                       help="Update prices for open positions")
    parser.add_argument("--reset", action="store_true",
                       help="Reset paper trading (start fresh)")
    parser.add_argument("--interval", type=int, default=5,
                       help="Full scan interval in minutes (default: 5)")
    parser.add_argument("--update-interval", type=int, default=1,
                       help="Price update interval in minutes (default: 1)")
    parser.add_argument("--close", type=str, default=None,
                       help="Close a position by ID (e.g. P0001)")
    parser.add_argument("--backtest", action="store_true",
                       help="Backtest against historical opportunity scans")
    args = parser.parse_args()

    os.makedirs(PAPER_DIR, exist_ok=True)
    setup_logging()

    if args.reset:
        for f in [STATE_FILE, TRADE_LOG]:
            if os.path.exists(f):
                os.remove(f)
        print("Paper trading reset. Starting fresh.")
        return

    trader = PaperTrader(bankroll=args.bankroll)

    if args.status:
        if trader.state.positions:
            trader.update_prices()
        print(trader.get_summary())
        return

    if args.update:
        trader.update_prices()
        trader.check_exits()
        print(trader.get_summary())
        return

    if args.backtest:
        trader.backtest_historical()
        return

    if args.close:
        pos = None
        for p in trader.state.positions:
            if p["id"] == args.close.upper():
                pos = p
                break
        if pos:
            trader.close_position(pos, pos.get("current_price", pos["entry_price"]), "manual")
            print(f"Closed {args.close}")
        else:
            print(f"Position {args.close} not found")
        print(trader.get_summary())
        return

    if args.scan_once:
        trader.scan_and_trade()
        print(trader.get_summary())
        return

    # Run continuous loop
    trader.run_loop(interval_min=args.interval, update_interval_min=args.update_interval)


if __name__ == "__main__":
    main()
