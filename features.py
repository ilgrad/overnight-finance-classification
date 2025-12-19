#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming feature builder for LOB (uses only past data).
No aggregations or windows looking into the future.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import sqrt
from typing import Deque, Dict, List, Tuple

import numpy as np

# Rolling windows (use only past values)
ROLL_WINDOWS = [5, 20, 100]
TICK_WINDOWS = [3, 5, 10, 20, 50]
RANGE_WINDOWS = [5, 20, 50]
OFI_LEVELS = 10  # OFI up to 10 levels
OFI_CUMSUM_WINDOWS = [5, 10, 20]  # windows for OFI cumsum
KAMA_CONFIGS = [(50, 5, 30), (300, 30, 150), (600, 100, 300)]  # (n, fast, slow)
KELTNER_WINDOWS = [10, 20, 50]
ETNA_SMOOTH_WINDOW = 49  # window size for anomaly smoothing
ETNA_ALPHA = 3.0  # coefficient for outlier detection


@dataclass
class RollingStat:
    window: int
    values: Deque[float] = field(default_factory=deque)
    s: float = 0.0
    s2: float = 0.0

    def push(self, v: float):
        self.values.append(v)
        self.s += v
        self.s2 += v * v
        if len(self.values) > self.window:
            old = self.values.popleft()
            self.s -= old
            self.s2 -= old * old

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return self.s / len(self.values)

    def std(self) -> float:
        n = len(self.values)
        if n == 0:
            return 0.0
        m = self.mean()
        var = max(self.s2 / n - m * m, 0.0)
        return sqrt(var)


@dataclass
class RollingCounter:
    window: int
    values: Deque[int] = field(default_factory=deque)
    s: int = 0

    def push(self, v: int):
        self.values.append(v)
        self.s += v
        if len(self.values) > self.window:
            old = self.values.popleft()
            self.s -= old

    def sum(self) -> int:
        return self.s


class StatefulFeatureBuilder:
    """Streaming feature calculator. Input: dict of columns for one row (raw LOB row)."""

    def __init__(self):
        self.prev_mid: float | None = None
        self.prev_mid_lag3: Deque[float] = deque(maxlen=3)
        self.prev_spread: float | None = None
        self.prev_imb: float | None = None
        self.prev_tick: int = 0
        self.tick_initialized = False

        self.roll_mid = {w: RollingStat(w) for w in ROLL_WINDOWS}
        self.roll_spread = {w: RollingStat(w) for w in ROLL_WINDOWS}
        self.roll_spread_rel = {w: RollingStat(w) for w in ROLL_WINDOWS}
        self.roll_imb_mean = {w: RollingStat(w) for w in ROLL_WINDOWS}
        self.roll_imb_std = {w: RollingStat(w) for w in ROLL_WINDOWS}
        self.roll_microprice = {w: RollingStat(w) for w in ROLL_WINDOWS}
        self.roll_mid1 = {w: RollingStat(w) for w in ROLL_WINDOWS}
        self.tick_changes = {w: RollingCounter(w) for w in TICK_WINDOWS}

        # OFI stores past qty/price by levels
        self.prev_bid_px = {}
        self.prev_ask_px = {}
        self.prev_bid_qty = {}
        self.prev_ask_qty = {}

        # EMA (short/long) for mid and imbalance
        self.ema_mid_fast = None
        self.ema_mid_slow = None
        self.ema_imb_fast = None
        self.ema_imb_slow = None
        self.alpha_mid_fast = 2 / (5 + 1)  # window ~5
        self.alpha_mid_slow = 2 / (20 + 1)  # window ~20
        self.alpha_imb_fast = 2 / (5 + 1)
        self.alpha_imb_slow = 2 / (20 + 1)

        # Rolling range storage
        self.range_mid = {w: deque(maxlen=w) for w in RANGE_WINDOWS}
        self.range_spread = {w: deque(maxlen=w) for w in RANGE_WINDOWS}
        self.range_microprice = {w: deque(maxlen=w) for w in RANGE_WINDOWS}

        # Additional states
        self.prev_mid1: float | None = None
        self.prev_mid1_prev: float | None = None
        self.prev_velocity: float | None = None
        self.imb1_history: Deque[float] = deque(maxlen=3)
        # Change-rate counters
        self.chg_counters = {
            "mid_price_mean": {w: RollingCounter(w) for w in [3, 5, 10]},
            "volume_imbalance": {w: RollingCounter(w) for w in [3, 5, 10]},
        }
        self.prev_vals = {
            "mid_price_mean": None,
            "volume_imbalance": None,
        }
        # EWMA states for spread_rel_mean and microprice_1
        self.ewma_spread_rel = {"span5": None, "span20": None}
        self.ewma_micro = {"span5": None, "span20": None}
        # MAD via EWMA(abs(x - ewma))
        self.ewma_spread_rel_mad = {"span5": None}
        self.ewma_micro_mad = {"span5": None}
        self.alpha_span5 = 2 / (5 + 1)
        self.alpha_span20 = 2 / (20 + 1)

        # ========== KAMA (Kaufman Adaptive Moving Average) ==========
        # For each config (n, fast, slow) we store:
        # - history: deque for change and volatility calculation
        # - kama_value: current KAMA value
        self.kama_states = {}
        for n, fast, slow in KAMA_CONFIGS:
            key = f"kama_{n}_{fast}_{slow}"
            self.kama_states[key] = {
                "history": deque(maxlen=n + 1),
                "kama": None,
                "n": n,
                "fast_alpha": 2 / (fast + 1),
                "slow_alpha": 2 / (slow + 1),
            }

        # ========== Keltner Channels ==========
        # EMA for mid_price_mean on different windows
        self.keltner_ema = {w: None for w in KELTNER_WINDOWS}
        self.keltner_alpha = {w: 2 / (w + 1) for w in KELTNER_WINDOWS}
        # ATR (True Range rolling mean)
        self.keltner_atr = {w: RollingStat(w) for w in KELTNER_WINDOWS}
        self.prev_mid_for_atr: float | None = None

        # ========== OFI Cumsum ==========
        # Rolling sum OFI for levels 1-3
        self.ofi_cumsum = {
            lvl: {w: RollingStat(w) for w in OFI_CUMSUM_WINDOWS}
            for lvl in range(1, 4)
        }

        # ========== Additional MAD for volume_imbalance ==========
        self.ewma_imb = {"span3": None, "span5": None, "span10": None, "span20": None}
        self.ewma_imb_mad = {"span3": None, "span5": None, "span10": None}
        self.alpha_span3 = 2 / (3 + 1)
        self.alpha_span10 = 2 / (10 + 1)

        # ========== New features from 0.461 solution ==========
        # Mid price ratio lags (for ratio_5, 10, 20, 50, 100)
        self.mid_history = deque(maxlen=101)  # store 101 values for ratio_100
        
        # Keltner position history (for change)
        self.keltner_pos_history = {w: deque(maxlen=6) for w in KELTNER_WINDOWS}  # 6 for diff(5)
        
        # Spread history (for pct_change 1-5)
        self.spread_history = deque(maxlen=6)
        
        # Volume ratio history (bid/ask volume ratio)
        self.vol_ratio_history = deque(maxlen=6)
        
        # KAMA values history (for cross signal)
        self.kama_history = {key: deque(maxlen=3) for key in self.kama_states.keys()}
        
        # Volume sums history
        self.prev_bid_vol_sum: float | None = None
        self.prev_ask_vol_sum: float | None = None
        
        # ========== Log returns history ==========
        self.mid_log_history = deque(maxlen=1001)  # for log_ret on 1000 steps
        
        # ========== Mid price position (rolling min/max) ==========
        self.mid_minmax_history = {w: deque(maxlen=w+1) for w in [3, 5, 10, 20, 50, 100, 1000]}
        
        # ========== ETNA-like Anomaly Smoothing ==========
        # History for outlier smoothing (MedianOutliers-like)
        self.etna_mid_history = deque(maxlen=ETNA_SMOOTH_WINDOW)
        self.etna_imb_history = deque(maxlen=ETNA_SMOOTH_WINDOW)
        self.etna_micro_history = deque(maxlen=ETNA_SMOOTH_WINDOW)
        self.etna_spread_history = deque(maxlen=ETNA_SMOOTH_WINDOW)

    @staticmethod
    def _core(row: Dict[str, float]) -> Dict[str, float]:
        # Level 1
        bid1 = row["bid_price_1"]
        ask1 = row["ask_price_1"]
        bq1 = row["bid_qty_1"]
        aq1 = row["ask_qty_1"]

        mid1 = (bid1 + ask1) / 2.0
        micro1 = (ask1 * bq1 + bid1 * aq1) / (bq1 + aq1 + 1e-9)

        bid_vol = 0.0
        ask_vol = 0.0
        mid_sum = 0.0
        spread_sum = 0.0

        for i in range(1, 21):
            bp = row[f"bid_price_{i}"]
            ap = row[f"ask_price_{i}"]
            bq = row[f"bid_qty_{i}"]
            aq = row[f"ask_qty_{i}"]
            bid_vol += bq
            ask_vol += aq
            mid_sum += (bp + ap) / 2.0
            spread_sum += ap - bp

        levels = 20.0
        mid_mean = mid_sum / levels
        spread_abs_mean = spread_sum / levels
        spread_rel_mean = spread_abs_mean / (mid_mean + 1e-9)
        volume_imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
        volume_imbalance_1 = (row["bid_qty_1"] - row["ask_qty_1"]) / (
            row["bid_qty_1"] + row["ask_qty_1"] + 1e-9
        )

        return {
            "mid_price_mean": mid_mean,
            "spread_abs_mean": spread_abs_mean,
            "spread_rel_mean": spread_rel_mean,
            "volume_imbalance": volume_imbalance,
            "volume_imbalance_1": volume_imbalance_1,
            "microprice_1": micro1,
            "mid_price_1": mid1,
            "bid_volume_sum": bid_vol,
            "ask_volume_sum": ask_vol,
        }

    def _ofi_features(self, row: Dict[str, float], levels: int = OFI_LEVELS) -> Dict[str, float]:
        """OFI on multiple levels (past values only)."""
        feats = {}
        total_ofi = 0.0
        for i in range(1, levels + 1):
            bp = row[f"bid_price_{i}"]
            ap = row[f"ask_price_{i}"]
            bq = row[f"bid_qty_{i}"]
            aq = row[f"ask_qty_{i}"]

            pbp = self.prev_bid_px.get(i, bp)
            pap = self.prev_ask_px.get(i, ap)
            pbq = self.prev_bid_qty.get(i, bq)
            paq = self.prev_ask_qty.get(i, aq)

            ofi = 0.0
            # Bid side
            if bp > pbp:
                ofi += bq
            elif bp == pbp:
                ofi += bq - pbq
            # Ask side
            if ap < pap:
                ofi -= aq
            elif ap == pap:
                ofi -= aq - paq

            feats[f"ofi_{i}"] = ofi
            total_ofi += ofi

            # update prev
            self.prev_bid_px[i] = bp
            self.prev_ask_px[i] = ap
            self.prev_bid_qty[i] = bq
            self.prev_ask_qty[i] = aq

        feats["ofi_total_1_5"] = total_ofi
        return feats

    def _ema_update(self, value: float, prev: float | None, alpha: float) -> float:
        return value if prev is None else alpha * value + (1 - alpha) * prev

    def build_row(self, row: Dict[str, float]) -> Dict[str, float]:
        f = self._core(row)
        f.update(self._ofi_features(row))

        mid = f["mid_price_mean"]
        spread = f["spread_abs_mean"]
        spread_rel = f["spread_rel_mean"]
        imb = f["volume_imbalance"]
        micro = f["microprice_1"]
        mid1 = f["mid_price_1"]

        # Lags and log-returns (past only)
        f["mid_price_mean_lag1"] = self.prev_mid if self.prev_mid is not None else 0.0

        if self.prev_mid is None:
            f["log_ret_mid_price_mean_1"] = 0.0
        else:
            f["log_ret_mid_price_mean_1"] = np.log(mid / (self.prev_mid + 1e-9))

        if len(self.prev_mid_lag3) < 3:
            f["log_ret_mid_price_mean_3"] = 0.0
        else:
            f["log_ret_mid_price_mean_3"] = np.log(
                mid / (self.prev_mid_lag3[0] + 1e-9)
            )

        if self.prev_spread is None:
            f["log_ret_spread_1"] = 0.0
        else:
            if spread <= 0.0 or self.prev_spread <= 0.0:
                f["log_ret_spread_1"] = 0.0
            else:
                ratio = spread / (self.prev_spread + 1e-9)
                f["log_ret_spread_1"] = np.log(max(ratio, 1e-9))

        if self.prev_imb is None:
            f["volume_imbalance_diff_1"] = 0.0
        else:
            f["volume_imbalance_diff_1"] = imb - self.prev_imb

        # Additional ratios
        f["imbalance_ratio"] = (
            (f["volume_imbalance_1"] / (imb + 1e-9)) if self.prev_imb is not None else 0.0
        )

        # Rolling (past values only - update states after calculation)
        for w, r in self.roll_mid.items():
            f[f"mid_price_mean_rolling_std_{w}"] = r.std()
            f[f"mid_price_mean_rolling_mean_{w}"] = r.mean()
        for w, r in self.roll_spread.items():
            f[f"spread_abs_mean_rolling_std_{w}"] = r.std()
            f[f"spread_abs_mean_rolling_mean_{w}"] = r.mean()
        for w, r in self.roll_spread_rel.items():
            f[f"spread_rel_mean_rolling_std_{w}"] = r.std()
            f[f"spread_rel_mean_rolling_mean_{w}"] = r.mean()
        for w, r in self.roll_imb_mean.items():
            f[f"volume_imbalance_rolling_mean_{w}"] = r.mean()
        for w, r in self.roll_imb_std.items():
            f[f"volume_imbalance_rolling_std_{w}"] = r.std()
        for w, r in self.roll_microprice.items():
            f[f"microprice_1_rolling_std_{w}"] = r.std()
            f[f"microprice_1_rolling_mean_{w}"] = r.mean()
        for w, r in self.roll_mid1.items():
            f[f"mid_price_1_rolling_std_{w}"] = r.std()
            f[f"mid_price_1_rolling_mean_{w}"] = r.mean()

        # Tick direction and sign changes
        tick = 0
        if self.prev_mid is not None:
            if mid > self.prev_mid:
                tick = 1
            elif mid < self.prev_mid:
                tick = -1
        f["tick_direction"] = tick
        # Sign change counted between prev tick and current
        change_flag = 1 if self.tick_initialized and tick != self.prev_tick else 0
        for w, rc in self.tick_changes.items():
            rc.push(change_flag)
            f[f"tick_direction_changes_{w}"] = rc.sum()

        # EMA signals
        self.ema_mid_fast = self._ema_update(mid, self.ema_mid_fast, self.alpha_mid_fast)
        self.ema_mid_slow = self._ema_update(mid, self.ema_mid_slow, self.alpha_mid_slow)
        self.ema_imb_fast = self._ema_update(imb, self.ema_imb_fast, self.alpha_imb_fast)
        self.ema_imb_slow = self._ema_update(imb, self.ema_imb_slow, self.alpha_imb_slow)

        f["ema_mid_fast"] = 0.0 if self.ema_mid_fast is None else self.ema_mid_fast
        f["ema_mid_slow"] = 0.0 if self.ema_mid_slow is None else self.ema_mid_slow
        f["ema_mid_diff"] = f["ema_mid_fast"] - f["ema_mid_slow"]
        f["ema_imb_fast"] = 0.0 if self.ema_imb_fast is None else self.ema_imb_fast
        f["ema_imb_slow"] = 0.0 if self.ema_imb_slow is None else self.ema_imb_slow
        f["ema_imb_diff"] = f["ema_imb_fast"] - f["ema_imb_slow"]

        # Rolling range and positions
        for w, dq in self.range_mid.items():
            dq.append(mid)
            rmin, rmax = min(dq), max(dq)
            f[f"mid_price_range_{w}"] = rmax - rmin
            f[f"mid_price_pos_{w}"] = (mid - rmin) / (rmax - rmin + 1e-9)

        for w, dq in self.range_spread.items():
            dq.append(spread)
            rmin, rmax = min(dq), max(dq)
            f[f"spread_range_{w}"] = rmax - rmin
            f[f"spread_pos_{w}"] = (spread - rmin) / (rmax - rmin + 1e-9)

        # Bollinger-like width based on std and ema
        for w in [5, 20]:
            std_mid = self.roll_mid[w].std()
            ema_mid = self.ema_mid_slow if w >= 20 else self.ema_mid_fast
            if ema_mid is None:
                ema_mid = mid
            f[f"bollinger_width_mid_{w}"] = 2 * std_mid / (abs(ema_mid) + 1e-9)

        # ATR-like metrics for spread
        for w in [5, 20]:
            f[f"atr_spread_{w}"] = self.roll_spread[w].mean()

        # Range/pos for microprice
        for w, dq in self.range_microprice.items():
            dq.append(micro)
            rmin, rmax = min(dq), max(dq)
            f[f"microprice_range_{w}"] = rmax - rmin
            f[f"microprice_pos_{w}"] = (micro - rmin) / (rmax - rmin + 1e-9)

        # EWMA and MAD for spread_rel_mean and microprice_1 (span 5/20 for EWMA, MAD span5)
        # Spread_rel_mean
        sr = spread_rel
        # EWMA span5
        prev = self.ewma_spread_rel["span5"]
        self.ewma_spread_rel["span5"] = sr if prev is None else self.alpha_span5 * sr + (1 - self.alpha_span5) * prev
        # EWMA span20
        prev = self.ewma_spread_rel["span20"]
        self.ewma_spread_rel["span20"] = sr if prev is None else self.alpha_span20 * sr + (1 - self.alpha_span20) * prev
        f["ewma_spread_rel_5"] = self.ewma_spread_rel["span5"] if self.ewma_spread_rel["span5"] is not None else 0.0
        f["ewma_spread_rel_20"] = self.ewma_spread_rel["span20"] if self.ewma_spread_rel["span20"] is not None else 0.0
        # MAD span5
        if self.ewma_spread_rel["span5"] is None:
            self.ewma_spread_rel_mad["span5"] = 0.0
        else:
            mad_prev = self.ewma_spread_rel_mad["span5"]
            delta = abs(sr - self.ewma_spread_rel["span5"])
            self.ewma_spread_rel_mad["span5"] = delta if mad_prev is None else self.alpha_span5 * delta + (1 - self.alpha_span5) * mad_prev
        f["ewma_spread_rel_mad_5"] = self.ewma_spread_rel_mad["span5"] if self.ewma_spread_rel_mad["span5"] is not None else 0.0

        # Microprice_1
        mc = micro
        prev = self.ewma_micro["span5"]
        self.ewma_micro["span5"] = mc if prev is None else self.alpha_span5 * mc + (1 - self.alpha_span5) * prev
        prev = self.ewma_micro["span20"]
        self.ewma_micro["span20"] = mc if prev is None else self.alpha_span20 * mc + (1 - self.alpha_span20) * prev
        f["ewma_micro_5"] = self.ewma_micro["span5"] if self.ewma_micro["span5"] is not None else 0.0
        f["ewma_micro_20"] = self.ewma_micro["span20"] if self.ewma_micro["span20"] is not None else 0.0
        # MAD span5
        if self.ewma_micro["span5"] is None:
            self.ewma_micro_mad["span5"] = 0.0
        else:
            mad_prev = self.ewma_micro_mad["span5"]
            delta = abs(mc - self.ewma_micro["span5"])
            self.ewma_micro_mad["span5"] = delta if mad_prev is None else self.alpha_span5 * delta + (1 - self.alpha_span5) * mad_prev
        f["ewma_micro_mad_5"] = self.ewma_micro_mad["span5"] if self.ewma_micro_mad["span5"] is not None else 0.0

        # Flags for local max/min breakout for mid_price_mean on windows 5/20/50
        for w, dq in self.range_mid.items():
            cur_max = max(dq) if dq else mid
            cur_min = min(dq) if dq else mid
            f[f"mid_price_break_max_{w}"] = int(mid > cur_max)
            f[f"mid_price_break_min_{w}"] = int(mid < cur_min)

        # Slope/gradient for bid/ask prices (levels 1 and 20, averaged gradient)
        bid_last = row["bid_price_20"]
        ask_last = row["ask_price_20"]
        bid_first = row["bid_price_1"]
        ask_first = row["ask_price_1"]
        f["bid_price_slope"] = bid_last - bid_first
        f["ask_price_slope"] = ask_last - ask_first
        # Gradient as avg increase per level
        f["bid_price_gradient"] = (bid_last - bid_first) / 19.0
        f["ask_price_gradient"] = (ask_last - ask_first) / 19.0
        f["price_slope_imbalance"] = f["bid_price_slope"] - f["ask_price_slope"]
        f["price_gradient_imbalance"] = f["bid_price_gradient"] - f["ask_price_gradient"]

        # Change-rate for mid_price_mean and volume_imbalance on windows 3/5/10
        for key, counters in self.chg_counters.items():
            prev_val = self.prev_vals[key]
            cur_val = mid if key == "mid_price_mean" else imb
            changed = 0 if prev_val is None else int(cur_val != prev_val)
            for w, rc in counters.items():
                rc.push(changed)
                f[f"{key}_chg_rate_{w}"] = rc.sum()
            self.prev_vals[key] = cur_val

        # Velocity / acceleration for mid_price_1
        if self.prev_mid1 is None or self.prev_mid1_prev is None:
            f["mid_price_1_velocity"] = 0.0
            f["mid_price_1_acceleration"] = 0.0
        else:
            vel = (mid1 - self.prev_mid1) / (self.prev_mid1 + 1e-9)
            f["mid_price_1_velocity"] = vel
            f["mid_price_1_acceleration"] = vel - (self.prev_mid1 - self.prev_mid1_prev) / (self.prev_mid1_prev + 1e-9)

        # Cumulative imbalance level 1 (3 ticks)
        self.imb1_history.append(f["volume_imbalance_1"])
        f["cumulative_imbalance_1"] = sum(self.imb1_history)

        # ========== KAMA (Kaufman Adaptive Moving Average) ==========
        for key, state in self.kama_states.items():
            history = state["history"]
            n = state["n"]
            fast_alpha = state["fast_alpha"]
            slow_alpha = state["slow_alpha"]

            history.append(mid)

            if len(history) > n:
                # Change: |price - price_n_periods_ago|
                change = abs(history[-1] - history[0])
                # Volatility: sum of |price_i - price_{i-1}| for last n periods
                volatility = sum(
                    abs(history[i] - history[i - 1]) for i in range(1, len(history))
                )
                # Efficiency Ratio
                er = change / (volatility + 1e-9)
                er = min(max(er, 0.0), 1.0)
                # Smoothing Constant
                sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2

                if state["kama"] is None:
                    state["kama"] = mid
                else:
                    state["kama"] = state["kama"] + sc * (mid - state["kama"])

            f[key] = state["kama"] if state["kama"] is not None else mid

        # KAMA spread (difference between fast and slow)
        kama_keys = list(self.kama_states.keys())
        if len(kama_keys) >= 2:
            f["kama_spread_fast_slow"] = f[kama_keys[0]] - f[kama_keys[-1]]
            f["kama_cross_signal"] = int(f[kama_keys[0]] > f[kama_keys[1]])

        # ========== Keltner Channels ==========
        # True Range = max(high-low, |high-close|, |low-close|)
        # For LOB we use spread as proxy for high-low
        if self.prev_mid_for_atr is not None:
            high_low = spread  # proxy
            high_close = abs(mid - self.prev_mid_for_atr)
            low_close = abs(mid - self.prev_mid_for_atr)
            true_range = max(high_low, high_close, low_close)
        else:
            true_range = spread

        for w in KELTNER_WINDOWS:
            # EMA update
            alpha = self.keltner_alpha[w]
            if self.keltner_ema[w] is None:
                self.keltner_ema[w] = mid
            else:
                self.keltner_ema[w] = alpha * mid + (1 - alpha) * self.keltner_ema[w]

            # ATR update
            self.keltner_atr[w].push(true_range)

            ema = self.keltner_ema[w]
            atr = self.keltner_atr[w].mean()

            # Keltner Channels
            upper = ema + 2 * atr
            lower = ema - 2 * atr
            f[f"keltner_upper_{w}"] = upper
            f[f"keltner_lower_{w}"] = lower
            f[f"keltner_mid_{w}"] = ema
            # Position within channel
            f[f"keltner_position_{w}"] = (mid - lower) / (upper - lower + 1e-9)

        self.prev_mid_for_atr = mid

        # ========== OFI Cumsum ==========
        for lvl in range(1, 4):
            ofi_val = f.get(f"ofi_{lvl}", 0.0)
            for w, rs in self.ofi_cumsum[lvl].items():
                rs.push(ofi_val)
                f[f"ofi_{lvl}_cumsum_{w}"] = rs.s  # sum over window

        # ========== EWMA MAD for volume_imbalance ==========
        for span, alpha in [("span3", self.alpha_span3), ("span5", self.alpha_span5),
                            ("span10", self.alpha_span10), ("span20", self.alpha_span20)]:
            prev = self.ewma_imb[span]
            self.ewma_imb[span] = imb if prev is None else alpha * imb + (1 - alpha) * prev
            f[f"ewma_volume_imbalance_{span}"] = self.ewma_imb[span] if self.ewma_imb[span] is not None else 0.0

        # MAD for span3/5/10
        for span, alpha in [("span3", self.alpha_span3), ("span5", self.alpha_span5), ("span10", self.alpha_span10)]:
            if self.ewma_imb[span] is not None:
                delta = abs(imb - self.ewma_imb[span])
                prev_mad = self.ewma_imb_mad[span]
                self.ewma_imb_mad[span] = delta if prev_mad is None else alpha * delta + (1 - alpha) * prev_mad
            else:
                self.ewma_imb_mad[span] = 0.0
            f[f"ewma_volume_imbalance_mad_{span}"] = self.ewma_imb_mad[span] if self.ewma_imb_mad[span] is not None else 0.0

        # ========== Additional ratio features ==========
        # Mid price ratio (ratio to past values)
        if len(self.prev_mid_lag3) >= 3:
            f["mid_price_ratio_3"] = mid / (self.prev_mid_lag3[0] + 1e-9)
        else:
            f["mid_price_ratio_3"] = 1.0

        # ========== New features from 0.461 solution ==========
        # Mid price ratio for different lags (5, 10, 20, 50, 100)
        self.mid_history.append(mid)
        for lag in [5, 10, 20, 50, 100]:
            if len(self.mid_history) > lag:
                f[f"mid_price_ratio_{lag}"] = mid / (self.mid_history[-lag - 1] + 1e-9)
            else:
                f[f"mid_price_ratio_{lag}"] = 1.0

        # Keltner position change (position change over 5 steps)
        for w in KELTNER_WINDOWS:
            keltner_pos = f.get(f"keltner_position_{w}", 0.5)
            self.keltner_pos_history[w].append(keltner_pos)
            if len(self.keltner_pos_history[w]) >= 6:
                f[f"keltner_position_change_{w}"] = keltner_pos - self.keltner_pos_history[w][-6]
            else:
                f[f"keltner_position_change_{w}"] = 0.0

        # Spread pct_change (1-5)
        self.spread_history.append(spread)
        for lag in [1, 2, 3, 4, 5]:
            if len(self.spread_history) > lag:
                prev_spread_val = self.spread_history[-lag - 1]
                f[f"spread_pct_change_{lag}"] = (spread - prev_spread_val) / (prev_spread_val + 1e-9)
            else:
                f[f"spread_pct_change_{lag}"] = 0.0

        # Volume ratio (bid/ask volume) and its lags
        vol_ratio = f["bid_volume_sum"] / (f["ask_volume_sum"] + 1e-9)
        f["volume_ratio"] = vol_ratio
        self.vol_ratio_history.append(vol_ratio)
        for lag in [1, 2, 3, 4, 5]:
            if len(self.vol_ratio_history) > lag:
                f[f"volume_ratio_{lag}"] = self.vol_ratio_history[-lag - 1]
            else:
                f[f"volume_ratio_{lag}"] = 1.0

        # KAMA cross signal and spread
        kama_values = {}
        for key, state in self.kama_states.items():
            kama_val = state["kama"]
            if kama_val is not None:
                kama_values[key] = kama_val
                self.kama_history[key].append(kama_val)
        
        # Cross signal: fast KAMA crosses slow KAMA
        if len(kama_values) >= 2:
            kama_keys = list(kama_values.keys())
            fast_kama = kama_values.get(kama_keys[0], mid)  # kama_50_5_30 (fast)
            slow_kama = kama_values.get(kama_keys[-1], mid)  # kama_600_100_300 (slow)
            
            # Spread between fast and slow
            f["kama_spread_fast_slow"] = fast_kama - slow_kama
            
            # Cross signal: 1 if fast > slow, -1 otherwise
            f["kama_cross_signal"] = 1.0 if fast_kama > slow_kama else -1.0
            
            # Position of mid relative to KAMA
            f["mid_kama_position"] = (mid - slow_kama) / (fast_kama - slow_kama + 1e-9)
        else:
            f["kama_spread_fast_slow"] = 0.0
            f["kama_cross_signal"] = 0.0
            f["mid_kama_position"] = 0.0

        # Volume imbalance pct_change
        if self.prev_imb is not None:
            f["volume_imbalance_pct_change_1"] = (imb - self.prev_imb) / (abs(self.prev_imb) + 1e-9)
        else:
            f["volume_imbalance_pct_change_1"] = 0.0
        
        # Volume sums pct_change (bid/ask)
        bid_vol_sum = f["bid_volume_sum"]
        ask_vol_sum = f["ask_volume_sum"]
        if hasattr(self, 'prev_bid_vol_sum') and self.prev_bid_vol_sum is not None:
            f["bid_volume_sum_pct_change_1"] = (bid_vol_sum - self.prev_bid_vol_sum) / (self.prev_bid_vol_sum + 1e-9)
            f["ask_volume_sum_pct_change_1"] = (ask_vol_sum - self.prev_ask_vol_sum) / (self.prev_ask_vol_sum + 1e-9)
        else:
            f["bid_volume_sum_pct_change_1"] = 0.0
            f["ask_volume_sum_pct_change_1"] = 0.0
        self.prev_bid_vol_sum = bid_vol_sum
        self.prev_ask_vol_sum = ask_vol_sum

        # ========== Log returns on different lags ==========
        log_mid = np.log(mid + 1e-9)
        self.mid_log_history.append(log_mid)
        for lag in [5, 10, 20, 50, 100, 1000]:
            if len(self.mid_log_history) > lag:
                f[f"log_ret_mid_price_mean_{lag}"] = log_mid - self.mid_log_history[-lag - 1]
            else:
                f[f"log_ret_mid_price_mean_{lag}"] = 0.0

        # ========== Mid price position (position in min-max range) ==========
        for w in [3, 5, 10, 20, 50, 100, 1000]:
            self.mid_minmax_history[w].append(mid)
            if len(self.mid_minmax_history[w]) >= 2:
                hist = list(self.mid_minmax_history[w])
                hist_past = hist[:-1]  # without current value
                roll_min = min(hist_past)
                roll_max = max(hist_past)
                f[f"mid_price_position_{w}"] = (mid - roll_min) / (roll_max - roll_min + 1e-9)
            else:
                f[f"mid_price_position_{w}"] = 0.5

        # ========== Mid price diff between levels ==========
        mid1 = f["mid_price_1"]
        for lvl in [2, 3, 4, 5]:
            bid_lvl = row[f"bid_price_{lvl}"]
            ask_lvl = row[f"ask_price_{lvl}"]
            mid_lvl = (bid_lvl + ask_lvl) / 2.0
            f[f"mid_price_diff_{lvl}"] = mid_lvl - mid1

        # ========== ETNA-like Anomaly Smoothing ==========
        # Outlier smoothing via window median (no future peeking)
        self.etna_mid_history.append(mid)
        self.etna_imb_history.append(imb)
        self.etna_micro_history.append(micro)
        self.etna_spread_history.append(spread)
        
        # Smoothed values (median of window)
        if len(self.etna_mid_history) >= 5:
            mid_arr = np.array(self.etna_mid_history)
            median_mid = np.median(mid_arr)
            std_mid = np.std(mid_arr) + 1e-9
            # Define outlier: |x - median| > alpha * std
            is_outlier = abs(mid - median_mid) > ETNA_ALPHA * std_mid
            f["mid_price_mean_etna_smooth"] = median_mid if is_outlier else mid
            f["mid_price_mean_is_outlier"] = 1.0 if is_outlier else 0.0
        else:
            f["mid_price_mean_etna_smooth"] = mid
            f["mid_price_mean_is_outlier"] = 0.0
        
        if len(self.etna_imb_history) >= 5:
            imb_arr = np.array(self.etna_imb_history)
            median_imb = np.median(imb_arr)
            std_imb = np.std(imb_arr) + 1e-9
            is_outlier = abs(imb - median_imb) > ETNA_ALPHA * std_imb
            f["volume_imbalance_etna_smooth"] = median_imb if is_outlier else imb
            f["volume_imbalance_is_outlier"] = 1.0 if is_outlier else 0.0
        else:
            f["volume_imbalance_etna_smooth"] = imb
            f["volume_imbalance_is_outlier"] = 0.0
        
        if len(self.etna_micro_history) >= 5:
            micro_arr = np.array(self.etna_micro_history)
            median_micro = np.median(micro_arr)
            std_micro = np.std(micro_arr) + 1e-9
            is_outlier = abs(micro - median_micro) > ETNA_ALPHA * std_micro
            f["microprice_1_etna_smooth"] = median_micro if is_outlier else micro
        else:
            f["microprice_1_etna_smooth"] = micro
        
        if len(self.etna_spread_history) >= 5:
            spread_arr = np.array(self.etna_spread_history)
            median_spread = np.median(spread_arr)
            std_spread = np.std(spread_arr) + 1e-9
            is_outlier = abs(spread - median_spread) > ETNA_ALPHA * std_spread
            f["spread_abs_mean_etna_smooth"] = median_spread if is_outlier else spread
        else:
            f["spread_abs_mean_etna_smooth"] = spread

        # VWAP of level 1
        bid1 = row["bid_price_1"]
        ask1 = row["ask_price_1"]
        bq1 = row["bid_qty_1"]
        aq1 = row["ask_qty_1"]
        vwap_1 = (bid1 * bq1 + ask1 * aq1) / (bq1 + aq1 + 1e-9)
        f["vwap_1"] = vwap_1
        f["mid_vwap_deviation"] = mid1 - vwap_1
        f["rel_mid_vwap_dev"] = (mid1 - vwap_1) / (vwap_1 + 1e-9)

        # Liquidity concentration (volume share in top-3 levels)
        bid_top3 = sum(row[f"bid_qty_{i}"] for i in range(1, 4))
        ask_top3 = sum(row[f"ask_qty_{i}"] for i in range(1, 4))
        bid_total = f["bid_volume_sum"]
        ask_total = f["ask_volume_sum"]
        bid_ratio = bid_top3 / (bid_total + 1e-9)
        ask_ratio = ask_top3 / (ask_total + 1e-9)
        f["liquidity_concentration"] = bid_ratio - ask_ratio

        # Momentum signal
        f["momentum_signal"] = (
            f["volume_imbalance_1"] *
            f["mid_price_1_velocity"] *
            np.sign(f["cumulative_imbalance_1"])
        )

        # Update states after feature calculation
        self.prev_mid = mid
        self.prev_mid_lag3.append(mid)
        self.prev_spread = spread
        self.prev_imb = imb
        self.prev_tick = tick
        self.tick_initialized = True
        self.prev_mid1_prev = self.prev_mid1
        self.prev_mid1 = mid1

        for r in self.roll_mid.values():
            r.push(mid)
        for r in self.roll_spread.values():
            r.push(spread)
        for r in self.roll_spread_rel.values():
            r.push(spread_rel)
        for r in self.roll_imb_mean.values():
            r.push(imb)
            # for std also need push
        for r in self.roll_imb_std.values():
            r.push(imb)
        for r in self.roll_microprice.values():
            r.push(micro)
        for r in self.roll_mid1.values():
            r.push(mid1)

        return f


def feature_order_from_dict(feature_dict: Dict[str, float]) -> List[str]:
    """Column order from features dict."""
    return list(feature_dict.keys())


def dict_row_from_polars(row: Tuple) -> Dict[str, float]:
    """Utility for polars row to dict conversion if needed."""
    return {k: row[idx] for idx, k in enumerate(row.keys())}

