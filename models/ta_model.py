"""
models.ta_model

Technical analysis (TA) indicator computation and aggregation
into a single ta_score ∈ [0, 1].

Indicators computed (all periods configurable via config.yaml → model.ta):
  - RSI  (Relative Strength Index)    : period = rsi_period
  - MACD (Moving Average Convergence) : fast, slow, signal periods
  - Bollinger Bands                   : period = bb_period, std = bb_std

Each indicator is normalised to [0, 1] (bullish = near 1, bearish = near 0)
and aggregated via simple average to produce ta_score.

Public interface:
  compute_ta_score(prices: pd.DataFrame, config: dict) -> float
    Returns ta_score ∈ [0, 1] for the most recent row in prices.

  compute_rsi(prices: pd.DataFrame, period: int) -> pd.Series
  compute_macd(prices: pd.DataFrame, fast: int, slow: int,
               signal: int) -> pd.DataFrame
  compute_bollinger(prices: pd.DataFrame, period: int,
                    std: float) -> pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_rsi(prices: pd.DataFrame, period: int) -> pd.Series:
    """Compute Wilder's RSI on the close column.

    Args:
        prices: DataFrame with a 'close' column.
        period: Lookback period (e.g. 14).

    Returns:
        pd.Series of RSI values (0–100), indexed like prices.
        Values before the first full period are NaN.
    """
    close = prices["close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder smoothing: EWM with com = period - 1  ↔  alpha = 1/period
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss          # inf when avg_loss == 0 & avg_gain > 0
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Flat market (both zero) → neutral 50
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi[both_zero] = 50.0

    return rsi.rename("rsi")


def compute_macd(
    prices: pd.DataFrame,
    fast: int,
    slow: int,
    signal: int,
) -> pd.DataFrame:
    """Compute MACD line, signal line, and histogram.

    Args:
        prices: DataFrame with a 'close' column.
        fast:   Fast EMA span (e.g. 12).
        slow:   Slow EMA span (e.g. 26).
        signal: Signal EMA span (e.g. 9).

    Returns:
        DataFrame with columns ['macd', 'signal', 'histogram'],
        indexed like prices.
    """
    close = prices["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram},
        index=prices.index,
    )


def compute_bollinger(
    prices: pd.DataFrame,
    period: int,
    std: float,
) -> pd.DataFrame:
    """Compute Bollinger Bands around the rolling mean of close.

    Args:
        prices: DataFrame with a 'close' column.
        period: Rolling window size (e.g. 20).
        std:    Number of standard deviations for the bands (e.g. 2.0).

    Returns:
        DataFrame with columns ['upper', 'middle', 'lower'],
        indexed like prices. Values before the first full window are NaN.
    """
    close = prices["close"]
    middle = close.rolling(window=period).mean()
    band = std * close.rolling(window=period).std(ddof=0)
    return pd.DataFrame(
        {"upper": middle + band, "middle": middle, "lower": middle - band},
        index=prices.index,
    )


def compute_ta_score(prices: pd.DataFrame, config: dict) -> float:
    """Aggregate RSI, MACD, and Bollinger into a single score ∈ [0, 1].

    Each indicator is normalised to [0, 1] (1 = bullish, 0 = bearish)
    then averaged.

    Normalisation:
      - RSI:      rsi / 100
      - MACD:     (histogram / abs_max_histogram + 1) / 2
                  (0.5 when steady-state or flat; > 0.5 while histogram
                  is proportionally positive vs. its historical peak)
      - Bollinger: %B = (close − lower) / (upper − lower), clipped [0, 1]

    Args:
        prices: DataFrame with a 'close' column, sorted oldest→newest.
        config: Loaded config dict (config.yaml).

    Returns:
        ta_score ∈ [0, 1].

    Raises:
        ValueError: If prices has fewer rows than the minimum required by
                    the configured indicator periods.
    """
    ta_cfg = config["model"]["ta"]
    rsi_period = ta_cfg["rsi_period"]
    macd_fast = ta_cfg["macd_fast"]
    macd_slow = ta_cfg["macd_slow"]
    macd_signal_period = ta_cfg["macd_signal"]
    bb_period = ta_cfg["bb_period"]
    bb_std = ta_cfg["bb_std"]

    min_required = max(rsi_period + 1, macd_slow + macd_signal_period, bb_period)
    if len(prices) < min_required:
        raise ValueError(
            f"Insufficient price data: need at least {min_required} rows, "
            f"got {len(prices)}"
        )

    # ── RSI ──────────────────────────────────────────────────────────────────
    rsi = compute_rsi(prices, rsi_period)
    rsi_score = float(rsi.iloc[-1]) / 100.0

    # ── MACD ─────────────────────────────────────────────────────────────────
    macd_df = compute_macd(prices, macd_fast, macd_slow, macd_signal_period)
    hist = macd_df["histogram"]
    hist_abs_max = float(hist.abs().max())
    if hist_abs_max > 1e-10:
        macd_score = (float(hist.iloc[-1]) / hist_abs_max + 1.0) / 2.0
    else:
        macd_score = 0.5

    # ── Bollinger %B ─────────────────────────────────────────────────────────
    bb = compute_bollinger(prices, bb_period, bb_std)
    last_close = float(prices["close"].iloc[-1])
    upper = float(bb["upper"].iloc[-1])
    lower = float(bb["lower"].iloc[-1])
    band_width = upper - lower
    if band_width > 1e-10:
        pct_b = (last_close - lower) / band_width
        bb_score = float(np.clip(pct_b, 0.0, 1.0))
    else:
        bb_score = 0.5

    return float(np.mean([rsi_score, macd_score, bb_score]))
