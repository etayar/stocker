"""
tests.test_models.test_ta_model

Tests for models.ta_model.
"""

import numpy as np
import pandas as pd
import pytest

from models.ta_model import (
    compute_bollinger,
    compute_macd,
    compute_rsi,
    compute_ta_score,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_CONFIG = {
    "model": {
        "ta": {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2.0,
        }
    }
}

# Minimum rows needed by the default config: max(15, 35, 20) = 35
_MIN_ROWS = 35


def _make_prices(n: int, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    """Linearly trending price DataFrame with a single 'close' column."""
    return pd.DataFrame({"close": [start + i * step for i in range(n)]})


# ── compute_rsi ───────────────────────────────────────────────────────────────

def test_rsi_overbought():
    """Strictly rising prices → RSI should be near 100 (> 70)."""
    prices = _make_prices(30, step=1.0)
    rsi = compute_rsi(prices, period=14)
    assert float(rsi.iloc[-1]) > 70


def test_rsi_oversold():
    """Strictly falling prices → RSI should be near 0 (< 30)."""
    prices = _make_prices(30, start=200.0, step=-1.0)
    rsi = compute_rsi(prices, period=14)
    assert float(rsi.iloc[-1]) < 30


def test_rsi_neutral_flat_market():
    """Flat prices → RSI returns neutral 50."""
    prices = _make_prices(30, step=0.0)
    rsi = compute_rsi(prices, period=14)
    assert float(rsi.iloc[-1]) == pytest.approx(50.0)


def test_rsi_returns_series_same_length():
    prices = _make_prices(30, step=1.0)
    rsi = compute_rsi(prices, period=14)
    assert isinstance(rsi, pd.Series)
    assert len(rsi) == len(prices)


# ── compute_macd ──────────────────────────────────────────────────────────────

def test_macd_returns_dataframe():
    prices = _make_prices(60, step=1.0)
    result = compute_macd(prices, fast=12, slow=26, signal=9)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"macd", "signal", "histogram"}
    assert len(result) == len(prices)


def test_macd_histogram_positive_on_uptrend():
    """During the early warm-up phase the histogram should be positive
    when prices are rising (fast EMA tracking the uptrend ahead of slow)."""
    prices = _make_prices(30, step=1.0)
    result = compute_macd(prices, fast=12, slow=26, signal=9)
    # At least some early histogram values should be positive for an uptrend
    assert result["histogram"].dropna().max() > 0


def test_macd_histogram_negative_on_downtrend():
    prices = _make_prices(30, start=200.0, step=-1.0)
    result = compute_macd(prices, fast=12, slow=26, signal=9)
    assert result["histogram"].dropna().min() < 0


# ── compute_bollinger ─────────────────────────────────────────────────────────

def test_bollinger_returns_dataframe():
    prices = _make_prices(60, step=1.0)
    result = compute_bollinger(prices, period=20, std=2.0)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"upper", "middle", "lower"}
    assert len(result) == len(prices)


def test_bollinger_band_ordering():
    """upper ≥ middle ≥ lower for every non-NaN row."""
    prices = _make_prices(60, step=1.0)
    bb = compute_bollinger(prices, period=20, std=2.0)
    valid = bb.dropna()
    assert (valid["upper"] >= valid["middle"]).all()
    assert (valid["middle"] >= valid["lower"]).all()


def test_bollinger_middle_is_rolling_mean():
    prices = _make_prices(40, step=0.5)
    bb = compute_bollinger(prices, period=20, std=2.0)
    expected_middle = prices["close"].rolling(20).mean()
    pd.testing.assert_series_equal(bb["middle"], expected_middle, check_names=False)


# ── compute_ta_score ──────────────────────────────────────────────────────────

def test_compute_ta_score_returns_float_in_range():
    prices = _make_prices(60, step=1.0)
    score = compute_ta_score(prices, _CONFIG)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_ta_score_bullish_market():
    """Strongly rising prices → RSI near 100, %B near 1 → score > 0.5."""
    prices = _make_prices(60, step=1.0)
    score = compute_ta_score(prices, _CONFIG)
    assert score > 0.5


def test_ta_score_bearish_market():
    """Strongly falling prices → RSI near 0, %B near 0 → score < 0.5."""
    prices = _make_prices(60, start=200.0, step=-1.0)
    score = compute_ta_score(prices, _CONFIG)
    assert score < 0.5


def test_insufficient_data_raises():
    """Fewer rows than required periods should raise ValueError."""
    prices = _make_prices(10, step=1.0)  # need 35, only 10
    with pytest.raises(ValueError, match="Insufficient"):
        compute_ta_score(prices, _CONFIG)


def test_ta_score_random_prices_in_range():
    """ta_score is always in [0, 1] regardless of price pattern."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        close = 100.0 + np.cumsum(rng.normal(0, 1, 60))
        prices = pd.DataFrame({"close": close})
        score = compute_ta_score(prices, _CONFIG)
        assert 0.0 <= score <= 1.0
