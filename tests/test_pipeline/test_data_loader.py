"""
tests.test_pipeline.test_data_loader

Tests for pipeline.data_loader.

yfinance calls are mocked so no network access occurs.
Scaler and fold logic are tested with synthetic DataFrames.
"""

from __future__ import annotations

import pathlib
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from pipeline.data_loader import (
    apply_scaler,
    build_walk_forward_folds,
    fetch_ticker,
    fit_scaler,
    load_scaler,
    save_scaler,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

_CONFIG = {
    "stock": {"start_date": "2020-01-01"},
}


def _make_ohlcv(n: int, start_price: float = 100.0) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with n rows, DatetimeIndex."""
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    close = np.linspace(start_price, start_price + n - 1, n)
    return pd.DataFrame(
        {
            "open":   close * 0.99,
            "high":   close * 1.01,
            "low":    close * 0.98,
            "close":  close,
            "volume": np.ones(n) * 1_000_000,
        },
        index=idx,
    )


def _mock_yf_download(ticker, start, progress, auto_adjust):
    """Simulate yf.download returning a plain DataFrame with Title-case columns."""
    df = _make_ohlcv(400)
    df.columns = [c.capitalize() for c in df.columns]
    return df


# ── fetch_ticker ───────────────────────────────────────────────────────────────

def test_fetch_ticker_returns_dataframe():
    with patch("pipeline.data_loader.yf.download", side_effect=_mock_yf_download):
        df = fetch_ticker("AAPL", _CONFIG)
    assert isinstance(df, pd.DataFrame)


def test_fetch_ticker_has_ohlcv_columns():
    with patch("pipeline.data_loader.yf.download", side_effect=_mock_yf_download):
        df = fetch_ticker("AAPL", _CONFIG)
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns, f"Missing column: {col}"


def test_fetch_ticker_columns_lowercase():
    with patch("pipeline.data_loader.yf.download", side_effect=_mock_yf_download):
        df = fetch_ticker("AAPL", _CONFIG)
    assert all(c == c.lower() for c in df.columns)


def test_fetch_ticker_sorted_oldest_to_newest():
    with patch("pipeline.data_loader.yf.download", side_effect=_mock_yf_download):
        df = fetch_ticker("AAPL", _CONFIG)
    assert df.index.is_monotonic_increasing


def test_fetch_ticker_datetime_index():
    with patch("pipeline.data_loader.yf.download", side_effect=_mock_yf_download):
        df = fetch_ticker("AAPL", _CONFIG)
    assert isinstance(df.index, pd.DatetimeIndex)


def test_fetch_ticker_no_nan_in_close():
    with patch("pipeline.data_loader.yf.download", side_effect=_mock_yf_download):
        df = fetch_ticker("AAPL", _CONFIG)
    assert df["close"].notna().all()


def test_fetch_ticker_empty_raises_value_error():
    with patch("pipeline.data_loader.yf.download", return_value=pd.DataFrame()):
        with pytest.raises(ValueError, match="no data"):
            fetch_ticker("FAKE", _CONFIG)


def test_fetch_ticker_flattens_multiindex_columns():
    """yfinance may return a MultiIndex — fetch_ticker must flatten it."""
    df = _make_ohlcv(50)
    df.columns = [c.capitalize() for c in df.columns]
    # Wrap in MultiIndex
    mi = pd.MultiIndex.from_tuples([(c, "AAPL") for c in df.columns])
    df.columns = mi
    with patch("pipeline.data_loader.yf.download", return_value=df):
        result = fetch_ticker("AAPL", _CONFIG)
    assert "close" in result.columns


# ── build_walk_forward_folds ───────────────────────────────────────────────────

def test_folds_returns_list():
    df = _make_ohlcv(400)
    folds = build_walk_forward_folds(df, 252, 64, 21)
    assert isinstance(folds, list)


def test_folds_correct_train_size():
    df = _make_ohlcv(400)
    folds = build_walk_forward_folds(df, 252, 64, 21)
    for train, _ in folds:
        assert len(train) == 252


def test_folds_correct_val_size():
    df = _make_ohlcv(400)
    folds = build_walk_forward_folds(df, 252, 64, 21)
    for _, val in folds:
        assert len(val) == 64


def test_folds_temporal_order():
    """Train must always precede val in time."""
    df = _make_ohlcv(400)
    folds = build_walk_forward_folds(df, 252, 64, 21)
    for train, val in folds:
        assert train.index[-1] < val.index[0]


def test_folds_within_fold_no_leakage():
    """Within each fold, train must end strictly before val begins (no leakage)."""
    df = _make_ohlcv(600)
    folds = build_walk_forward_folds(df, 252, 64, 21)
    for train, val in folds:
        assert train.index[-1] < val.index[0], "Train leaks into val window"


def test_folds_empty_when_data_too_short():
    df = _make_ohlcv(100)
    folds = build_walk_forward_folds(df, 252, 64, 21)
    assert folds == []


def test_folds_at_least_one_fold():
    """Exactly train+val rows → exactly one fold."""
    df = _make_ohlcv(252 + 64)
    folds = build_walk_forward_folds(df, 252, 64, 21)
    assert len(folds) >= 1


def test_folds_step_controls_count():
    """More folds with smaller step."""
    df = _make_ohlcv(600)
    folds_small_step = build_walk_forward_folds(df, 252, 64, 10)
    folds_large_step = build_walk_forward_folds(df, 252, 64, 50)
    assert len(folds_small_step) > len(folds_large_step)


# ── fit_scaler / apply_scaler ──────────────────────────────────────────────────

def test_fit_scaler_returns_scaler():
    df = _make_ohlcv(100)
    scaler = fit_scaler(df)
    assert hasattr(scaler, "transform")


def test_apply_scaler_close_in_unit_range():
    """Scaled close values must be in [0, 1] for data from the same range."""
    df = _make_ohlcv(200)
    train, val = df.iloc[:100], df.iloc[100:]
    scaler = fit_scaler(train)
    scaled = apply_scaler(train, scaler)
    assert scaled["close"].min() >= -1e-9
    assert scaled["close"].max() <= 1.0 + 1e-9


def test_apply_scaler_does_not_mutate_input():
    df = _make_ohlcv(100)
    original_close = df["close"].copy()
    scaler = fit_scaler(df)
    apply_scaler(df, scaler)
    pd.testing.assert_series_equal(df["close"], original_close)


def test_apply_scaler_preserves_other_columns():
    df = _make_ohlcv(100)
    scaler = fit_scaler(df)
    scaled = apply_scaler(df, scaler)
    pd.testing.assert_series_equal(scaled["volume"], df["volume"])


def test_scaler_fitted_on_train_only_prevents_leakage():
    """Val close values may exceed [0,1] if they go outside the train range."""
    train = _make_ohlcv(100, start_price=100.0)
    val   = _make_ohlcv(50,  start_price=500.0)  # prices way above train range
    scaler = fit_scaler(train)
    scaled_val = apply_scaler(val, scaler)
    # Values above training range will be > 1 — this is EXPECTED (no leakage)
    assert scaled_val["close"].max() > 1.0


# ── save_scaler / load_scaler ──────────────────────────────────────────────────

def test_save_load_scaler_roundtrip(tmp_path):
    df = _make_ohlcv(100)
    scaler = fit_scaler(df)
    path = str(tmp_path / "scaler.pkl")
    save_scaler(scaler, path)
    loaded = load_scaler(path)
    # Both scalers should produce identical output
    result1 = scaler.transform(df[["close"]])
    result2 = loaded.transform(df[["close"]])
    np.testing.assert_array_almost_equal(result1, result2)


def test_save_scaler_creates_parent_dirs(tmp_path):
    df = _make_ohlcv(50)
    scaler = fit_scaler(df)
    path = str(tmp_path / "a" / "b" / "scaler.pkl")
    save_scaler(scaler, path)
    assert pathlib.Path(path).exists()


def test_load_scaler_raises_type_error_on_wrong_object(tmp_path):
    """load_scaler must reject files that don't contain a MinMaxScaler."""
    import joblib

    # Save something that is NOT a MinMaxScaler to a file
    bad_path = tmp_path / "bad_scaler.pkl"
    joblib.dump({"not": "a scaler"}, bad_path)

    with pytest.raises(TypeError, match="MinMaxScaler"):
        load_scaler(str(bad_path))
