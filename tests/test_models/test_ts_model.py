"""
tests.test_models.test_ts_model

Tests for models.ts_model.

The implementation is a Ridge regression on rolling log-return windows.
Score formula: sigmoid(predicted_cumulative_return / return_std)
  → > 0.5 bullish, < 0.5 bearish, = 0.5 neutral (predicted return == 0).
"""

import pathlib

import numpy as np
import pandas as pd
import pytest

from models.ts_model import (
    TSModel,
    _build_dataset,
    _log_returns,
    _sigmoid,
    load_model,
    predict,
    save_model,
    train,
)

# ── fixtures / helpers ────────────────────────────────────────────────────────

_CONFIG = {
    "model": {"ts": {"window_size": 30}},
    "stock": {"prediction_horizon": 5},
}


def _make_prices(n: int, start: float = 100.0, step: float = 0.0) -> pd.DataFrame:
    """Linearly trending DataFrame with a single 'close' column."""
    return pd.DataFrame({"close": [start + i * step for i in range(n)]})


# ── _sigmoid ──────────────────────────────────────────────────────────────────

def test_sigmoid_zero_returns_half():
    assert _sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_large_positive_approaches_one():
    assert _sigmoid(1000.0) == pytest.approx(1.0, abs=1e-6)


def test_sigmoid_large_negative_approaches_zero():
    assert _sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-6)


def test_sigmoid_symmetry():
    for z in [0.1, 1.0, 3.0, 10.0]:
        assert _sigmoid(z) + _sigmoid(-z) == pytest.approx(1.0, abs=1e-10)


# ── _log_returns ──────────────────────────────────────────────────────────────

def test_log_returns_flat_prices_are_zero():
    prices = _make_prices(10, step=0.0)
    r = _log_returns(prices)
    assert np.allclose(r, 0.0)


def test_log_returns_length():
    prices = _make_prices(50, step=1.0)
    r = _log_returns(prices)
    assert len(r) == 49  # n_prices - 1


# ── _build_dataset ────────────────────────────────────────────────────────────

def test_build_dataset_shapes():
    returns = np.ones(50)
    X, y = _build_dataset(returns, window=10, horizon=5)
    # n_samples = 50 - 10 - 5 + 1 = 36
    assert X.shape == (36, 10)
    assert y.shape == (36,)


def test_build_dataset_feature_values():
    """X[i] should contain exactly returns[i : i+window]."""
    returns = np.arange(20, dtype=float)
    X, _ = _build_dataset(returns, window=5, horizon=3)
    np.testing.assert_array_equal(X[0], returns[0:5])
    np.testing.assert_array_equal(X[1], returns[1:6])


def test_build_dataset_target_values():
    """y[i] should be sum of returns[i+window : i+window+horizon]."""
    returns = np.arange(20, dtype=float)
    _, y = _build_dataset(returns, window=5, horizon=3)
    assert y[0] == pytest.approx(returns[5:8].sum())


def test_build_dataset_insufficient_data_raises():
    returns = np.ones(10)
    with pytest.raises(ValueError, match="Not enough data"):
        _build_dataset(returns, window=10, horizon=5)


# ── train ─────────────────────────────────────────────────────────────────────

def test_train_produces_model():
    model = train(_make_prices(100, step=1.0), _CONFIG)
    assert model is not None
    assert isinstance(model, TSModel)


def test_train_model_attributes():
    model = train(_make_prices(100, step=1.0), _CONFIG)
    assert model.window_size == 30
    assert model.horizon == 5
    assert model.model is not None
    assert isinstance(model.return_std, float)


def test_train_flat_prices_return_std_defaults_to_one():
    """When all returns are zero, return_std falls back to 1.0."""
    model = train(_make_prices(100, step=0.0), _CONFIG)
    assert model.return_std == pytest.approx(1.0)


def test_train_insufficient_data_raises():
    # window=30, horizon=5 → need ≥ 36 rows; provide 20
    with pytest.raises(ValueError):
        train(_make_prices(20, step=1.0), _CONFIG)


# ── predict ───────────────────────────────────────────────────────────────────

def test_predict_returns_float_in_range():
    prices = _make_prices(100, step=1.0)
    model = train(prices, _CONFIG)
    score = predict(model, prices, horizon=5)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_predict_neutral_on_flat_prices():
    """Flat prices → all returns = 0 → predicted return = 0 → score = 0.5."""
    prices = _make_prices(100, step=0.0)
    model = train(prices, _CONFIG)
    score = predict(model, prices, horizon=5)
    assert score == pytest.approx(0.5, abs=1e-6)


def test_predict_with_bullish_trend():
    """Strongly rising prices → Ridge predicts positive return → score > 0.5."""
    prices = _make_prices(100, start=100.0, step=2.0)
    model = train(prices, _CONFIG)
    assert predict(model, prices, horizon=5) > 0.5


def test_predict_with_bearish_trend():
    """Strongly falling prices → Ridge predicts negative return → score < 0.5."""
    prices = _make_prices(100, start=300.0, step=-1.0)
    model = train(prices, _CONFIG)
    assert predict(model, prices, horizon=5) < 0.5


def test_predict_insufficient_data_raises():
    model = train(_make_prices(100, step=1.0), _CONFIG)
    with pytest.raises(ValueError, match="at least"):
        predict(model, _make_prices(10, step=1.0), horizon=5)


def test_predict_random_prices_always_in_range():
    """Score stays in [0, 1] for arbitrary random-walk price paths."""
    rng = np.random.default_rng(42)
    train_close = 100.0 + np.cumsum(rng.normal(0, 1, 200))
    pred_close = 100.0 + np.cumsum(rng.normal(0, 1, 50))
    # clip to keep prices positive (log-returns undefined for non-positive)
    pred_close = np.clip(pred_close, 0.01, None)
    model = train(pd.DataFrame({"close": train_close}), _CONFIG)
    score = predict(model, pd.DataFrame({"close": pred_close}), horizon=5)
    assert 0.0 <= score <= 1.0


# ── save / load ───────────────────────────────────────────────────────────────

def test_save_and_load_model(tmp_path):
    """Round-trip: loaded model produces identical predictions."""
    prices = _make_prices(100, step=1.0)
    model = train(prices, _CONFIG)

    path = str(tmp_path / "ts_model.pkl")
    save_model(model, path)
    loaded = load_model(path)

    assert predict(model, prices, horizon=5) == pytest.approx(
        predict(loaded, prices, horizon=5)
    )


def test_save_model_creates_parent_dirs(tmp_path):
    model = train(_make_prices(100, step=1.0), _CONFIG)
    path = tmp_path / "a" / "b" / "c" / "model.pkl"
    save_model(model, str(path))
    assert path.exists()


def test_load_model_returns_ts_model(tmp_path):
    model = train(_make_prices(100, step=1.0), _CONFIG)
    path = str(tmp_path / "model.pkl")
    save_model(model, path)
    loaded = load_model(path)
    assert isinstance(loaded, TSModel)


def test_loaded_model_preserves_attributes(tmp_path):
    prices = _make_prices(100, step=1.0)
    model = train(prices, _CONFIG)
    path = str(tmp_path / "model.pkl")
    save_model(model, path)
    loaded = load_model(path)

    assert loaded.window_size == model.window_size
    assert loaded.horizon == model.horizon
    assert loaded.return_std == pytest.approx(model.return_std)
