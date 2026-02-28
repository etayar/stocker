"""
tests.test_models.test_ts_model

Tests for the Chronos-based models.ts_model.

Chronos / PyTorch / network access are never used directly.
All pipeline and model objects are MagicMock instances injected via patch.

Tested surface:
  - _sigmoid          (pure math)
  - _ts_score_from_forecast  (pure math)
  - _is_fresh         (filesystem logic)
  - load_base_model   (import guard + ChronosPipeline.from_pretrained)
  - fine_tune_general (orchestration, saves files)
  - get_ticker_model  (cache check, load vs rebuild paths)
  - predict           (public API, end-to-end mocked)
"""

from __future__ import annotations

import json
import math
import pathlib
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

import models.ts_model as ts_mod
from models.ts_model import (
    _is_fresh,
    _sigmoid,
    _ts_score_from_forecast,
    predict,
)

# ── Config fixture ─────────────────────────────────────────────────────────────

_CONFIG = {
    "stock": {"start_date": "2020-01-01"},
    "data": {
        "universe": ["AAPL", "MSFT"],
        "walk_forward": {
            "train_window_days": 252,
            "val_window_days":   64,
            "step_days":         21,
        },
    },
    "model": {
        "ts": {
            "chronos_base":       "amazon/chronos-t5-small",
            "cache_ttl_days":     7,
            "general_model_dir":  "models/general",
            "cache_dir":          "models/cache",
            "context_length":     64,
            "num_samples":        5,
            "n_training_epochs":  1,
            "learning_rate":      1e-4,
        }
    },
    "scheduler": {"retrain_interval_days": 14},
}


def _make_price_df(n: int = 300, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    close = np.linspace(start, start + n - 1, n)
    return pd.DataFrame({"close": close, "volume": np.ones(n)}, index=idx)


# ── Helper to make a fake forecast tensor ──────────────────────────────────────

def _make_forecast(samples: list[list[float]]):
    """Construct a fake forecast tensor shaped [1, num_samples, horizon]."""
    import torch
    data = torch.tensor([samples], dtype=torch.float32)  # [1, S, H]
    return data


# ── _sigmoid ──────────────────────────────────────────────────────────────────

def test_sigmoid_zero_returns_half():
    assert _sigmoid(0.0) == pytest.approx(0.5)


def test_sigmoid_large_positive_near_one():
    assert _sigmoid(1000.0) == pytest.approx(1.0, abs=1e-6)


def test_sigmoid_large_negative_near_zero():
    assert _sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-6)


def test_sigmoid_symmetry():
    for z in [0.5, 1.0, 5.0]:
        assert _sigmoid(z) + _sigmoid(-z) == pytest.approx(1.0, abs=1e-10)


# ── _ts_score_from_forecast ────────────────────────────────────────────────────

def test_score_from_forecast_bullish():
    """Median forecast above current price → score > 0.5."""
    import torch
    forecast = _make_forecast([[110.0, 115.0, 112.0]])  # all > 100
    score = _ts_score_from_forecast(forecast, current_price=100.0, return_std=0.01)
    assert score > 0.5


def test_score_from_forecast_bearish():
    """Median forecast below current price → score < 0.5."""
    import torch
    forecast = _make_forecast([[85.0, 88.0, 86.0]])   # all < 100
    score = _ts_score_from_forecast(forecast, current_price=100.0, return_std=0.01)
    assert score < 0.5


def test_score_from_forecast_neutral_on_flat():
    """Forecast == current price → log_return = 0 → sigmoid(0) = 0.5."""
    import torch
    forecast = _make_forecast([[100.0, 100.0, 100.0]])
    score = _ts_score_from_forecast(forecast, current_price=100.0, return_std=0.01)
    assert score == pytest.approx(0.5, abs=1e-6)


def test_score_from_forecast_near_zero_std_returns_half():
    """return_std < 1e-10 → neutral score."""
    import torch
    forecast = _make_forecast([[110.0]])
    score = _ts_score_from_forecast(forecast, current_price=100.0, return_std=0.0)
    assert score == 0.5


def test_score_from_forecast_uses_last_horizon_day():
    """Score is based on the *last* day of the horizon, not the first."""
    import torch
    # Last day is 90 (below 100 → bearish)
    forecast = _make_forecast([[110.0, 105.0, 90.0]])
    score = _ts_score_from_forecast(forecast, current_price=100.0, return_std=0.01)
    assert score < 0.5


def test_score_in_unit_interval():
    import torch
    for _ in range(10):
        samples = [[float(np.random.uniform(50, 200))] for _ in range(5)]
        forecast = _make_forecast(samples)
        score = _ts_score_from_forecast(forecast, current_price=100.0, return_std=0.01)
        assert 0.0 <= score <= 1.0


# ── _is_fresh ─────────────────────────────────────────────────────────────────

def test_is_fresh_missing_dir_returns_false(tmp_path):
    assert not _is_fresh(tmp_path / "nonexistent", ttl_days=7)


def test_is_fresh_missing_checkpoint_returns_false(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    meta = {"fine_tuned_at": datetime.now(timezone.utc).isoformat()}
    (model_dir / "metadata.json").write_text(json.dumps(meta))
    # No model_state.pt → not fresh
    assert not _is_fresh(model_dir, ttl_days=7)


def test_is_fresh_recent_model_returns_true(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    meta = {"fine_tuned_at": datetime.now(timezone.utc).isoformat()}
    (model_dir / "metadata.json").write_text(json.dumps(meta))
    (model_dir / "model_state.pt").touch()
    assert _is_fresh(model_dir, ttl_days=7)


def test_is_fresh_stale_model_returns_false(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    stale_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    meta = {"fine_tuned_at": stale_time}
    (model_dir / "metadata.json").write_text(json.dumps(meta))
    (model_dir / "model_state.pt").touch()
    assert not _is_fresh(model_dir, ttl_days=7)


def test_is_fresh_last_trained_key_also_accepted(tmp_path):
    """General model uses 'last_trained' key instead of 'fine_tuned_at'."""
    model_dir = tmp_path / "general"
    model_dir.mkdir()
    meta = {"last_trained": datetime.now(timezone.utc).isoformat()}
    (model_dir / "metadata.json").write_text(json.dumps(meta))
    (model_dir / "model_state.pt").touch()
    assert _is_fresh(model_dir, ttl_days=365)


# ── load_base_model ────────────────────────────────────────────────────────────

def test_load_base_model_raises_without_chronos():
    with patch.object(ts_mod, "_CHRONOS_AVAILABLE", False):
        with pytest.raises(ImportError, match="chronos-forecasting"):
            ts_mod.load_base_model(_CONFIG)


def test_load_base_model_calls_from_pretrained():
    mock_pipeline = MagicMock()
    with patch("models.ts_model.ChronosPipeline") as MockChronos, \
         patch.object(ts_mod, "_CHRONOS_AVAILABLE", True):
        MockChronos.from_pretrained.return_value = mock_pipeline
        ts_mod.load_base_model(_CONFIG)
    MockChronos.from_pretrained.assert_called_once_with(
        "amazon/chronos-t5-small",
        device_map="cpu",
        dtype=ts_mod.torch.float32,
    )


# ── fine_tune_general ─────────────────────────────────────────────────────────

def test_fine_tune_general_saves_metadata(tmp_path):
    """fine_tune_general must write metadata.json with correct structure."""
    cfg = {
        **_CONFIG,
        "model": {
            **_CONFIG["model"],
            "ts": {
                **_CONFIG["model"]["ts"],
                "general_model_dir": str(tmp_path / "general"),
            },
        },
    }
    universe = ["AAPL"]
    df = _make_price_df(400)
    folds = [(df.iloc[:252], df.iloc[252:316])]

    mock_pipeline = MagicMock()
    mock_pipeline.model.model.state_dict.return_value = {}
    mock_pipeline.tokenizer.config.prediction_length = 64

    with ExitStack() as stack:
        stack.enter_context(patch.object(ts_mod, "_CHRONOS_AVAILABLE", True))
        stack.enter_context(patch("models.ts_model.load_base_model", return_value=mock_pipeline))
        stack.enter_context(patch("models.ts_model.fetch_ticker", return_value=df))
        stack.enter_context(patch("models.ts_model.build_walk_forward_folds", return_value=folds))
        stack.enter_context(patch("models.ts_model._fine_tune", return_value=mock_pipeline))
        stack.enter_context(patch("models.ts_model.fit_scaler", return_value=MagicMock()))
        stack.enter_context(patch("models.ts_model.save_scaler"))
        stack.enter_context(patch("models.ts_model.torch.save"))
        ts_mod.fine_tune_general(universe, cfg)

    meta_path = pathlib.Path(cfg["model"]["ts"]["general_model_dir"]) / "metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["universe"] == universe
    assert meta["n_tickers"] == 1
    assert "last_trained" in meta


def test_fine_tune_general_skips_bad_tickers(tmp_path):
    """If fetch_ticker raises for a ticker, it should be skipped (no crash)."""
    cfg = {
        **_CONFIG,
        "model": {"ts": {**_CONFIG["model"]["ts"], "general_model_dir": str(tmp_path / "g")}},
    }
    df = _make_price_df(400)
    folds = [(df.iloc[:252], df.iloc[252:316])]

    mock_pipeline = MagicMock()
    mock_pipeline.tokenizer.config.prediction_length = 64

    def _fetch_side_effect(ticker, config):
        if ticker == "BAD":
            raise ValueError("bad ticker")
        return df

    with ExitStack() as stack:
        stack.enter_context(patch.object(ts_mod, "_CHRONOS_AVAILABLE", True))
        stack.enter_context(patch("models.ts_model.load_base_model", return_value=mock_pipeline))
        stack.enter_context(patch("models.ts_model.fetch_ticker", side_effect=_fetch_side_effect))
        stack.enter_context(patch("models.ts_model.build_walk_forward_folds", return_value=folds))
        stack.enter_context(patch("models.ts_model._fine_tune", return_value=mock_pipeline))
        stack.enter_context(patch("models.ts_model.fit_scaler", return_value=MagicMock()))
        stack.enter_context(patch("models.ts_model.save_scaler"))
        stack.enter_context(patch("models.ts_model.torch.save"))
        ts_mod.fine_tune_general(["BAD", "AAPL"], cfg)   # should not raise


# ── get_ticker_model ──────────────────────────────────────────────────────────

def test_get_ticker_model_uses_cache_when_fresh(tmp_path):
    """If cache is fresh, no fine-tuning should occur."""
    ticker_dir = tmp_path / "cache" / "AAPL"
    ticker_dir.mkdir(parents=True)
    meta = {"fine_tuned_at": datetime.now(timezone.utc).isoformat()}
    (ticker_dir / "metadata.json").write_text(json.dumps(meta))
    (ticker_dir / "model_state.pt").touch()

    cfg = {
        **_CONFIG,
        "model": {"ts": {**_CONFIG["model"]["ts"],
                          "cache_dir": str(tmp_path / "cache"),
                          "general_model_dir": str(tmp_path / "general")}},
    }

    mock_pipeline = MagicMock()
    with patch("models.ts_model._load_pipeline", return_value=mock_pipeline) as mock_load, \
         patch("models.ts_model._fine_tune") as mock_ft, \
         patch.object(ts_mod, "_CHRONOS_AVAILABLE", True):
        result = ts_mod.get_ticker_model("AAPL", cfg)

    mock_load.assert_called_once()
    mock_ft.assert_not_called()
    assert result is mock_pipeline


def test_get_ticker_model_rebuilds_when_stale(tmp_path):
    """Stale cache triggers fetch → fine-tune → save."""
    ticker_dir = tmp_path / "cache" / "AAPL"
    ticker_dir.mkdir(parents=True)
    stale_time = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    meta = {"fine_tuned_at": stale_time}
    (ticker_dir / "metadata.json").write_text(json.dumps(meta))
    (ticker_dir / "model_state.pt").touch()

    cfg = {
        **_CONFIG,
        "model": {"ts": {**_CONFIG["model"]["ts"],
                          "cache_dir": str(tmp_path / "cache"),
                          "general_model_dir": str(tmp_path / "general")}},
    }

    df = _make_price_df(400)
    folds = [(df.iloc[:252], df.iloc[252:316])]
    mock_pipeline = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(patch.object(ts_mod, "_CHRONOS_AVAILABLE", True))
        stack.enter_context(patch("models.ts_model.fetch_ticker", return_value=df))
        stack.enter_context(patch("models.ts_model.build_walk_forward_folds", return_value=folds))
        stack.enter_context(patch("models.ts_model.load_base_model", return_value=mock_pipeline))
        mock_ft = stack.enter_context(patch("models.ts_model._fine_tune", return_value=mock_pipeline))
        stack.enter_context(patch("models.ts_model.fit_scaler", return_value=MagicMock()))
        stack.enter_context(patch("models.ts_model.save_scaler"))
        stack.enter_context(patch("models.ts_model.torch.save"))
        ts_mod.get_ticker_model("AAPL", cfg)

    mock_ft.assert_called_once()


def test_get_ticker_model_raises_if_no_folds(tmp_path):
    """If there's not enough data to form any fold, raise ValueError."""
    cfg = {
        **_CONFIG,
        "model": {"ts": {**_CONFIG["model"]["ts"],
                          "cache_dir": str(tmp_path / "cache"),
                          "general_model_dir": str(tmp_path / "general")}},
    }
    with ExitStack() as stack:
        stack.enter_context(patch.object(ts_mod, "_CHRONOS_AVAILABLE", True))
        stack.enter_context(patch("models.ts_model.fetch_ticker", return_value=_make_price_df(10)))
        stack.enter_context(patch("models.ts_model.build_walk_forward_folds", return_value=[]))
        stack.enter_context(patch("models.ts_model.load_base_model", return_value=MagicMock()))
        with pytest.raises(ValueError, match="Not enough price history"):
            ts_mod.get_ticker_model("AAPL", cfg)


def test_get_ticker_model_saves_metadata(tmp_path):
    """After rebuilding, metadata.json must exist in the cache dir."""
    cfg = {
        **_CONFIG,
        "model": {"ts": {**_CONFIG["model"]["ts"],
                          "cache_dir": str(tmp_path / "cache"),
                          "general_model_dir": str(tmp_path / "general")}},
    }
    df = _make_price_df(400)
    folds = [(df.iloc[:252], df.iloc[252:316])]
    mock_pipeline = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(patch.object(ts_mod, "_CHRONOS_AVAILABLE", True))
        stack.enter_context(patch("models.ts_model.fetch_ticker", return_value=df))
        stack.enter_context(patch("models.ts_model.build_walk_forward_folds", return_value=folds))
        stack.enter_context(patch("models.ts_model.load_base_model", return_value=mock_pipeline))
        stack.enter_context(patch("models.ts_model._fine_tune", return_value=mock_pipeline))
        stack.enter_context(patch("models.ts_model.fit_scaler", return_value=MagicMock()))
        stack.enter_context(patch("models.ts_model.save_scaler"))
        stack.enter_context(patch("models.ts_model.torch.save"))
        ts_mod.get_ticker_model("AAPL", cfg)

    meta_path = tmp_path / "cache" / "AAPL" / "metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["ticker"] == "AAPL"
    assert "fine_tuned_at" in meta
    assert "data_until" in meta


# ── predict ───────────────────────────────────────────────────────────────────

def test_predict_returns_float(tmp_path):
    import torch
    cfg = {**_CONFIG, "model": {"ts": {**_CONFIG["model"]["ts"],
                                        "cache_dir": str(tmp_path / "cache"),
                                        "general_model_dir": str(tmp_path / "g")}}}
    df = _make_price_df(100)
    forecast = torch.tensor([[[105.0] * 5] * 5])  # [1, 5, 5]
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = forecast

    with ExitStack() as stack:
        stack.enter_context(patch.object(ts_mod, "_CHRONOS_AVAILABLE", True))
        stack.enter_context(patch("models.ts_model.get_ticker_model", return_value=mock_pipeline))
        stack.enter_context(patch("models.ts_model.fetch_ticker", return_value=df))
        stack.enter_context(patch.object(ts_mod, "torch", torch))
        result = predict("AAPL", 5, cfg)

    assert isinstance(result, float)


def test_predict_returns_value_in_unit_interval(tmp_path):
    import torch
    cfg = {**_CONFIG, "model": {"ts": {**_CONFIG["model"]["ts"],
                                        "cache_dir": str(tmp_path / "cache"),
                                        "general_model_dir": str(tmp_path / "g")}}}
    df = _make_price_df(100)
    forecast = torch.tensor([[[102.0] * 5] * 5])
    mock_pipeline = MagicMock()
    mock_pipeline.predict.return_value = forecast

    with ExitStack() as stack:
        stack.enter_context(patch.object(ts_mod, "_CHRONOS_AVAILABLE", True))
        stack.enter_context(patch("models.ts_model.get_ticker_model", return_value=mock_pipeline))
        stack.enter_context(patch("models.ts_model.fetch_ticker", return_value=df))
        stack.enter_context(patch.object(ts_mod, "torch", torch))
        score = predict("AAPL", 5, cfg)

    assert 0.0 <= score <= 1.0


def test_predict_raises_without_chronos():
    with patch.object(ts_mod, "_CHRONOS_AVAILABLE", False):
        with pytest.raises(ImportError):
            predict("AAPL", 5, _CONFIG)


def test_predict_neutral_on_tiny_data(tmp_path):
    """With fewer than 2 price rows, score must be 0.5 (neutral)."""
    cfg = {**_CONFIG, "model": {"ts": {**_CONFIG["model"]["ts"],
                                        "cache_dir": str(tmp_path / "cache"),
                                        "general_model_dir": str(tmp_path / "g")}}}
    df = _make_price_df(1)   # only 1 row
    mock_pipeline = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(patch.object(ts_mod, "_CHRONOS_AVAILABLE", True))
        stack.enter_context(patch("models.ts_model.get_ticker_model", return_value=mock_pipeline))
        stack.enter_context(patch("models.ts_model.fetch_ticker", return_value=df))
        score = predict("AAPL", 5, cfg)

    assert score == 0.5
