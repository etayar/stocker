"""
tests.test_pipeline.test_runner

Tests for pipeline.runner (run_pipeline and load_config).

All external I/O and heavy ML operations are mocked so the suite runs
instantly without network access, a database, or model weights.

Patching strategy
-----------------
Fetchers / preprocessors are stub modules (docstring only), so they are
imported in runner.py as module objects and called via attribute access
(e.g. _price_fetcher.fetch_prices).  We patch those attributes with
create=True so the attribute is added temporarily to the real module object.

Model functions (ts_train, ts_predict, …) are imported into runner.py as
named bindings, so we patch them at pipeline.runner.<name>.
"""

from __future__ import annotations

import math
import pathlib
from contextlib import ExitStack
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from pipeline.runner import load_config, run_pipeline

# ── Shared config / paths ─────────────────────────────────────────────────────

_PROJECT_ROOT = pathlib.Path(__file__).parents[2]   # stocker/
_CONFIG_PATH  = str(_PROJECT_ROOT / "config.yaml")

_CONFIG = {
    "stock": {
        "ticker": "AAPL",
        "start_date": "2020-01-01",
        "prediction_horizon": 5,
    },
    "data": {
        "price_source": "yfinance",
        "news_source": "newsapi",
        "universe": ["AAPL"],
        "walk_forward": {
            "train_window_days": 252,
            "val_window_days":   64,
            "step_days":         21,
        },
    },
    "model": {
        "ts": {
            "chronos_base":      "amazon/chronos-t5-small",
            "cache_ttl_days":    7,
            "general_model_dir": "models/general",
            "cache_dir":         "models/cache",
            "context_length":    64,
            "num_samples":       5,
        },
        "sentiment": {
            "model_name": "ProsusAI/finbert",
            "max_length": 512,
            "batch_size": 16,
        },
        "ta": {
            "rsi_period": 14,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "bb_period": 20,
            "bb_std": 2.0,
        },
    },
    "scoring": {
        "components": ["ts", "llms", "ta"],
        "buy_threshold": 0.65,
        "sell_threshold": 0.35,
    },
}

# ── Fixture: patch every external call in run_pipeline ────────────────────────

_DUMMY_PRICES   = pd.DataFrame({"close": [100.0 + i for i in range(60)]})
_DUMMY_ARTICLES = [{"text": "Strong earnings beat expectations."}]

# Fixed component scores used across multiple tests.
_TS_SCORE   = 0.70
_LLMS_SCORE = 0.60
_TA_SCORE   = 0.80
_EXPECTED_COMPOUND = (_TS_SCORE * _LLMS_SCORE * _TA_SCORE) ** (1.0 / 3.0)


@pytest.fixture()
def pipeline_mocks():
    """Patch every I/O and ML call that run_pipeline makes.

    Yields a dict of the fixed scores so tests can compute expected values.
    """
    with ExitStack() as stack:
        # -- Fetchers (stub modules, create=True adds the attr temporarily) --
        stack.enter_context(patch(
            "data.fetchers.price_fetcher.fetch_prices",
            return_value=_DUMMY_PRICES,
            create=True,
        ))
        stack.enter_context(patch(
            "data.fetchers.news_fetcher.fetch_news",
            return_value=_DUMMY_ARTICLES,
            create=True,
        ))


        # -- Preprocessors (stub modules, create=True) --
        stack.enter_context(patch(
            "pipeline.preprocessors.price_preprocessor.preprocess_prices",
            return_value=(_DUMMY_PRICES, None),
            create=True,
        ))
        stack.enter_context(patch(
            "pipeline.preprocessors.text_preprocessor.preprocess_articles",
            return_value=_DUMMY_ARTICLES,
            create=True,
        ))

        # -- Model functions (bound into runner.py via `from … import`) --
        stack.enter_context(patch("pipeline.runner.ts_predict",
                                  return_value=_TS_SCORE))
        stack.enter_context(patch("pipeline.runner.load_sentiment_model",
                                  return_value=MagicMock()))
        stack.enter_context(patch("pipeline.runner.score_articles",
                                  return_value=_LLMS_SCORE))
        stack.enter_context(patch("pipeline.runner.compute_ta_score",
                                  return_value=_TA_SCORE))

        yield {"ts": _TS_SCORE, "llms": _LLMS_SCORE, "ta": _TA_SCORE}


# ── run_pipeline tests ────────────────────────────────────────────────────────

_EXPECTED_KEYS = {
    "ticker", "run_date", "horizon",
    "ts_score", "llms_score", "ta_score",
    "compound_score", "signal",
}


def test_run_pipeline_returns_expected_keys(pipeline_mocks):
    result = run_pipeline("AAPL", 5, _CONFIG)
    assert set(result.keys()) == _EXPECTED_KEYS


def test_run_pipeline_scores_in_range(pipeline_mocks):
    result = run_pipeline("AAPL", 5, _CONFIG)
    for key in ("ts_score", "llms_score", "ta_score", "compound_score"):
        assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"


def test_run_pipeline_signal_valid(pipeline_mocks):
    result = run_pipeline("AAPL", 5, _CONFIG)
    assert result["signal"] in {"BUY", "SELL", "HOLD"}


def test_run_pipeline_ticker_and_horizon_in_result(pipeline_mocks):
    result = run_pipeline("TSLA", 10, _CONFIG)
    assert result["ticker"] == "TSLA"
    assert result["horizon"] == 10


def test_run_pipeline_run_date_is_iso_date(pipeline_mocks):
    result = run_pipeline("AAPL", 5, _CONFIG)
    # Should parse without error and equal today's UTC date
    parsed = date.fromisoformat(result["run_date"])
    assert isinstance(parsed, date)


def test_run_pipeline_with_mocked_components(pipeline_mocks):
    """Compound score must equal the geometric mean of the three mocked scores."""
    result = run_pipeline("AAPL", 5, _CONFIG)
    assert result["ts_score"]   == pytest.approx(_TS_SCORE)
    assert result["llms_score"] == pytest.approx(_LLMS_SCORE)
    assert result["ta_score"]   == pytest.approx(_TA_SCORE)
    assert result["compound_score"] == pytest.approx(_EXPECTED_COMPOUND, rel=1e-6)


def test_run_pipeline_signal_above_buy_threshold(pipeline_mocks):
    """All-high scores → compound above buy_threshold → BUY."""
    with patch("pipeline.runner.ts_predict",      return_value=0.9), \
         patch("pipeline.runner.score_articles",   return_value=0.9), \
         patch("pipeline.runner.compute_ta_score", return_value=0.9):
        result = run_pipeline("AAPL", 5, _CONFIG)
    assert result["signal"] == "BUY"


def test_run_pipeline_signal_below_sell_threshold(pipeline_mocks):
    """All-low scores → compound below sell_threshold → SELL."""
    with patch("pipeline.runner.ts_predict",      return_value=0.1), \
         patch("pipeline.runner.score_articles",   return_value=0.1), \
         patch("pipeline.runner.compute_ta_score", return_value=0.1):
        result = run_pipeline("AAPL", 5, _CONFIG)
    assert result["signal"] == "SELL"


# ── load_config tests ─────────────────────────────────────────────────────────

def test_load_config_reads_yaml():
    config = load_config(_CONFIG_PATH)
    for key in ("stock", "data", "model", "scoring", "api", "database"):
        assert key in config, f"Missing top-level key: {key}"


def test_load_config_stock_keys():
    config = load_config(_CONFIG_PATH)
    assert "ticker" in config["stock"]
    assert "prediction_horizon" in config["stock"]


def test_load_config_scoring_thresholds():
    config = load_config(_CONFIG_PATH)
    assert 0 < config["scoring"]["sell_threshold"] < config["scoring"]["buy_threshold"] < 1


def test_load_config_env_override_ticker(monkeypatch):
    monkeypatch.setenv("STOCK_TICKER", "TSLA")
    config = load_config(_CONFIG_PATH)
    assert config["stock"]["ticker"] == "TSLA"


def test_load_config_env_override_horizon_is_int(monkeypatch):
    monkeypatch.setenv("PREDICTION_HORIZON", "10")
    config = load_config(_CONFIG_PATH)
    assert config["stock"]["prediction_horizon"] == 10
    assert isinstance(config["stock"]["prediction_horizon"], int)


def test_load_config_env_override_price_source(monkeypatch):
    monkeypatch.setenv("PRICE_SOURCE", "polygon")
    config = load_config(_CONFIG_PATH)
    assert config["data"]["price_source"] == "polygon"


def test_load_config_env_override_news_source(monkeypatch):
    monkeypatch.setenv("NEWS_SOURCE", "finnhub")
    config = load_config(_CONFIG_PATH)
    assert config["data"]["news_source"] == "finnhub"


def test_load_config_no_env_vars_gives_yaml_defaults(monkeypatch):
    """Without env overrides the raw YAML values come through."""
    for var in ("STOCK_TICKER", "PREDICTION_HORIZON", "PRICE_SOURCE", "NEWS_SOURCE"):
        monkeypatch.delenv(var, raising=False)
    config = load_config(_CONFIG_PATH)
    assert config["stock"]["ticker"] == "AAPL"
    assert config["data"]["price_source"] == "yfinance"
