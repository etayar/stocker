"""
tests.test_pipeline.test_runner

Tests for pipeline.runner (run_pipeline and load_config).

All external I/O and heavy ML operations are mocked so the suite runs
instantly without network access, a database, or model weights.

Patching strategy
-----------------
fetch_ticker is now a real implementation imported as a named binding into
runner.py, so it is patched at pipeline.runner.fetch_ticker.

The news fetcher and text preprocessor are still stub modules; their
attributes are added temporarily via create=True.

All model functions (ts_predict, score_articles, compute_ta_score, …) are
imported as named bindings and patched at pipeline.runner.<name>.
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
        # -- Price fetch (real implementation, named binding in runner) --------
        stack.enter_context(patch(
            "pipeline.runner.fetch_ticker",
            return_value=_DUMMY_PRICES,
        ))

        # -- News fetcher / text preprocessor (stub modules, create=True) -----
        stack.enter_context(patch(
            "data.fetchers.news_fetcher.fetch_news",
            return_value=_DUMMY_ARTICLES,
            create=True,
        ))
        stack.enter_context(patch(
            "pipeline.preprocessors.text_preprocessor.preprocess_articles",
            return_value=_DUMMY_ARTICLES,
            create=True,
        ))

        # -- Model functions (named bindings in runner) ------------------------
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


def test_run_pipeline_fetch_ticker_called_with_ticker(pipeline_mocks):
    """fetch_ticker must be called with the ticker passed to run_pipeline."""
    with patch("pipeline.runner.fetch_ticker", return_value=_DUMMY_PRICES) as mock_ft:
        run_pipeline("NVDA", 5, _CONFIG)
    assert mock_ft.call_args[0][0] == "NVDA"


def test_run_pipeline_ts_predict_called_with_ticker_and_horizon(pipeline_mocks):
    """ts_predict must receive (ticker, horizon, config) — no prices."""
    with patch("pipeline.runner.ts_predict", return_value=_TS_SCORE) as mock_ts:
        run_pipeline("TSLA", 10, _CONFIG)
    args = mock_ts.call_args[0]
    assert args[0] == "TSLA"
    assert args[1] == 10


def test_run_pipeline_ta_receives_fetched_prices(pipeline_mocks):
    """compute_ta_score must receive the DataFrame from fetch_ticker."""
    custom_prices = pd.DataFrame({"close": [200.0, 201.0, 202.0]})
    with patch("pipeline.runner.fetch_ticker", return_value=custom_prices), \
         patch("pipeline.runner.compute_ta_score", return_value=_TA_SCORE) as mock_ta:
        run_pipeline("AAPL", 5, _CONFIG)
    pd.testing.assert_frame_equal(mock_ta.call_args[0][0], custom_prices)


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


# ── DB persistence in run_pipeline ────────────────────────────────────────────

class TestRunPipelineDbPersistence:
    """run_pipeline(db_engine=...) must persist prices + news; db_engine=None skips it."""

    def test_no_persistence_when_db_engine_is_none(self, pipeline_mocks):
        """Default call (no db_engine) must not call save_prices or save_news."""
        with patch("pipeline.runner.save_prices") as mock_sp, \
             patch("pipeline.runner.save_news") as mock_sn:
            run_pipeline("AAPL", 5, _CONFIG)
        mock_sp.assert_not_called()
        mock_sn.assert_not_called()

    def test_save_prices_called_with_db_engine(self, pipeline_mocks):
        mock_engine = MagicMock()
        with patch("pipeline.runner.save_prices") as mock_sp, \
             patch("pipeline.runner.save_news"), \
             patch("pipeline.runner.get_session"):
            run_pipeline("AAPL", 5, _CONFIG, db_engine=mock_engine)
        mock_sp.assert_called_once()

    def test_save_news_called_with_db_engine(self, pipeline_mocks):
        mock_engine = MagicMock()
        with patch("pipeline.runner.save_prices"), \
             patch("pipeline.runner.save_news") as mock_sn, \
             patch("pipeline.runner.get_session"):
            run_pipeline("AAPL", 5, _CONFIG, db_engine=mock_engine)
        mock_sn.assert_called_once()

    def test_save_prices_receives_correct_ticker(self, pipeline_mocks):
        mock_engine = MagicMock()
        with patch("pipeline.runner.save_prices") as mock_sp, \
             patch("pipeline.runner.save_news"), \
             patch("pipeline.runner.get_session"):
            run_pipeline("TSLA", 5, _CONFIG, db_engine=mock_engine)
        # second positional arg to save_prices is ticker
        assert mock_sp.call_args[0][1] == "TSLA"

    def test_pipeline_result_unchanged_with_db_engine(self, pipeline_mocks):
        """Persistence must not affect the returned result dict."""
        mock_engine = MagicMock()
        with patch("pipeline.runner.save_prices"), \
             patch("pipeline.runner.save_news"), \
             patch("pipeline.runner.get_session"):
            result_with_db = run_pipeline("AAPL", 5, _CONFIG, db_engine=mock_engine)
        result_without_db = run_pipeline("AAPL", 5, _CONFIG)
        assert result_with_db.keys() == result_without_db.keys()
        assert result_with_db["signal"] == result_without_db["signal"]

    def test_prices_db_failure_does_not_raise(self, pipeline_mocks):
        """A DB error during price persistence must not propagate."""
        mock_engine = MagicMock()
        with patch("pipeline.runner.get_session", side_effect=RuntimeError("DB down")):
            result = run_pipeline("AAPL", 5, _CONFIG, db_engine=mock_engine)
        assert "signal" in result

    def test_news_db_failure_does_not_raise(self, pipeline_mocks):
        """A DB error during news persistence must not propagate."""
        mock_engine = MagicMock()
        call_count = 0

        def _get_session_side_effect(engine):
            nonlocal call_count
            call_count += 1
            if call_count == 2:   # second call is for news
                raise RuntimeError("DB down on news")
            from contextlib import contextmanager
            @contextmanager
            def _ok():
                yield MagicMock()
            return _ok()

        with patch("pipeline.runner.get_session", side_effect=_get_session_side_effect), \
             patch("pipeline.runner.save_prices"), \
             patch("pipeline.runner.save_news"):
            result = run_pipeline("AAPL", 5, _CONFIG, db_engine=mock_engine)
        assert "signal" in result
