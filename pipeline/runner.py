"""
pipeline.runner

Top-level pipeline orchestrator. Wires together all fetch, preprocess,
model, and scoring steps into a single callable.

Reads configuration from config.yaml (merged with env overrides).

Public interface:
  run_pipeline(ticker: str, horizon: int, config: dict) -> dict
    Executes the full pipeline for one stock.
    Returns a result dict:
    {
      "ticker": str,
      "run_date": str (ISO-8601),
      "horizon": int,
      "ts_score": float,
      "llms_score": float,
      "ta_score": float,
      "compound_score": float,
      "signal": "BUY" | "SELL" | "HOLD",
    }

  load_config(config_path: str = "config.yaml") -> dict
    Loads YAML config and applies env var overrides.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import yaml

# ── Stub module imports (news fetcher / text preprocessor not yet implemented) ──
# Imported as module objects so tests can patch individual attributes with
# create=True without needing the functions to exist yet.
import data.fetchers.news_fetcher as _news_fetcher
import pipeline.preprocessors.text_preprocessor as _text_preprocessor

# ── Implemented imports ────────────────────────────────────────────────────────
# Named bindings so tests can patch at pipeline.runner.<name>.
from pipeline.data_loader import fetch_ticker
from models.ts_model import predict as ts_predict
from models.sentiment_model import load_sentiment_model, score_articles
from models.ta_model import compute_ta_score
from models.scorer import compound_score, interpret_signal

# ── Env-var → config-key mapping ───────────────────────────────────────────────
_STR_OVERRIDES: dict[str, tuple[str, str]] = {
    "STOCK_TICKER": ("stock", "ticker"),
    "PRICE_SOURCE": ("data",  "price_source"),
    "NEWS_SOURCE":  ("data",  "news_source"),
}
_INT_OVERRIDES: dict[str, tuple[str, str]] = {
    "PREDICTION_HORIZON": ("stock", "prediction_horizon"),
}


def load_config(config_path: str = "config.yaml") -> dict:
    """Load config.yaml and apply environment-variable overrides.

    Override mappings (env var → config path):
      STOCK_TICKER        → stock.ticker           (str)
      PREDICTION_HORIZON  → stock.prediction_horizon (int)
      PRICE_SOURCE        → data.price_source      (str)
      NEWS_SOURCE         → data.news_source       (str)

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config dict with any env overrides applied.
    """
    with open(config_path) as f:
        config: dict = yaml.safe_load(f)

    for env_var, (section, key) in _STR_OVERRIDES.items():
        val = os.environ.get(env_var)
        if val is not None:
            config[section][key] = val

    for env_var, (section, key) in _INT_OVERRIDES.items():
        val = os.environ.get(env_var)
        if val is not None:
            config[section][key] = int(val)

    return config


def run_pipeline(ticker: str, horizon: int, config: dict) -> dict:
    """Execute the full Stocker pipeline for one stock.

    Pipeline steps:
      1. Fetch price history           (pipeline.data_loader.fetch_ticker)
      2. Fetch news/text data          (data.fetchers.news_fetcher — stub)
      3. Preprocess articles           (pipeline.preprocessors.text_preprocessor — stub)
      4. Compute ts_score              (models.ts_model.predict — Chronos, self-contained)
      5. Compute llms_score            (models.sentiment_model: load → score_articles)
      6. Compute ta_score              (models.ta_model: compute_ta_score)
      7. Compound via geometric mean   (models.scorer: compound_score)
      8. Interpret signal              (models.scorer: interpret_signal)

    Args:
        ticker:  Stock ticker symbol (e.g. "AAPL").
        horizon: Prediction horizon in days.
        config:  Loaded config dict (from load_config()).

    Returns:
        Result dict::

            {
              "ticker":         str,
              "run_date":       str  (ISO-8601 date, UTC),
              "horizon":        int,
              "ts_score":       float ∈ [0, 1],
              "llms_score":     float ∈ [0, 1],
              "ta_score":       float ∈ [0, 1],
              "compound_score": float ∈ [0, 1],
              "signal":         "BUY" | "SELL" | "HOLD",
            }
    """
    news_source = config["data"]["news_source"]

    # ── 1: Fetch prices (real implementation) ─────────────────────────────────
    prices = fetch_ticker(ticker, config)

    # ── 2 & 3: Fetch + preprocess news (stubs — not yet implemented) ──────────
    articles_raw = _news_fetcher.fetch_news(ticker, news_source)
    articles     = _text_preprocessor.preprocess_articles(articles_raw, ticker, config)

    # ── 4: TS score (Chronos — fetches its own price data internally) ─────────
    ts_score_val = ts_predict(ticker, horizon, config)

    # ── 5: LLMS score ─────────────────────────────────────────────────────────
    sent_model     = load_sentiment_model(config)
    llms_score_val = score_articles(sent_model, articles, config)

    # ── 6: TA score ───────────────────────────────────────────────────────────
    ta_score_val = compute_ta_score(prices, config)

    # ── 7 & 8: Compound + signal ──────────────────────────────────────────────
    cmpd   = compound_score([ts_score_val, llms_score_val, ta_score_val])
    signal = interpret_signal(cmpd, config)

    return {
        "ticker":         ticker,
        "run_date":       datetime.now(timezone.utc).date().isoformat(),
        "horizon":        horizon,
        "ts_score":       ts_score_val,
        "llms_score":     llms_score_val,
        "ta_score":       ta_score_val,
        "compound_score": cmpd,
        "signal":         signal,
    }
