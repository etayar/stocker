"""
models.ts_model

Three-tier Chronos-based time-series model for stock price forecasting.

Model hierarchy (from CLAUDE.md):
  1. Base   : Pretrained amazon/chronos-t5-small (HuggingFace, frozen)
  2. General: Chronos fine-tuned on a broad stock universe, rebuilt every 14 days.
              Saved to:  models/general/model_state.pt + scaler.pkl + metadata.json
  3. Ticker : General model fine-tuned on one specific ticker, built on demand.
              Cache TTL: config model.ts.cache_ttl_days (default 7 days).
              Saved to:  models/cache/{TICKER}/model_state.pt + scaler.pkl + metadata.json

Public interface:
  load_base_model(config)  → ChronosPipeline (pretrained, unmodified)
  fine_tune_general(universe, config)  → None  (trains + saves general model)
  get_ticker_model(ticker, config)     → ChronosPipeline (fine-tuned, from cache or fresh)
  predict(ticker, horizon, config)     → float ∈ [0, 1]  ← only method called by runner
"""

from __future__ import annotations

import json
import logging
import math
import pathlib
from datetime import datetime, timezone

import numpy as np

try:
    import torch
    from chronos import ChronosPipeline
    _CHRONOS_AVAILABLE = True
except ImportError:
    _CHRONOS_AVAILABLE = False
    torch = None          # type: ignore[assignment]
    ChronosPipeline = None  # type: ignore[assignment]

from pipeline.data_loader import (
    build_walk_forward_folds,
    fetch_ticker,
    fit_scaler,
    load_scaler,
    save_scaler,
)

logger = logging.getLogger(__name__)

# ── Internal helpers ───────────────────────────────────────────────────────────

def _require_chronos() -> None:
    if not _CHRONOS_AVAILABLE:
        raise ImportError(
            "chronos-forecasting and torch are required for the TS model. "
            "Install with: pip install chronos-forecasting"
        )


def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid, clipped to avoid overflow."""
    return float(1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, z)))))


def _ts_score_from_forecast(
    forecast: "torch.Tensor",
    current_price: float,
    return_std: float,
) -> float:
    """Convert a Chronos forecast distribution into a [0, 1] buy-likelihood score.

    Score formula:
        median_end_price  = median of forecast samples for the last horizon day
        log_return        = log(median_end_price / current_price)
        z                 = log_return / return_std
        ts_score          = sigmoid(z)

    > 0.5 → model expects price rise (bullish)
    < 0.5 → model expects price fall (bearish)
    = 0.5 → neutral (predicted return ≈ 0 or return_std is near zero)

    Parameters
    ----------
    forecast      : Tensor [batch=1, num_samples, horizon] from ChronosPipeline.predict.
    current_price : Most recent observed price.
    return_std    : Std of recent log-returns (historical volatility proxy).
    """
    if return_std < 1e-10:
        return 0.5

    # Median across samples for the last day of the forecast horizon
    last_day_samples = forecast[0, :, -1]           # [num_samples]
    median_end = float(last_day_samples.median())

    if median_end <= 0.0 or current_price <= 0.0:
        return 0.5

    log_return = math.log(median_end / current_price)
    return _sigmoid(log_return / return_std)


def _metadata_path(model_dir: pathlib.Path) -> pathlib.Path:
    return model_dir / "metadata.json"


def _checkpoint_path(model_dir: pathlib.Path) -> pathlib.Path:
    return model_dir / "model_state.pt"


def _scaler_path(model_dir: pathlib.Path) -> pathlib.Path:
    return model_dir / "scaler.pkl"


def _is_fresh(model_dir: pathlib.Path, ttl_days: int) -> bool:
    """Return True if the model at *model_dir* is younger than *ttl_days*."""
    meta_path = _metadata_path(model_dir)
    if not meta_path.exists() or not _checkpoint_path(model_dir).exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        key = "fine_tuned_at" if "fine_tuned_at" in meta else "last_trained"
        trained_at = datetime.fromisoformat(meta[key])
        # Make timezone-aware for comparison
        if trained_at.tzinfo is None:
            trained_at = trained_at.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - trained_at).total_seconds() / 86400
        return age_days < ttl_days
    except Exception:
        return False


def _save_pipeline(pipeline: "ChronosPipeline", model_dir: pathlib.Path) -> None:
    """Save fine-tuned T5 weights to *model_dir*."""
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pipeline.model.model.state_dict(), _checkpoint_path(model_dir))


def _load_pipeline(model_dir: pathlib.Path, config: dict) -> "ChronosPipeline":
    """Load a fine-tuned pipeline by restoring weights into a fresh base model."""
    pipeline = load_base_model(config)
    state = torch.load(
        _checkpoint_path(model_dir),
        map_location="cpu",
        weights_only=True,
    )
    pipeline.model.model.load_state_dict(state)
    return pipeline


def _fine_tune(
    pipeline: "ChronosPipeline",
    folds: list,
    config: dict,
) -> "ChronosPipeline":
    """Fine-tune *pipeline* on walk-forward *folds* using the Chronos T5 training loop.

    Each fold's training data is used as the context and the first
    ``prediction_length`` rows of the val data are used as the target.
    Folds whose val window is shorter than the required prediction_length are skipped.

    Parameters
    ----------
    pipeline : ChronosPipeline to fine-tune (modified in-place, also returned).
    folds    : list of (train_df, val_df) from build_walk_forward_folds.
    config   : Loaded config dict.

    Returns
    -------
    The same pipeline, with updated T5 weights.
    """
    ts_cfg = config["model"]["ts"]
    n_epochs = int(ts_cfg.get("n_training_epochs", 3))
    lr       = float(ts_cfg.get("learning_rate", 1e-4))

    t5  = pipeline.model.model
    tok = pipeline.tokenizer
    pred_len = tok.config.prediction_length  # 64 for chronos-t5-small

    optimizer = torch.optim.Adam(t5.parameters(), lr=lr)
    t5.train()

    for epoch in range(n_epochs):
        total_loss = 0.0
        n_batches  = 0
        for train_df, val_df in folds:
            # Need at least pred_len rows in val for a complete training target
            if len(val_df) < pred_len:
                continue

            close_ctx    = train_df["close"].values.astype(float)
            close_target = val_df["close"].values[:pred_len].astype(float)

            ctx    = torch.tensor([close_ctx],    dtype=torch.float32)
            target = torch.tensor([close_target], dtype=torch.float32)

            input_ids, attn_mask, scale = tok.context_input_transform(ctx)
            label_ids, _               = tok.label_input_transform(target, scale)

            output = t5(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=label_ids,
            )
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()

            total_loss += output.loss.item()
            n_batches  += 1

        if n_batches:
            logger.debug("epoch %d/%d  avg_loss=%.4f", epoch + 1, n_epochs,
                         total_loss / n_batches)

    t5.eval()
    return pipeline


# ── Public interface ───────────────────────────────────────────────────────────

def load_base_model(config: dict) -> "ChronosPipeline":
    """Load the pretrained Chronos model from HuggingFace (or local cache).

    Parameters
    ----------
    config : Loaded config dict. Reads ``model.ts.chronos_base``.

    Returns
    -------
    ChronosPipeline with frozen weights (as-downloaded, no fine-tuning).

    Raises
    ------
    ImportError : If chronos-forecasting is not installed.
    """
    _require_chronos()
    model_name = config["model"]["ts"]["chronos_base"]
    pipeline = ChronosPipeline.from_pretrained(
        model_name,
        device_map="cpu",
        dtype=torch.float32,
    )
    pipeline.model.model.eval()
    return pipeline


def fine_tune_general(universe: list[str], config: dict) -> None:
    """Fine-tune Chronos on the full *universe* of tickers and save to disk.

    Flow:
      1. Load pretrained base model.
      2. For each ticker in universe: fetch prices → build walk-forward folds.
      3. Fine-tune on the concatenation of all folds.
      4. Save model weights, a fitted scaler (last ticker), and metadata.json.

    Parameters
    ----------
    universe : List of ticker symbols, e.g. ``["AAPL", "MSFT", "NVDA"]``.
    config   : Loaded config dict.
    """
    _require_chronos()
    ts_cfg  = config["model"]["ts"]
    wf_cfg  = config["data"]["walk_forward"]
    general_dir = pathlib.Path(ts_cfg["general_model_dir"])

    train_w = int(wf_cfg["train_window_days"])
    val_w   = int(wf_cfg["val_window_days"])
    step    = int(wf_cfg["step_days"])

    logger.info("fine_tune_general: loading base model …")
    pipeline = load_base_model(config)

    all_folds: list = []
    last_scaler = None

    for ticker in universe:
        try:
            df = fetch_ticker(ticker, config)
        except Exception as exc:
            logger.warning("Skipping %s during general training: %s", ticker, exc)
            continue

        folds = build_walk_forward_folds(df, train_w, val_w, step)
        if not folds:
            logger.warning("No valid folds for %s — skipping.", ticker)
            continue

        # Fit scaler on the first fold's train data (one per ticker)
        last_scaler = fit_scaler(folds[0][0])
        all_folds.extend(folds)

    if not all_folds:
        raise RuntimeError("No training folds produced from universe — aborting.")

    logger.info("fine_tune_general: %d folds across %d tickers", len(all_folds), len(universe))
    pipeline = _fine_tune(pipeline, all_folds, config)

    # Save model + scaler + metadata
    _save_pipeline(pipeline, general_dir)
    if last_scaler is not None:
        save_scaler(last_scaler, str(_scaler_path(general_dir)))

    meta = {
        "last_trained": datetime.now(timezone.utc).isoformat(),
        "universe":     universe,
        "n_tickers":    len(universe),
    }
    _metadata_path(general_dir).write_text(json.dumps(meta, indent=2))
    logger.info("fine_tune_general: saved to %s", general_dir)


def get_ticker_model(ticker: str, config: dict) -> "ChronosPipeline":
    """Return a fine-tuned ChronosPipeline for *ticker*, with automatic caching.

    Flow (from CLAUDE.md):
      1. Check models/cache/{ticker}/metadata.json
         - Exists AND age < cache_ttl_days → load and return immediately.
         - Missing or stale → rebuild:
      2. Fetch price history via yfinance.
      3. Build walk-forward folds.
      4. Load general model (if available) or base Chronos.
      5. Fine-tune on ticker data.
      6. Save model + scaler + metadata to models/cache/{ticker}/.
      7. Return fine-tuned pipeline.

    Parameters
    ----------
    ticker : Stock symbol.
    config : Loaded config dict.

    Returns
    -------
    Fine-tuned ChronosPipeline ready for inference.
    """
    _require_chronos()
    ts_cfg     = config["model"]["ts"]
    wf_cfg     = config["data"]["walk_forward"]
    ttl        = int(ts_cfg.get("cache_ttl_days", 7))
    cache_dir  = pathlib.Path(ts_cfg["cache_dir"]) / ticker.upper()
    general_dir = pathlib.Path(ts_cfg["general_model_dir"])

    # ── 1. Cache hit ──────────────────────────────────────────────────────────
    if _is_fresh(cache_dir, ttl):
        logger.info("get_ticker_model: cache hit for %s (< %d days old)", ticker, ttl)
        return _load_pipeline(cache_dir, config)

    logger.info("get_ticker_model: rebuilding model for %s …", ticker)

    # ── 2. Fetch price history ────────────────────────────────────────────────
    df = fetch_ticker(ticker, config)

    # ── 3. Build walk-forward folds ───────────────────────────────────────────
    train_w = int(wf_cfg["train_window_days"])
    val_w   = int(wf_cfg["val_window_days"])
    step    = int(wf_cfg["step_days"])
    folds   = build_walk_forward_folds(df, train_w, val_w, step)

    if not folds:
        raise ValueError(
            f"Not enough price history for '{ticker}' to build any walk-forward fold "
            f"(need at least {train_w + val_w} rows, got {len(df)})."
        )

    # ── 4. Load general model or base Chronos ─────────────────────────────────
    if _is_fresh(general_dir, ttl_days=365):  # general model never expires automatically
        logger.info("get_ticker_model: loading general model from %s", general_dir)
        pipeline = _load_pipeline(general_dir, config)
    else:
        logger.info("get_ticker_model: no general model found — using base Chronos")
        pipeline = load_base_model(config)

    # ── 5. Fine-tune on ticker data ───────────────────────────────────────────
    pipeline = _fine_tune(pipeline, folds, config)

    # ── 6. Save to cache ──────────────────────────────────────────────────────
    _save_pipeline(pipeline, cache_dir)
    scaler = fit_scaler(folds[0][0])
    save_scaler(scaler, str(_scaler_path(cache_dir)))

    meta = {
        "ticker":        ticker.upper(),
        "fine_tuned_at": datetime.now(timezone.utc).isoformat(),
        "data_until":    str(df.index[-1].date()),
    }
    _metadata_path(cache_dir).write_text(json.dumps(meta, indent=2))

    logger.info("get_ticker_model: saved ticker model to %s", cache_dir)
    return pipeline


def predict(ticker: str, horizon: int, config: dict) -> float:
    """Compute the TS buy-likelihood score for *ticker* over *horizon* days.

    This is the only function called by the rest of the system.

    Steps:
      1. Retrieve (or build) a fine-tuned pipeline via get_ticker_model.
      2. Fetch recent price history for context.
      3. Run Chronos inference → sample distribution over next *horizon* days.
      4. Derive score ∈ [0, 1] from the forecast distribution.

    Parameters
    ----------
    ticker  : Stock ticker symbol.
    horizon : Number of days ahead to forecast.
    config  : Loaded config dict.

    Returns
    -------
    ts_score ∈ [0, 1]:
      > 0.5 → bullish (model expects price rise)
      < 0.5 → bearish (model expects price fall)
      = 0.5 → neutral
    """
    _require_chronos()
    ts_cfg = config["model"]["ts"]

    # ── 1. Get fine-tuned model ───────────────────────────────────────────────
    pipeline = get_ticker_model(ticker, config)

    # ── 2. Fetch recent price history for context ─────────────────────────────
    df = fetch_ticker(ticker, config)
    close = df["close"].values.astype(float)

    if len(close) < 2:
        return 0.5  # not enough data for a meaningful prediction

    # Historical return std for score normalisation
    log_returns = np.log(close[1:] / close[:-1])
    return_std  = float(log_returns.std())

    # ── 3. Run Chronos inference ──────────────────────────────────────────────
    context_length = int(ts_cfg.get("context_length", 512))
    num_samples    = int(ts_cfg.get("num_samples", 20))
    ctx_prices     = close[-context_length:]
    ctx            = torch.tensor(np.array([ctx_prices]), dtype=torch.float32)

    with torch.no_grad():
        forecast = pipeline.predict(
            inputs=ctx,
            prediction_length=horizon,
            num_samples=num_samples,
            limit_prediction_length=False,
        )
    # forecast shape: [1, num_samples, horizon]

    # ── 4. Convert distribution → [0, 1] score ───────────────────────────────
    current_price = float(close[-1])
    return _ts_score_from_forecast(forecast, current_price, return_std)
