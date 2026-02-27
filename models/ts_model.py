"""
models.ts_model

Time-series model that forecasts short-term price direction
and converts it into a buy-likelihood score in [0, 1].

Model type is configurable via config.yaml → model.ts.model_type:
  - lstm    : LSTM neural network (PyTorch)
  - arima   : ARIMA statistical model (statsmodels)
  - xgboost : gradient boosting on lag features

Training:
  train(prices: pd.DataFrame, config: dict) -> TSModel

Inference:
  predict(model: TSModel, prices: pd.DataFrame, horizon: int) -> float
    Returns ts_score ∈ [0, 1]:
      > 0.5 → bullish (expect price rise over horizon)
      < 0.5 → bearish (expect price fall)
      = 0.5 → neutral

Persistence:
  save_model(model, path: str)
  load_model(path: str) -> TSModel
"""

from __future__ import annotations

import pathlib
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass
class TSModel:
    """Ridge regression trained to predict N-day cumulative log-returns.

    Attributes:
        model:       Fitted sklearn Ridge instance.
        window_size: Number of past log-return days used as features.
        horizon:     Days ahead the model was trained to predict.
        return_std:  Std of training log-returns; used to normalise the
                     predicted return before applying sigmoid so that the
                     score is calibrated relative to typical daily movement.
    """

    model: object   # sklearn Ridge (typed as object — no import at call-site)
    window_size: int
    horizon: int
    return_std: float


# ── internal helpers ──────────────────────────────────────────────────────────

def _log_returns(prices: pd.DataFrame) -> np.ndarray:
    """Compute log returns from the close column."""
    close = prices["close"].values.astype(float)
    return np.log(close[1:] / close[:-1])


def _build_dataset(
    returns: np.ndarray,
    window: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble (X, y) from a 1-D returns array.

    X[i] = returns[i : i+window]               — past window log-returns
    y[i] = sum(returns[i+window : i+window+horizon])  — future cumulative return

    Args:
        returns: 1-D array of log-returns.
        window:  Lookback size in days.
        horizon: Forward lookahead in days.

    Returns:
        Tuple (X, y) with shapes (n_samples, window) and (n_samples,).

    Raises:
        ValueError: If returns is too short to produce at least one sample.
    """
    n = len(returns)
    n_samples = n - window - horizon + 1
    if n_samples <= 0:
        raise ValueError(
            f"Not enough data: need at least {window + horizon} return values "
            f"({window + horizon + 1} price rows), got {n}"
        )
    X = np.stack([returns[i : i + window] for i in range(n_samples)])
    y = np.array(
        [returns[i + window : i + window + horizon].sum() for i in range(n_samples)]
    )
    return X, y


def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid, clipped to avoid overflow."""
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0))))


# ── public interface ──────────────────────────────────────────────────────────

def train(prices: pd.DataFrame, config: dict) -> TSModel:
    """Fit a Ridge regression model on rolling window log-return features.

    Features are the last `window_size` daily log-returns.
    The regression target is the sum of log-returns over the next `horizon`
    days — i.e. log(close[t+horizon] / close[t]).

    Args:
        prices: DataFrame with a 'close' column, sorted oldest → newest.
        config: Loaded config dict. Reads:
                  model.ts.window_size
                  stock.prediction_horizon

    Returns:
        Fitted TSModel.

    Raises:
        ValueError: If prices is too short for even one training sample.
    """
    window = config["model"]["ts"]["window_size"]
    horizon = config["stock"]["prediction_horizon"]

    returns = _log_returns(prices)
    return_std = float(returns.std())
    if return_std < 1e-10:
        return_std = 1.0  # flat/near-flat series → default normalization

    X, y = _build_dataset(returns, window, horizon)

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    return TSModel(
        model=model,
        window_size=window,
        horizon=horizon,
        return_std=return_std,
    )


def predict(model_obj: TSModel, prices: pd.DataFrame, horizon: int) -> float:
    """Predict a buy-likelihood score from the most recent price window.

    Score formula:
        predicted_return = Ridge.predict(last_window_returns)
        z                = predicted_return / model.return_std
        ts_score         = sigmoid(z)

    This maps a bullish prediction to > 0.5, bearish to < 0.5,
    and neutral (predicted return ≈ 0) to ≈ 0.5.

    Args:
        model_obj: Fitted TSModel.
        prices:    DataFrame with 'close' column (at least window_size+1 rows).
        horizon:   Intended prediction horizon in days.  The model was
                   trained for model_obj.horizon; this parameter is accepted
                   for interface consistency and future use.

    Returns:
        ts_score ∈ [0, 1].

    Raises:
        ValueError: If prices has fewer rows than window_size + 1.
    """
    min_rows = model_obj.window_size + 1
    if len(prices) < min_rows:
        raise ValueError(
            f"Need at least {min_rows} price rows to predict, got {len(prices)}"
        )
    returns = _log_returns(prices)
    features = returns[-model_obj.window_size :]
    predicted_return = float(model_obj.model.predict(features.reshape(1, -1))[0])
    return _sigmoid(predicted_return / model_obj.return_std)


def save_model(model_obj: TSModel, path: str) -> None:
    """Persist a TSModel to disk via pickle.

    Args:
        model_obj: TSModel to save.
        path:      Destination file path.  Parent directories are created
                   automatically.
    """
    dest = pathlib.Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        pickle.dump(model_obj, f)


def load_model(path: str) -> TSModel:
    """Load a TSModel previously saved with save_model.

    Args:
        path: File path written by save_model.

    Returns:
        Restored TSModel.
    """
    with pathlib.Path(path).open("rb") as f:
        return pickle.load(f)
