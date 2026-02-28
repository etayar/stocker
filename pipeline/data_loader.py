"""
pipeline.data_loader

Fetches and prepares price data for the TS model pipeline.

Public interface:
  fetch_ticker(ticker, config)
    → pd.DataFrame  OHLCV, lowercase columns, sorted oldest → newest,
                    DatetimeIndex, fetched up to today via yfinance.

  build_walk_forward_folds(df, train_window_days, val_window_days, step_days)
    → list[tuple[pd.DataFrame, pd.DataFrame]]
      Temporally ordered folds; val windows never overlap.
      NEVER shuffles. NEVER uses future data.

  fit_scaler(train_df)  → sklearn scaler fitted on train close only.
  apply_scaler(df, scaler) → new DataFrame with scaled close column.
  save_scaler(scaler, path) / load_scaler(path)  → pickle persistence.
"""

from __future__ import annotations

import pathlib
import pickle
from typing import List, Tuple

import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


# ── Fetch ──────────────────────────────────────────────────────────────────────

def fetch_ticker(ticker: str, config: dict) -> pd.DataFrame:
    """Download full OHLCV history for *ticker* up to today using yfinance.

    Parameters
    ----------
    ticker : Stock symbol, e.g. "AAPL".
    config : Loaded config dict. Reads ``stock.start_date`` for the
             beginning of the historical window.

    Returns
    -------
    DataFrame with columns ``open, high, low, close, volume``,
    DatetimeIndex (UTC), sorted oldest → newest.

    Raises
    ------
    ValueError : If yfinance returns an empty DataFrame.
    """
    start_date = config["stock"]["start_date"]

    raw = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"yfinance returned no data for ticker '{ticker}'.")

    # Flatten MultiIndex columns produced by yfinance when auto_adjust=True
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Normalize to lowercase
    raw.columns = [c.lower() for c in raw.columns]

    # Keep only OHLCV columns that exist
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in raw.columns]
    df = raw[keep].copy()

    # Ensure DatetimeIndex and chronological order
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.dropna(subset=["close"], inplace=True)

    return df


# ── Walk-forward folds ─────────────────────────────────────────────────────────

def build_walk_forward_folds(
    df: pd.DataFrame,
    train_window_days: int,
    val_window_days: int,
    step_days: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Slice *df* into temporally ordered (train, val) fold pairs.

    Each fold's val window starts immediately after its train window.
    Consecutive folds are offset by *step_days* rows.

    Rules (matches CLAUDE.md):
    - NEVER shuffle rows.
    - NEVER let val data of one fold appear as train data of a later fold
      within the same fold's val window (non-overlapping val windows).
    - NEVER look ahead: train always precedes val in time.

    Parameters
    ----------
    df              : Full price DataFrame, sorted oldest → newest.
    train_window_days : Number of rows per train slice.
    val_window_days   : Number of rows per val slice.
    step_days         : Row stride between fold start positions.

    Returns
    -------
    List of (train_df, val_df) tuples.  Empty list if df is too short for
    even one fold.
    """
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    n = len(df)
    start = 0

    while True:
        train_end = start + train_window_days
        val_end   = train_end + val_window_days

        if val_end > n:
            break  # not enough rows left for a complete fold

        train_df = df.iloc[start:train_end]
        val_df   = df.iloc[train_end:val_end]

        folds.append((train_df, val_df))
        start += step_days

    return folds


# ── Scaler ─────────────────────────────────────────────────────────────────────

def fit_scaler(train_df: pd.DataFrame) -> MinMaxScaler:
    """Fit a MinMaxScaler on the close column of *train_df*.

    The scaler is fitted on **train data only** to prevent data leakage.
    Apply to val/test data using :func:`apply_scaler`.

    Parameters
    ----------
    train_df : Training fold DataFrame with a ``close`` column.

    Returns
    -------
    Fitted MinMaxScaler instance.
    """
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(train_df[["close"]])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler) -> pd.DataFrame:
    """Transform the ``close`` column of *df* using a pre-fitted *scaler*.

    All other columns (open, high, low, volume) are preserved unchanged.

    Parameters
    ----------
    df     : DataFrame with a ``close`` column.
    scaler : Fitted scaler from :func:`fit_scaler`.

    Returns
    -------
    New DataFrame with the ``close`` column replaced by its scaled values.
    The original *df* is not modified.
    """
    out = df.copy()
    out["close"] = scaler.transform(df[["close"]])
    return out


def save_scaler(scaler: MinMaxScaler, path: str) -> None:
    """Persist *scaler* to disk via pickle.

    Parent directories are created automatically.

    Parameters
    ----------
    scaler : Fitted scaler to save.
    path   : Destination file path (e.g. ``models/cache/AAPL/scaler.pkl``).
    """
    dest = pathlib.Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str) -> MinMaxScaler:
    """Load a scaler previously saved with :func:`save_scaler`.

    Parameters
    ----------
    path : File path written by save_scaler.

    Returns
    -------
    Restored MinMaxScaler instance.
    """
    with pathlib.Path(path).open("rb") as f:
        return pickle.load(f)
