"""
pipeline.preprocessors.price_preprocessor

Cleans and transforms raw OHLCV price data for use by ts_model and ta_model.

Steps:
  1. Forward-fill missing trading days (holidays, weekends already absent
     from yfinance but gaps may exist in other sources).
  2. Drop rows with all-null OHLCV values.
  3. Compute log returns: log(close_t / close_{t-1}).
  4. Build rolling window feature matrix for TS model
     (window size from config.yaml → model.ts.window_size).
  5. Normalise features to [0, 1] using min-max scaling fitted on
     the training slice only (prevent lookahead).

Public interface:
  preprocess_prices(df: pd.DataFrame, config: dict,
                    fit_scaler: bool = True) -> tuple[pd.DataFrame, Scaler]
    Returns (processed_df, fitted_scaler).
    Pass fit_scaler=False and supply a pre-fitted scaler for inference.
"""
