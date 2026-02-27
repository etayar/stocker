"""
data.fetchers.price_fetcher

Retrieves historical OHLCV (Open, High, Low, Close, Volume) price data
for a given ticker and date range.

Supported sources (configured via config.yaml → data.price_source):
  - yfinance   : free, no API key required
  - alpha_vantage: requires ALPHA_VANTAGE_API_KEY env var
  - polygon    : requires POLYGON_API_KEY env var

Public interface:
  fetch_prices(ticker: str, start: str, end: str, source: str) -> pd.DataFrame
    Returns a DataFrame indexed by date with columns:
    [open, high, low, close, volume, adjusted_close]
"""
