"""
tests.test_data.test_price_fetcher

Tests for data.fetchers.price_fetcher.

Test cases to implement:
  - test_fetch_prices_yfinance_returns_dataframe
      Verify fetch_prices returns a DataFrame with expected columns
      when source="yfinance".
  - test_fetch_prices_columns
      Assert columns [open, high, low, close, volume] are present.
  - test_fetch_prices_date_range
      Assert returned index is within [start, end].
  - test_fetch_prices_no_future_dates
      Assert no rows have date > today.
  - test_fetch_prices_unknown_source_raises
      Assert ValueError is raised for unsupported source string.
  - test_fetch_prices_invalid_ticker_raises
      Assert appropriate exception for non-existent ticker.
"""
