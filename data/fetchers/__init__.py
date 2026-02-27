"""
data.fetchers package

Pluggable fetchers for price and news/text data.
Each fetcher implements a common interface so the pipeline
can swap sources (yfinance ↔ Alpha Vantage ↔ Polygon, etc.)
without changing downstream code.
"""
