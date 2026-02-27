"""
pipeline.preprocessors package

Data cleaning and feature engineering steps applied before models run:
  - price_preprocessor : normalises OHLCV data, handles missing values,
                         builds lag features and rolling windows
  - text_preprocessor  : cleans raw article text, deduplicates,
                         filters by relevance to ticker
"""
