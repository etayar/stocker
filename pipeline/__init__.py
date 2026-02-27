"""
pipeline package

Orchestrates the end-to-end Stocker pipeline:
  1. Fetch price data          (data.fetchers.price_fetcher)
  2. Fetch news/text data      (data.fetchers.news_fetcher)
  3. Preprocess price data     (pipeline.preprocessors.price_preprocessor)
  4. Preprocess text data      (pipeline.preprocessors.text_preprocessor)
  5. Score: TS model           (models.ts_model)
  6. Score: Sentiment model    (models.sentiment_model)
  7. Score: TA model           (models.ta_model)
  8. Compound score            (models.scorer)
  9. Persist results           (data.storage.db)
"""
