"""
pipeline.preprocessors.text_preprocessor

Cleans and filters raw news articles and social posts before
sentiment scoring.

Steps:
  1. Deduplicate articles by URL and near-duplicate title detection.
  2. Filter articles to those mentioning the ticker or company name.
  3. Strip HTML tags, special characters, and excessive whitespace.
  4. Truncate to config.yaml → model.sentiment.max_length tokens
     (FinBERT has a 512-token limit).
  5. Combine title + body into a single text field for scoring.

Public interface:
  preprocess_articles(articles: list[dict], ticker: str,
                      config: dict) -> list[dict]
    Returns cleaned and filtered article dicts, same schema as
    news_fetcher output: [title, body, text, published_at, source, url].
    Adds a 'text' key = cleaned concatenation of title + body.
"""
