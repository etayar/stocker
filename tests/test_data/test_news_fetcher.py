"""
tests.test_data.test_news_fetcher

Tests for data.fetchers.news_fetcher.

Test cases to implement:
  - test_fetch_news_returns_list
      Verify fetch_news returns a list.
  - test_fetch_news_article_schema
      Each article dict must have keys: title, body, published_at, source, url.
  - test_fetch_news_date_filter
      Articles are within the requested date range.
  - test_fetch_news_unknown_source_raises
      ValueError for unsupported source string.
  - test_fetch_news_missing_api_key_raises
      EnvironmentError (or similar) when required key is absent.
"""
