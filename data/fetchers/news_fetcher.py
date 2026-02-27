"""
data.fetchers.news_fetcher

Retrieves news articles and social media posts related to a given ticker
for use in LLM sentiment scoring.

Supported sources (configured via config.yaml → data.news_source):
  - newsapi  : requires NEWSAPI_KEY env var
  - finnhub  : requires FINNHUB_API_KEY env var
  - reddit   : requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET,
                        REDDIT_USER_AGENT env vars (PRAW)

Public interface:
  fetch_news(ticker: str, start: str, end: str, source: str) -> list[dict]
    Returns a list of article dicts with keys:
    [title, body, published_at, source, url]
"""
