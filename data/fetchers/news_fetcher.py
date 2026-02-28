"""
data.fetchers.news_fetcher

Retrieves news articles and social-media posts related to a given ticker
for use in LLM sentiment scoring.

Supported sources (configured via config.yaml → data.news_source):
  newsapi  : requires NEWSAPI_KEY env var
  finnhub  : requires FINNHUB_API_KEY env var
  reddit   : requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET,
                      REDDIT_USER_AGENT env vars (PRAW)

Public interface:
  fetch_news(ticker, source, lookback_days=30) -> list[dict]

Every article dict has these keys:
  title        : str   headline / post title
  body         : str   article body or post text (may be empty string)
  published_at : str   ISO-8601 datetime string (UTC)
  source       : str   source name ("newsapi", "finnhub", "reddit")
  url          : str   canonical URL (empty string when unavailable)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

# ── Optional library imports (guarded so the module loads without them) ────────
# Tests patch these at module level: data.fetchers.news_fetcher.NewsApiClient etc.

try:
    from newsapi import NewsApiClient
except ImportError:
    NewsApiClient = None  # type: ignore[assignment, misc]

try:
    import finnhub
except ImportError:
    finnhub = None  # type: ignore[assignment]

try:
    import praw
except ImportError:
    praw = None  # type: ignore[assignment]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _lookback_dates(lookback_days: int) -> tuple[str, str]:
    """Return (from_date_iso, to_date_iso) for the last *lookback_days* days."""
    now   = datetime.now(timezone.utc)
    start = now - timedelta(days=lookback_days)
    return start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")


def _require_env(var: str) -> str:
    """Return env var value or raise EnvironmentError."""
    val = os.environ.get(var, "").strip()
    if not val:
        raise EnvironmentError(
            f"Environment variable '{var}' is required but not set. "
            f"Add it to your .env file."
        )
    return val


# ── Source backends ────────────────────────────────────────────────────────────

def _fetch_newsapi(ticker: str, lookback_days: int) -> list[dict]:
    """Fetch articles from NewsAPI using the /everything endpoint."""
    if NewsApiClient is None:
        raise ImportError("newsapi-python is required. Install with: pip install newsapi-python")

    api_key = _require_env("NEWSAPI_KEY")
    from_date, to_date = _lookback_dates(lookback_days)

    client   = NewsApiClient(api_key=api_key)
    response = client.get_everything(
        q=ticker,
        from_param=from_date,
        to=to_date,
        language="en",
        sort_by="publishedAt",
        page_size=100,
    )

    articles = []
    for item in response.get("articles", []):
        articles.append({
            "title":        item.get("title") or "",
            "body":         item.get("content") or item.get("description") or "",
            "published_at": item.get("publishedAt") or "",
            "source":       "newsapi",
            "url":          item.get("url") or "",
        })
    return articles


def _fetch_finnhub(ticker: str, lookback_days: int) -> list[dict]:
    """Fetch company news from Finnhub."""
    if finnhub is None:
        raise ImportError("finnhub-python is required. Install with: pip install finnhub-python")

    api_key = _require_env("FINNHUB_API_KEY")
    from_date, to_date = _lookback_dates(lookback_days)

    client = finnhub.Client(api_key=api_key)
    items  = client.company_news(ticker.upper(), _from=from_date, to=to_date)

    articles = []
    for item in (items or []):
        ts = item.get("datetime")
        published = (
            datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else ""
        )
        articles.append({
            "title":        item.get("headline") or "",
            "body":         item.get("summary") or "",
            "published_at": published,
            "source":       "finnhub",
            "url":          item.get("url") or "",
        })
    return articles


def _fetch_reddit(ticker: str, lookback_days: int) -> list[dict]:
    """Fetch posts mentioning *ticker* from finance subreddits via PRAW."""
    if praw is None:
        raise ImportError("praw is required. Install with: pip install praw")

    client_id     = _require_env("REDDIT_CLIENT_ID")
    client_secret = _require_env("REDDIT_CLIENT_SECRET")
    user_agent    = _require_env("REDDIT_USER_AGENT")

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    cutoff     = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    subreddits = ["stocks", "investing", "wallstreetbets"]
    query      = ticker.upper()

    articles = []
    for sub_name in subreddits:
        subreddit = reddit.subreddit(sub_name)
        for post in subreddit.search(query, sort="new", time_filter="month", limit=50):
            created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
            if created < cutoff:
                continue
            articles.append({
                "title":        post.title or "",
                "body":         post.selftext or "",
                "published_at": created.isoformat(),
                "source":       "reddit",
                "url":          f"https://reddit.com{post.permalink}",
            })
    return articles


# ── Public interface ───────────────────────────────────────────────────────────

_BACKENDS = {
    "newsapi": _fetch_newsapi,
    "finnhub": _fetch_finnhub,
    "reddit":  _fetch_reddit,
}


def fetch_news(
    ticker: str,
    source: str,
    lookback_days: int = 30,
) -> list[dict]:
    """Fetch recent news articles for *ticker* from *source*.

    Parameters
    ----------
    ticker        : Stock symbol, e.g. "AAPL".
    source        : One of "newsapi", "finnhub", "reddit".
    lookback_days : How many calendar days back to fetch (default 30).

    Returns
    -------
    List of article dicts, each with keys:
    ``title``, ``body``, ``published_at``, ``source``, ``url``.
    Empty list if the source returns no results.

    Raises
    ------
    ValueError       : Unknown source string.
    EnvironmentError : Required API key env var is missing.
    ImportError      : Required client library not installed.
    """
    source = source.lower().strip()
    if source not in _BACKENDS:
        raise ValueError(
            f"Unknown news source '{source}'. "
            f"Supported: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[source](ticker, lookback_days)
