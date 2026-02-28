"""
tests.test_data.test_news_fetcher

Tests for data.fetchers.news_fetcher.

All external API calls (NewsAPI, Finnhub, Reddit/PRAW) are mocked so no
network access or real API keys are required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from data.fetchers.news_fetcher import fetch_news

# ── Article schema ─────────────────────────────────────────────────────────────

_REQUIRED_KEYS = {"title", "body", "published_at", "source", "url"}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_newsapi_response(n: int = 3) -> dict:
    """Fake NewsAPI /everything response."""
    return {
        "status": "ok",
        "totalResults": n,
        "articles": [
            {
                "title":       f"Apple headline {i}",
                "content":     f"Body text {i}",
                "description": f"Description {i}",
                "publishedAt": f"2026-02-{i+1:02d}T10:00:00Z",
                "url":         f"https://example.com/news/{i}",
            }
            for i in range(n)
        ],
    }


def _make_finnhub_response(n: int = 3) -> list:
    """Fake Finnhub company_news response."""
    base_ts = int(datetime(2026, 2, 1, 10, 0, tzinfo=timezone.utc).timestamp())
    return [
        {
            "headline": f"AAPL news {i}",
            "summary":  f"Finnhub body {i}",
            "datetime": base_ts + i * 86400,
            "url":      f"https://finnhub.io/news/{i}",
        }
        for i in range(n)
    ]


def _make_reddit_post(i: int) -> MagicMock:
    """Fake PRAW Submission object."""
    post = MagicMock()
    post.title      = f"AAPL reddit post {i}"
    post.selftext   = f"Reddit body {i}"
    post.created_utc = datetime(2026, 2, i + 1, 10, tzinfo=timezone.utc).timestamp()
    post.permalink  = f"/r/stocks/comments/{i}/aapl_post_{i}/"
    return post


# ── NewsAPI ────────────────────────────────────────────────────────────────────

class TestFetchNewsNewsAPI:
    def _call(self, monkeypatch, n=3):
        monkeypatch.setenv("NEWSAPI_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.get_everything.return_value = _make_newsapi_response(n)
        with patch("data.fetchers.news_fetcher.NewsApiClient", return_value=mock_client):
            return fetch_news("AAPL", "newsapi")

    def test_returns_list(self, monkeypatch):
        assert isinstance(self._call(monkeypatch), list)

    def test_article_schema(self, monkeypatch):
        articles = self._call(monkeypatch)
        for a in articles:
            assert _REQUIRED_KEYS.issubset(a.keys()), f"Missing keys: {_REQUIRED_KEYS - a.keys()}"

    def test_correct_count(self, monkeypatch):
        articles = self._call(monkeypatch, n=5)
        assert len(articles) == 5

    def test_source_field_is_newsapi(self, monkeypatch):
        articles = self._call(monkeypatch)
        assert all(a["source"] == "newsapi" for a in articles)

    def test_title_populated(self, monkeypatch):
        articles = self._call(monkeypatch)
        assert all(isinstance(a["title"], str) for a in articles)

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("NEWSAPI_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="NEWSAPI_KEY"):
            fetch_news("AAPL", "newsapi")

    def test_empty_response_returns_empty_list(self, monkeypatch):
        monkeypatch.setenv("NEWSAPI_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.get_everything.return_value = {"articles": []}
        with patch("data.fetchers.news_fetcher.NewsApiClient", return_value=mock_client):
            assert fetch_news("AAPL", "newsapi") == []


# ── Finnhub ────────────────────────────────────────────────────────────────────

class TestFetchNewsFinnhub:
    def _call(self, monkeypatch, n=3):
        monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.company_news.return_value = _make_finnhub_response(n)
        with patch("data.fetchers.news_fetcher.finnhub") as mock_module:
            mock_module.Client.return_value = mock_client
            return fetch_news("AAPL", "finnhub")

    def test_returns_list(self, monkeypatch):
        assert isinstance(self._call(monkeypatch), list)

    def test_article_schema(self, monkeypatch):
        articles = self._call(monkeypatch)
        for a in articles:
            assert _REQUIRED_KEYS.issubset(a.keys())

    def test_correct_count(self, monkeypatch):
        assert len(self._call(monkeypatch, n=4)) == 4

    def test_source_field_is_finnhub(self, monkeypatch):
        articles = self._call(monkeypatch)
        assert all(a["source"] == "finnhub" for a in articles)

    def test_published_at_is_iso_string(self, monkeypatch):
        articles = self._call(monkeypatch)
        for a in articles:
            datetime.fromisoformat(a["published_at"])  # must not raise

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="FINNHUB_API_KEY"):
            fetch_news("AAPL", "finnhub")

    def test_none_response_returns_empty_list(self, monkeypatch):
        monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.company_news.return_value = None
        with patch("data.fetchers.news_fetcher.finnhub") as mock_module:
            mock_module.Client.return_value = mock_client
            assert fetch_news("AAPL", "finnhub") == []


# ── Reddit ─────────────────────────────────────────────────────────────────────

class TestFetchNewsReddit:
    def _call(self, monkeypatch, posts=None):
        monkeypatch.setenv("REDDIT_CLIENT_ID",     "cid")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "csec")
        monkeypatch.setenv("REDDIT_USER_AGENT",    "stocker/1.0")

        if posts is None:
            posts = [_make_reddit_post(i) for i in range(3)]

        mock_reddit   = MagicMock()
        mock_sub      = MagicMock()
        mock_sub.search.return_value = posts
        mock_reddit.subreddit.return_value = mock_sub

        with patch("data.fetchers.news_fetcher.praw") as mock_praw:
            mock_praw.Reddit.return_value = mock_reddit
            return fetch_news("AAPL", "reddit")

    def test_returns_list(self, monkeypatch):
        assert isinstance(self._call(monkeypatch), list)

    def test_article_schema(self, monkeypatch):
        articles = self._call(monkeypatch)
        for a in articles:
            assert _REQUIRED_KEYS.issubset(a.keys())

    def test_source_field_is_reddit(self, monkeypatch):
        articles = self._call(monkeypatch)
        assert all(a["source"] == "reddit" for a in articles)

    def test_url_contains_reddit(self, monkeypatch):
        articles = self._call(monkeypatch)
        assert all("reddit.com" in a["url"] for a in articles)

    def test_missing_client_id_raises(self, monkeypatch):
        monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "csec")
        monkeypatch.setenv("REDDIT_USER_AGENT",    "stocker/1.0")
        with pytest.raises(EnvironmentError, match="REDDIT_CLIENT_ID"):
            fetch_news("AAPL", "reddit")


# ── Unknown source ─────────────────────────────────────────────────────────────

def test_fetch_news_unknown_source_raises():
    with pytest.raises(ValueError, match="Unknown news source"):
        fetch_news("AAPL", "bloomberg")


def test_fetch_news_source_case_insensitive(monkeypatch):
    """Source string should be normalised to lowercase."""
    monkeypatch.setenv("NEWSAPI_KEY", "test-key")
    mock_client = MagicMock()
    mock_client.get_everything.return_value = {"articles": []}
    with patch("data.fetchers.news_fetcher.NewsApiClient", return_value=mock_client):
        result = fetch_news("AAPL", "NewsAPI")   # mixed case
    assert result == []


# ── lookback_days ──────────────────────────────────────────────────────────────

def test_fetch_news_lookback_days_passed_to_backend(monkeypatch):
    """lookback_days must control the from-date sent to the API."""
    monkeypatch.setenv("NEWSAPI_KEY", "test-key")
    mock_client = MagicMock()
    mock_client.get_everything.return_value = {"articles": []}
    with patch("data.fetchers.news_fetcher.NewsApiClient", return_value=mock_client):
        fetch_news("AAPL", "newsapi", lookback_days=7)
    _, kwargs = mock_client.get_everything.call_args
    # from_param should be 7 days ago — just verify it was passed
    assert "from_param" in kwargs
