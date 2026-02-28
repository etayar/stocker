"""
pipeline.preprocessors.text_preprocessor

Cleans and filters raw news articles / social posts before sentiment scoring.

Steps (applied in order):
  1. Deduplicate by URL; skip articles whose URLs have already been seen.
     Articles with no URL are kept but not deduplicated against each other.
  2. Filter to articles that mention the ticker symbol or a known alias in
     their title or body (case-insensitive).
  3. Strip HTML tags and normalise whitespace (collapse runs of
     spaces/newlines into a single space, strip leading/trailing space).
  4. Combine title + body into a single ``text`` field
     (``"<title>. <body>"``).
  5. Truncate ``text`` to ``model.sentiment.max_length`` *characters*
     (FinBERT's actual token limit is 512 sub-word tokens; using characters
     is a safe conservative proxy without requiring a tokeniser here).

Public interface:
  preprocess_articles(articles, ticker, config) -> list[dict]
    Returns cleaned article dicts.  Each dict preserves all original keys
    and gains a ``text`` field = cleaned title + body concatenation.
    Articles that are entirely empty after cleaning are dropped.
"""

from __future__ import annotations

import re
from typing import Any


# ── HTML stripping ─────────────────────────────────────────────────────────────

_TAG_RE       = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    """Remove HTML tags and normalise whitespace."""
    text = _TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


# ── Ticker alias lookup ────────────────────────────────────────────────────────

# A small hard-coded map of common tickers → company name fragments.
# Extend via config if needed; the ticker symbol itself is always checked.
_TICKER_ALIASES: dict[str, list[str]] = {
    "AAPL":  ["apple"],
    "MSFT":  ["microsoft"],
    "GOOGL": ["google", "alphabet"],
    "GOOG":  ["google", "alphabet"],
    "AMZN":  ["amazon"],
    "TSLA":  ["tesla"],
    "NVDA":  ["nvidia"],
    "META":  ["meta", "facebook"],
    "NFLX":  ["netflix"],
    "AMD":   ["amd", "advanced micro"],
}


def _ticker_mentioned(text: str, ticker: str) -> bool:
    """Return True if *ticker* or a known alias appears in *text*."""
    lower = text.lower()
    if ticker.lower() in lower:
        return True
    for alias in _TICKER_ALIASES.get(ticker.upper(), []):
        if alias in lower:
            return True
    return False


# ── Public interface ───────────────────────────────────────────────────────────

def preprocess_articles(
    articles: list[dict[str, Any]],
    ticker: str,
    config: dict,
) -> list[dict[str, Any]]:
    """Clean and filter *articles* for sentiment scoring of *ticker*.

    Parameters
    ----------
    articles : Raw article dicts from ``fetch_news``.
               Expected keys: ``title``, ``body``, ``published_at``,
               ``source``, ``url``.
    ticker   : Stock symbol used for relevance filtering (e.g. "AAPL").
    config   : Loaded config dict. Reads
               ``model.sentiment.max_length`` for text truncation.

    Returns
    -------
    List of cleaned article dicts.  Each dict retains all original keys
    and gains a ``text`` key (cleaned ``"<title>. <body>"`` string).
    Articles that are empty after cleaning or do not mention the ticker
    are dropped.
    """
    max_length: int = int(
        config.get("model", {}).get("sentiment", {}).get("max_length", 512)
    )

    seen_urls: set[str] = set()
    result: list[dict[str, Any]] = []

    for article in articles:
        # ── 1. Deduplicate by URL ─────────────────────────────────────────────
        url = (article.get("url") or "").strip()
        if url:
            if url in seen_urls:
                continue
            seen_urls.add(url)

        # ── 2. Clean title and body ───────────────────────────────────────────
        title = _strip_html(article.get("title") or "")
        body  = _strip_html(article.get("body")  or "")

        # ── 3. Build combined text ────────────────────────────────────────────
        if title and body:
            combined = f"{title}. {body}"
        else:
            combined = title or body

        if not combined:
            continue  # nothing left after cleaning → drop

        # ── 4. Filter: must mention ticker ────────────────────────────────────
        if not _ticker_mentioned(combined, ticker):
            continue

        # ── 5. Truncate ───────────────────────────────────────────────────────
        text = combined[:max_length]

        out = {**article, "title": title, "body": body, "text": text}
        result.append(out)

    return result
