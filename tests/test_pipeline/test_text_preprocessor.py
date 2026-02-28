"""
tests.test_pipeline.test_text_preprocessor

Tests for pipeline.preprocessors.text_preprocessor.preprocess_articles.

No external I/O — all inputs are in-process dicts.
"""

from __future__ import annotations

import pytest

from pipeline.preprocessors.text_preprocessor import preprocess_articles

# ── Config / helpers ───────────────────────────────────────────────────────────

_CONFIG = {
    "model": {
        "sentiment": {
            "max_length": 512,
        }
    }
}


def _article(
    title: str = "Apple reports strong earnings",
    body:  str = "AAPL beat estimates.",
    url:   str = "https://example.com/1",
    published_at: str = "2026-02-01T10:00:00Z",
    source: str = "newsapi",
) -> dict:
    return {
        "title":        title,
        "body":         body,
        "url":          url,
        "published_at": published_at,
        "source":       source,
    }


# ── Output structure ───────────────────────────────────────────────────────────

def test_returns_list():
    result = preprocess_articles([_article()], "AAPL", _CONFIG)
    assert isinstance(result, list)


def test_output_has_text_key():
    result = preprocess_articles([_article()], "AAPL", _CONFIG)
    assert len(result) == 1
    assert "text" in result[0]


def test_output_preserves_original_keys():
    result = preprocess_articles([_article()], "AAPL", _CONFIG)
    for key in ("title", "body", "published_at", "source", "url"):
        assert key in result[0]


def test_text_is_title_plus_body():
    art = _article(title="Apple is great", body="Earnings beat.")
    result = preprocess_articles([art], "AAPL", _CONFIG)
    assert result[0]["text"] == "Apple is great. Earnings beat."


def test_text_only_title_when_body_empty():
    art = _article(title="Apple rises", body="")
    result = preprocess_articles([art], "AAPL", _CONFIG)
    assert result[0]["text"] == "Apple rises"


def test_text_only_body_when_title_empty():
    art = _article(title="", body="AAPL surged today.")
    result = preprocess_articles([art], "AAPL", _CONFIG)
    assert result[0]["text"] == "AAPL surged today."


# ── HTML stripping ─────────────────────────────────────────────────────────────

def test_strips_html_tags():
    art = _article(title="<b>AAPL</b> soars", body="<p>Good news.</p>")
    result = preprocess_articles([art], "AAPL", _CONFIG)
    assert "<b>" not in result[0]["text"]
    assert "<p>" not in result[0]["text"]
    assert "AAPL" in result[0]["text"]


def test_normalises_whitespace():
    art = _article(title="  AAPL   beats   \n\t estimates  ", body="")
    result = preprocess_articles([art], "AAPL", _CONFIG)
    assert "  " not in result[0]["text"]


# ── Deduplication ─────────────────────────────────────────────────────────────

def test_deduplicates_by_url():
    arts = [
        _article(url="https://example.com/1"),
        _article(url="https://example.com/1"),  # duplicate
    ]
    result = preprocess_articles(arts, "AAPL", _CONFIG)
    assert len(result) == 1


def test_different_urls_both_kept():
    arts = [
        _article(url="https://example.com/1"),
        _article(url="https://example.com/2"),
    ]
    result = preprocess_articles(arts, "AAPL", _CONFIG)
    assert len(result) == 2


def test_no_url_articles_not_deduplicated_against_each_other():
    """Articles without a URL should all be kept (can't deduplicate them)."""
    arts = [
        _article(title="AAPL news one",   url=""),
        _article(title="AAPL news two",   url=""),
        _article(title="AAPL news three", url=""),
    ]
    result = preprocess_articles(arts, "AAPL", _CONFIG)
    assert len(result) == 3


# ── Ticker filtering ──────────────────────────────────────────────────────────

def test_keeps_article_mentioning_ticker_symbol():
    art = _article(title="AAPL hits all-time high", body="")
    result = preprocess_articles([art], "AAPL", _CONFIG)
    assert len(result) == 1


def test_keeps_article_mentioning_alias():
    art = _article(title="Apple beats estimates", body="")
    result = preprocess_articles([art], "AAPL", _CONFIG)
    assert len(result) == 1


def test_drops_article_not_mentioning_ticker():
    art = _article(title="Google quarterly results", body="Alphabet reports strong growth.")
    result = preprocess_articles([art], "AAPL", _CONFIG)
    assert len(result) == 0


def test_filter_is_case_insensitive():
    art = _article(title="aapl earnings beat", body="")
    result = preprocess_articles([art], "AAPL", _CONFIG)
    assert len(result) == 1


def test_ticker_in_body_passes_filter():
    art = _article(title="Tech stocks rally", body="AAPL led the gains.")
    result = preprocess_articles([art], "AAPL", _CONFIG)
    assert len(result) == 1


# ── Truncation ────────────────────────────────────────────────────────────────

def test_text_truncated_to_max_length():
    cfg = {"model": {"sentiment": {"max_length": 20}}}
    art = _article(title="AAPL " + "x" * 100, body="")
    result = preprocess_articles([art], "AAPL", cfg)
    assert len(result[0]["text"]) <= 20


def test_short_text_not_truncated():
    art = _article(title="AAPL up", body="")
    result = preprocess_articles([art], "AAPL", _CONFIG)
    assert result[0]["text"] == "AAPL up"


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_input_returns_empty_list():
    assert preprocess_articles([], "AAPL", _CONFIG) == []


def test_all_empty_articles_dropped():
    arts = [_article(title="", body="", url="")]
    result = preprocess_articles(arts, "AAPL", _CONFIG)
    assert len(result) == 0


def test_multiple_articles_mixed():
    arts = [
        _article(title="AAPL beats earnings", url="https://a.com/1"),
        _article(title="<b>AAPL</b> rises", url="https://a.com/2"),
        _article(title="Google quarterly results", body="Alphabet reports strong growth.",
                 url="https://a.com/3"),  # filtered — no AAPL mention
        _article(title="AAPL beats earnings", url="https://a.com/1"),  # duplicate
    ]
    result = preprocess_articles(arts, "AAPL", _CONFIG)
    assert len(result) == 2


def test_missing_config_keys_use_defaults():
    """Should not crash when model/sentiment/max_length is absent from config."""
    result = preprocess_articles([_article()], "AAPL", {})
    assert len(result) == 1
