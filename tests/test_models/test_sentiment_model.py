"""
tests.test_models.test_sentiment_model

Tests for models.sentiment_model.

All tests mock the pipeline so no model weights need to be downloaded.
The scoring logic (_extract_score, score_text, score_articles) is fully
covered without transformers installed.
load_sentiment_model is tested by patching hf_pipeline at the module level.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from models.sentiment_model import (
    SentimentModel,
    _extract_score,
    load_sentiment_model,
    score_articles,
    score_text,
)

# ── Fixtures / helpers ────────────────────────────────────────────────────────

_CONFIG = {
    "model": {
        "sentiment": {
            "model_name": "ProsusAI/finbert",
            "max_length": 512,
            "batch_size": 16,
        }
    }
}


def _result(pos: float, neg: float, neu: float) -> list[dict]:
    """Build a single-text pipeline result (list[dict])."""
    return [
        {"label": "positive", "score": pos},
        {"label": "negative", "score": neg},
        {"label": "neutral",  "score": neu},
    ]


def _mock_model(pos: float = 0.5, neg: float = 0.3, neu: float = 0.2) -> SentimentModel:
    """SentimentModel whose pipe returns a fixed single-text result."""
    mock_pipe = MagicMock(return_value=_result(pos, neg, neu))
    return SentimentModel(pipe=mock_pipe, max_length=512)


# ── _extract_score ────────────────────────────────────────────────────────────

def test_extract_score_positive_dominant():
    score = _extract_score(_result(0.9, 0.05, 0.05))
    assert score == pytest.approx(0.9 / 0.95)


def test_extract_score_negative_dominant():
    score = _extract_score(_result(0.05, 0.9, 0.05))
    assert score == pytest.approx(0.05 / 0.95)


def test_extract_score_pure_neutral_returns_half():
    """When P(positive) = P(negative) = 0 the formula is 0/0 → default 0.5."""
    score = _extract_score(_result(0.0, 0.0, 1.0))
    assert score == pytest.approx(0.5)


def test_extract_score_balanced_returns_half():
    score = _extract_score(_result(0.4, 0.4, 0.2))
    assert score == pytest.approx(0.5)


def test_extract_score_result_in_range():
    for pos, neg, neu in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.33, 0.33, 0.34)]:
        assert 0.0 <= _extract_score(_result(pos, neg, neu)) <= 1.0


# ── score_text ────────────────────────────────────────────────────────────────

def test_score_text_positive():
    model = _mock_model(pos=0.9, neg=0.05, neu=0.05)
    assert score_text(model, "Record profits drive stock to all-time high") > 0.5


def test_score_text_negative():
    model = _mock_model(pos=0.05, neg=0.9, neu=0.05)
    assert score_text(model, "Company files for bankruptcy amid fraud allegations") < 0.5


def test_score_text_neutral():
    """Equal positive and negative probabilities → score == 0.5."""
    model = _mock_model(pos=0.05, neg=0.05, neu=0.9)
    assert score_text(model, "Company announces Q3 earnings date") == pytest.approx(0.5)


def test_score_text_returns_float_in_range():
    for pos, neg, neu in [
        (0.8, 0.1, 0.1),
        (0.1, 0.8, 0.1),
        (0.1, 0.1, 0.8),
        (0.0, 0.0, 1.0),
        (0.5, 0.5, 0.0),
    ]:
        model = _mock_model(pos=pos, neg=neg, neu=neu)
        s = score_text(model, "any text")
        assert 0.0 <= s <= 1.0


def test_score_text_passes_truncation_and_max_length():
    model = _mock_model()
    score_text(model, "test input")
    _, kwargs = model.pipe.call_args
    assert kwargs.get("truncation") is True
    assert kwargs.get("max_length") == 512


# ── score_articles ────────────────────────────────────────────────────────────

def test_score_articles_empty_list():
    model = _mock_model()
    assert score_articles(model, [], _CONFIG) == pytest.approx(0.5)


def test_score_articles_aggregation():
    """Mean of two article scores."""
    mock_pipe = MagicMock(return_value=[
        _result(0.8, 0.1, 0.1),   # score = 0.8 / 0.9
        _result(0.2, 0.6, 0.2),   # score = 0.2 / 0.8
    ])
    model = SentimentModel(pipe=mock_pipe, max_length=512)
    articles = [{"text": "Great earnings!"}, {"text": "Stock crashes!"}]

    result = score_articles(model, articles, _CONFIG)
    expected = float(np.mean([0.8 / 0.9, 0.2 / 0.8]))
    assert result == pytest.approx(expected, abs=1e-6)


def test_score_articles_uses_text_key():
    """'text' key takes priority over title + body."""
    mock_pipe = MagicMock(return_value=[_result(0.9, 0.05, 0.05)])
    model = SentimentModel(pipe=mock_pipe, max_length=512)
    articles = [{"text": "explicit text", "title": "ignored", "body": "ignored"}]

    score_articles(model, articles, _CONFIG)

    texts_passed = mock_pipe.call_args[0][0]
    assert texts_passed == ["explicit text"]


def test_score_articles_falls_back_to_title_body():
    """When 'text' is absent, concatenate title + body."""
    mock_pipe = MagicMock(return_value=[_result(0.5, 0.3, 0.2)])
    model = SentimentModel(pipe=mock_pipe, max_length=512)
    articles = [{"title": "Earnings", "body": "beat expectations"}]

    score_articles(model, articles, _CONFIG)

    texts_passed = mock_pipe.call_args[0][0]
    assert texts_passed == ["Earnings beat expectations"]


def test_score_articles_single_article():
    mock_pipe = MagicMock(return_value=[_result(0.7, 0.2, 0.1)])
    model = SentimentModel(pipe=mock_pipe, max_length=512)
    score = score_articles(model, [{"text": "Strong buy signal"}], _CONFIG)
    assert score == pytest.approx(0.7 / 0.9, abs=1e-6)


def test_score_articles_passes_batch_size():
    mock_pipe = MagicMock(return_value=[_result(0.5, 0.3, 0.2)])
    model = SentimentModel(pipe=mock_pipe, max_length=512)
    score_articles(model, [{"text": "x"}], _CONFIG)
    _, kwargs = mock_pipe.call_args
    assert kwargs.get("batch_size") == 16


# ── load_sentiment_model ──────────────────────────────────────────────────────

def test_load_sentiment_model_returns_model():
    """Patching hf_pipeline at module level to avoid downloading weights."""
    fake_pipe = MagicMock()
    with patch("models.sentiment_model.hf_pipeline", MagicMock(return_value=fake_pipe)):
        model = load_sentiment_model(_CONFIG)
    assert model is not None
    assert isinstance(model, SentimentModel)
    assert model.max_length == 512


def test_load_sentiment_model_passes_correct_model_name():
    with patch("models.sentiment_model.hf_pipeline") as mock_hf:
        mock_hf.return_value = MagicMock()
        load_sentiment_model(_CONFIG)
    _, kwargs = mock_hf.call_args
    assert kwargs.get("model") == "ProsusAI/finbert"


def test_load_sentiment_model_raises_if_transformers_missing():
    """hf_pipeline=None simulates transformers not being installed."""
    with patch("models.sentiment_model.hf_pipeline", None):
        with pytest.raises(ImportError, match="transformers"):
            load_sentiment_model(_CONFIG)
