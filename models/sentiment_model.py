"""
models.sentiment_model

FinBERT-based LLM sentiment scoring for financial news and social posts.

Uses ProsusAI/finbert (preferred over generic BERT for finance domain).
Model name is configurable via config.yaml → model.sentiment.model_name.

FinBERT outputs three classes: positive, negative, neutral.
These are mapped to a continuous score in [0, 1]:
  score = P(positive) / (P(positive) + P(negative))
  (neutral articles score 0.5)

Public interface:
  load_sentiment_model(config: dict) -> SentimentModel
    Loads the HuggingFace model and tokenizer.

  score_articles(model: SentimentModel, articles: list[dict],
                 config: dict) -> float
    Accepts a list of article dicts (from news_fetcher).
    Returns a single aggregated llms_score ∈ [0, 1] averaged
    across all articles in the window.

  score_text(model: SentimentModel, text: str) -> float
    Scores a single text snippet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Lazy guard: transformers is only needed when load_sentiment_model() is called.
# All scoring functions work on any callable pipe, so tests can mock freely.
try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None  # type: ignore[assignment]


@dataclass
class SentimentModel:
    """Wraps a HuggingFace text-classification pipeline for finance sentiment.

    Attributes:
        pipe:       A callable pipeline that accepts text and kwargs and returns
                    classification results (transformers.Pipeline or mock).
        max_length: Token limit passed to the pipeline on every call.
    """

    pipe: object
    max_length: int


def load_sentiment_model(config: dict) -> SentimentModel:
    """Load the FinBERT model from HuggingFace Hub.

    Args:
        config: Loaded config dict. Reads model.sentiment.model_name
                and model.sentiment.max_length.

    Returns:
        SentimentModel ready for inference.

    Raises:
        ImportError: If the `transformers` package is not installed.
    """
    if hf_pipeline is None:
        raise ImportError(
            "transformers is required for sentiment scoring. "
            "Install it with: pip install transformers"
        )
    cfg = config["model"]["sentiment"]
    pipe = hf_pipeline(
        "text-classification",
        model=cfg["model_name"],
        top_k=None,  # return scores for all three classes
    )
    return SentimentModel(pipe=pipe, max_length=cfg["max_length"])


def _extract_score(result: list[dict]) -> float:
    """Map a pipeline result for one text to a continuous score in [0, 1].

    Formula:
        score = P(positive) / (P(positive) + P(negative))

    When neutral dominates (denominator ≈ 0) the score is 0.5 — no
    directional signal, matching the neutral midpoint of the [0, 1] range.

    Args:
        result: List of dicts like
                [{"label": "positive", "score": 0.9}, ...].

    Returns:
        float ∈ [0, 1].
    """
    pos = next((r["score"] for r in result if r["label"] == "positive"), 0.0)
    neg = next((r["score"] for r in result if r["label"] == "negative"), 0.0)
    denom = pos + neg
    if denom < 1e-10:
        return 0.5
    return float(pos / denom)


def score_text(model: SentimentModel, text: str) -> float:
    """Score a single text snippet.

    Args:
        model: Loaded SentimentModel.
        text:  Raw text (news headline, body paragraph, social post, …).

    Returns:
        llms_score ∈ [0, 1]:
          P(positive) / (P(positive) + P(negative))
          → 0.5 when sentiment is neutral or balanced.
    """
    # Single-string call with top_k=None → pipeline returns list[dict]
    result = model.pipe(text, truncation=True, max_length=model.max_length)
    return _extract_score(result)


def score_articles(
    model: SentimentModel,
    articles: list[dict],
    config: dict,
) -> float:
    """Score a batch of articles and return the mean llms_score.

    Each article dict should have a 'text' key with pre-processed text.
    If absent, 'title' and 'body' are concatenated as fallback.

    Args:
        model:    Loaded SentimentModel.
        articles: List of article dicts (from news_fetcher /
                  text_preprocessor).
        config:   Loaded config dict.

    Returns:
        Mean llms_score ∈ [0, 1]. Returns 0.5 for an empty list.
    """
    if not articles:
        return 0.5

    texts = [
        a.get("text") or f"{a.get('title', '')} {a.get('body', '')}".strip()
        for a in articles
    ]
    batch_size = config["model"]["sentiment"]["batch_size"]

    # Batch call with list[str] → pipeline returns list[list[dict]]
    results = model.pipe(
        texts,
        truncation=True,
        max_length=model.max_length,
        batch_size=batch_size,
    )
    scores = [_extract_score(r) for r in results]
    return float(np.mean(scores))
