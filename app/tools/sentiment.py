from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from app.tools.registry import ToolResult

_ANALYZER = SentimentIntensityAnalyzer()

def sentiment_score(text: str) -> ToolResult:
    """
    VADER sentiment tool.
    Returns compound score in [-1, 1] + breakdown (pos/neu/neg).
    """
    if not text or not text.strip():
        return ToolResult(ok=False, error="Empty text")

    scores = _ANALYZER.polarity_scores(text)

    # VADER returns:
    # - compound: overall sentiment in [-1, 1]
    # - pos/neu/neg: proportions in [0, 1]
    compound = float(scores.get("compound", 0.0))
    pos = float(scores.get("pos", 0.0))
    neu = float(scores.get("neu", 0.0))
    neg = float(scores.get("neg", 0.0))

    # Simple confidence heuristic: more non-neutral => higher confidence.
    # (Not a "true" statistical confidence, but useful for ranking/alerts.)
    confidence = min(0.95, 0.25 + (1.0 - neu))

    return ToolResult(
        ok=True,
        data={
            "score": compound,
            "confidence": confidence,
            "pos": pos,
            "neu": neu,
            "neg": neg,
        },
    )
