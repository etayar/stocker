from app.tools.registry import ToolResult

def sentiment_score(text: str) -> ToolResult:
    """
    v0 sentiment tool: simple deterministic heuristic.
    Later we will replace this with VADER or a transformer model.
    """
    if not text or not text.strip():
        return ToolResult(ok=False, error="Empty text")

    positive_words = {"beat", "growth", "record", "upgrade", "profit", "strong", "surge"}
    negative_words = {"miss", "lawsuit", "downgrade", "fraud", "weak", "loss", "crash"}

    words = {w.strip(".,!?;:()[]{}\"'").lower() for w in text.split()}
    pos = len(words & positive_words)
    neg = len(words & negative_words)

    raw = pos - neg

    # Normalize to [-1, 1] in a simple way
    if raw >= 3:
        score = 1.0
    elif raw <= -3:
        score = -1.0
    else:
        score = raw / 3.0

    confidence = 0.3 + min(0.6, (pos + neg) * 0.1)

    return ToolResult(ok=True, data={"score": score, "confidence": confidence, "pos_hits": pos, "neg_hits": neg})
