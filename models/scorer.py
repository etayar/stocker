"""
models.scorer

Compound scorer that combines individual component scores into a
single buy/sell signal via geometric mean.

Formula:
  For n components q_1, q_2, ..., q_n (each ∈ [0, 1]):
    f(q_1, ..., q_n) = (q_1 * q_2 * ... * q_n)^(1/n)

  Properties:
    - If any q_i ≈ 0 the compound score is dragged to near 0
      (one danger signal prevents a buy action).
    - If q_1 = q_2 = ... = q_n = q then f = q (preserves uniform scores).

Default: f(ts_score, llms_score, ta_score) = (ts · llms · ta)^(1/3)

Public interface:
  compound_score(scores: list[float]) -> float
    Accepts an arbitrary list of component scores.
    Returns a compound score ∈ [0, 1].

  interpret_signal(score: float, config: dict) -> str
    Returns "BUY", "SELL", or "HOLD" based on thresholds in config.yaml:
      score >= scoring.buy_threshold  → "BUY"
      score <= scoring.sell_threshold → "SELL"
      otherwise                       → "HOLD"
"""

from math import prod


def compound_score(scores: list[float]) -> float:
    """Return the geometric mean of scores.

    Args:
        scores: Non-empty list of floats, each in [0, 1].

    Returns:
        Geometric mean ∈ [0, 1].

    Raises:
        ValueError: If scores is empty or contains values outside [0, 1].
    """
    if not scores:
        raise ValueError("scores must be a non-empty list")

    for s in scores:
        if not (0.0 <= s <= 1.0):
            raise ValueError(f"All scores must be in [0, 1], got {s}")

    n = len(scores)
    return prod(scores) ** (1.0 / n)


def interpret_signal(score: float, config: dict) -> str:
    """Map a compound score to a trading signal.

    Args:
        score:  Compound score ∈ [0, 1].
        config: Loaded config dict (from config.yaml).
                Expected keys: config["scoring"]["buy_threshold"]
                               config["scoring"]["sell_threshold"]

    Returns:
        "BUY"  if score >= buy_threshold
        "SELL" if score <= sell_threshold
        "HOLD" otherwise
    """
    scoring = config["scoring"]
    buy_threshold = scoring["buy_threshold"]
    sell_threshold = scoring["sell_threshold"]

    if score >= buy_threshold:
        return "BUY"
    if score <= sell_threshold:
        return "SELL"
    return "HOLD"
