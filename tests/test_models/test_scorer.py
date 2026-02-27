"""
tests.test_models.test_scorer

Tests for models.scorer.
"""

import math
import pytest

from models.scorer import compound_score, interpret_signal


# ── compound_score ─────────────────────────────────────────────────────────────

def test_compound_score_all_ones():
    assert compound_score([1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_compound_score_all_zeros():
    assert compound_score([0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_compound_score_uniform():
    for q in [0.0, 0.2, 0.5, 0.75, 1.0]:
        assert compound_score([q, q, q]) == pytest.approx(q)


def test_compound_score_one_near_zero_drags_down():
    score = compound_score([0.9, 0.9, 0.01])
    assert score < 0.3  # far below the 0.9 components


def test_compound_score_returns_float_in_range():
    import random
    random.seed(42)
    for _ in range(100):
        scores = [random.random() for _ in range(random.randint(1, 6))]
        result = compound_score(scores)
        assert 0.0 <= result <= 1.0


def test_compound_score_two_components():
    a, b = 0.4, 0.9
    expected = math.sqrt(a * b)
    assert compound_score([a, b]) == pytest.approx(expected)


def test_compound_score_empty_raises():
    with pytest.raises(ValueError):
        compound_score([])


def test_compound_score_out_of_range_raises():
    with pytest.raises(ValueError):
        compound_score([0.5, 1.1])

    with pytest.raises(ValueError):
        compound_score([-0.1, 0.5])


# ── interpret_signal ───────────────────────────────────────────────────────────

_CONFIG = {
    "scoring": {
        "buy_threshold": 0.65,
        "sell_threshold": 0.35,
    }
}


def test_interpret_signal_buy():
    assert interpret_signal(0.65, _CONFIG) == "BUY"
    assert interpret_signal(1.0, _CONFIG) == "BUY"
    assert interpret_signal(0.8, _CONFIG) == "BUY"


def test_interpret_signal_sell():
    assert interpret_signal(0.35, _CONFIG) == "SELL"
    assert interpret_signal(0.0, _CONFIG) == "SELL"
    assert interpret_signal(0.1, _CONFIG) == "SELL"


def test_interpret_signal_hold():
    assert interpret_signal(0.5, _CONFIG) == "HOLD"
    assert interpret_signal(0.36, _CONFIG) == "HOLD"
    assert interpret_signal(0.64, _CONFIG) == "HOLD"
