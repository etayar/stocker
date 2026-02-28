"""
tests.conftest

Session-wide pytest fixtures shared across all test modules.
"""

from __future__ import annotations

import pytest

from api.limiter import limiter


@pytest.fixture(autouse=True)
def reset_rate_limits():
    """Reset in-memory rate-limit counters before every test.

    Prevents the per-IP counters from accumulating across tests and
    inadvertently triggering 429 responses in unrelated test cases.
    """
    limiter.reset()
    yield
