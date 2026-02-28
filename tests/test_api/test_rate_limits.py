"""
tests.test_api.test_rate_limits

Verifies that every public endpoint enforces its per-IP rate limit and
returns HTTP 429 once the limit is exceeded.

The autouse fixture in tests/conftest.py resets the in-memory limiter
storage before each test, so counters never bleed between tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.limiter import LIMIT_ASK, LIMIT_SCORES, LIMIT_SIGNAL
from api.main import app
from api.schemas import QueryParams

# ── Shared mock data ──────────────────────────────────────────────────────────

_MOCK_RESULT = {
    "ticker":         "AAPL",
    "run_date":       "2026-02-27",
    "horizon":        5,
    "ts_score":       0.70,
    "llms_score":     0.60,
    "ta_score":       0.80,
    "compound_score": round((0.70 * 0.60 * 0.80) ** (1 / 3), 6),
    "signal":         "BUY",
}

_MOCK_QUERY       = QueryParams(ticker="AAPL", horizon=5, granularity="daily")
_MOCK_EXPLANATION = "Strong BUY signal."


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_limit(limit_str: str) -> int:
    """Extract the numeric count from a limit string like '10/minute'."""
    return int(limit_str.split("/")[0])


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Module-scoped client; anthropic_client pre-injected for /ask tests."""
    with TestClient(app) as c:
        app.state.anthropic_client = MagicMock()
        yield c


# ── GET /signal ───────────────────────────────────────────────────────────────

class TestSignalRateLimit:
    _LIMIT = _parse_limit(LIMIT_SIGNAL)

    def test_requests_within_limit_succeed(self, client):
        with patch("api.routes.signal.run_pipeline", return_value=_MOCK_RESULT):
            for _ in range(self._LIMIT):
                r = client.get("/signal?ticker=AAPL")
        assert r.status_code == 200

    def test_exceeding_limit_returns_429(self, client):
        with patch("api.routes.signal.run_pipeline", return_value=_MOCK_RESULT):
            for _ in range(self._LIMIT):
                client.get("/signal?ticker=AAPL")
        # (self._LIMIT + 1)-th request — limiter fires before pipeline runs
        r = client.get("/signal?ticker=AAPL")
        assert r.status_code == 429

    def test_429_response_has_error_key(self, client):
        with patch("api.routes.signal.run_pipeline", return_value=_MOCK_RESULT):
            for _ in range(self._LIMIT):
                client.get("/signal?ticker=AAPL")
        r = client.get("/signal?ticker=AAPL")
        assert "error" in r.json()

    def test_limit_resets_after_counter_cleared(self, client):
        """After limiter.reset(), requests should succeed again."""
        from api.limiter import limiter
        with patch("api.routes.signal.run_pipeline", return_value=_MOCK_RESULT):
            for _ in range(self._LIMIT):
                client.get("/signal?ticker=AAPL")
        r = client.get("/signal?ticker=AAPL")
        assert r.status_code == 429

        limiter.reset()  # simulate new time window

        with patch("api.routes.signal.run_pipeline", return_value=_MOCK_RESULT):
            r = client.get("/signal?ticker=AAPL")
        assert r.status_code == 200


# ── POST /ask ─────────────────────────────────────────────────────────────────

class TestAskRateLimit:
    _LIMIT = _parse_limit(LIMIT_ASK)

    def _post(self, client):
        return client.post("/ask", json={"prompt": "Should I buy AAPL?"})

    def test_requests_within_limit_succeed(self, client):
        with patch("api.routes.ask.parse_query",    return_value=_MOCK_QUERY), \
             patch("api.routes.ask.run_pipeline",   return_value=_MOCK_RESULT), \
             patch("api.routes.ask.explain_signal", return_value=_MOCK_EXPLANATION):
            for _ in range(self._LIMIT):
                r = self._post(client)
        assert r.status_code == 200

    def test_exceeding_limit_returns_429(self, client):
        with patch("api.routes.ask.parse_query",    return_value=_MOCK_QUERY), \
             patch("api.routes.ask.run_pipeline",   return_value=_MOCK_RESULT), \
             patch("api.routes.ask.explain_signal", return_value=_MOCK_EXPLANATION):
            for _ in range(self._LIMIT):
                self._post(client)
        # Limiter fires before handler — no mocks needed
        r = self._post(client)
        assert r.status_code == 429

    def test_429_response_has_error_key(self, client):
        with patch("api.routes.ask.parse_query",    return_value=_MOCK_QUERY), \
             patch("api.routes.ask.run_pipeline",   return_value=_MOCK_RESULT), \
             patch("api.routes.ask.explain_signal", return_value=_MOCK_EXPLANATION):
            for _ in range(self._LIMIT):
                self._post(client)
        r = self._post(client)
        assert "error" in r.json()


# ── GET /scores/{ticker} ──────────────────────────────────────────────────────

class TestScoresRateLimit:
    _LIMIT = _parse_limit(LIMIT_SCORES)

    def test_requests_within_limit_succeed(self, client):
        # No DB engine → endpoint returns [] immediately; fast to loop
        for _ in range(self._LIMIT):
            r = client.get("/scores/AAPL")
        assert r.status_code == 200

    def test_exceeding_limit_returns_429(self, client):
        for _ in range(self._LIMIT):
            client.get("/scores/AAPL")
        r = client.get("/scores/AAPL")
        assert r.status_code == 429

    def test_429_response_has_error_key(self, client):
        for _ in range(self._LIMIT):
            client.get("/scores/AAPL")
        r = client.get("/scores/AAPL")
        assert "error" in r.json()
