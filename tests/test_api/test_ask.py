"""
tests.test_api.test_ask

Tests for POST /ask using FastAPI's TestClient (httpx).

All external dependencies are mocked:
  - app.state.anthropic_client  → MagicMock injected at module scope
  - api.routes.ask.parse_query  → patched per-test
  - api.routes.ask.run_pipeline → patched per-test
  - api.routes.ask.explain_signal → patched per-test
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import QueryParams

# ── Shared mock data ───────────────────────────────────────────────────────────

_MOCK_QUERY = QueryParams(ticker="AAPL", horizon=7, granularity="daily")

_MOCK_RESULT = {
    "ticker":         "AAPL",
    "run_date":       "2026-02-27",
    "horizon":        7,
    "ts_score":       0.70,
    "llms_score":     0.60,
    "ta_score":       0.80,
    "compound_score": round((0.70 * 0.60 * 0.80) ** (1 / 3), 6),
    "signal":         "BUY",
}

_MOCK_EXPLANATION = "AAPL shows a BUY signal based on strong TS and TA scores."

_ASK_FIELDS = {
    "ticker", "run_date", "horizon",
    "ts_score", "llms_score", "ta_score",
    "compound_score", "signal",
    "query_params", "explanation",
}

_PROMPT = "Should I buy Apple stock for the next week?"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _all_mocked(parse_rv=_MOCK_QUERY, pipeline_rv=_MOCK_RESULT, explain_rv=_MOCK_EXPLANATION):
    """Return a list of (target, kwargs) patch specs for the three collaborators."""
    return [
        ("api.routes.ask.parse_query",   {"return_value": parse_rv}),
        ("api.routes.ask.run_pipeline",  {"return_value": pipeline_rv}),
        ("api.routes.ask.explain_signal", {"return_value": explain_rv}),
    ]


# ── Client fixture ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient with a fake anthropic_client pre-injected."""
    with TestClient(app) as c:
        # Inject a MagicMock so the route doesn't raise 503 during tests.
        app.state.anthropic_client = MagicMock()
        yield c


# ── 200 happy-path ─────────────────────────────────────────────────────────────

def test_ask_returns_200(client):
    with patch("api.routes.ask.parse_query", return_value=_MOCK_QUERY), \
         patch("api.routes.ask.run_pipeline", return_value=_MOCK_RESULT), \
         patch("api.routes.ask.explain_signal", return_value=_MOCK_EXPLANATION):
        response = client.post("/ask", json={"prompt": _PROMPT})
    assert response.status_code == 200


def test_ask_response_schema(client):
    """Response must contain all AskResponse fields with correct types."""
    with patch("api.routes.ask.parse_query", return_value=_MOCK_QUERY), \
         patch("api.routes.ask.run_pipeline", return_value=_MOCK_RESULT), \
         patch("api.routes.ask.explain_signal", return_value=_MOCK_EXPLANATION):
        response = client.post("/ask", json={"prompt": _PROMPT})

    data = response.json()
    assert _ASK_FIELDS == set(data.keys())

    # Types
    assert isinstance(data["ticker"], str)
    assert isinstance(data["horizon"], int)
    assert isinstance(data["explanation"], str)
    assert isinstance(data["query_params"], dict)
    for score_key in ("ts_score", "llms_score", "ta_score", "compound_score"):
        assert 0.0 <= data[score_key] <= 1.0
    assert data["signal"] in {"BUY", "SELL", "HOLD"}


def test_ask_query_params_in_response(client):
    """query_params must reflect what parse_query returned."""
    with patch("api.routes.ask.parse_query", return_value=_MOCK_QUERY), \
         patch("api.routes.ask.run_pipeline", return_value=_MOCK_RESULT), \
         patch("api.routes.ask.explain_signal", return_value=_MOCK_EXPLANATION):
        response = client.post("/ask", json={"prompt": _PROMPT})

    qp = response.json()["query_params"]
    assert qp["ticker"] == "AAPL"
    assert qp["horizon"] == 7
    assert qp["granularity"] == "daily"


def test_ask_explanation_in_response(client):
    with patch("api.routes.ask.parse_query", return_value=_MOCK_QUERY), \
         patch("api.routes.ask.run_pipeline", return_value=_MOCK_RESULT), \
         patch("api.routes.ask.explain_signal", return_value=_MOCK_EXPLANATION):
        response = client.post("/ask", json={"prompt": _PROMPT})

    assert response.json()["explanation"] == _MOCK_EXPLANATION


# ── Argument forwarding ────────────────────────────────────────────────────────

def test_ask_passes_prompt_to_parse_query(client):
    with patch("api.routes.ask.parse_query", return_value=_MOCK_QUERY) as mock_parse, \
         patch("api.routes.ask.run_pipeline", return_value=_MOCK_RESULT), \
         patch("api.routes.ask.explain_signal", return_value=_MOCK_EXPLANATION):
        client.post("/ask", json={"prompt": _PROMPT})

    assert mock_parse.call_args[0][0] == _PROMPT


def test_ask_passes_ticker_and_horizon_to_pipeline(client):
    custom_query = QueryParams(ticker="TSLA", horizon=14, granularity="daily")
    custom_result = {**_MOCK_RESULT, "ticker": "TSLA", "horizon": 14}
    with patch("api.routes.ask.parse_query", return_value=custom_query), \
         patch("api.routes.ask.run_pipeline", return_value=custom_result) as mock_rp, \
         patch("api.routes.ask.explain_signal", return_value=_MOCK_EXPLANATION):
        client.post("/ask", json={"prompt": "Buy Tesla for 2 weeks?"})

    args = mock_rp.call_args[0]
    assert args[0] == "TSLA"
    assert args[1] == 14


def test_ask_passes_result_and_prompt_to_explain(client):
    with patch("api.routes.ask.parse_query", return_value=_MOCK_QUERY), \
         patch("api.routes.ask.run_pipeline", return_value=_MOCK_RESULT), \
         patch("api.routes.ask.explain_signal", return_value=_MOCK_EXPLANATION) as mock_ex:
        client.post("/ask", json={"prompt": _PROMPT})

    args = mock_ex.call_args[0]
    assert args[0] == _MOCK_RESULT   # result dict
    assert args[1] == _PROMPT        # original prompt


# ── Validation errors ──────────────────────────────────────────────────────────

def test_ask_missing_body_returns_422(client):
    response = client.post("/ask")
    assert response.status_code == 422


def test_ask_empty_prompt_returns_422(client):
    response = client.post("/ask", json={"prompt": ""})
    assert response.status_code == 422


def test_ask_prompt_too_long_returns_422(client):
    response = client.post("/ask", json={"prompt": "x" * 1001})
    assert response.status_code == 422


# ── 503 — agent unavailable ────────────────────────────────────────────────────

def test_ask_returns_503_when_no_anthropic_client(client):
    """If anthropic_client is None, /ask must return 503."""
    original = app.state.anthropic_client
    app.state.anthropic_client = None
    try:
        response = client.post("/ask", json={"prompt": _PROMPT})
    finally:
        app.state.anthropic_client = original
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


# ── 400 — parse failure ────────────────────────────────────────────────────────

def test_ask_returns_400_on_parse_value_error(client):
    with patch(
        "api.routes.ask.parse_query",
        side_effect=ValueError("Could not extract ticker"),
    ):
        response = client.post("/ask", json={"prompt": "gibberish input"})
    assert response.status_code == 400
    assert "ticker" in response.json()["detail"]


def test_ask_returns_502_on_parse_generic_error(client):
    with patch(
        "api.routes.ask.parse_query",
        side_effect=RuntimeError("API timeout"),
    ):
        response = client.post("/ask", json={"prompt": _PROMPT})
    assert response.status_code == 502


# ── 500 — pipeline failure ─────────────────────────────────────────────────────

def test_ask_returns_500_on_pipeline_error(client):
    with patch("api.routes.ask.parse_query", return_value=_MOCK_QUERY), \
         patch(
             "api.routes.ask.run_pipeline",
             side_effect=RuntimeError("data source unavailable"),
         ):
        response = client.post("/ask", json={"prompt": _PROMPT})
    assert response.status_code == 500


def test_ask_500_does_not_leak_exception_detail(client):
    """500 responses must NOT expose internal error messages to the caller."""
    with patch("api.routes.ask.parse_query", return_value=_MOCK_QUERY), \
         patch(
             "api.routes.ask.run_pipeline",
             side_effect=RuntimeError("super secret DB connection string"),
         ):
        response = client.post("/ask", json={"prompt": _PROMPT})
    assert "super secret DB connection string" not in response.json().get("detail", "")


def test_ask_502_does_not_leak_exception_detail(client):
    """502 responses from agent errors must NOT expose internal error messages."""
    with patch(
        "api.routes.ask.parse_query",
        side_effect=RuntimeError("internal API key: sk-1234"),
    ):
        response = client.post("/ask", json={"prompt": _PROMPT})
    assert response.status_code == 502
    assert "sk-1234" not in response.json().get("detail", "")


def test_ask_returns_400_on_pipeline_value_error(client):
    with patch("api.routes.ask.parse_query", return_value=_MOCK_QUERY), \
         patch(
             "api.routes.ask.run_pipeline",
             side_effect=ValueError("Unknown ticker: FAKE"),
         ):
        response = client.post("/ask", json={"prompt": _PROMPT})
    assert response.status_code == 400


# ── Explanation fallback ───────────────────────────────────────────────────────

def test_ask_explanation_fallback_on_agent_error(client):
    """If explain_signal raises, the route must still return 200 with a fallback."""
    with patch("api.routes.ask.parse_query", return_value=_MOCK_QUERY), \
         patch("api.routes.ask.run_pipeline", return_value=_MOCK_RESULT), \
         patch(
             "api.routes.ask.explain_signal",
             side_effect=RuntimeError("explain failed"),
         ):
        response = client.post("/ask", json={"prompt": _PROMPT})

    assert response.status_code == 200
    data = response.json()
    # Fallback string must mention the signal and compound score.
    assert _MOCK_RESULT["signal"] in data["explanation"]
