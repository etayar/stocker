"""
tests.test_api.test_signal

Tests for api.routes.signal using FastAPI's TestClient (httpx).

run_pipeline is patched at api.routes.signal.run_pipeline so no real data
fetching, ML inference, or database access occurs.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from api.main import app
from data.storage.db import ScoreRecord, get_scores, get_session, init_db

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

_SIGNAL_FIELDS = {
    "ticker", "run_date", "horizon",
    "ts_score", "llms_score", "ta_score",
    "compound_score", "signal",
}

# ── Client fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Single TestClient per module — lifespan runs once."""
    with TestClient(app) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ── GET /signal ───────────────────────────────────────────────────────────────

def test_get_signal_returns_200(client):
    with patch("api.routes.signal.run_pipeline", return_value=_MOCK_RESULT):
        response = client.get("/signal?ticker=AAPL")
    assert response.status_code == 200


def test_get_signal_response_schema(client):
    """Response JSON must contain all SignalResponse fields with correct types."""
    with patch("api.routes.signal.run_pipeline", return_value=_MOCK_RESULT):
        response = client.get("/signal?ticker=AAPL")

    data = response.json()
    assert _SIGNAL_FIELDS == set(data.keys())

    # Type checks
    assert isinstance(data["ticker"], str)
    assert isinstance(data["run_date"], str)
    assert isinstance(data["horizon"], int)
    for score_key in ("ts_score", "llms_score", "ta_score", "compound_score"):
        assert 0.0 <= data[score_key] <= 1.0, f"{score_key} out of range"
    assert data["signal"] in {"BUY", "SELL", "HOLD"}


def test_get_signal_custom_horizon(client):
    mock = {**_MOCK_RESULT, "horizon": 10}
    with patch("api.routes.signal.run_pipeline", return_value=mock):
        response = client.get("/signal?ticker=AAPL&horizon=10")
    assert response.status_code == 200
    assert response.json()["horizon"] == 10


def test_get_signal_ticker_in_response(client):
    mock = {**_MOCK_RESULT, "ticker": "TSLA"}
    with patch("api.routes.signal.run_pipeline", return_value=mock):
        response = client.get("/signal?ticker=TSLA")
    assert response.json()["ticker"] == "TSLA"


def test_get_signal_missing_ticker_returns_422(client):
    """ticker is required — omitting it must yield 422 Unprocessable Entity."""
    response = client.get("/signal")
    assert response.status_code == 422


def test_get_signal_horizon_zero_returns_422(client):
    """horizon must be >= 1."""
    response = client.get("/signal?ticker=AAPL&horizon=0")
    assert response.status_code == 422


def test_get_signal_invalid_ticker_returns_error(client):
    """run_pipeline raising ValueError → 400 error response."""
    with patch(
        "api.routes.signal.run_pipeline",
        side_effect=ValueError("Unknown ticker: XXXXXXXXX"),
    ):
        response = client.get("/signal?ticker=XXXXXXXXX")
    assert response.status_code == 400
    assert "XXXXXXXXX" in response.json()["detail"]


def test_get_signal_pipeline_exception_returns_500(client):
    """Unexpected pipeline failure → 500 error response."""
    with patch(
        "api.routes.signal.run_pipeline",
        side_effect=RuntimeError("data source unavailable"),
    ):
        response = client.get("/signal?ticker=AAPL")
    assert response.status_code == 500


def test_get_signal_500_does_not_leak_exception_detail(client):
    """500 responses must NOT expose internal error messages to the caller."""
    with patch(
        "api.routes.signal.run_pipeline",
        side_effect=RuntimeError("super secret internal error"),
    ):
        response = client.get("/signal?ticker=AAPL")
    assert "super secret internal error" not in response.json().get("detail", "")


# ── Security: ticker format validation (path traversal prevention) ─────────────

def test_get_signal_path_traversal_ticker_returns_400(client):
    """A ticker with path-traversal chars must be rejected (9 chars, passes length check)."""
    response = client.get("/signal?ticker=../../etc")  # 9 chars, invalid chars
    assert response.status_code == 400


def test_get_signal_special_chars_in_ticker_returns_400(client):
    """Shell metacharacters in ticker must be rejected."""
    response = client.get("/signal?ticker=AAPL%3Bx")  # "AAPL;x" URL-encoded
    assert response.status_code == 400


def test_get_signal_ticker_too_long_returns_422(client):
    """Tickers longer than 10 characters return 422 from FastAPI Query validation."""
    response = client.get("/signal?ticker=TOOLONGTICKER")
    assert response.status_code == 422


def test_get_signal_lowercase_ticker_is_normalised(client):
    """Lowercase ticker must be uppercased and accepted."""
    with patch("api.routes.signal.run_pipeline", return_value=_MOCK_RESULT):
        response = client.get("/signal?ticker=aapl")
    assert response.status_code == 200


def test_get_signal_passes_horizon_to_pipeline(client):
    """The horizon query param must reach run_pipeline as an int."""
    with patch("api.routes.signal.run_pipeline", return_value=_MOCK_RESULT) as mock_rp:
        client.get("/signal?ticker=AAPL&horizon=7")
    _, call_kwargs = mock_rp.call_args
    args = mock_rp.call_args[0]
    # run_pipeline(ticker, horizon, config) — positional
    assert args[1] == 7


def test_get_signal_passes_ticker_to_pipeline(client):
    with patch("api.routes.signal.run_pipeline", return_value=_MOCK_RESULT) as mock_rp:
        client.get("/signal?ticker=NVDA")
    assert mock_rp.call_args[0][0] == "NVDA"


# ── GET /scores/{ticker} ──────────────────────────────────────────────────────

def test_get_scores_returns_list(client):
    response = client.get("/scores/AAPL")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_scores_empty_when_no_db_engine(client):
    """When db_engine is None (no credentials), endpoint returns []."""
    response = client.get("/scores/TSLA")
    assert response.json() == []


def test_get_scores_custom_limit_accepted(client):
    """limit query param should be accepted without error."""
    response = client.get("/scores/AAPL?limit=10")
    assert response.status_code == 200


def test_get_scores_limit_zero_returns_422(client):
    """limit must be >= 1."""
    response = client.get("/scores/AAPL?limit=0")
    assert response.status_code == 422


# ── DB integration tests ──────────────────────────────────────────────────────

@pytest.fixture
def client_with_db(client):
    """Function-scoped: reuse module client but inject a fresh SQLite engine.

    StaticPool ensures all connections share the same in-memory database so
    tables created by init_db() are visible to every subsequent query.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    original_engine = getattr(app.state, "db_engine", None)
    app.state.db_engine = engine
    yield client
    engine.dispose()
    app.state.db_engine = original_engine


def test_get_signal_persists_score_to_db(client_with_db):
    """Calling /signal should write a ScoreRecord to the DB."""
    with patch("api.routes.signal.run_pipeline", return_value=_MOCK_RESULT):
        response = client_with_db.get("/signal?ticker=AAPL")
    assert response.status_code == 200

    engine = app.state.db_engine
    with get_session(engine) as session:
        records = get_scores("AAPL", 10, session)
    assert len(records) == 1
    assert records[0].ticker == "AAPL"
    assert records[0].signal == _MOCK_RESULT["signal"]


def test_get_scores_returns_persisted_records(client_with_db):
    """After /signal populates the DB, /scores/{ticker} should return the record."""
    with patch("api.routes.signal.run_pipeline", return_value=_MOCK_RESULT):
        client_with_db.get("/signal?ticker=AAPL")

    response = client_with_db.get("/scores/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["ticker"] == "AAPL"
    assert data[0]["signal"] == _MOCK_RESULT["signal"]


def test_get_scores_respects_limit(client_with_db):
    """With 3 records in DB, limit=2 should return only 2."""
    for date in ("2026-02-25", "2026-02-26", "2026-02-27"):
        mock = {**_MOCK_RESULT, "run_date": date}
        with patch("api.routes.signal.run_pipeline", return_value=mock):
            client_with_db.get("/signal?ticker=AAPL")

    response = client_with_db.get("/scores/AAPL?limit=2")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_get_scores_filters_by_ticker(client_with_db):
    """Scores for TSLA must not appear when querying AAPL."""
    aapl_mock = {**_MOCK_RESULT, "ticker": "AAPL"}
    tsla_mock = {**_MOCK_RESULT, "ticker": "TSLA"}

    with patch("api.routes.signal.run_pipeline", return_value=aapl_mock):
        client_with_db.get("/signal?ticker=AAPL")
    with patch("api.routes.signal.run_pipeline", return_value=tsla_mock):
        client_with_db.get("/signal?ticker=TSLA")

    response = client_with_db.get("/scores/AAPL")
    data = response.json()
    assert all(r["ticker"] == "AAPL" for r in data)
    assert len(data) == 1
