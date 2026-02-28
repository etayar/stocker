"""
tests.test_data.test_db

Unit tests for data.storage.db using an in-memory SQLite database.

No running PostgreSQL required — SQLAlchemy's dialect differences are
minimal for the queries used here.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import StaticPool

from data.storage.db import (
    Base,
    ScoreRecord,
    get_engine,
    get_scores,
    get_session,
    init_db,
    save_score,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def engine():
    """SQLite in-memory engine with all tables created.

    StaticPool ensures every connection reuses the same in-memory database,
    so tables created by init_db() are visible to all subsequent queries.
    """
    e = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(e)
    yield e
    e.dispose()


@pytest.fixture
def session(engine):
    """Open session that is always rolled back after each test."""
    with get_session(engine) as s:
        yield s


# ── Shared mock result ────────────────────────────────────────────────────────

def _make_result(
    ticker="AAPL",
    run_date="2026-02-27",
    horizon=5,
    ts_score=0.70,
    llms_score=0.60,
    ta_score=0.80,
    compound_score=0.6952,
    signal="BUY",
) -> dict:
    return {
        "ticker":         ticker,
        "run_date":       run_date,
        "horizon":        horizon,
        "ts_score":       ts_score,
        "llms_score":     llms_score,
        "ta_score":       ta_score,
        "compound_score": compound_score,
        "signal":         signal,
    }


# ── init_db ───────────────────────────────────────────────────────────────────

class TestInitDb:
    def test_creates_scores_table(self, engine):
        tables = inspect(engine).get_table_names()
        assert "scores" in tables

    def test_creates_prices_table(self, engine):
        tables = inspect(engine).get_table_names()
        assert "prices" in tables

    def test_creates_news_table(self, engine):
        tables = inspect(engine).get_table_names()
        assert "news" in tables

    def test_safe_to_call_twice(self, engine):
        """Second call must not raise (idempotent)."""
        init_db(engine)
        tables = inspect(engine).get_table_names()
        assert "scores" in tables


# ── save_score ────────────────────────────────────────────────────────────────

class TestSaveScore:
    def test_inserts_new_record(self, engine):
        result = _make_result()
        with get_session(engine) as session:
            save_score(result, session)

        with get_session(engine) as session:
            records = get_scores("AAPL", 10, session)
        assert len(records) == 1

    def test_all_fields_stored(self, engine):
        result = _make_result()
        with get_session(engine) as session:
            save_score(result, session)

        with get_session(engine) as session:
            records = get_scores("AAPL", 10, session)
        r = records[0]
        assert r.ticker         == "AAPL"
        assert r.run_date       == "2026-02-27"
        assert r.horizon        == 5
        assert r.ts_score       == pytest.approx(0.70)
        assert r.llms_score     == pytest.approx(0.60)
        assert r.ta_score       == pytest.approx(0.80)
        assert r.compound_score == pytest.approx(0.6952)
        assert r.signal         == "BUY"

    def test_upsert_on_duplicate_key(self, engine):
        """Re-saving (ticker, run_date, horizon) updates existing row — no duplicates."""
        result = _make_result(ts_score=0.70, signal="BUY")
        with get_session(engine) as session:
            save_score(result, session)

        updated = _make_result(ts_score=0.90, signal="HOLD")
        with get_session(engine) as session:
            save_score(updated, session)

        with get_session(engine) as session:
            records = get_scores("AAPL", 10, session)
        assert len(records) == 1
        assert records[0].ts_score == pytest.approx(0.90)
        assert records[0].signal   == "HOLD"

    def test_returns_score_record_instance(self, engine):
        result = _make_result()
        with get_session(engine) as session:
            record = save_score(result, session)
        assert isinstance(record, ScoreRecord)

    def test_created_at_is_set(self, engine):
        result = _make_result()
        with get_session(engine) as session:
            save_score(result, session)

        with get_session(engine) as session:
            records = get_scores("AAPL", 10, session)
        assert records[0].created_at is not None


# ── get_scores ────────────────────────────────────────────────────────────────

class TestGetScores:
    def test_returns_empty_for_unknown_ticker(self, engine):
        with get_session(engine) as session:
            records = get_scores("UNKNOWN", 10, session)
        assert records == []

    def test_returns_saved_records(self, engine):
        result = _make_result()
        with get_session(engine) as session:
            save_score(result, session)

        with get_session(engine) as session:
            records = get_scores("AAPL", 10, session)
        assert len(records) == 1

    def test_newest_first(self, engine):
        """Records must come back ordered by run_date DESC."""
        for date in ("2026-02-25", "2026-02-27", "2026-02-26"):
            with get_session(engine) as session:
                save_score(_make_result(run_date=date), session)

        with get_session(engine) as session:
            records = get_scores("AAPL", 10, session)
        dates = [r.run_date for r in records]
        assert dates == sorted(dates, reverse=True)

    def test_respects_limit(self, engine):
        for date in ("2026-02-25", "2026-02-26", "2026-02-27"):
            with get_session(engine) as session:
                save_score(_make_result(run_date=date), session)

        with get_session(engine) as session:
            records = get_scores("AAPL", 2, session)
        assert len(records) == 2

    def test_filters_by_ticker(self, engine):
        with get_session(engine) as session:
            save_score(_make_result(ticker="AAPL"), session)
        with get_session(engine) as session:
            save_score(_make_result(ticker="TSLA"), session)

        with get_session(engine) as session:
            records = get_scores("AAPL", 10, session)
        assert all(r.ticker == "AAPL" for r in records)
        assert len(records) == 1


# ── get_session ───────────────────────────────────────────────────────────────

class TestGetSession:
    def test_commits_on_success(self, engine):
        result = _make_result()
        with get_session(engine) as session:
            save_score(result, session)

        # Open a fresh session to verify the data persisted
        with get_session(engine) as session:
            records = get_scores("AAPL", 10, session)
        assert len(records) == 1

    def test_rolls_back_on_error(self, engine):
        result = _make_result()
        try:
            with get_session(engine) as session:
                save_score(result, session)
                raise RuntimeError("forced error")
        except RuntimeError:
            pass

        # Rollback means no row should have been committed
        with get_session(engine) as session:
            records = get_scores("AAPL", 10, session)
        assert len(records) == 0


# ── get_engine ────────────────────────────────────────────────────────────────

class TestGetEngine:
    def test_returns_none_when_no_credentials(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.delenv("DB_PASSWORD",  raising=False)
        engine = get_engine({})
        assert engine is None

    def test_uses_database_url_env_var(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
        monkeypatch.delenv("DB_PASSWORD", raising=False)
        engine = get_engine({})
        assert engine is not None
        engine.dispose()

    def test_uses_db_password_with_config(self, monkeypatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        monkeypatch.setenv("DB_PASSWORD", "secret")
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "stocker",
                "user": "stocker_user",
            }
        }
        engine = get_engine(config)
        assert engine is not None
        # Don't try to connect — just verify an engine object was returned
        engine.dispose()
