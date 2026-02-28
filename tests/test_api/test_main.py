"""
tests.test_api.test_main

Tests for api.main lifespan behaviour: scheduler wiring and rate-limiter setup.

All background-scheduler and Chronos calls are mocked so no real threads or
ML work occurs during the test suite.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_mock_scheduler(running: bool = True) -> MagicMock:
    m = MagicMock()
    m.running = running
    return m


# ── Scheduler wiring ──────────────────────────────────────────────────────────

class TestSchedulerLifespan:
    def test_start_scheduler_called_on_startup(self):
        """start_scheduler must be called once during lifespan startup."""
        mock_sched = _make_mock_scheduler()
        with patch("api.main.start_scheduler", return_value=mock_sched) as mock_start, \
             patch("api.main.stop_scheduler"):
            with TestClient(app):
                mock_start.assert_called_once()

    def test_start_scheduler_receives_config(self):
        """start_scheduler must receive the loaded config dict."""
        mock_sched = _make_mock_scheduler()
        with patch("api.main.start_scheduler", return_value=mock_sched) as mock_start, \
             patch("api.main.stop_scheduler"):
            with TestClient(app) as client:
                config = app.state.config
            mock_start.assert_called_once_with(config)

    def test_app_state_scheduler_is_set(self):
        """app.state.scheduler must be the object returned by start_scheduler."""
        mock_sched = _make_mock_scheduler()
        with patch("api.main.start_scheduler", return_value=mock_sched), \
             patch("api.main.stop_scheduler"):
            with TestClient(app):
                assert app.state.scheduler is mock_sched

    def test_stop_scheduler_called_on_shutdown(self):
        """stop_scheduler must be called with the running scheduler on teardown."""
        mock_sched = _make_mock_scheduler()
        with patch("api.main.start_scheduler", return_value=mock_sched), \
             patch("api.main.stop_scheduler") as mock_stop:
            with TestClient(app):
                pass
        mock_stop.assert_called_once_with(mock_sched)

    def test_scheduler_failure_does_not_crash_startup(self):
        """If start_scheduler raises, the app must still start successfully."""
        with patch("api.main.start_scheduler", side_effect=RuntimeError("APScheduler error")), \
             patch("api.main.stop_scheduler"):
            with TestClient(app) as client:
                response = client.get("/health")
            assert response.status_code == 200

    def test_scheduler_none_when_startup_fails(self):
        """app.state.scheduler must be None when start_scheduler fails."""
        with patch("api.main.start_scheduler", side_effect=RuntimeError("fail")), \
             patch("api.main.stop_scheduler"):
            with TestClient(app):
                assert app.state.scheduler is None

    def test_stop_scheduler_not_called_when_scheduler_is_none(self):
        """stop_scheduler must not be called if startup failed (scheduler is None)."""
        with patch("api.main.start_scheduler", side_effect=RuntimeError("fail")), \
             patch("api.main.stop_scheduler") as mock_stop:
            with TestClient(app):
                pass
        mock_stop.assert_not_called()


# ── Rate limiter setup ────────────────────────────────────────────────────────

class TestRateLimiterSetup:
    def test_limiter_registered_on_app_state(self):
        """app.state.limiter must be set so slowapi can inject response headers."""
        from api.limiter import limiter as expected_limiter
        with patch("api.main.start_scheduler", return_value=_make_mock_scheduler()), \
             patch("api.main.stop_scheduler"):
            with TestClient(app):
                assert app.state.limiter is expected_limiter

    def test_rate_limit_exceeded_handler_registered(self):
        """RateLimitExceeded must have a registered exception handler."""
        from slowapi.errors import RateLimitExceeded
        assert RateLimitExceeded in app.exception_handlers
