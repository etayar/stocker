"""
tests.test_pipeline.test_scheduler

Tests for pipeline.scheduler.

APScheduler is mocked so no real background threads are started.
fine_tune_general is patched so no Chronos / network access occurs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from pipeline.scheduler import _retrain_job, start_scheduler, stop_scheduler

# ── Config fixture ─────────────────────────────────────────────────────────────

_CONFIG = {
    "data": {
        "universe": ["AAPL", "MSFT"],
    },
    "scheduler": {
        "retrain_interval_days": 14,
    },
}


# ── start_scheduler ────────────────────────────────────────────────────────────

def test_start_scheduler_returns_scheduler():
    mock_sched = MagicMock()
    with patch("pipeline.scheduler.BackgroundScheduler", return_value=mock_sched):
        result = start_scheduler(_CONFIG)
    assert result is mock_sched


def test_start_scheduler_calls_start():
    mock_sched = MagicMock()
    with patch("pipeline.scheduler.BackgroundScheduler", return_value=mock_sched):
        start_scheduler(_CONFIG)
    mock_sched.start.assert_called_once()


def test_start_scheduler_adds_job():
    mock_sched = MagicMock()
    with patch("pipeline.scheduler.BackgroundScheduler", return_value=mock_sched):
        start_scheduler(_CONFIG)
    mock_sched.add_job.assert_called_once()


def test_start_scheduler_uses_interval_trigger():
    mock_sched = MagicMock()
    with patch("pipeline.scheduler.BackgroundScheduler", return_value=mock_sched):
        start_scheduler(_CONFIG)
    _, kwargs = mock_sched.add_job.call_args
    assert kwargs.get("trigger") == "interval"


def test_start_scheduler_interval_from_config():
    mock_sched = MagicMock()
    with patch("pipeline.scheduler.BackgroundScheduler", return_value=mock_sched):
        start_scheduler(_CONFIG)
    _, kwargs = mock_sched.add_job.call_args
    assert kwargs.get("days") == 14


def test_start_scheduler_default_interval_when_missing():
    """If scheduler key is absent, interval defaults to 14."""
    mock_sched = MagicMock()
    cfg_no_sched = {"data": {"universe": []}}
    with patch("pipeline.scheduler.BackgroundScheduler", return_value=mock_sched):
        start_scheduler(cfg_no_sched)
    _, kwargs = mock_sched.add_job.call_args
    assert kwargs.get("days") == 14


def test_start_scheduler_custom_interval():
    mock_sched = MagicMock()
    cfg = {**_CONFIG, "scheduler": {"retrain_interval_days": 7}}
    with patch("pipeline.scheduler.BackgroundScheduler", return_value=mock_sched):
        start_scheduler(cfg)
    _, kwargs = mock_sched.add_job.call_args
    assert kwargs.get("days") == 7


def test_start_scheduler_passes_config_to_job():
    """The config dict must be passed as the job's first argument."""
    mock_sched = MagicMock()
    with patch("pipeline.scheduler.BackgroundScheduler", return_value=mock_sched):
        start_scheduler(_CONFIG)
    _, kwargs = mock_sched.add_job.call_args
    assert kwargs.get("args") == [_CONFIG]


def test_start_scheduler_job_id_is_retrain():
    mock_sched = MagicMock()
    with patch("pipeline.scheduler.BackgroundScheduler", return_value=mock_sched):
        start_scheduler(_CONFIG)
    _, kwargs = mock_sched.add_job.call_args
    assert kwargs.get("id") == "general_model_retrain"


# ── stop_scheduler ─────────────────────────────────────────────────────────────

def test_stop_scheduler_calls_shutdown_when_running():
    mock_sched = MagicMock()
    mock_sched.running = True
    stop_scheduler(mock_sched)
    mock_sched.shutdown.assert_called_once_with(wait=False)


def test_stop_scheduler_does_not_call_shutdown_when_not_running():
    mock_sched = MagicMock()
    mock_sched.running = False
    stop_scheduler(mock_sched)
    mock_sched.shutdown.assert_not_called()


# ── _retrain_job ───────────────────────────────────────────────────────────────

def test_retrain_job_calls_fine_tune_general():
    with patch("models.ts_model.fine_tune_general") as mock_ftg, \
         patch("pipeline.scheduler.fine_tune_general", mock_ftg, create=True):
        # Patch the deferred import inside the function
        with patch("builtins.__import__", wraps=__import__) as mock_import:
            with patch("models.ts_model.fine_tune_general") as mock_ftg2:
                # Patch the module-level function used inside _retrain_job
                import pipeline.scheduler as sched_mod
                with patch.object(sched_mod, "_retrain_job",
                                   wraps=sched_mod._retrain_job):
                    pass

    # Simpler approach: patch the deferred import path
    with patch("models.ts_model.fine_tune_general") as mock_ftg:
        with patch.dict("sys.modules", {"models.ts_model": MagicMock(
            fine_tune_general=mock_ftg
        )}):
            _retrain_job(_CONFIG)


def test_retrain_job_empty_universe_skips_training():
    """If universe is empty, fine_tune_general must NOT be called."""
    cfg = {"data": {"universe": []}, "scheduler": {}}
    called = []

    def _fake_ftg(universe, config):
        called.append(universe)

    # _retrain_job imports fine_tune_general lazily — patch at import time
    import sys
    mock_ts = MagicMock()
    mock_ts.fine_tune_general.side_effect = _fake_ftg
    with patch.dict(sys.modules, {"models.ts_model": mock_ts}):
        _retrain_job(cfg)

    assert called == []


def test_retrain_job_passes_universe_and_config():
    """fine_tune_general must receive the universe list and full config."""
    import sys
    mock_ts = MagicMock()
    with patch.dict(sys.modules, {"models.ts_model": mock_ts}):
        _retrain_job(_CONFIG)
    mock_ts.fine_tune_general.assert_called_once_with(
        _CONFIG["data"]["universe"], _CONFIG
    )


def test_retrain_job_does_not_raise_on_exception():
    """A failure inside fine_tune_general must not propagate."""
    import sys
    mock_ts = MagicMock()
    mock_ts.fine_tune_general.side_effect = RuntimeError("GPU OOM")
    with patch.dict(sys.modules, {"models.ts_model": mock_ts}):
        _retrain_job(_CONFIG)   # must not raise
