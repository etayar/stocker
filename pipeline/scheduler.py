"""
pipeline.scheduler

APScheduler background job that periodically retrains the general Chronos model
on the configured stock universe.

Public interface:
  start_scheduler(config)  → BackgroundScheduler (already started)
  stop_scheduler(scheduler) → None

The scheduler runs fine_tune_general() every ``scheduler.retrain_interval_days``
days (default 14) and updates models/general/metadata.json on completion.

Usage (called from api/main.py lifespan):
  from pipeline.scheduler import start_scheduler, stop_scheduler

  @asynccontextmanager
  async def lifespan(app):
      config = load_config(...)
      scheduler = start_scheduler(config)
      yield
      stop_scheduler(scheduler)
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)


# ── Job function ──────────────────────────────────────────────────────────────

def _retrain_job(config: dict) -> None:
    """Background job executed by APScheduler every N days.

    Imports are deferred to avoid loading Chronos at import time.
    Errors are caught and logged so a single failure never kills the scheduler.
    """
    from models.ts_model import fine_tune_general  # noqa: PLC0415  (deferred import)

    universe = config["data"].get("universe", [])
    if not universe:
        logger.warning("scheduler: universe is empty — skipping retrain.")
        return

    logger.info("scheduler: starting general model retrain on %d tickers …", len(universe))
    try:
        fine_tune_general(universe, config)
        logger.info("scheduler: general model retrain complete.")
    except Exception:
        logger.exception("scheduler: general model retrain FAILED.")


# ── Public interface ──────────────────────────────────────────────────────────

def start_scheduler(config: dict) -> BackgroundScheduler:
    """Create, configure, and start the background retraining scheduler.

    The first job run is triggered after one full interval, not immediately
    on startup — this prevents a heavy training job from blocking app startup.

    Parameters
    ----------
    config : Loaded config dict. Reads:
               ``scheduler.retrain_interval_days``  (default 14)

    Returns
    -------
    A running BackgroundScheduler instance.  Keep the reference to stop it
    later via :func:`stop_scheduler`.
    """
    interval_days = int(
        config.get("scheduler", {}).get("retrain_interval_days", 14)
    )

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        _retrain_job,
        args=[config],
        trigger="interval",
        days=interval_days,
        id="general_model_retrain",
        name="General Chronos retrain",
        replace_existing=True,
    )
    scheduler.start()
    logger.info(
        "scheduler: started — general model retrain every %d days.", interval_days
    )
    return scheduler


def stop_scheduler(scheduler: BackgroundScheduler) -> None:
    """Gracefully shut down *scheduler* if it is running.

    Parameters
    ----------
    scheduler : BackgroundScheduler returned by :func:`start_scheduler`.
    """
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("scheduler: stopped.")
