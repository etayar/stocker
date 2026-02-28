"""
api.routes.signal

Signal route — triggers the full Stocker pipeline for a given ticker
and returns the compound buy/sell signal.

Endpoints:
  GET /signal
    Query params:
      ticker  : str  (e.g. "AAPL")          required
      horizon : int  (days ahead, default=5) optional
    Response: SignalResponse schema

  GET /scores/{ticker}
    Path param: ticker : str
    Query params:
      limit : int (number of historical records, default=30) optional
    Response: list[ScoreRecord] schema

All heavy computation is delegated to pipeline.runner.run_pipeline().
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, HTTPException, Query, Request

from api.schemas import HistoricalScoreRecord, SignalResponse
from data.storage.db import get_session, get_scores, save_score
from pipeline.runner import run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()

_TICKER_RE = re.compile(r"^[A-Z0-9.]{1,10}$")


@router.get("/signal", response_model=SignalResponse)
async def get_signal(
    request: Request,
    ticker: str = Query(
        ...,
        description="Stock ticker symbol (e.g. AAPL)",
        min_length=1,
        max_length=10,
    ),
    horizon: int = Query(
        5,
        description="Prediction horizon in days",
        ge=1,
        le=365,
    ),
) -> dict:
    """Run the full pipeline for *ticker* and return the compound signal.

    Raises 400 for invalid input (e.g. unknown ticker).
    Raises 500 for unexpected pipeline failures.
    """
    ticker_up = ticker.upper().strip()
    if not _TICKER_RE.match(ticker_up):
        raise HTTPException(
            status_code=400,
            detail="Invalid ticker format. Must be 1-10 uppercase letters, digits, or dots.",
        )

    try:
        config = request.app.state.config
        result = run_pipeline(ticker_up, horizon, config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Pipeline error ticker=%s horizon=%d: %s", ticker_up, horizon, exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail="An internal error occurred. Please try again later."
        ) from exc

    db_engine = getattr(request.app.state, "db_engine", None)
    if db_engine is not None:
        try:
            with get_session(db_engine) as session:
                save_score(result, session)
        except Exception as exc:
            logger.warning("Failed to persist score: %s", exc)

    return result


def _record_to_dict(record) -> dict:
    """Serialise a ScoreRecord ORM row to a dict matching HistoricalScoreRecord."""
    return {
        "id":             record.id,
        "ticker":         record.ticker,
        "run_date":       record.run_date,
        "horizon":        record.horizon,
        "ts_score":       record.ts_score,
        "llms_score":     record.llms_score,
        "ta_score":       record.ta_score,
        "compound_score": record.compound_score,
        "signal":         record.signal,
    }


@router.get(
    "/scores/{ticker}",
    response_model=list[HistoricalScoreRecord],
)
async def get_scores_endpoint(
    ticker: str,
    request: Request,
    limit: int = Query(30, description="Max records to return", ge=1, le=500),
) -> list:
    """Return historical pipeline results for *ticker* from the database."""
    db_engine = getattr(request.app.state, "db_engine", None)
    if db_engine is None:
        return []
    try:
        with get_session(db_engine) as session:
            records = get_scores(ticker.upper(), limit, session)
        return [_record_to_dict(r) for r in records]
    except Exception as exc:
        logger.error("Failed to fetch scores for ticker=%s: %s", ticker, exc, exc_info=True)
        return []
