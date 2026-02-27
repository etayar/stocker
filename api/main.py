"""
api.main

FastAPI application factory and startup/shutdown lifecycle hooks.

Creates the FastAPI app, registers all routers, initialises the
database connection, and loads the config on startup.

Run with:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints (mounted via routers):
  GET  /health          → liveness check
  GET  /signal          → run pipeline, return compound score + signal
  GET  /scores/{ticker} → retrieve historical scores from DB
  POST /ask             → free-text prompt → signal + explanation (agent)
"""

from __future__ import annotations

import logging
import os
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes.ask import router as ask_router
from api.routes.signal import router as signal_router
from pipeline.runner import load_config

logger = logging.getLogger(__name__)

# Resolve config.yaml relative to this file so the app works from any CWD.
_CONFIG_PATH = str(pathlib.Path(__file__).parents[1] / "config.yaml")


def _make_anthropic_client():
    """Return an Anthropic client if the SDK and API key are available, else None."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set — POST /ask will return 503.")
        return None
    try:
        import anthropic  # noqa: PLC0415
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        logger.warning("anthropic package not installed — POST /ask will return 503.")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config and init optional Anthropic client once at startup."""
    app.state.config = load_config(_CONFIG_PATH)
    app.state.anthropic_client = _make_anthropic_client()
    yield
    # Nothing to tear down yet (no persistent connections).


app = FastAPI(
    title="Stocker",
    description="Time-series + LLM sentiment + technical analysis → buy/sell signal.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["ops"])
async def health() -> dict:
    """Liveness check — always returns 200 if the process is up."""
    return {"status": "ok"}


app.include_router(signal_router, tags=["signals"])
app.include_router(ask_router, tags=["agent"])
