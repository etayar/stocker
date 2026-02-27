"""
api.main

FastAPI application factory and startup/shutdown lifecycle hooks.

Creates the FastAPI app, registers all routers, initialises the
database connection, and loads the config on startup.

Run with:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints (mounted via routers):
  GET /health          → liveness check
  GET /signal          → run pipeline, return compound score + signal
  GET /scores/{ticker} → retrieve historical scores from DB
"""
