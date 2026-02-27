"""
api.schemas

Pydantic request and response models for the Stocker API.

Models:
  SignalRequest
    Fields: ticker (str), horizon (int, default=5)

  ComponentScores
    Fields: ts_score (float), llms_score (float), ta_score (float)

  SignalResponse
    Fields:
      ticker         : str
      run_date       : datetime
      horizon        : int
      ts_score       : float  ∈ [0, 1]
      llms_score     : float  ∈ [0, 1]
      ta_score       : float  ∈ [0, 1]
      compound_score : float  ∈ [0, 1]
      signal         : Literal["BUY", "SELL", "HOLD"]

  HistoricalScoreRecord
    Fields: same as SignalResponse plus id (int)

  ErrorResponse
    Fields: detail (str)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ComponentScores(BaseModel):
    """Individual component scores before compounding."""

    ts_score:   float = Field(ge=0.0, le=1.0)
    llms_score: float = Field(ge=0.0, le=1.0)
    ta_score:   float = Field(ge=0.0, le=1.0)


class SignalResponse(BaseModel):
    """Full pipeline result returned by GET /signal."""

    ticker:         str
    run_date:       str                         # ISO-8601 date string, e.g. "2026-02-27"
    horizon:        int = Field(ge=1)
    ts_score:       float = Field(ge=0.0, le=1.0)
    llms_score:     float = Field(ge=0.0, le=1.0)
    ta_score:       float = Field(ge=0.0, le=1.0)
    compound_score: float = Field(ge=0.0, le=1.0)
    signal:         Literal["BUY", "SELL", "HOLD"]


class HistoricalScoreRecord(SignalResponse):
    """A persisted pipeline result retrieved from the database."""

    id: int


class ErrorResponse(BaseModel):
    """Generic error body."""

    detail: str


# ── Agent schemas ──────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    """Free-text prompt sent to POST /ask."""

    prompt: str = Field(min_length=1, max_length=1000)


class QueryParams(BaseModel):
    """Structured parameters extracted from the user prompt by the agent."""

    ticker:      str
    horizon:     int = Field(ge=1, le=365)
    granularity: str = "daily"


class AskResponse(SignalResponse):
    """Pipeline result plus agent-generated explanation."""

    query_params: QueryParams
    explanation:  str
