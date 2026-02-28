"""
data.storage.db

Database session management and ORM model definitions.

Reads connection string from DATABASE_URL environment variable,
with host/port/name/user fallback from config.yaml.

ORM models:
  - PriceRecord   : daily OHLCV row keyed by (ticker, date)
  - NewsRecord    : article keyed by (ticker, published_at, url)
  - ScoreRecord   : pipeline output keyed by (ticker, run_date, horizon)
                    fields: ts_score, llms_score, ta_score, compound_score

Public interface:
  get_engine(config: dict) -> Engine | None
  get_session(engine: Engine) -> ContextManager[Session]
  init_db(engine: Engine) -> None
  save_score(result: dict, session: Session) -> ScoreRecord
  get_scores(ticker: str, limit: int, session: Session) -> list[ScoreRecord]
  save_prices(df, ticker: str, session: Session) -> int
  save_news(articles: list[dict], ticker: str, session: Session) -> int
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

logger = logging.getLogger(__name__)


# ── ORM base ──────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── ORM models ────────────────────────────────────────────────────────────────

class ScoreRecord(Base):
    """One pipeline result per (ticker, run_date, horizon)."""

    __tablename__ = "scores"
    __table_args__ = (
        UniqueConstraint("ticker", "run_date", "horizon", name="uq_scores_ticker_date_horizon"),
    )

    id:             Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker:         Mapped[str]   = mapped_column(String(10), nullable=False, index=True)
    run_date:       Mapped[str]   = mapped_column(String(10), nullable=False)   # ISO-8601 date
    horizon:        Mapped[int]   = mapped_column(Integer, nullable=False)
    ts_score:       Mapped[float] = mapped_column(Float, nullable=False)
    llms_score:     Mapped[float] = mapped_column(Float, nullable=False)
    ta_score:       Mapped[float] = mapped_column(Float, nullable=False)
    compound_score: Mapped[float] = mapped_column(Float, nullable=False)
    signal:         Mapped[str]   = mapped_column(String(4), nullable=False)
    created_at:     Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class PriceRecord(Base):
    """Daily OHLCV row for a ticker."""

    __tablename__ = "prices"
    __table_args__ = (
        UniqueConstraint("ticker", "date", name="uq_prices_ticker_date"),
    )

    id:     Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str]   = mapped_column(String(10), nullable=False, index=True)
    date:   Mapped[str]   = mapped_column(String(10), nullable=False)   # ISO-8601 date
    open:   Mapped[float] = mapped_column(Float, nullable=True)
    high:   Mapped[float] = mapped_column(Float, nullable=True)
    low:    Mapped[float] = mapped_column(Float, nullable=True)
    close:  Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=True)


class NewsRecord(Base):
    """One article per (ticker, published_at, url)."""

    __tablename__ = "news"
    __table_args__ = (
        UniqueConstraint("ticker", "published_at", "url", name="uq_news_ticker_time_url"),
    )

    id:           Mapped[int]   = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker:       Mapped[str]   = mapped_column(String(10), nullable=False, index=True)
    published_at: Mapped[str]   = mapped_column(String(35), nullable=False)  # ISO-8601 datetime
    url:          Mapped[str]   = mapped_column(String(2048), nullable=False)
    title:        Mapped[str]   = mapped_column(String(512), nullable=True)
    body:         Mapped[str]   = mapped_column(String, nullable=True)
    source:       Mapped[str]   = mapped_column(String(64), nullable=True)


# ── Public helpers ────────────────────────────────────────────────────────────

def get_engine(config: dict):
    """Return a SQLAlchemy Engine, or None if no credentials are available.

    Resolution order:
      1. DATABASE_URL environment variable (full connection string).
      2. Construct from config["database"] + DB_PASSWORD environment variable.

    Args:
        config: Loaded config dict (from pipeline.runner.load_config).

    Returns:
        sqlalchemy.Engine if credentials are available, otherwise None.
    """
    database_url = os.environ.get("DATABASE_URL", "")
    if database_url:
        return create_engine(database_url, pool_pre_ping=True)

    db_password = os.environ.get("DB_PASSWORD", "")
    if not db_password:
        logger.warning(
            "No database credentials found. "
            "Set DATABASE_URL or DB_PASSWORD to enable persistence."
        )
        return None

    db_cfg = config.get("database", {})
    host = db_cfg.get("host", "localhost")
    port = db_cfg.get("port", 5432)
    name = db_cfg.get("name", "stocker")
    user = db_cfg.get("user", "stocker_user")
    url = f"postgresql+psycopg2://{user}:{db_password}@{host}:{port}/{name}"
    return create_engine(url, pool_pre_ping=True)


@contextmanager
def get_session(engine) -> Generator[Session, None, None]:
    """Yield a SQLAlchemy Session; commit on exit, roll back on error.

    Args:
        engine: A SQLAlchemy Engine returned by get_engine().

    Yields:
        sqlalchemy.orm.Session
    """
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine) -> None:
    """Create all tables if they do not already exist.

    Safe to call repeatedly (uses CREATE TABLE IF NOT EXISTS semantics).

    Args:
        engine: A SQLAlchemy Engine returned by get_engine().
    """
    Base.metadata.create_all(engine)


def save_score(result: dict, session: Session) -> ScoreRecord:
    """Upsert a pipeline result into the scores table.

    If a row already exists for (ticker, run_date, horizon), its scores are
    updated in place. Otherwise a new row is inserted.

    Args:
        result:  Dict returned by pipeline.runner.run_pipeline().
        session: An open SQLAlchemy Session.

    Returns:
        The inserted or updated ScoreRecord ORM instance.
    """
    ticker   = result["ticker"]
    run_date = result["run_date"]
    horizon  = result["horizon"]

    stmt = select(ScoreRecord).where(
        ScoreRecord.ticker   == ticker,
        ScoreRecord.run_date == run_date,
        ScoreRecord.horizon  == horizon,
    )
    record = session.scalars(stmt).first()

    if record is None:
        record = ScoreRecord(
            ticker=ticker,
            run_date=run_date,
            horizon=horizon,
        )
        session.add(record)

    record.ts_score       = result["ts_score"]
    record.llms_score     = result["llms_score"]
    record.ta_score       = result["ta_score"]
    record.compound_score = result["compound_score"]
    record.signal         = result["signal"]

    session.flush()
    return record


def get_scores(ticker: str, limit: int, session: Session) -> list[ScoreRecord]:
    """Return historical score records for *ticker*, newest first.

    Args:
        ticker:  Ticker symbol (e.g. "AAPL").
        limit:   Maximum number of records to return.
        session: An open SQLAlchemy Session.

    Returns:
        List of ScoreRecord instances ordered by run_date DESC, id DESC.
    """
    stmt = (
        select(ScoreRecord)
        .where(ScoreRecord.ticker == ticker)
        .order_by(ScoreRecord.run_date.desc(), ScoreRecord.id.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt).all())


def save_prices(df, ticker: str, session: Session) -> int:
    """Upsert daily OHLCV rows from *df* into the prices table.

    Fetches all existing rows for *ticker* in one query, then updates or
    inserts each row from the DataFrame.  NaN values for optional columns
    (open, high, low, volume) are stored as NULL.

    Args:
        df:      pandas DataFrame with DatetimeIndex and columns including
                 ``close`` (required) plus any of ``open, high, low, volume``.
        ticker:  Stock symbol (e.g. "AAPL").
        session: An open SQLAlchemy Session.

    Returns:
        Number of rows processed.
    """
    import math  # noqa: PLC0415

    # Load all existing dates for this ticker into a dict for O(1) lookup.
    existing: dict[str, PriceRecord] = {
        r.date: r
        for r in session.scalars(
            select(PriceRecord).where(PriceRecord.ticker == ticker)
        ).all()
    }

    def _float(val) -> float | None:
        try:
            f = float(val)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    count = 0
    for ts, row in df.iterrows():
        date_str = ts.date().isoformat()
        record   = existing.get(date_str)
        if record is None:
            record = PriceRecord(ticker=ticker, date=date_str)
            session.add(record)
            existing[date_str] = record

        record.close  = _float(row.get("close"))
        record.open   = _float(row.get("open"))
        record.high   = _float(row.get("high"))
        record.low    = _float(row.get("low"))
        record.volume = _float(row.get("volume"))
        count += 1

    session.flush()
    return count


def save_news(articles: list[dict], ticker: str, session: Session) -> int:
    """Upsert news articles into the news table.

    Deduplicates by (ticker, published_at, url).  Articles without a URL
    are keyed only on (ticker, published_at) and may not deduplicate
    correctly if the same article appears multiple times.

    Args:
        articles: List of article dicts with keys ``title``, ``body``,
                  ``published_at``, ``source``, ``url``.
        ticker:   Stock symbol (e.g. "AAPL").
        session:  An open SQLAlchemy Session.

    Returns:
        Number of articles processed.
    """
    # Load existing records for this ticker into a dict for O(1) lookup.
    existing: dict[tuple[str, str], NewsRecord] = {
        (r.published_at, r.url): r
        for r in session.scalars(
            select(NewsRecord).where(NewsRecord.ticker == ticker)
        ).all()
    }

    count = 0
    for article in articles:
        pub = (article.get("published_at") or "").strip()
        url = (article.get("url") or "").strip()
        key = (pub, url)

        record = existing.get(key)
        if record is None:
            record = NewsRecord(ticker=ticker, published_at=pub, url=url)
            session.add(record)
            existing[key] = record

        record.title  = (article.get("title")  or "")[:512]
        record.body   = article.get("body")   or ""
        record.source = (article.get("source") or "")[:64]
        count += 1

    session.flush()
    return count
