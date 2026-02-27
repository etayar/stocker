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
  get_engine(config: dict) -> sqlalchemy.Engine
  get_session(engine) -> sqlalchemy.orm.Session
  init_db(engine)          # Creates all tables if not present
"""
