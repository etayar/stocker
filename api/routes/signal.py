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
