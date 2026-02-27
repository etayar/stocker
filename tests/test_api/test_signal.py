"""
tests.test_api.test_signal

Tests for api.routes.signal using FastAPI TestClient (httpx).

Test cases to implement:
  - test_get_signal_returns_200
      GET /signal?ticker=AAPL returns HTTP 200.
  - test_get_signal_response_schema
      Response JSON matches SignalResponse schema.
  - test_get_signal_custom_horizon
      GET /signal?ticker=AAPL&horizon=10 returns horizon=10 in response.
  - test_get_signal_missing_ticker_returns_422
      GET /signal (no ticker) returns HTTP 422 Unprocessable Entity.
  - test_get_signal_invalid_ticker_returns_error
      GET /signal?ticker=XXXXXXXXX returns an error response.
  - test_get_scores_returns_list
      GET /scores/AAPL returns a list of historical score records.
  - test_health_check
      GET /health returns HTTP 200 and {"status": "ok"}.
"""
