"""
api.limiter

Singleton rate-limiter instance shared across all route modules.

Import this module to apply @limiter.limit() decorators.
The app registers it via app.state.limiter in api.main.

Limits (per remote IP):
  GET /signal          10 / minute  — full ML pipeline, CPU-heavy
  POST /ask             5 / minute  — Anthropic + pipeline, most expensive
  GET /scores/{ticker} 60 / minute  — lightweight DB query
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Exposed as module-level constants so routes and tests can reference them
# without hard-coding the strings in multiple places.
LIMIT_SIGNAL = "10/minute"
LIMIT_ASK    = "5/minute"
LIMIT_SCORES = "60/minute"
