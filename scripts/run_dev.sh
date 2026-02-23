#!/usr/bin/env bash
set -e

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# Use venv python explicitly (avoids system python mismatch)
PY="./.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "Error: venv python not found at $PY"
  echo "Did you create the venv? Try: python3 -m venv .venv"
  exit 1
fi

# Check if port is already in use
PID="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true)"

if [ -n "$PID" ]; then
  echo "Port $PORT is already in use by PID $PID"
  echo "Process:"
  ps -p "$PID" -o pid,comm,args

  if [ "${1:-}" = "--kill" ]; then
    echo "Killing PID $PID..."
    kill -9 "$PID"
    echo "Killed. Starting server..."
  else
    echo ""
    echo "To stop it, run:"
    echo "  kill -9 $PID"
    echo "Or restart using:"
    echo "  $0 --kill"
    echo "Or run on another port, e.g.:"
    echo "  PORT=8001 $0"
    exit 1
  fi
fi

exec "$PY" -m uvicorn app.main:app --reload --host "$HOST" --port "$PORT"
