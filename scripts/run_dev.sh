#!/usr/bin/env bash
set -e

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
PY="./.venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "Error: venv python not found at $PY"
  exit 1
fi

PIDS="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true)"

if [ -n "$PIDS" ]; then
  echo "Port $PORT is already in use by PID(s):"
  echo "$PIDS"
  echo ""
  ps -p $PIDS -o pid,comm,args || true

  if [ "${1:-}" = "--kill" ]; then
    echo ""
    echo "Killing PID(s)..."
    kill -9 $PIDS
    echo "Killed. Starting server..."
  else
    echo ""
    echo "To stop them, run:"
    for p in $PIDS; do
      echo "  kill -9 $p"
    done
    echo ""
    echo "Or restart using:"
    echo "  $0 --kill"
    echo ""
    echo "Or run on another port:"
    echo "  PORT=8001 $0"
    exit 1
  fi
fi

exec "$PY" -m uvicorn app.main:app --reload --host "$HOST" --port "$PORT"
