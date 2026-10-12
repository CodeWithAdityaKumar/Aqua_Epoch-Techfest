#!/usr/bin/env bash
set -euo pipefail

# Optional shell convenience wrapper. The cross-platform entry point is
# python -m backend.demo; it reads AQUA_EPOCH_HOST, AQUA_EPOCH_PORT,
# AQUA_EPOCH_ARTIFACTS_DIR and AQUA_EPOCH_HISTORY_DB itself.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  : "${PYTHON_BIN}"
else
  PYTHON_BIN=""
  for candidate in "$ROOT/.venv/bin/python" python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import fastapi, uvicorn, joblib, sklearn' >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
  if [[ -z "$PYTHON_BIN" ]]; then
    printf 'No Python runtime with FastAPI, Uvicorn, joblib and scikit-learn was found.\nActivate .venv or set PYTHON_BIN explicitly.\n' >&2
    exit 2
  fi
fi
exec "$PYTHON_BIN" -m backend.demo "$@"
