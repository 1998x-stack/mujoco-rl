#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -x "$ROOT/.venv/bin/python" ]] || { echo 'Run ./run.sh first.' >&2; exit 2; }
if [[ "$(uname -s)" == Darwin && -x "$ROOT/.venv/bin/mjpython" ]]; then
  exec "$ROOT/.venv/bin/mjpython" -m src.play "$@"
fi
exec "$ROOT/.venv/bin/python" -m src.play "$@"
