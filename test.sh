#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -x "$ROOT/.venv/bin/python" ]] || { echo 'Run ./run.sh first.' >&2; exit 2; }
"$ROOT/.venv/bin/python" -m unittest discover -s tests -v
"$ROOT/.venv/bin/python" -m src.check_env --steps 20
