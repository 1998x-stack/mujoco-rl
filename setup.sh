#!/usr/bin/env bash
# Project-local Python/bootstrap for macOS and Linux. No sudo and no shell modifications.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
case "$(uname -s)" in Darwin|Linux) ;; *) echo 'For Windows use setup.ps1 or run.cmd.' >&2; exit 2 ;; esac
mkdir -p "$ROOT/.tools/bin"
if [[ -x "$ROOT/.tools/bin/uv" ]]; then
  UV="$ROOT/.tools/bin/uv"
elif command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  command -v curl >/dev/null 2>&1 || { echo 'curl is needed to install uv. Install curl, or install uv separately.' >&2; exit 2; }
  echo '[setup] Downloading official uv installer (first run requires internet).'
  installer="$(mktemp "${TMPDIR:-/tmp}/uv-install.XXXXXXXX")"
  trap 'rm -f "$installer"' EXIT
  curl --fail --location --silent --show-error --retry 3 --connect-timeout 30 \
    https://astral.sh/uv/install.sh --output "$installer"
  env UV_UNMANAGED_INSTALL="$ROOT/.tools/bin" sh "$installer"
  UV="$ROOT/.tools/bin/uv"
  [[ -x "$UV" ]] || { echo 'uv installation failed.' >&2; exit 3; }
fi
"$UV" --version
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$UV" venv --python 3.11 --managed-python "$ROOT/.venv"
fi
if ! "$ROOT/.venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Expected Python 3.11"'; then
  echo 'Existing .venv is not Python 3.11. Rename it and rerun.' >&2
  exit 4
fi
"$UV" pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements.txt"
echo "[setup] Ready: $ROOT/.venv/bin/python"
