#!/usr/bin/env bash
# Recreate the Python venv for the Art Collection index on THIS machine.
# The venv is machine-specific and gitignored, so it must be rebuilt after a move.
# Usage:  bash _index/setup.sh          (run from anywhere)
#         PYTHON=python3.13 bash ...    (override interpreter)
set -euo pipefail

INDEX="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3.12}"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: '$PY' not found. Install it first, e.g.:  brew install python@3.12" >&2
  echo "(or set PYTHON=... to point at another 3.12+ interpreter)" >&2
  exit 1
fi

if [ -d "$INDEX/.venv" ]; then
  echo "A .venv already exists at $INDEX/.venv"
  read -r -p "Delete and rebuild it? [y/N] " ans
  [ "$ans" = "y" ] || { echo "Aborted."; exit 0; }
  rm -rf "$INDEX/.venv"
fi

echo "Creating venv with $("$PY" --version) ..."
"$PY" -m venv "$INDEX/.venv"
"$INDEX/.venv/bin/pip" install --quiet --upgrade pip
echo "Installing dependencies (torch is large — a few minutes) ..."
"$INDEX/.venv/bin/pip" install -r "$INDEX/requirements.txt"

echo
echo "Done. Quick test:"
echo "  \"$INDEX/.venv/bin/python\" \"$INDEX/search.py\" \"misty forest\" -n 3"
echo "Bring up the gallery + aliases with:  bash \"$INDEX/install-service.sh\""
