#!/usr/bin/env bash
# MoScribe dev launcher → http://127.0.0.1:8770
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip -q install --upgrade pip
  .venv/bin/pip -q install -e ".[dev]"
fi

exec .venv/bin/moscribe --port "${MOSCRIBE_PORT:-8770}" --host "${MOSCRIBE_HOST:-127.0.0.1}" "$@"
