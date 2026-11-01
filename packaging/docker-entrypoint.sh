#!/bin/sh
# Container entrypoint: fetch models into the data volume once, then serve.
set -e

DATA="${MOSCRIBE_DATA:-/data}"
mkdir -p "$DATA"

if [ "${MOSCRIBE_SKIP_FETCH:-0}" != "1" ]; then
  if [ ! -f "$DATA/models/vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf" ] || \
     [ ! -f "$DATA/models/llm/ggml-model-i2_s.gguf" ]; then
    echo "→ first start: downloading models into $DATA (about 2.8 GB, once)"
    python -m moscribe.fetch || echo "! model download failed — fix network and restart"
  fi
fi

exec moscribe --host "${MOSCRIBE_HOST:-0.0.0.0}" --port "${MOSCRIBE_PORT:-8770}" --no-browser "$@"
