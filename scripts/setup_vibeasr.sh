#!/usr/bin/env bash
# Build the VibeASR.cpp inference server and fetch the quantized models.
set -euo pipefail
ROOT="${VIBEASR_ROOT:-$HOME/apps/VibeASR.cpp}"

if [ ! -d "$ROOT" ]; then
  git clone --recursive https://github.com/microsoft/VibeASR.cpp.git "$ROOT"
fi

cd "$ROOT"
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target asr_stream_server asr_infer -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

python3 - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("microsoft/VibeVoice-ASR-BitNet", local_dir="models/vibeasr",
                      allow_patterns=["*.gguf"])
print("models →", p)
PY

ls -lh models/vibeasr/*.gguf build/bin/asr_stream_server
