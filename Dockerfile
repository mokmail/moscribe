# MoScribe in a container — the fastest way to run it on a server or NAS.
#
#   docker build -t moscribe .
#   docker run --rm -p 8770:8770 -v moscribe-data:/data moscribe
#
# Models download into the volume on first start (~2.8 GB, once).

# ---------- native engines ----------
FROM ubuntu:22.04 AS engines
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/microsoft/VibeASR.cpp /src
WORKDIR /src
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF \
    && cmake --build build --target asr_stream_server -j"$(nproc)" \
    && cmake -S . -B build-llm -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON -DGGML_OPENMP=OFF \
    && cmake --build build-llm --target llama-server -j"$(nproc)" \
    && mkdir -p /out/bin \
    && cp build/bin/asr_stream_server build-llm/bin/llama-server /out/bin/ \
    && (find /src -name 'libllama.so*' -o -name 'libggml.so*' | xargs -r -I{} cp -n {} /out/bin/ || true) \
    && chmod +x /out/bin/*

# ---------- app ----------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    MOSCRIBE_DATA=/data \
    MOSCRIBE_HOST=0.0.0.0 \
    OMP_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE THIRD_PARTY.md ./
COPY moscribe ./moscribe
COPY moscribe_cli.py ./
COPY scripts ./scripts

RUN pip install --no-cache-dir . \
    && mkdir -p /app/bin /data

COPY --from=engines /out/bin /app/bin

COPY packaging/docker-entrypoint.sh /usr/local/bin/moscribe-entrypoint
RUN chmod +x /usr/local/bin/moscribe-entrypoint

EXPOSE 8770
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8770/api/health',timeout=4)"

ENTRYPOINT ["moscribe-entrypoint"]
