"""VibeScribe — a local web app around VibeASR.cpp (VibeVoice-ASR-BitNet).

Backend duties:
  * supervise one persistent `asr_stream_server` process (see asr_engine.py)
  * accept audio uploads, decode + chunk them, stream tokens over a WebSocket
  * accept live microphone windows over a WebSocket, stitch overlapping text
  * keep a small transcript history on disk
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import psutil
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .asr_engine import (
    ENGINE,
    SAMPLE_RATE,
    EngineConfig,
    EngineError,
    decode_to_float,
    plan_chunks,
    stitch_overlap,
    tidy,
    tmpdir,
    write_wav,
)
from .diarization import (
    DEFAULT_THRESHOLD,
    SpeakerCluster,
    describe_status,
    diarize_audio,
    group_turns,
    render_turns,
    split_text_by_speaker,
    vad_segments,
)
from .llm import LLM_ENGINE, build_messages
from .paths import DATA_DIR, app_dir, ensure_dirs
from .spk_embed import EMBEDDER

ROOT = app_dir()
STATIC = ROOT / "static"
DATA = DATA_DIR
ensure_dirs()
JOBS_DIR = DATA / "jobs"
HISTORY_FILE = DATA / "history.json"
MAX_UPLOAD_BYTES = 600 * 1024 * 1024
UPLOAD_DIR = DATA / "audio"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

DATA.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="VibeScribe", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

JOBS: dict[str, dict] = {}

# ---------------------------------------------------------------- vitals
# One sampler keeps CPU/RAM numbers honest: cpu_percent(interval=None) reports
# the average since the previous call, so polling it from a request handler
# would hand out meaningless zeroes.
VITALS: dict = {"cpu": 0.0, "ram": 0.0, "used": 0, "total": 0, "cores": 1,
                "load": 0.0, "rss": 0, "cpu_hist": [], "ram_hist": []}
VITALS_LOCK = asyncio.Lock()
HIST = 48


async def _vitals_sampler() -> None:
    proc = psutil.Process()
    psutil.cpu_percent(None)
    proc.cpu_percent(None)
    while True:
        try:
            cpu = psutil.cpu_percent(None)
            vm = psutil.virtual_memory()
            async with VITALS_LOCK:
                VITALS.update(
                    cpu=round(cpu, 1),
                    ram=round(vm.percent, 1),
                    used=int(vm.total - vm.available),
                    total=int(vm.total),
                    cores=psutil.cpu_count(logical=True) or 1,
                    load=round(os.getloadavg()[0], 2),
                    rss=int(proc.memory_info().rss),
                )
                VITALS["cpu_hist"] = (VITALS["cpu_hist"] + [VITALS["cpu"]])[-HIST:]
                VITALS["ram_hist"] = (VITALS["ram_hist"] + [VITALS["ram"]])[-HIST:]
        except Exception:  # noqa: BLE001 - vitals are cosmetic
            pass
        await asyncio.sleep(1.5)


# --------------------------------------------------------------------- helpers

def cfg_from_form(threads: int, decoding: str, temperature: float, top_p: float,
                  context: str, prompt_format: str) -> EngineConfig:
    return EngineConfig(
        threads=max(1, min(16, int(threads))),
        greedy=decoding != "sampling",
        temperature=max(0.0, min(2.0, temperature)),
        top_p=max(0.05, min(1.0, top_p)),
        context=(context or "").strip(),
        prompt_format="json" if prompt_format == "json" else "text",
    )


def load_history() -> list[dict]:
    try:
        return json.loads(HISTORY_FILE.read_text())
    except Exception:
        return []


def save_history(items: list[dict]) -> None:
    HISTORY_FILE.write_text(json.dumps(items[:100], ensure_ascii=False, indent=1))


def add_history(entry: dict) -> None:
    items = load_history()
    items.insert(0, entry)
    save_history(items)


def _safe_error(exc: Exception) -> JSONResponse:
    return JSONResponse({"error": str(exc)}, status_code=400)


def plan_chunks_by_speaker(spans: list[tuple[float, float, int]], total_dur: float,
                           chunk_sec: float, pad: float = 0.10) -> list[tuple[float, float, int]]:
    """Cut the audio at speaker turns instead of on a fixed grid.

    Windows that straddle two voices are the reason per-chunk speaker labels used
    to be a coin flip: half the words belong to someone else. Aligning each
    window to a diarised turn means the label is right by construction, and the
    text no longer has to be split after the fact.
    """
    if not spans:
        return []
    merged: list[list] = []
    for s, e, lab in spans:
        if merged and merged[-1][2] == lab and s - merged[-1][1] < 0.45:
            merged[-1][1] = e
        else:
            merged.append([float(s), float(e), lab])

    out: list[tuple[float, float, int]] = []
    min_len = 1.0
    for s, e, lab in merged:
        if e - s < min_len and out:
            out[-1] = (out[-1][0], max(out[-1][1], e), out[-1][2])
            continue
        cur = s
        while e - cur > chunk_sec:
            out.append((cur, cur + chunk_sec, lab))
            cur += chunk_sec
        if e - cur >= 0.3:
            out.append((cur, e, lab))

    padded: list[tuple[float, float, int]] = []
    for i, (a, b, lab) in enumerate(out):
        nxt = out[i + 1][0] if i + 1 < len(out) else total_dur
        prev_end = out[i - 1][1] if i > 0 else 0.0
        # Pad into the pause, but never past the middle of it — otherwise the
        # next speaker's first word lands in this window.
        a2 = max(0.0, a - min(pad, max(0.0, (a - prev_end) / 2)))
        b2 = min(total_dur, b + min(pad, max(0.0, (nxt - b) / 2)))
        if padded and a2 < padded[-1][1]:
            a2 = padded[-1][1]
        padded.append((a2, b2, lab))
    return [(a, b, lab) for a, b, lab in padded if b - a >= 0.3]


# ------------------------------------------------------------------- REST API

@app.on_event("startup")
async def start_vitals():
    asyncio.create_task(_vitals_sampler())


@app.get("/api/system")
async def system_stats():
    async with VITALS_LOCK:
        return dict(VITALS)


@app.on_event("startup")
async def warm_text_model():
    """Load the BitNet model in the background so the first AI click is fast."""
    async def _warm():
        await asyncio.sleep(2)
        try:
            await LLM_ENGINE.ensure()
        except Exception:  # noqa: BLE001 - optional feature
            pass
    asyncio.create_task(_warm())


@app.get("/api/health")
async def health():
    return {"ok": True, "engine": ENGINE.info(), "diarizer": describe_status(),
            "text_model": LLM_ENGINE.info(), "time": time.time()}


@app.get("/api/ai/health")
async def ai_health():
    return LLM_ENGINE.info()


@app.get("/api/ai/models")
async def ai_models():
    """GGUFs available on disk plus the active configuration."""
    return {"models": LLM_ENGINE.list_models(), "config": dict(LLM_ENGINE.config),
            "active": str(LLM_ENGINE.model), "threads": LLM_ENGINE.threads(),
            "max_threads": min(16, os.cpu_count() or 8)}


@app.post("/api/ai/config")
async def ai_config(payload: dict):
    """Persist generation settings and optionally reload the model now."""
    cfg = LLM_ENGINE.update_config(payload or {})
    result = {"config": cfg}
    if payload.get("reload"):
        result["reload"] = await LLM_ENGINE.restart()
    else:
        result["info"] = LLM_ENGINE.info()
    return result


@app.post("/api/ai/stream")
async def ai_stream(payload: dict):
    """Stream a summary/minutes/rewrite for a transcript, token by token."""
    action = str(payload.get("action", "summarize"))
    transcript = str(payload.get("transcript", "")).strip()
    extra = str(payload.get("extra", ""))
    speakers = payload.get("speakers") or {}

    if not transcript:
        return JSONResponse({"error": "empty transcript"}, status_code=400)

    messages = build_messages(action, transcript, extra, speakers)

    async def events():
        started = time.perf_counter()
        try:
            async for piece in LLM_ENGINE.stream(messages):
                yield f"data: {json.dumps({'delta': piece})}\n\n"
            yield f"data: {json.dumps({'done': True, 'elapsed': round(time.perf_counter() - started, 2)})}\n\n"
        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.post("/api/engine/warmup")
async def warmup(payload: dict | None = None):
    p = payload or {}
    cfg = cfg_from_form(int(p.get("threads", 4)), p.get("decoding", "greedy"),
                        float(p.get("temperature", 0.7)), float(p.get("top_p", 0.9)),
                        p.get("context", ""), p.get("prompt_format", "text"))
    try:
        await ENGINE.ensure(cfg)
    except EngineError as exc:
        return _safe_error(exc)
    return ENGINE.info()


@app.get("/api/history")
async def history():
    return {"items": load_history()}


@app.delete("/api/history/{item_id}")
async def delete_history(item_id: str):
    items = [i for i in load_history() if i.get("id") != item_id]
    save_history(items)
    shutil.rmtree(JOBS_DIR / item_id, ignore_errors=True)
    return {"deleted": item_id, "remaining": len(items)}


@app.post("/api/transcribe")
async def create_job(
    file: UploadFile = File(...),
    threads: int = Form(4),
    decoding: str = Form("greedy"),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    context: str = Form(""),
    prompt_format: str = Form("text"),
    chunk_sec: float = Form(12.0),
    overlap_sec: float = Form(2.0),
    speaker_separation: str = Form("off"),
    speaker_sensitivity: float = Form(0.78),
    max_speakers: int = Form(6),
):
    """Upload audio, split it into windows, return a job id for the WebSocket."""
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "audio").suffix or ".bin"
    src = job_dir / f"source{suffix}"
    size = 0
    with src.open("wb") as out:
        while True:
            block = await file.read(1 << 20)
            if not block:
                break
            size += len(block)
            if size > MAX_UPLOAD_BYTES:
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, "file too large (max 600 MB)")
            out.write(block)
    if size == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, "empty upload")

    try:
        audio = await asyncio.to_thread(decode_to_float, src)
    except EngineError as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        return _safe_error(exc)

    spans = None
    if speaker_separation == "on" and EMBEDDER.available:
        try:
            spans = await asyncio.to_thread(
                diarize_audio, audio, SAMPLE_RATE,
                max(0.55, min(0.95, float(speaker_sensitivity))),
                max(2, min(8, int(max_speakers))))
        except Exception:  # noqa: BLE001 - fall back to a fixed grid
            spans = None

    if spans:
        plan = plan_chunks_by_speaker(spans, len(audio) / SAMPLE_RATE,
                                      max(4.0, min(30.0, chunk_sec)))
    else:
        plan = [(a / SAMPLE_RATE, b / SAMPLE_RATE, None)
                for a, b in plan_chunks(len(audio), max(4.0, min(30.0, chunk_sec)),
                                        max(0.0, min(5.0, overlap_sec)), SAMPLE_RATE, audio)]

    chunks = []
    for idx, (a_s, b_s, spk) in enumerate(plan):
        a, b = int(a_s * SAMPLE_RATE), int(b_s * SAMPLE_RATE)
        path = job_dir / f"chunk_{idx:04d}.wav"
        await asyncio.to_thread(write_wav, path, audio[a:b])
        chunks.append({"index": idx, "start": round(a_s, 2), "end": round(b_s, 2),
                       "speaker": spk, "path": str(path)})

    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "filename": file.filename,
        "duration": len(audio) / SAMPLE_RATE,
        "chunks": chunks,
        "cfg": cfg_from_form(threads, decoding, temperature, top_p, context, prompt_format),
        "created": time.time(),
        "bytes": size,
        "speaker_separation": speaker_separation == "on",
        "spans": spans,
        "speaker_threshold": max(0.55, min(0.95, float(speaker_sensitivity))),
        "max_speakers": max(2, min(8, int(max_speakers))),
    }
    return {"job_id": job_id, "duration": len(audio) / SAMPLE_RATE,
            "chunks": len(chunks), "sr": SAMPLE_RATE,
            "chunk_spans": [{"index": c["index"], "start": round(c["start"], 2),
                             "end": round(c["end"], 2)} for c in chunks]}


@app.post("/api/transcribe-sync")
async def transcribe_sync(
    file: UploadFile = File(...),
    threads: int = Form(4),
    decoding: str = Form("greedy"),
    temperature: float = Form(0.7),
    top_p: float = Form(0.9),
    context: str = Form(""),
    prompt_format: str = Form("text"),
    chunk_sec: float = Form(12.0),
    overlap_sec: float = Form(2.0),
    speaker_separation: str = Form("off"),
    speaker_sensitivity: float = Form(0.78),
    max_speakers: int = Form(6),
):
    """One-shot transcription (no streaming) — handy for scripts and CI."""
    meta = await create_job(file, threads, decoding, temperature, top_p, context,
                            prompt_format, chunk_sec, overlap_sec, speaker_separation,
                            speaker_sensitivity, max_speakers)
    if isinstance(meta, JSONResponse):
        return meta
    job = JOBS[meta["job_id"]]
    text, segs, total_rtf = await _run_job(job, on_token=None, on_chunk=None)
    job["status"] = "done"
    _record(job, text, segs)
    return {"text": text, "segments": segs, "turns": job.get("turns", []),
            "speakers": job.get("speakers", 0),
            "duration": job["duration"],
            "rtf": round(total_rtf, 3), "job_id": job["id"]}


# ---------------------------------------------------------------- job runner

def _record(job: dict, text: str, segments: list[dict]) -> None:
    add_history({
        "id": job["id"],
        "title": job.get("filename") or "transcript",
        "mode": job.get("mode", "file"),
        "created": job["created"],
        "duration": round(job["duration"], 2),
        "rtf": round(sum(s.get("rtf", 0) for s in segments), 3),
        "words": len(text.split()),
        "text": text,
        "segments": segments,
    })


async def _run_job(job: dict, on_token=None, on_chunk=None,
                   should_cancel=None) -> tuple[str, list[dict], float]:
    committed = ""
    segments: list[dict] = []
    total_rtf = 0.0
    for chunk in job["chunks"]:
        if should_cancel and should_cancel():
            break
        idx = chunk["index"]

        async def tok(piece: str, _idx=idx):
            if on_token:
                await on_token(_idx, piece)

        result = await ENGINE.transcribe(chunk["path"], job["cfg"], on_token=tok)
        raw = tidy(result.get("text", ""))
        part = stitch_overlap(committed, raw) if committed else raw
        part = tidy(part)
        committed = (committed + (" " if committed and part else "") + part).strip()
        stats = result.get("stats") or {}
        total_rtf += float(stats.get("rtf", 0))
        seg = {"index": idx, "start": round(chunk["start"], 2), "end": round(chunk["end"], 2),
               "text": part, "raw": raw, "speaker": chunk.get("speaker"),
               **{k: stats.get(k) for k in ("rtf", "tokens")}}
        segments.append(seg)
        if on_chunk:
            await on_chunk(seg, committed)

    # Speaker labels: windows are already cut at speaker turns when the speaker
    # model was available, so most segments arrive labelled. Anything unlabelled
    # is split across the diarised spans of its own time range.
    if job.get("speaker_separation") and segments:
        try:
            spans = job.get("spans")
            if not spans:
                if on_chunk:
                    await on_chunk({"phase": "diarizing"}, committed)
                full = await asyncio.to_thread(_read_source_audio, job)
                spans = await asyncio.to_thread(
                    diarize_audio, full, SAMPLE_RATE,
                    job.get("speaker_threshold", DEFAULT_THRESHOLD),
                    job.get("max_speakers", 6))
            detailed: list[dict] = []
            for seg in segments:
                speaker = seg.get("speaker")
                if speaker is not None and speaker >= 0:
                    detailed.append(seg)
                    continue
                pieces = split_text_by_speaker(seg["text"], seg["start"], seg["end"], spans)
                if not pieces:
                    detailed.append(seg)
                    continue
                for j, piece in enumerate(pieces):
                    detailed.append({**seg, "piece": j, "speaker": piece["speaker"],
                                     "text": piece["text"],
                                     "start": piece["start"], "end": piece["end"]})
            segments = detailed
            turns = group_turns(segments)
            job["turns"] = turns
            job["speakers"] = len({t["speaker"] for t in turns if t.get("speaker", -1) >= 0})
            committed = render_turns(turns, labels=True)
        except Exception as exc:  # noqa: BLE001 - diarisation is best-effort
            job["diarization_error"] = f"{type(exc).__name__}: {exc}"

    return committed, segments, total_rtf


def _read_source_audio(job: dict) -> np.ndarray:
    """Decode the uploaded file back to float32 mono at SAMPLE_RATE."""
    import wave
    src_path = next(Path(job["chunks"][0]["path"]).parent.glob("source*"))
    with wave.open(str(src_path)) as w:
        raw_frames = w.readframes(w.getnframes())
        width = w.getsampwidth()
    if width == 2:
        return np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
    if width == 4:
        return np.frombuffer(raw_frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    raise ValueError(f"unsupported sample width: {width}")


@app.websocket("/ws/jobs/{job_id}")
async def job_socket(ws: WebSocket, job_id: str):
    await ws.accept()
    job = JOBS.get(job_id)
    if not job:
        await ws.send_json({"type": "error", "message": "unknown job"})
        await ws.close()
        return

    cancel = {"flag": False}

    async def reader():
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("type") == "cancel":
                    cancel["flag"] = True
                    break
        except Exception:
            pass

    reader_task = asyncio.create_task(reader())
    job["status"] = "running"
    started = time.perf_counter()
    await ws.send_json({"type": "start", "job_id": job_id, "duration": job["duration"],
                        "chunks": len(job["chunks"]), "filename": job.get("filename")})

    async def on_token(idx: int, piece: str):
        await ws.send_json({"type": "token", "chunk": idx, "text": piece})

    async def on_chunk(seg: dict, committed: str):
        if "phase" in seg:
            await ws.send_json({"type": "phase", "phase": seg["phase"]})
            return
        await ws.send_json({"type": "chunk_done", "segment": seg, "text": committed})

    try:
        text, segments, _ = await _run_job(job, on_token, on_chunk,
                                           lambda: cancel["flag"])
    except EngineError as exc:
        job["status"] = "error"
        await ws.send_json({"type": "error", "message": str(exc)})
        await ws.close()
        return
    except Exception as exc:  # noqa: BLE001 - surface anything to the UI
        job["status"] = "error"
        await ws.send_json({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        await ws.close()
        return
    finally:
        reader_task.cancel()

    elapsed = time.perf_counter() - started
    job["status"] = "done" if not cancel["flag"] else "cancelled"
    if text:
        _record(job, text, segments)
    await ws.send_json({
        "type": "done", "text": text, "segments": segments,
        "turns": job.get("turns", []), "speakers": job.get("speakers", 0),
        "duration": job["duration"], "elapsed": round(elapsed, 2),
        "rtf": round(elapsed / job["duration"], 3) if job["duration"] else None,
        "cancelled": cancel["flag"],
    })
    await ws.close()


# --------------------------------------------------------------- live mic feed

@app.websocket("/ws/live")
async def live_socket(ws: WebSocket):
    """Browser sends overlapping 24 kHz mono WAV windows; we stitch them."""
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=8)
    state = {"cfg": EngineConfig(threads=4, greedy=True), "committed": "",
             "seq": 0, "words": 0, "started": time.time(), "mode": "live",
             "diarize": False, "cluster": None, "elapsed": 0.0,
             "segments": [], "spans": []}

    await ws.send_json({"type": "ready", "sr": SAMPLE_RATE,
                        "engine": ENGINE.info()["state"],
                        "diarizer": EMBEDDER.available})

    async def worker():
        while True:
            item = await queue.get()
            if item is None:
                break
            seq, raw_bytes = item
            path = tmpdir() / f"live_{uuid.uuid4().hex[:8]}.wav"
            try:
                path.write_bytes(raw_bytes)
                pieces: list[dict] = []
                win_dur = 0.0
                if state["diarize"] and state["cluster"] is not None:
                    try:
                        import wave
                        with wave.open(str(path)) as w:
                            raw = w.readframes(w.getnframes())
                            width = w.getsampwidth()
                            win_dur = w.getnframes() / w.getframerate()
                        if width == 2:
                            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                        elif width == 4:
                            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                        else:
                            samples = None
                        if samples is not None and samples.size > SAMPLE_RATE // 2:
                            spans = await asyncio.to_thread(vad_segments, samples, SAMPLE_RATE)
                            if spans:
                                clips = [samples[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)]
                                         for a, b in spans]
                                embs = await asyncio.to_thread(EMBEDDER.embed_many, clips, SAMPLE_RATE)
                                local = []
                                for (a, b), emb in zip(spans, embs):
                                    lab = state["cluster"].add(emb, b - a)
                                    local.append((a, b, lab))
                                state["spans"] = local
                    except Exception:  # noqa: BLE001 - diarisation is best-effort
                        state["spans"] = []

                async def tok(piece: str):
                    await ws.send_json({"type": "token", "seq": seq, "text": piece})
                result = await ENGINE.transcribe(str(path), state["cfg"], on_token=tok)
                raw = tidy(result.get("text", ""))
                part = stitch_overlap(state["committed"], raw) if state["committed"] else raw
                part = tidy(part)
                if part:
                    state["committed"] = (state["committed"] + " " + part).strip()
                    state["words"] += len(part.split())
                    if state["diarize"] and state.get("spans"):
                        pieces = split_text_by_speaker(part, 0.0, win_dur or 1.0, state["spans"])
                    if not pieces:
                        pieces = [{"speaker": -1, "text": part,
                                   "start": round(state["elapsed"], 2), "end": None}]
                    for pc in pieces:
                        state["segments"].append({**pc, "window": seq})

                turns = group_turns(state["segments"]) if state["diarize"] else []
                text = render_turns(turns, labels=True) if turns else state["committed"]
                await ws.send_json({
                    "type": "window_done", "seq": seq, "raw": raw, "appended": part,
                    "text": text, "pieces": pieces, "turns": turns,
                    "speakers": state["cluster"].speaker_count if state["cluster"] else 0,
                    "speaker": pieces[0]["speaker"] if pieces else None,
                    "rtf": (result.get("stats") or {}).get("rtf"),
                })
            except EngineError as exc:
                await ws.send_json({"type": "error", "seq": seq, "message": str(exc)})
            except Exception as exc:  # noqa: BLE001
                await ws.send_json({"type": "error", "seq": seq,
                                    "message": f"{type(exc).__name__}: {exc}"})
            finally:
                path.unlink(missing_ok=True)
                queue.task_done()

    worker_task = asyncio.create_task(worker())
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                if state.get("finalizing"):
                    continue
                state["seq"] += 1
                try:
                    queue.put_nowait((state["seq"], msg["bytes"]))
                except asyncio.QueueFull:
                    await ws.send_json({"type": "error",
                                        "message": "capture is faster than inference — window dropped"})
                continue
            if msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except Exception:
                    continue
                t = data.get("type")
                if t == "config":
                    state["cfg"] = cfg_from_form(
                        int(data.get("threads", 4)), data.get("decoding", "greedy"),
                        float(data.get("temperature", 0.7)), float(data.get("top_p", 0.9)),
                        data.get("context", ""), data.get("prompt_format", "text"))
                    state["diarize"] = data.get("speaker_separation") == "on"
                    threshold = max(0.55, min(0.95, float(data.get("speaker_sensitivity", DEFAULT_THRESHOLD))))
                    max_spk = max(2, min(8, int(data.get("max_speakers", 6))))
                    if state["diarize"]:
                        if state["cluster"] is None:
                            state["cluster"] = SpeakerCluster(threshold=threshold,
                                                              max_speakers=max_spk)
                        else:
                            state["cluster"].threshold = threshold
                            state["cluster"].max_speakers = max_spk
                    else:
                        state["cluster"] = None
                        state["segments"] = []
                        state["spans"] = []
                elif t == "reset":
                    state["committed"] = ""
                    state["words"] = 0
                    state["segments"] = []
                    state["spans"] = []
                    state["elapsed"] = 0.0
                    if state["cluster"] is not None:
                        state["cluster"] = SpeakerCluster(
                            threshold=state["cluster"].threshold,
                            max_speakers=state["cluster"].max_speakers)
                    await ws.send_json({"type": "reset_ok"})
                elif t == "tick":
                    state["elapsed"] = float(data.get("elapsed", state["elapsed"]))
                elif t == "finalize":
                    # the client may still have windows in flight: let the worker
                    # finish them before the socket goes away, otherwise the last
                    # sentence is silently dropped
                    state["finalizing"] = True
                    try:
                        await asyncio.wait_for(queue.join(), timeout=120)
                    except asyncio.TimeoutError:
                        pass
                    turns = group_turns(state["segments"]) if state["diarize"] else []
                    text = (render_turns(turns, labels=True) if turns else state["committed"]).strip()
                    await ws.send_json({
                        "type": "final", "text": text, "turns": turns,
                        "speakers": state["cluster"].speaker_count if state["cluster"] else 0,
                        "windows": state["seq"], "words": len(text.split()),
                        "duration": round(time.time() - state["started"], 1),
                    })
                    break
    except WebSocketDisconnect:
        pass
    finally:
        worker_task.cancel()
        turns = group_turns(state["segments"]) if state["diarize"] else []
        text = (render_turns(turns, labels=True) if turns else state["committed"]).strip()
        if text:
            add_history({
                "id": uuid.uuid4().hex[:12], "title": "Live dictation",
                "mode": "live", "created": state["started"],
                "duration": round(time.time() - state["started"], 1),
                "rtf": None, "words": len(text.split()), "text": text,
                "segments": state["segments"], "turns": turns,
                "speakers": len({t["speaker"] for t in turns if t.get("speaker", -1) >= 0}),
            })


# ------------------------------------------------------------------- static UI

@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.middleware("http")
async def no_cache_static(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store"
    return response


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
