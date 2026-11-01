"""VibeScribe engine: wraps the VibeASR.cpp `asr_stream_server` binary.

The C++ server keeps the VAE tokenizer + BitNet LM resident and speaks a tiny
line protocol on stdin/stdout:

    stdin  : one audio file path per line, "EXIT" to quit,
             "CONTEXT:<text>" / "FORMAT:<text>" to update prompt options
    stdout : "---READY---" once loaded, then one *token piece* per line,
             then "---END---" after each chunk, "---ACK---" for control cmds
    stderr : human logs + a timing line per chunk:
             "  Audio: 12.00s | Tokens: 42 | RTF: 0.3120 | VAE: ... | Total: ..."

This module keeps that process alive, serialises access with an asyncio lock,
and turns the byte-level token stream into clean incremental UTF-8 text.
"""

from __future__ import annotations

import asyncio
import codecs
import os
import re
import tempfile
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000
_RTF_RE = re.compile(r"Audio:\s*([\d.]+)s.*?Tokens:\s*(\d+).*?RTF:\s*([\d.]+).*?Total:\s*([\d.]+)ms")


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else default


from .paths import APP_DIR, DATA_DIR, bin_dir, exe, find_exe, models_dir  # noqa: E402

_LEGACY_ROOT = Path.home() / "apps" / "VibeASR.cpp"

# Resolution order: explicit env → bundled app → user data → the dev checkout.
VIBEASR_ROOT = _env_path("VIBEASR_ROOT", APP_DIR)
_BIN_CANDIDATES = [
    Path(os.environ["VIBEASR_BIN"]).expanduser() if os.environ.get("VIBEASR_BIN") else None,
    bin_dir() / exe("asr_stream_server"),
    DATA_DIR / "bin" / exe("asr_stream_server"),
    _LEGACY_ROOT / "build" / "bin" / "asr_stream_server",
]
BIN = next((c for c in _BIN_CANDIDATES if c and c.exists()), _BIN_CANDIDATES[1])


def resolve_bin() -> Path:
    """Re-check on demand: a first-run download lands after import."""
    if BIN.exists():
        return BIN
    found = find_exe("asr_stream_server")
    if found:
        return found
    legacy = _LEGACY_ROOT / "build" / "bin" / "asr_stream_server"
    return legacy if legacy.exists() else BIN

_MODEL_CANDIDATES = [
    Path(os.environ["VIBEASR_MODELS"]).expanduser() if os.environ.get("VIBEASR_MODELS") else None,
    models_dir() / "vibeasr",
    DATA_DIR / "models" / "vibeasr",
    _LEGACY_ROOT / "models" / "vibeasr",
]
MODELS_DIR = next((c for c in _MODEL_CANDIDATES if c and c.exists()), _MODEL_CANDIDATES[1])
VAE_MODEL = MODELS_DIR / "vibeasr-vae-encoder-i8_s.gguf"
LM_MODEL = MODELS_DIR / "vibeasr-lm-i2_s-embed-q6_k.gguf"


@dataclass(frozen=True)
class EngineConfig:
    threads: int = 4
    greedy: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    context: str = ""
    prompt_format: str = "text"  # "text" | "json"
    max_tokens: int = 4096

    def key(self) -> tuple:
        return (self.threads, self.greedy, round(self.temperature, 3),
                round(self.top_p, 3), self.context, self.prompt_format, self.max_tokens)


class EngineError(RuntimeError):
    pass


class AsrEngine:
    """One persistent inference process, shared by every request."""

    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._cfg: EngineConfig | None = None
        self._lock = asyncio.Lock()          # serialises chunk inference
        self._proc_lock = asyncio.Lock()     # serialises start/stop
        self._stderr_task: asyncio.Task | None = None
        self._stderr_tail: deque[str] = deque(maxlen=400)
        self.state = "stopped"               # stopped | loading | ready | error
        self.load_ms: float | None = None
        self.last_error: str | None = None
        self.chunks_done = 0
        self.model_id: str | None = None

    # ---------------------------------------------------------------- info
    def binary_ok(self) -> bool:
        return BIN.exists() and os.access(BIN, os.X_OK)

    def models_ok(self) -> bool:
        return VAE_MODEL.exists() and LM_MODEL.exists()

    def info(self) -> dict:
        return {
            "state": self.state,
            "binary": str(BIN),
            "binary_ok": self.binary_ok(),
            "models_ok": self.models_ok(),
            "vae_model": str(VAE_MODEL),
            "lm_model": str(LM_MODEL),
            "load_ms": self.load_ms,
            "last_error": self.last_error,
            "chunks_done": self.chunks_done,
            "config": None if self._cfg is None else {
                "threads": self._cfg.threads,
                "greedy": self._cfg.greedy,
                "temperature": self._cfg.temperature,
                "top_p": self._cfg.top_p,
                "prompt_format": self._cfg.prompt_format,
            },
            "sample_rate": SAMPLE_RATE,
        }

    # ------------------------------------------------------------- process
    async def _read_stderr(self, proc: asyncio.subprocess.Process) -> None:
        assert proc.stderr is not None
        while True:
            line = await proc.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").rstrip()
            if text:
                self._stderr_tail.append(text)

    async def stop(self) -> None:
        async with self._proc_lock:
            await self._stop_locked()

    async def _stop_locked(self) -> None:
        """Stop the child process. Caller must hold _proc_lock (asyncio locks
        are not reentrant, so ensure() cannot call stop())."""
        proc, self._proc = self._proc, None
        self.state = "stopped"
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.write(b"EXIT\n")
                await proc.stdin.drain()
                proc.stdin.close()
        except Exception:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

    async def ensure(self, cfg: EngineConfig) -> None:
        """Start the process (or restart it if the config needs new CLI flags)."""
        if not self.binary_ok():
            raise EngineError(f"asr_stream_server binary missing: {BIN}")
        if not self.models_ok():
            raise EngineError(
                "models missing — run scripts/fetch_models.sh "
                f"(expected {VAE_MODEL.name} + {LM_MODEL.name})")

        alive = self._proc is not None and self._proc.returncode is None
        if alive and self._cfg is not None and self._cfg.key() == cfg.key():
            return

        async with self._proc_lock:
            alive = self._proc is not None and self._proc.returncode is None
            if alive and self._cfg is not None and self._cfg.key() == cfg.key():
                return
            # Hot-updatable knobs don't need a restart.
            if alive and self._cfg is not None and self._soft_equal(self._cfg, cfg):
                await self._hot_update(cfg)
                self._cfg = cfg
                return

            if alive:
                await self._stop_locked()

            cmd = [str(BIN), "--vae-model", str(VAE_MODEL), "--lm-model", str(LM_MODEL),
                   "-t", str(cfg.threads), "--max-tokens", str(cfg.max_tokens),
                   "--prompt-format", cfg.prompt_format]
            cmd += ["--greedy"] if cfg.greedy else [
                "--temperature", str(cfg.temperature), "--top-p", str(cfg.top_p)]
            if cfg.context:
                cmd += ["--context", cfg.context]

            self.state = "loading"
            self.last_error = None
            self._stderr_tail.clear()
            t0 = time.perf_counter()
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)

            self._stderr_task = asyncio.create_task(self._read_stderr(proc))
            assert proc.stdout is not None
            try:
                while True:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=180)
                    if not line:
                        raise EngineError("engine exited during load")
                    if b"---READY---" in line:
                        break
            except asyncio.TimeoutError:
                proc.kill()
                raise EngineError("engine did not become ready within 180s")

            self._proc = proc
            self._cfg = cfg
            self.load_ms = (time.perf_counter() - t0) * 1000
            self.state = "ready"

    @staticmethod
    def _soft_equal(a: EngineConfig, b: EngineConfig) -> bool:
        """True when the diff can be applied through stdin instead of a restart."""
        hard = ("threads", "greedy", "temperature", "top_p", "max_tokens", "prompt_format")
        return all(getattr(a, f) == getattr(b, f) for f in hard)

    async def _hot_update(self, cfg: EngineConfig) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(f"CONTEXT:{cfg.context}\n".encode())
        await self._proc.stdin.drain()

    # ---------------------------------------------------------- inference
    async def transcribe(self, wav_path: str, cfg: EngineConfig,
                         on_token=None) -> dict:
        """Transcribe one WAV file; streams text pieces to `on_token`."""
        async with self._lock:
            await self.ensure(cfg)
            proc = self._proc
            if proc is None or proc.returncode is not None or proc.stdin is None \
                    or proc.stdout is None:
                raise EngineError("engine process is not running")

            mark = len(self._stderr_tail)
            proc.stdin.write((wav_path + "\n").encode())
            await proc.stdin.drain()

            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            parts: list[str] = []
            error: str | None = None

            while True:
                line = await proc.stdout.readline()
                if not line:
                    raise EngineError("engine process died mid-chunk")
                if line.endswith(b"\n"):
                    line = line[:-1]
                if line == b"---END---":
                    break
                if line.startswith(b"[ERROR]"):
                    error = line.decode("utf-8", "replace")
                    continue
                if not line:
                    continue
                text = decoder.decode(line, final=False)
                if text:
                    parts.append(text)
                    if on_token:
                        await on_token(text)
            tail = decoder.decode(b"", final=True)
            if tail:
                parts.append(tail)
                if on_token:
                    await on_token(tail)

            self.chunks_done += 1
            stats = await self._wait_stats(mark)
            if error:
                raise EngineError(error)
            return {"text": "".join(parts), "stats": stats}

    async def _wait_stats(self, mark: int, timeout: float = 5.0) -> dict:
        """The C++ side prints its timing line to stderr right after ---END---."""
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            for line in list(self._stderr_tail)[mark:]:
                m = _RTF_RE.search(line)
                if m:
                    return {"audio_sec": float(m.group(1)), "tokens": int(m.group(2)),
                            "rtf": float(m.group(3)), "total_ms": float(m.group(4))}
            await asyncio.sleep(0.02)
        return {}


# ------------------------------------------------------------------ audio IO

def decode_to_float(path: str | Path) -> np.ndarray:
    """Any input format -> mono float32 @ 24 kHz, via ffmpeg."""
    import subprocess
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
           "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise EngineError(msg[-1] if msg else "ffmpeg could not decode the audio")
    if not proc.stdout:
        raise EngineError("audio file decoded to 0 samples")
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def write_wav(path: str | Path, samples: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def _quietest_boundary(audio: np.ndarray, target: int, search: int, win: int) -> int:
    start = max(0, target - search)
    end = min(len(audio), target + search)
    step = max(1, win // 2)
    best_pos, best_rms = target, float("inf")
    for pos in range(start, max(start + 1, end - win), step):
        window = audio[pos:pos + win]
        if window.size == 0:
            break
        rms = float(np.sqrt(np.mean(window ** 2)))
        if rms < best_rms:
            best_rms, best_pos = rms, pos + win // 2
    return best_pos if best_rms < 0.01 else target


def plan_chunks(total: int, chunk_sec: float = 12.0, overlap_sec: float = 2.0,
                sr: int = SAMPLE_RATE, audio: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Cut the signal into overlapping windows, snapping cuts to quiet spots."""
    if total <= 0:
        return []
    chunk, overlap, win = int(chunk_sec * sr), int(overlap_sec * sr), int(0.3 * sr)
    search = int(1.0 * sr)
    if total <= chunk:
        return [(0, total)]
    spans: list[tuple[int, int]] = []
    start = 0
    while start < total:
        nominal_end = min(start + chunk, total)
        end = nominal_end
        if audio is not None and nominal_end < total:
            end = _quietest_boundary(audio, nominal_end, search, win)
            end = max(end, start + int(1.0 * sr))  # never produce a stub
        spans.append((start, min(end, total)))
        if end >= total:
            break
        start = max(0, end - overlap)
    return spans


# ------------------------------------------------------------ text stitching

def _words(text: str) -> list[str]:
    return text.split()


_PUNCT = re.compile(r"^\W+|\W+$")
_TIGHT = re.compile(r"([.!?])([A-Z])")


def tidy(text: str) -> str:
    """Insert the space the model sometimes omits after sentence punctuation.

    ``night.Before`` tokenises as one word, which breaks overlap stitching and
    reads badly; normalising here fixes both at once.
    """
    return _TIGHT.sub(r"\1 \2", (text or "").strip())


def _norm(word: str) -> str:
    """Comparison form: case-folded, punctuation stripped from both ends."""
    return _PUNCT.sub("", word).lower()


def stitch_overlap(tail: str, new: str, max_words: int = 40, min_run: int = 3) -> str:
    """Drop the part of `new` that repeats the end of `tail`.

    Windows overlap in audio, so window N+1 starts by re-saying what window N
    finished with. A plain suffix/prefix comparison is not enough: the overlap
    usually trails off into ASR drift (``… Perfect. The main question is when I.``
    vs ``Perfect. The main question is whether we ship …``), so the shared span
    is *inside* the tail, not anchored at its end.

    Instead we slide every offset of the tail over the head of the new text and
    keep the longest matching run. That run length is how much of `new` to drop.
    """
    old, new_w = _words(tail), _words(new)
    if not old or not new_w:
        return new.strip()
    tail_w = old[-max_words:]
    a = [_norm(w) for w in tail_w]
    b = [_norm(w) for w in new_w[:max_words]]

    best_run, best_off = 0, 0
    for off in range(len(a)):
        run = 0
        while (off + run < len(a) and run < len(b) and a[off + run] and a[off + run] == b[run]):
            run += 1
        if run > best_run:
            best_run, best_off = run, off

    reaches_end = best_off + best_run >= len(a)
    if (reaches_end and best_run >= 2) or best_run >= 5:
        return " ".join(new_w[best_run:]).strip()
    return new.strip()


def tmpdir() -> Path:
    d = Path(tempfile.gettempdir()) / "vibescribe"
    d.mkdir(parents=True, exist_ok=True)
    return d


ENGINE = AsrEngine()
