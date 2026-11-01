"""Local text generation for VibeScribe — summarizing, minutes, rewriting.

Runs BitNet b1.58 2B-4T (microsoft/bitnet) through the llama.cpp build that ships
with VibeASR.cpp: its ggml already carries the I2_S kernels and registers
``LLM_ARCH_BITNET_B158``, so no second toolchain is needed. The server speaks the
OpenAI chat API, which keeps prompt/template handling in one place.

Everything stays on this machine: a child process on 127.0.0.1, no keys.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from .paths import APP_DIR, DATA_DIR, bin_dir, exe, find_exe, models_dir  # noqa: E402

ROOT = APP_DIR
MODELS_DIR = models_dir() / "llm"
CONFIG_FILE = DATA_DIR / "llm.json"
_BIN_CANDIDATES = [
    bin_dir() / exe("llama-server"),
    DATA_DIR / "bin" / exe("llama-server"),
    Path.home() / "apps" / "VibeASR.cpp" / "build-llm" / "bin" / "llama-server",
]
DEFAULT_BIN = next((c for c in _BIN_CANDIDATES if c.exists()), _BIN_CANDIDATES[0])
DEFAULT_MODEL = MODELS_DIR / "ggml-model-i2_s.gguf"
PORT = int(os.environ.get("VIBESCRIBE_LLM_PORT", "8771"))

SYSTEM = ("You are a precise meeting assistant. Use only the transcript you are "
          "given. Never invent names, numbers or decisions. If something is not "
          "in the transcript, say it is not mentioned.")


def _prefix(name: str, text: str) -> tuple[str, str]:
    label = f"{name}: " if name else ""
    return label, text.strip()


def build_messages(action: str, transcript: str, extra: str = "",
                   speakers: dict[str, str] | None = None) -> list[dict]:
    """Prompt for each action. Short and explicit — a 2B model rewards brevity."""
    lines = []
    for raw in transcript.splitlines():
        line = raw.strip()
        if not line:
            continue
        lines.append(line)
    body = "\n".join(lines)
    if speakers:
        legend = "; ".join(f"Speaker {int(k) + 1} is {v}" for k, v in speakers.items() if v)
        if legend:
            body = f"Speakers: {legend}\n\n{body}"

    tasks = {
        "summarize": "Summarise this transcript in at most 5 bullet points. "
                     "One sentence each, mention who said what when it matters.",
        "minutes": "Write short meeting minutes with these headings:\n"
                   "Decisions\nAction items (owner, if named)\nOpen questions\n"
                   "Use bullets under each heading. Omit a heading if the transcript has nothing for it.",
        "actions": "List only the action items. One bullet per item, in the form "
                   "'- <task> — <owner if named>'. If no actions are stated, say so.",
        "rewrite": "Rewrite this transcript as clean readable prose. Remove filler words and "
                   "false starts, fix punctuation, keep every fact and keep the speaker "
                   "attribution as 'Name: ...' lines.",
        "translate_en": "Translate this transcript into English. Keep speaker labels and names.",
        "translate_de": "Translate this transcript into German (formal register, Austrian usage "
                        "where it applies). Keep speaker labels and names.",
        "custom": (extra or "Summarise this transcript.").strip(),
    }
    task = tasks.get(action, tasks["summarize"])
    # No "answer in German" instruction: measured on a German transcript, the 2.4B
    # model echoes the transcript or invents facts in German, while its English
    # summaries are accurate. Translation stays a separate, explicit action.
    if action == "custom" and extra:
        task += "\nAnswer in the language of the request."
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{task}\n\nTRANSCRIPT:\n{body}"},
    ]


class LLM:
    """Supervises the llama-server child process and streams completions."""

    DEFAULTS = {
        "model": "",            # empty -> DEFAULT_MODEL
        "threads": 0,           # 0 -> min(8, cpu)
        "ctx": 8192,
        "temperature": 0.30,
        "max_tokens": 700,
        "repeat_penalty": 1.08,
    }

    def __init__(self, binary: Path | None = None, model: Path | None = None, port: int = PORT):
        self.binary = Path(os.environ.get("VIBESCRIBE_LLM_BIN", binary or DEFAULT_BIN))
        self.port = port
        self._proc: subprocess.Popen | None = None
        self._lock = asyncio.Lock()
        self._last_error: str | None = None
        self._fetch_tried = False
        self._started_at: float | None = None
        self.config: dict = dict(self.DEFAULTS)
        self._env_model = Path(os.environ["VIBESCRIBE_LLM_MODEL"]) if os.environ.get("VIBESCRIBE_LLM_MODEL") else None
        self._default_model = model or DEFAULT_MODEL
        self._load_config()

    def _auto_fetch(self) -> None:
        """A missing engine binary is a download, not an error the user should read."""
        if self._fetch_tried:
            return
        self._fetch_tried = True
        if os.environ.get("MOSCRIBE_AUTO_FETCH", "1").lower() in ("0", "false", "no", "off"):
            return
        try:
            from .fetch import fetch_binaries

            fetch_binaries()
        except Exception:  # noqa: BLE001 - offline is a real possibility
            pass

    def resolve_binary(self) -> Path:
        """Re-check on demand: a first-run download lands after import."""
        if self.binary.exists():
            return self.binary
        found = find_exe("llama-server")
        if found:
            self.binary = found
        return self.binary

    # -- configuration -----------------------------------------------------
    def _load_config(self) -> None:
        try:
            saved = json.loads(CONFIG_FILE.read_text())
            for key, value in (saved or {}).items():
                if key in self.DEFAULTS:
                    self.config[key] = value
        except Exception:  # noqa: BLE001 - missing/corrupt config is fine
            pass

    def _save_config(self) -> None:
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(self.config, indent=1))

    @property
    def model(self) -> Path:
        """Active model: env override > configured path > packaged default."""
        if self._env_model:
            return self._env_model
        chosen = (self.config.get("model") or "").strip()
        if chosen:
            candidate = Path(chosen).expanduser()
            return candidate if candidate.is_absolute() else (MODELS_DIR / candidate)
        return self._default_model

    def list_models(self) -> list[dict]:
        """GGUF files the user can pick from (models/llm plus the packaged default)."""
        found: list[dict] = []
        seen: set[str] = set()
        for path in sorted(MODELS_DIR.glob("*.gguf")) if MODELS_DIR.exists() else []:
            try:
                size_mb = round(path.stat().st_size / 1e6, 1)      # broken symlink / removed file
            except OSError:
                continue
            try:
                active = path.resolve() == self.model.resolve()
            except OSError:
                active = False
            found.append({"name": path.name, "path": str(path), "mb": size_mb, "active": active})
            seen.add(str(path))
        default = self._default_model
        if default.exists() and str(default.resolve()) not in seen:
            found.insert(0, {"name": default.name, "path": str(default),
                             "mb": round(default.stat().st_size / 1e6, 1),
                             "active": default.resolve() == self.model.resolve()})
        return found

    def update_config(self, patch: dict) -> dict:
        """Validate, persist and return the new configuration (no restart here)."""
        out = dict(self.config)
        if "model" in patch:
            out["model"] = str(patch.get("model") or "").strip()
        if "threads" in patch:
            out["threads"] = max(1, min(16, int(patch["threads"])))
        if "ctx" in patch:
            out["ctx"] = max(512, min(32768, int(patch["ctx"])))
        if "temperature" in patch:
            out["temperature"] = max(0.0, min(2.0, float(patch["temperature"])))
        if "max_tokens" in patch:
            out["max_tokens"] = max(32, min(4096, int(patch["max_tokens"])))
        if "repeat_penalty" in patch:
            out["repeat_penalty"] = max(1.0, min(1.5, float(patch["repeat_penalty"])))
        self.config = out
        self._save_config()
        return out

    def threads(self) -> int:
        t = int(self.config.get("threads") or 0)
        return t if t > 0 else max(2, min(8, os.cpu_count() or 4))

    # -- lifecycle ---------------------------------------------------------
    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _ready(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base}/health", timeout=3) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False

    def info(self) -> dict:
        alive = self._proc is not None and self._proc.poll() is None
        return {
            "ok": self._ready(),
            "running": alive,
            "binary": str(self.resolve_binary()),
            "binary_ok": self.resolve_binary().exists(),
            "model": str(self.model),
            "model_ok": self.model.exists(),
            "model_mb": round(self.model.stat().st_size / 1e6, 1) if self.model.exists() else 0,
            "port": self.port,
            "threads": self.threads(),
            "config": dict(self.config),
            "models": self.list_models(),
            "error": self._last_error,
        }

    async def ensure(self) -> bool:
        """Start the server once; cheap no-op when *our* server is answering."""
        if self._ready() and self._proc is not None and self._proc.poll() is None:
            return True
        async with self._lock:
            if self._ready() and self._proc is not None and self._proc.poll() is None:
                return True
            if self._ready():
                # something answers on our port but it is not our child: a server
                # left behind by an earlier app run, still holding the old threads
                # and context. Settings would silently do nothing until it is gone.
                self._kill_foreign()
                for _ in range(10):
                    if not self._ready():
                        break
                    await asyncio.sleep(0.2)
            self.resolve_binary()
            if not self.binary.exists():
                self._auto_fetch()
                self.resolve_binary()
            if not self.binary.exists() or not self.model.exists():
                missing = []
                if not self.binary.exists():
                    missing.append(f"llama-server at {self.binary}")
                if not self.model.exists():
                    missing.append(f"{self.model.name} at {self.model}")
                self._last_error = ("text engine not installed: " + "; ".join(missing)
                                    + " — run `moscribe --fetch-models`, or just restart the app"
                                    " and it will download them")
                return False
            try:
                if self._proc and self._proc.poll() is None:
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass

            log = DATA_DIR / "llama-server.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            self._log_path = log
            model_path = self.model
            self._proc = subprocess.Popen(
                [str(self.binary), "-m", str(model_path), "--host", "127.0.0.1",
                 "--port", str(self.port), "-t", str(self.threads()),
                 "-c", str(int(self.config.get("ctx") or 4096))],
                stdout=log.open("ab"), stderr=subprocess.STDOUT, start_new_session=True)
            self._started_at = time.time()

            for _ in range(120):                       # model load takes a few seconds
                if self._proc.poll() is not None:
                    self._last_error = self._exit_reason(self._proc.returncode)
                    return False
                if self._ready():
                    self._last_error = None
                    return True
                await asyncio.sleep(0.5)
            self._last_error = (f"llama-server did not answer within 60 s — {self._log_tail() or 'no output'}"
                                f" (log: {getattr(self, '_log_path', '?')})")
            return False

    def _log_tail(self, lines: int = 4) -> str:
        path = getattr(self, "_log_path", None)
        if not path or not path.exists():
            return ""
        try:
            text = path.read_text(errors="replace").strip().splitlines()
            useful = [ln.strip() for ln in text if ln.strip()
                      and not ln.startswith(("llama_", "load_", "print_info", "ggml_"))]
            return " / ".join(useful[-lines:])[-300:]
        except OSError:
            return ""

    def _exit_reason(self, code: int | None) -> str:
        """Say what actually went wrong instead of blaming missing files."""
        tail = self._log_tail().lower()
        hint = ""
        if "out of memory" in tail or "alloc" in tail or "failed to allocate" in tail:
            hint = " — the context you asked for does not fit in memory; lower it in Settings → Model"
        elif "address already in use" in tail or "bind" in tail:
            hint = f" — port {self.port} is taken; set another one with VIBESCRIBE_LLM_PORT"
        return (f"llama-server exited with code {code}{hint}"
                f" — {self._log_tail() or 'no output'} (log: {getattr(self, '_log_path', '?')})")

    def _pids_on_port(self) -> list[int]:
        """PIDs listening on our port. A server left behind by an earlier app
        run still answers /health, so ensure() would happily report success while
        the *old* threads/ctx keep serving — it has to be reaped explicitly."""
        try:
            out = subprocess.run(["lsof", "-ti", f"tcp:{self.port}"],
                                 capture_output=True, text=True, timeout=5).stdout
            return [int(p) for p in out.split() if p.strip().isdigit()]
        except Exception:  # noqa: BLE001
            return []

    def stop(self) -> None:
        """Terminate the child we started (and any orphan on our port)."""
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
        self._kill_foreign()

    def _kill_foreign(self) -> None:
        for pid in self._pids_on_port():
            if pid == os.getpid():
                continue
            try:
                cmd = subprocess.run(["ps", "-o", "command=", "-p", str(pid)],
                                     capture_output=True, text=True, timeout=5).stdout
                if "llama-server" not in cmd:
                    continue                      # not ours — leave it alone
                os.kill(pid, 15)
            except Exception:  # noqa: BLE001
                pass

    async def restart(self) -> dict:
        """Apply the current config: stop whatever serves the port, then start."""
        async with self._lock:
            try:
                if self._proc and self._proc.poll() is None:
                    self._proc.terminate()
                    self._proc.wait(timeout=8)
            except Exception:  # noqa: BLE001
                pass
            self._proc = None
            self._kill_foreign()
            for _ in range(20):                    # wait for the port to free up
                if not self._ready():
                    break
                await asyncio.sleep(0.25)
        started = time.perf_counter()
        ok = await self.ensure()
        return {"ok": ok, "load_s": round(time.perf_counter() - started, 2),
                "error": None if ok else self._last_error, "info": self.info()}

    # -- inference ---------------------------------------------------------
    async def stream(self, messages: list[dict], max_tokens: int | None = None,
                     temperature: float | None = None):
        """Yield content deltas. Raises RuntimeError when the server is unavailable."""
        if not await self.ensure():
            raise RuntimeError(self._last_error or "text model unavailable")
        payload = json.dumps({
            "messages": messages, "stream": True,
            "temperature": float(temperature if temperature is not None
                                 else self.config.get("temperature", 0.3)),
            "max_tokens": int(max_tokens or self.config.get("max_tokens") or 700),
            "repeat_penalty": float(self.config.get("repeat_penalty") or 1.08),
        }).encode()

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()

        def worker() -> None:
            try:
                req = urllib.request.Request(
                    f"{self.base}/v1/chat/completions", data=payload,
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as resp:
                    for raw in resp:
                        if stop.is_set():
                            break
                        line = raw.decode("utf-8", "ignore").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        piece = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
                        if piece:
                            loop.call_soon_threadsafe(queue.put_nowait, ("delta", piece))
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, ("error", f"{type(exc).__name__}: {exc}"))
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

        threading.Thread(target=worker, daemon=True).start()
        try:
            while True:
                kind, value = await queue.get()
                if kind == "delta":
                    yield value
                elif kind == "error":
                    raise RuntimeError(value)
                else:
                    break
        finally:
            stop.set()


LLM_ENGINE = LLM()
