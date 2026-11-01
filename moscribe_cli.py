"""MoScribe launcher — `moscribe` on the command line, or the packaged app icon.

Starts the local server, opens a browser, and explains what is missing instead of
throwing a stack trace at the user.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

APP_NAME = "MoScribe"
DEFAULT_PORT = 8770


def _free_port(host: str, start: int) -> int:
    for port in range(start, start + 30):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
            except OSError:
                continue
            return port
    return start


def _banner(url: str, paths: dict) -> None:
    print(f"\n  {APP_NAME}")
    print(f"  {url}")
    print(f"  data     {paths['data_dir']}")
    print(f"  models   {paths['models_dir']}")
    print("  stop with Ctrl+C\n", flush=True)


def _open_later(url: str, delay: float = 1.2) -> None:
    def go() -> None:
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - a missing browser is not fatal
            pass

    threading.Thread(target=go, daemon=True).start()


def _status() -> tuple[bool, list[str]]:
    """Are the native binaries and models in place? Ask the engine, not a guess."""
    notes: list[str] = []
    ok = True
    try:
        from moscribe.asr_engine import LM_MODEL, VAE_MODEL, resolve_bin

        BIN = resolve_bin()
    except Exception as exc:  # noqa: BLE001 - a broken import is itself the problem
        return False, [f"engine module failed to load: {type(exc).__name__}: {exc}"]

    if not BIN.exists():
        ok = False
        notes.append(f"missing binary: asr_stream_server (expected {BIN})")
    for path, label in ((VAE_MODEL, "ASR encoder"), (LM_MODEL, "ASR language model")):
        if not path.exists():
            ok = False
            notes.append(f"missing model: {label} ({path.name})")

    try:
        from moscribe.llm import LLM_ENGINE

        # ask the live engine: it already applies VIBESCRIBE_LLM_BIN / model config
        binary, model = LLM_ENGINE.resolve_binary(), LLM_ENGINE.model
        if not binary.exists():
            ok = False
            notes.append(f"missing binary: llama-server (expected {binary})")
        if not model.exists():
            ok = False
            notes.append(f"missing model: text model ({model.name})")
    except Exception as exc:  # noqa: BLE001
        ok = False
        notes.append(f"text engine failed to load: {type(exc).__name__}")
    return ok, notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="moscribe", description=f"{APP_NAME} — local speech to text")
    ap.add_argument("--host", default=os.environ.get("MOSCRIBE_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MOSCRIBE_PORT", DEFAULT_PORT)))
    ap.add_argument("--data-dir", help="where history, jobs and downloads live")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--check", action="store_true", help="report binaries and models, then exit")
    ap.add_argument("--fetch-models", action="store_true", help="download models, then exit")
    ap.add_argument("--no-fetch", action="store_true",
                    help="do not download missing models/binaries on start")
    ap.add_argument("--version", action="store_true")
    args = ap.parse_args(argv)

    if args.data_dir:
        os.environ["MOSCRIBE_DATA"] = str(Path(args.data_dir).expanduser())

    from moscribe import main as app_module  # imported late: honours --data-dir
    from moscribe import paths

    if args.version:
        from moscribe import __version__ as v

        print(f"{APP_NAME} {v}")
        return 0

    if args.fetch_models:
        from moscribe.fetch import main as fetch_main

        return fetch_main([])

    ok, notes = _status()
    if args.check:
        print(f"{APP_NAME} — {'ready' if ok else 'incomplete'}")
        for p, v in paths.describe().items():
            print(f"  {p:<10} {v}")
        for n in notes:
            print(f"  ! {n}")
        return 0 if ok else 1
    auto_fetch = os.environ.get("MOSCRIBE_AUTO_FETCH", "1").lower() not in ("0", "false", "no", "off")
    if not ok and auto_fetch and not args.no_fetch:
        print(f"\n  {APP_NAME} is missing its models or engines — downloading them once.")
        for n in notes:
            print(f"    · {n}")
        from moscribe.fetch import main as fetch_main

        try:
            fetch_main([])
        except SystemExit:
            pass
        except Exception as exc:  # noqa: BLE001 - never block the UI on a download
            print(f"    ! download failed: {type(exc).__name__}: {exc}")
        ok, notes = _status()

    if not ok:
        print(f"{APP_NAME}: setup is incomplete", file=sys.stderr)
        for n in notes:
            print(f"  ! {n}", file=sys.stderr)
        print("\n  retry the download with: moscribe --fetch-models\n", file=sys.stderr)

    paths.ensure_dirs()
    port = _free_port(args.host, args.port) if args.port else _free_port(args.host, DEFAULT_PORT)
    host_part = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{host_part}:{port}"
    _banner(url, paths.describe())
    if not args.no_browser:
        _open_later(url)

    import uvicorn

    try:
        uvicorn.run(app_module.app, host=args.host, port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
