"""Where things live, in a source checkout and inside a frozen bundle.

Three kinds of path matter:
  * APP_DIR  — code + binaries + any models shipped inside the app
  * DATA     — user-writable state (history, jobs, settings); never the bundle
  * models   — bundled first, user data second (first-run download lands there)

Every location can be overridden with an environment variable, which is what the
installers and the Docker image use.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APP_NAME = "MoScribe"
APP_SLUG = "moscribe"


def _frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """Directory holding the app's own files (bin/, static/, models/)."""
    if _frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent


APP_DIR = app_dir()


def user_data_dir() -> Path:
    raw = os.environ.get("MOSCRIBE_DATA") or os.environ.get("VIBESCRIBE_DATA")
    if raw:
        return Path(raw).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_SLUG


DATA_DIR = user_data_dir()


def _first_existing(*candidates: Path) -> Path:
    for c in candidates:
        if c and c.exists():
            return c
    return candidates[0]


def models_dir() -> Path:
    """Root of all model files.

    Order: explicit env → models shipped in the app → a dev checkout beside the
    package → downloads in user data. Downloads never land inside the bundle.
    """
    env = os.environ.get("MOSCRIBE_MODELS")
    if env:
        return Path(env).expanduser()
    bundled = APP_DIR / "models"
    if bundled.exists() and any(bundled.iterdir()):
        return bundled
    checkout = APP_DIR.parent / "models"
    if not _frozen() and checkout.exists() and any(checkout.iterdir()):
        return checkout
    return DATA_DIR / "models"


def bin_dir() -> Path:
    return Path(os.environ.get("MOSCRIBE_BIN", APP_DIR / "bin"))


def exe(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def _migrate_dev_data() -> None:
    """A checkout used to keep history beside the code; carry it over once."""
    legacy = APP_DIR.parent / "data"
    if not legacy.exists() or legacy == DATA_DIR:
        return
    for name in ("history.json", "llm.json"):
        src, dst = legacy / name, DATA_DIR / name
        if src.exists() and not dst.exists():
            try:
                dst.write_bytes(src.read_bytes())
            except OSError:
                pass
    legacy_jobs = legacy / "jobs"
    if legacy_jobs.exists() and not any((DATA_DIR / "jobs").iterdir()):
        try:
            (DATA_DIR / "jobs").mkdir(parents=True, exist_ok=True)
            for job in legacy_jobs.iterdir():
                target = DATA_DIR / "jobs" / job.name
                if not target.exists():
                    shutil.copytree(job, target)
        except OSError:
            pass


def ensure_dirs() -> None:
    (DATA_DIR / "jobs").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "audio").mkdir(parents=True, exist_ok=True)
    _migrate_dev_data()


def describe() -> dict:
    return {
        "app_dir": str(APP_DIR),
        "data_dir": str(DATA_DIR),
        "models_dir": str(models_dir()),
        "frozen": _frozen(),
    }
