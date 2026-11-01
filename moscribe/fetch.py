"""First-run downloads: model weights (and prebuilt binaries when published).

Nothing here needs credentials — every model is public on Hugging Face. Downloads
resume via `.part` files, so an interrupted 950 MB fetch is not wasted.
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from . import paths

HF = "https://huggingface.co"

MODELS: list[tuple[str, str, float]] = [
    ("vibeasr/vibeasr-vae-encoder-i8_s.gguf",
     f"{HF}/microsoft/VibeVoice-ASR-BitNet/resolve/main/vibeasr-vae-encoder-i8_s.gguf?download=true", 671),
    ("vibeasr/vibeasr-lm-i2_s-embed-q6_k.gguf",
     f"{HF}/microsoft/VibeVoice-ASR-BitNet/resolve/main/vibeasr-lm-i2_s-embed-q6_k.gguf?download=true", 947),
    ("llm/ggml-model-i2_s.gguf",
     f"{HF}/microsoft/BitNet-b1.58-2B-4T-gguf/resolve/main/ggml-model-i2_s.gguf?download=true", 1133),
    ("spk/wespeaker-resnet34.onnx",
     f"{HF}/onnx-community/wespeaker-voxceleb-resnet34-LM/resolve/main/onnx/model.onnx?download=true", 25),
]

BIN_PLATFORMS = {
    ("Darwin", "arm64"): "macos-arm64",
    ("Darwin", "x86_64"): "macos-x64",
    ("Linux", "x86_64"): "linux-x64",
    ("Windows", "AMD64"): "windows-x64",
}


def _platform_tag() -> str | None:
    import platform

    return BIN_PLATFORMS.get((platform.system(), platform.machine()))


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit != "GB" else f"{n:.1f} GB"
        n /= 1024
    return f"{n:.1f} GB"


def _download(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "moscribe"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0) + have
            mode = "ab" if have and r.status == 206 else "wb"
            if mode == "wb":
                have = 0
            done = have
            with part.open(mode) as fh:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
                    done += len(block)
                    if total:
                        pct = done * 100 / total
                        bar = "=" * int(pct / 3)
                        print(f"\r  {label:<22} [{bar:<33}] {pct:5.1f}%  {human(done)}", end="", flush=True)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"\n  download failed ({exc.code}) for {url}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"\n  network error for {url}: {exc.reason}") from exc
    part.replace(dest)
    print(f"\r  {label:<22} done — {human(dest.stat().st_size)}" + " " * 20)


def fetch_models(force: bool = False) -> int:
    root = paths.models_dir()
    print(f"  models → {root}")
    missing = 0
    for rel, url, _mb in MODELS:
        dest = root / rel
        if dest.exists() and dest.stat().st_size > 1024 and not force:
            print(f"  {dest.name:<30} present ({human(dest.stat().st_size)})")
            continue
        print(f"  {dest.name:<30} {_mb:.0f} MB")
        _download(url, dest, dest.name)
        missing += 1
    return missing


def fetch_binaries() -> bool:
    """Prebuilt engine binaries from this project's releases, when they exist."""
    tag = _platform_tag()
    if not tag:
        print("  no prebuilt binaries for this platform — build from source (see docs)")
        return False
    already = [p for p in (paths.bin_dir(), paths.DATA_DIR / "bin")
               if (p / paths.exe("asr_stream_server")).exists() and (p / paths.exe("llama-server")).exists()]
    if already:
        print(f"  binaries present in {already[0]}")
        return True

    override = os.environ.get("MOSCRIBE_BIN_URL")
    url = override or (f"https://github.com/mokmail/moscribe/releases/latest/download/"
                       f"moscribe-{tag}-bin.zip")
    dest = paths.DATA_DIR / "bin"
    dest.mkdir(parents=True, exist_ok=True)
    tmp = paths.DATA_DIR / "bin.zip"
    print(f"  binaries → {dest}")
    try:
        _download(url, tmp, "engine binaries")
    except SystemExit:
        print("  (no published bundle for this platform — build from source, see docs/BUILD.md)")
        return False
    with zipfile.ZipFile(tmp) as z:
        z.extractall(dest)
    tmp.unlink(missing_ok=True)
    for f in dest.iterdir():
        if not f.name.endswith(".zip"):
            f.chmod(f.stat().st_mode | 0o755)
    return True


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in args
    print("\n  MoScribe setup\n")
    paths.ensure_dirs()
    fetch_models(force=force)
    fetch_binaries()
    print("\n  done — start the app with: moscribe\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
