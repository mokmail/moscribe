"""Make the native engine binaries portable for packaging.

They are linked against `@rpath/libllama.dylib` / `libggml.dylib` but built with
rpaths pointing at the build tree — inside a bundle that path does not exist. This
copies the libs next to the executables and points the binaries at `@loader_path`
(macOS) or `$ORIGIN` (Linux). Windows builds already drop the DLLs beside the exe.

    python packaging/prepare_bin.py --src ~/apps/VibeASR.cpp --out bin
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

EXES = ["asr_stream_server", "llama-server"]
LIB_GLOBS = ["**/libllama*.dylib", "**/libggml*.dylib", "**/libllama*.so*", "**/libggml*.so*"]
WIN_GLOBS = ["**/llama.dll", "**/ggml*.dll", "**/*.dll"]


def _find(root: Path, patterns: list[str]) -> list[Path]:
    seen: dict[str, Path] = {}
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file() and p.name not in seen:
                seen[p.name] = p
    return list(seen.values())


def _mac_fix(exe: Path, out: Path) -> None:
    libs = sorted(out.glob("*.dylib"))
    for path in [exe, *libs]:
        try:
            names = subprocess.run(["otool", "-l", str(path)], capture_output=True, text=True).stdout
            existing = set()
            for line in names.splitlines():
                line = line.strip()
                if not line.startswith("path "):
                    continue
                value = line[5:].split(" (offset")[0].strip()  # otool appends offsets
                existing.add(value)
                # absolute rpaths point into the build tree; inside a bundle they are dead
                if value.startswith("/") and not value.startswith(("/usr", "/System")):
                    subprocess.run(["install_name_tool", "-delete_rpath", value, str(path)],
                                   capture_output=True)
            if "@loader_path" not in existing:
                subprocess.run(["install_name_tool", "-add_rpath", "@loader_path", str(path)],
                               capture_output=True)
            for lib in libs:
                subprocess.run(["install_name_tool", "-change", f"@rpath/{lib.name}",
                                f"@loader_path/{lib.name}", str(path)], capture_output=True)
        except FileNotFoundError:
            print("  ! install_name_tool not found — bundle may rely on build-tree rpaths")


def _linux_fix(exe: Path, out: Path) -> None:
    try:
        subprocess.run(["patchelf", "--set-rpath", "$ORIGIN", str(exe)], check=False)
    except FileNotFoundError:
        print("  ! patchelf not installed — set rpath manually or ship LD_LIBRARY_PATH")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="root of the VibeASR.cpp checkout")
    ap.add_argument("--out", default="bin")
    args = ap.parse_args()

    src = Path(args.src).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sub = ".exe" if os.name == "nt" else ""

    found = 0
    for name in EXES:
        candidates = list(src.glob(f"**/build*/**/{name}{sub}")) + list(src.glob(f"**/{name}{sub}"))
        hit = next((c for c in candidates if c.is_file()), None)
        if not hit:
            print(f"  ! {name} not built under {src}")
            continue
        shutil.copy2(hit, out / hit.name)
        print(f"  {name} <- {hit.relative_to(src)}")
        found += 1

    libs = _find(src, WIN_GLOBS if os.name == "nt" else LIB_GLOBS)
    for lib in libs:
        shutil.copy2(lib, out / lib.name)
    if libs:
        print(f"  copied {len(libs)} shared lib(s)")

    for exe in (out / f"{n}{sub}" for n in EXES):
        if not exe.exists():
            continue
        exe.chmod(exe.stat().st_mode | 0o755)
        if sys.platform == "darwin":
            _mac_fix(exe, out)
        elif sys.platform.startswith("linux"):
            _linux_fix(exe, out)

    print(f"  {found}/{len(EXES)} engine binaries ready in {out}")
    return 0 if found == len(EXES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
