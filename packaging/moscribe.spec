# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build for MoScribe.

Produces a one-dir bundle: the executable plus `static/` and `bin/` beside it, so
`paths.app_dir()` resolves exactly the way it does in a checkout.

    pyinstaller --clean --noconfirm packaging/moscribe.spec

Native engine binaries are copied from $MOSCRIBE_BIN_DIR when present; a bundle
without them still starts and tells the user to run `moscribe --fetch-models`.
"""
import os
import sys
from pathlib import Path

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - injected by PyInstaller
APP = ROOT / "moscribe"

datas = [(str(APP / "static"), "static")]

bin_dir = Path(os.environ.get("MOSCRIBE_BIN_DIR", ROOT / "bin"))
if bin_dir.is_dir():
    datas.append((str(bin_dir), "bin"))

models = ROOT / "models"
if os.environ.get("MOSCRIBE_BUNDLE_MODELS") == "1" and models.is_dir():
    datas.append((str(models), "models"))

hidden = [
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "moscribe.main",
    "moscribe.fetch",
]

a = Analysis(
    [str(ROOT / "moscribe_cli.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pandas", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MoScribe",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=sys.platform != "darwin",   # no stray terminal window on macOS
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=os.environ.get("MACOS_SIGN_IDENTITY") or None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="MoScribe",
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="MoScribe.app",
        icon=os.environ.get("MOSCRIBE_ICON") or None,
        bundle_identifier="at.kmail.moscribe",
        info_plist={
            "CFBundleName": "MoScribe",
            "CFBundleDisplayName": "MoScribe",
            "CFBundleShortVersionString": "1.0.0",
            "LSMinimumSystemVersion": "12.0",
            "NSMicrophoneUsageDescription": "MoScribe transcribes speech on this machine only.",
            "NSHighResolutionCapable": True,
        },
    )
