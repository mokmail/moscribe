#!/usr/bin/env bash
# Wrap dist/MoScribe.app into a compressed DMG with a drag-to-Applications link.
#
#   ./packaging/macos/make_dmg.sh [version]
#
# Builds the .app first when it is missing (needs .venv with pyinstaller).
set -euo pipefail

VERSION="${1:-1.0.0}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$ROOT/dist/MoScribe.app"
OUT="$ROOT/dist/MoScribe-${VERSION}-macos-$(uname -m).dmg"
PY="$ROOT/.venv/bin/python"

if [ ! -d "$APP" ]; then
  echo "→ dist/MoScribe.app is missing; building it now"
  if [ ! -x "$PY" ]; then
    echo "   no .venv found. Set one up first:" >&2
    echo "     python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
    echo "     .venv/bin/pyinstaller --clean --noconfirm packaging/moscribe.spec" >&2
    exit 1
  fi
  "$PY" packaging/make_icons.py
  "$PY" -c "import PyInstaller" >/dev/null 2>&1 || {
    echo "   installing pyinstaller + pillow into .venv"
    "$PY" -m pip install -q pyinstaller pillow
  }
  [ -d "$ROOT/bin" ] && [ -n "$(ls -A "$ROOT/bin" 2>/dev/null)" ] || {
    echo "   ! bin/ is empty — the bundle will ship without engine binaries."
    echo "     build them: python packaging/prepare_bin.py --src ~/apps/VibeASR.cpp --out bin"
  }
  ( cd "$ROOT" && "$PY" -m PyInstaller --clean --noconfirm packaging/moscribe.spec )
fi

[ -d "$APP" ] || { echo "pyinstaller did not produce $APP" >&2; exit 1; }

STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/MoScribe.app"
ln -s /Applications "$STAGE/Applications"

# the mounted disk shows the kmail mark, not the generic disk icon
ICNS="$ROOT/moscribe/static/brand/app.icns"
if [ -f "$ICNS" ]; then
  cp "$ICNS" "$STAGE/.VolumeIcon.icns"
  command -v SetFile >/dev/null && SetFile -a C "$STAGE" || true
fi

# ad-hoc signature: enough for Gatekeeper to accept a locally built bundle
codesign --force --deep --sign - "$STAGE/MoScribe.app" 2>/dev/null || echo "  (codesign skipped)"

rm -f "$OUT"
hdiutil create -volname "MoScribe $VERSION" -srcfolder "$STAGE" -ov -format UDZO "$OUT" >/dev/null
rm -rf "$STAGE"
echo "built $OUT"
