#!/usr/bin/env bash
# Wrap dist/MoScribe.app into a compressed DMG with a drag-to-Applications link.
#   packaging/macos/make_dmg.sh [version]
set -euo pipefail

VERSION="${1:-1.0.0}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$ROOT/dist/MoScribe.app"
OUT="$ROOT/dist/MoScribe-${VERSION}-macos-$(uname -m).dmg"

[ -d "$APP" ] || { echo "missing $APP — run pyinstaller first" >&2; exit 1; }

STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/MoScribe.app"
ln -s /Applications "$STAGE/Applications"

# ad-hoc signature: enough for Gatekeeper to accept a locally built bundle
codesign --force --deep --sign - "$STAGE/MoScribe.app" 2>/dev/null || \
  echo "  (codesign skipped)"

rm -f "$OUT"
hdiutil create -volname "MoScribe $VERSION" -srcfolder "$STAGE" -ov -format UDZO "$OUT" >/dev/null
rm -rf "$STAGE"
echo "built $OUT"
