"""Generate platform icon files from the brand mark.

    python packaging/make_icons.py

Writes moscribe/static/brand/app.ico (Windows), app.icns (macOS) and app-512.png.
The mark is a 553 px square with a transparent ground, so it tiles cleanly.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

LANCZOS = Image.Resampling.LANCZOS

BRAND = Path(__file__).resolve().parent.parent / "moscribe" / "static" / "brand"
SRC = BRAND / "logo-mark.png"


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}")
        return 1
    img = Image.open(SRC).convert("RGBA")
    side = max(img.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))

    canvas.resize((512, 512), LANCZOS).save(BRAND / "app-512.png")
    canvas.save(BRAND / "app.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                                          (64, 64), (128, 128), (256, 256)])
    print("  app.ico, app-512.png written")

    if sys.platform == "darwin" and shutil.which("iconutil"):
        iconset = BRAND / "app.iconset"
        shutil.rmtree(iconset, ignore_errors=True)
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            canvas.resize((size, size), LANCZOS).save(iconset / f"icon_{size}x{size}.png")
            canvas.resize((size * 2, size * 2), LANCZOS).save(iconset / f"icon_{size}x{size}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(BRAND / "app.icns")], check=False)
        shutil.rmtree(iconset, ignore_errors=True)
        print("  app.icns written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
