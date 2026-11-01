"""Generate platform icon files from the brand mark.

    python packaging/make_icons.py

The mark is a black chevron inside a red/green brush ring on a transparent ground.
Dropped raw into a Dock or taskbar it reads as a floating scribble, so every icon is
composed the way the platforms expect: a rounded-square tile with the mark inset.

Writes moscribe/static/brand/{app.icns,app.ico,app-512.png}.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

LANCZOS = Image.Resampling.LANCZOS

BRAND = Path(__file__).resolve().parent.parent / "moscribe" / "static" / "brand"
SRC = BRAND / "logo-mark.png"
SIDE = 1024
INSET = 64          # macOS-style margin around the tile
MARK_RATIO = 0.74   # how much of the tile the mark fills


def squircle_mask(size: int, radius: float) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def build_tile() -> Image.Image:
    mark = Image.open(SRC).convert("RGBA")
    inner = SIDE - 2 * INSET
    radius = int(inner * 0.224)          # macOS squircle proportions

    canvas = Image.new("RGBA", (SIDE, SIDE), (0, 0, 0, 0))

    # soft drop shadow, so the tile sits on a desktop instead of floating
    shadow = Image.new("RGBA", (SIDE, SIDE), (0, 0, 0, 0))
    mask = squircle_mask(inner, radius)
    shadow.paste(Image.new("RGBA", (inner, inner), (0, 0, 0, 70)), (INSET, INSET + 14), mask)
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(18)))

    # the tile itself: near-white with a hairline edge
    tile = Image.new("RGBA", (inner, inner), (0, 0, 0, 0))
    tile.paste(Image.new("RGBA", (inner, inner), (255, 255, 255, 255)), (0, 0), mask)
    ImageDraw.Draw(tile).rounded_rectangle([1, 1, inner - 2, inner - 2], radius=radius,
                                           outline=(15, 21, 18, 26), width=2)
    canvas.alpha_composite(tile, (INSET, INSET))

    # the mark, centred on the tile
    target = int(inner * MARK_RATIO)
    ratio = target / max(mark.size)
    scaled = mark.resize((max(1, int(mark.width * ratio)), max(1, int(mark.height * ratio))), LANCZOS)
    canvas.alpha_composite(scaled, ((SIDE - scaled.width) // 2, (SIDE - scaled.height) // 2))
    return canvas


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}")
        return 1

    tile = build_tile()
    tile.resize((512, 512), LANCZOS).save(BRAND / "app-512.png")
    tile.save(BRAND / "app.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                                        (64, 64), (128, 128), (256, 256)])
    print("  app.ico, app-512.png written")

    if sys.platform == "darwin" and shutil.which("iconutil"):
        iconset = BRAND / "app.iconset"
        shutil.rmtree(iconset, ignore_errors=True)
        iconset.mkdir()
        for size in (16, 32, 64, 128, 256, 512):
            tile.resize((size, size), LANCZOS).save(iconset / f"icon_{size}x{size}.png")
            tile.resize((size * 2, size * 2), LANCZOS).save(iconset / f"icon_{size}x{size}@2x.png")
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(BRAND / "app.icns")], check=False)
        shutil.rmtree(iconset, ignore_errors=True)
        print("  app.icns written")
    else:
        print("  (iconutil not available — skipping .icns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
