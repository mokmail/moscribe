"""Knock the white ground out of the supplied brushstroke logo.

Source: source-logo.png (the 512px upload). Writes logo-mark.png plus the
hi-dpi and icon sizes next to it. Re-run after a new source arrives.

    python prepare_logo.py [source.png] [outdir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

LANCZOS = Image.Resampling.LANCZOS

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "source-logo.png")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else ".")

a = np.array(Image.open(SRC).convert("RGBA")).astype(np.int16)
rgb = a[..., :3]

lum = rgb.mean(axis=2)
sat = rgb.max(axis=2) - rgb.min(axis=2)

# The artwork ships on an opaque white ground and the app is light, so drop it
# to alpha 0 — scaling alpha by how white-and-neutral a pixel is keeps the
# antialiased fringe instead of cutting it off.
cut = np.clip((lum - 232) / 20.0, 0, 1) * np.clip((30 - sat) / 30.0, 0, 1)

out = a.copy()
out[..., 3] = np.clip(a[..., 3] * (1 - cut), 0, 255)
fringe = cut > 0.01  # un-premultiply the white that was mixed into the edge
for c in range(3):
    ch = out[..., c]
    ch[fringe] = np.clip((ch[fringe] - 255 * cut[fringe]) / np.maximum(1e-6, 1 - cut[fringe]), 0, 255)

img = Image.fromarray(out.astype(np.uint8), "RGBA")
img = img.crop(img.split()[3].getbbox())

# square canvas with 4% room, so the brush ring is never flush with the edge
side = round(max(img.size) * 1.08)
canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))

canvas.save(OUT / "logo-mark.png")
for px in (76, 114, 152):
    canvas.resize((px, px), LANCZOS).save(OUT / f"logo-mark-{px}.png")
canvas.resize((32, 32), LANCZOS).save(OUT / "favicon-32.png")
canvas.resize((180, 180), LANCZOS).save(OUT / "apple-touch-icon.png")

alpha = np.array(canvas.split()[3])
print(f"bbox {img.size} -> {canvas.size}, transparent {(alpha == 0).mean():.1%}")
