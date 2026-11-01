"""Fail if the built wheel is missing the interface or brand assets."""
from __future__ import annotations

import glob
import zipfile

wheels = glob.glob("dist/*.whl")
if not wheels:
    print("no wheel in dist/")
    raise SystemExit(1)

wheel = wheels[0]
names = zipfile.ZipFile(wheel).namelist()
required = (
    "moscribe/static/index.html",
    "moscribe/static/app.js",
    "moscribe/static/styles.css",
    "moscribe/static/brand/logo-mark.png",
)
missing = [r for r in required if not any(n.endswith(r) for n in names)]
for m in missing:
    print(f"missing from wheel: {m}")

print(f"{wheel}: {len(names)} files, {len(missing)} missing")
raise SystemExit(1 if missing else 0)
