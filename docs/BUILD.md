# Building MoScribe

Everything here runs on a normal machine with a compiler, or on GitHub Actions with
none. The release workflow does all of it for you — this document is for when it
does not.

## Prerequisites

| Need | Why | macOS | Linux | Windows |
| --- | --- | --- | --- | --- |
| Python 3.10+ | the app | `brew install python` | `apt install python3` | python.org |
| CMake 3.20+ | engine build | `brew install cmake` | `apt install cmake` | winget |
| C++17 compiler | engine build | Xcode CLT | `build-essential` | Visual Studio 2022 |
| Git | fetching VibeASR.cpp | ✓ | ✓ | ✓ |

## 1. Build the native engines

```bash
git clone --depth 1 https://github.com/microsoft/VibeASR.cpp
cd VibeASR.cpp

# streaming ASR server
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF
cmake --build build --target asr_stream_server -j4

# llama.cpp chat server (ships inside the same checkout)
cmake -S . -B build-llm -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON -DGGML_OPENMP=OFF
cmake --build build-llm --target llama-server -j4
```

Out: `build/bin/asr_stream_server` and `build-llm/bin/llama-server`.

## 2. Make the binaries portable

They link `@rpath/libllama.dylib` and `libggml.dylib`, and the build tree rpaths
point at absolute paths — which do not exist inside an installer. This copies the
libs beside the executables and rewrites the rpaths to `@loader_path` (macOS) or
`$ORIGIN` (Linux):

```bash
python packaging/prepare_bin.py --src ~/apps/VibeASR.cpp --out bin
otool -l bin/asr_stream_server | grep -A2 LC_RPATH    # should show only @loader_path
```

Windows: the DLLs land next to the exe already; the script copies them into `bin/`.

## 3. Bundle the app

```bash
pip install -e ".[dev]"          # pulls pyinstaller + pillow
python packaging/make_icons.py   # app.ico / app.icns from the brand mark
pyinstaller --clean --noconfirm packaging/moscribe.spec
```

Output: `dist/MoScribe/` (and `dist/MoScribe.app` on macOS). Layout inside:

```
MoScribe                    the launcher (the .app on macOS)
_internal/static/           the interface
_internal/bin/              engines + their dylibs
_internal/…                 Python runtime
```

`MOSCRIBE_BIN_DIR` points the spec at another binary directory;
`MOSCRIBE_BUNDLE_MODELS=1` embeds `models/` too (about 2.8 GB — only for offline builds).

## 4. Installers

| Platform | Command | Result |
| --- | --- | --- |
| macOS | `./packaging/macos/make_dmg.sh 1.0.0` | `dist/MoScribe-1.0.0-macos-arm64.dmg` |
| Windows | `iscc packaging\windows\moscribe.iss` | `MoScribe-1.0.0-windows-x64-setup.exe` |
| Linux | `tar -C dist -czf dist/MoScribe-1.0.0-linux-x64.tar.gz MoScribe` | tarball |

The DMG is ad-hoc signed. For distribution outside your own machine, sign with a
Developer ID (`MACOS_SIGN_IDENTITY` is read by the spec) and notarise:

```bash
xcrun notarytool submit dist/MoScribe-1.0.0-macos-arm64.dmg \
  --apple-id you@example.com --team-id TEAMID --password "$APP_PASSWORD" --wait
```

Windows: unsigned installers trigger SmartScreen until you use a code-signing
certificate (`signtool sign /fd sha256 /tr ...`).

## 5. Docker

```bash
docker build -t moscribe .
docker run --rm -p 8770:8770 -v moscribe-data:/data moscribe
```

The image builds the engines in a builder stage, then serves with the models in the
`/data` volume. First start downloads ~2.8 GB into the volume.

## 6. Release

```bash
git tag v1.0.0 && git push origin v1.0.0
```

`.github/workflows/release.yml` builds all four targets, uploads artifacts and
publishes a GitHub Release with:

- `MoScribe-<v>-macos-{arm64,x64}.dmg`
- `MoScribe-<v>-windows-x64-setup.exe` and `.zip`
- `MoScribe-<v>-linux-x64.tar.gz`
- `moscribe-<platform>-bin.zip` — binaries only, which `moscribe --fetch-models`
  pulls for a pip install

### Status of each target

| Target | State |
| --- | --- |
| macOS arm64 | **verified locally** — bundle launches, engine starts from `_internal/bin`, transcription works, DMG mounts |
| macOS x64, Linux x64, Windows x64 | pipeline written, first CI run still to confirm the C++ build flags |

The matrix marks the unverified targets `continue-on-error: true`, so a failure
there does not block a release; the job log says which step failed.

## Troubleshooting

- **`missing binary: asr_stream_server`** — the bundle was built without `bin/`, or
  the engines failed to compile. Run `moscribe --fetch-models` (it pulls the published
  `-bin.zip`) or repeat step 1.
- **Engine dies at start, `Library not loaded`** — rpaths were not rewritten; redo
  step 2 and check `otool -l`.
- **First transcription is slow** — models load lazily; the app warms them at boot,
  and `POST /api/engine/warmup` does it manually.
- **`moscribe` not on PATH after pip install** — use `python -m moscribe_cli`.
