# MoScribe

**Local speech to text with speaker separation and on-device summaries.**
Nothing leaves your machine — no API keys, no uploads, no accounts.

![MoScribe](docs/screenshot.png)

MoScribe turns a microphone or an audio file into a transcript that knows *who* is
speaking, then writes summaries, minutes and action items with a small model that
runs on the same CPU. It is one local web app: a FastAPI server and a no-build-step
interface, wrapping [VibeASR.cpp](https://github.com/microsoft/VibeASR.cpp) for
recognition and llama.cpp for text. Built by [kmail.at](https://kmail.at).

## Install

| Platform | Package | Then |
| --- | --- | --- |
| macOS (Apple silicon / Intel) | `MoScribe-<version>-macos-*.dmg` | drag to Applications |
| Windows 10/11 (x64) | `MoScribe-<version>-windows-x64-setup.exe` | run the installer |
| Linux (x64) | `MoScribe-<version>-linux-x64.tar.gz` | unpack, `./MoScribe` |
| Docker (server / NAS) | `docker compose up -d` | open `http://localhost:8770` |
| From source (any OS) | `pip install -e ".[dev]"` | `moscribe` |

Latest artifacts: [Releases](https://github.com/mokmail/moscribe/releases).

**First run downloads the models — about 2.8 GB, once:**

```bash
moscribe --fetch-models     # or:  python -m moscribe.fetch
moscribe --check            # confirm binaries + models are in place
```

The weights come straight from Hugging Face (VibeVoice-ASR-BitNet, BitNet b1.58 2B-4T,
WeSpeaker ResNet34). It is a download, not an upload.

## Use

Start it (`moscribe`, or the app icon); it opens `http://127.0.0.1:8770`.

- **Live** — press the orb or <kbd>Space</kbd>. Speech lands every 8–10 s. Press
  again to stop: the tail window still transcribes, and a `transcribing` chip stays
  up until the transcript is genuinely final.
- **Upload** — drop wav, mp3, m4a, flac, ogg or mp4 (max 600 MB).
- **Speakers** — WeSpeaker voiceprints plus clustering give each voice a colour and
  a name you can rename inline; names carry into copy/export.
- **Prompts** — ten presets (meeting, stand-up, interview, lecture, Austrian public
  sector, clinical, legal, support, podcast, workshop) or your own context, to bias
  recognition toward names and jargon.
- **Layouts** — the transcript follows the template: chat bubbles for a meeting,
  prose for a lecture, Q&A for an interview, cards or a compact log. Override in
  Settings → Prompt.
- **AI actions** — summarize, minutes, action items, clean up, translate, or ask your
  own question. Results stream into the side panel; copy, download, mail or share.
- **Machine panel** — live CPU/RAM sparklines, so you can see what the models cost.

Press <kbd>?</kbd> in the header for the in-app guide.

## How it works

```
mic / file
   │  24 kHz mono; windows cut at the quietest point near a boundary
   ▼
VAD ──► WeSpeaker ResNet34 (256-d voiceprints) ──► clustering = speakers
   │                                                    │
   ▼                                                    │
VibeASR.cpp (VibeVoice-ASR-BitNet, I8_S encoder + I2_S LM) ◄┘
   │  uploads are cut at speaker turns, so a label is right by construction
   ▼
stitching (sliding longest-run merge across the 2 s overlap)
   ▼
BitNet b1.58 2B-4T via llama.cpp → summaries, minutes, action items
```

| Piece | What it is | Size |
| --- | --- | --- |
| `asr_stream_server` | VibeASR.cpp streaming server (BitNet ASR) | 0.2 MB + dylibs |
| ASR weights | VibeVoice-ASR-BitNet VAE + LM (GGUF) | 1.6 GB |
| Speaker model | WeSpeaker ResNet34-LM (ONNX) | 25 MB |
| Text model | BitNet b1.58 2B-4T (GGUF, I2_S) | 1.1 GB |

About 0.3–0.6× real time for recognition on Apple silicon CPU; roughly 12 tokens/s
from the text model. Plain CPU code — no CUDA, no Metal requirement.

### Speaker separation

1. **VAD** — short-time energy with an adaptive noise floor splits audio into
   utterances at pauses.
2. **Embedding** — L2-normalised 256-d voiceprint per utterance (Kaldi-style 80-bin
   fbank, 16 kHz, per-utterance mean normalisation). Measured margin on four
   synthetic voices: +0.11 cosine within-vs-across speaker.
3. **Clustering** — average-linkage agglomerative + spherical k-means refinement,
   hard speaker cap, pruning of tiny clusters. Spans under 0.8 s never define a
   voice; they inherit their nearest neighbour.
4. **Alignment** — uploads are cut into windows *at speaker turns*, so a window
   belongs to one person by construction. Live windows can span two voices; those
   are split word-proportionally over the window's diarised spans.

Live dictation uses a stable online clusterer (persistent centroids, creation-order
labels) — re-clustering renumbered speakers mid-transcript.

### Text actions

Summaries, minutes, action items, cleanup and translation run on **BitNet b1.58 2B-4T**
through the llama.cpp build that ships with VibeASR.cpp (its ggml already registers
`LLM_ARCH_BITNET_B158` and has I2_S kernels, so no second toolchain). The app starts
`llama-server` on 127.0.0.1:8771 and warms it at boot.

Actions: `summarize`, `minutes`, `actions`, `rewrite`, `translate_en`, `translate_de`,
`custom`. English output is accurate and keeps speaker names. German *generation* is
not usable at 2.4 B — the model echoes or invents facts — so output stays English and
translation is marked experimental. Point `VIBESCRIBE_LLM_MODEL` at a larger
multilingual instruct GGUF (e.g. Qwen2.5-7B-Instruct Q4) for real German.

## Privacy

Audio, transcripts, summaries and history live in your user data directory:

| OS | Path |
| --- | --- |
| macOS | `~/Library/Application Support/MoScribe` |
| Windows | `%LOCALAPPDATA%\MoScribe` |
| Linux | `~/.local/share/moscribe` (or `$XDG_DATA_HOME`) |
| Docker | `/data` (the volume) |

Deleting a session in History removes its files. The only network access is the
one-time model download.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | engine + speaker-model status |
| `GET /api/system` | CPU/RAM/load samples + short history |
| `POST /api/engine/warmup` | load models ahead of the first request |
| `POST /api/transcribe` | multipart upload → job id (progress over WS) |
| `POST /api/transcribe-sync` | upload → finished transcript (scripts, CI) |
| `WS /ws/jobs/{id}` | token / chunk / phase / done stream |
| `WS /ws/live` | 24 kHz mono WAV windows in, turns + `final` out |
| `POST /api/ai/stream` | `{action, transcript, extra, speakers}` → SSE |
| `GET|DELETE /api/history[/*]` | transcript history |

Form fields (`/api/transcribe`, `/api/transcribe-sync`): `threads`, `decoding`
(`greedy`|`sampling`), `temperature`, `top_p`, `context`, `prompt_format`
(`text`|`json`), `chunk_sec`, `overlap_sec`, `speaker_separation` (`on`|`off`),
`speaker_sensitivity` (0.55–0.95, default 0.78), `max_speakers` (2–8, default 6).

## Configuration

| Variable | Effect |
| --- | --- |
| `MOSCRIBE_DATA` | data directory (history, jobs, downloads) |
| `MOSCRIBE_MODELS` | model root |
| `MOSCRIBE_BIN` | engine binary directory |
| `MOSCRIBE_HOST` / `MOSCRIBE_PORT` | bind address / port |
| `VIBEASR_BIN`, `VIBEASR_MODELS` | override individual engine paths |
| `VIBESCRIBE_LLM_BIN`, `VIBESCRIBE_LLM_MODEL`, `VIBESCRIBE_LLM_PORT` | text engine |
| `MOSCRIBE_SKIP_FETCH=1` | Docker: never download models at start |

## Develop

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/moscribe --port 8770 --no-browser
```

Frontend is `moscribe/static/` — plain HTML/CSS/JS, no bundler, no npm. Backend is
`moscribe/`, one module per job:

| File | Role |
| --- | --- |
| `main.py` | routes, job runner, live WebSocket |
| `asr_engine.py` | supervises `asr_stream_server`, chunk planning, stitching |
| `diarization.py` | VAD, span grouping, clustering, turn building |
| `spk_embed.py` | WeSpeaker ONNX embedder (Kaldi-style fbank) |
| `llm.py` | supervises `llama-server`, prompt templates per action |
| `paths.py` | app/data/model locations, frozen vs checkout |
| `fetch.py` | first-run downloads |
| `moscribe_cli.py` | the `moscribe` command |

Building the engines, packaging installers and the release pipeline:
[docs/BUILD.md](docs/BUILD.md). Checks:

```bash
.venv/bin/python -m py_compile moscribe/*.py
node --check moscribe/static/app.js
.venv/bin/python scripts/test_diarization.py     # synthesises a 3-voice conversation
```

## License

MIT for MoScribe's own code — see [LICENSE](LICENSE). Upstream models and libraries
keep their own terms: [THIRD_PARTY.md](THIRD_PARTY.md). The kmail mark is not covered
by the MIT grant.
