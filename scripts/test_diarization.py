"""Diarisation regression test — no fixtures, no network.

Builds a 3-voice conversation with macOS `say`, then checks that:
  * every utterance maps to exactly one speaker label,
  * three distinct speakers are found,
  * the same person keeps the same label across the recording.

Run from the repo root:  .venv/bin/python scripts/test_diarization.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moscribe.diarization import diarize_audio, group_turns, render_turns, vad_segments  # noqa: E402
from moscribe.spk_embed import EMBEDDER  # noqa: E402

SR = 24000
VOICES = {"A": "Samantha (English (US))", "B": "Daniel (English (UK))", "C": "Moira"}
SCRIPT = [
    ("A", "Good morning everyone, thanks for joining the call on such short notice."),
    ("B", "No problem at all, I had a look at the numbers you sent over last night."),
    ("C", "Before we dive in, can we confirm who is taking the minutes today?"),
    ("A", "I will do that, and I will share the document before the end of the day."),
    ("B", "Perfect. The main question is whether we ship the migration in November."),
    ("C", "My team can support it, but we need two extra weeks for the load testing."),
]


def synth(voice: str, text: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        aiff, wav = Path(tmp) / "a.aiff", Path(tmp) / "a.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@24000", "-c", "1",
                        str(aiff), str(wav)], check=True)
        with wave.open(str(wav)) as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def build() -> tuple[np.ndarray, list[tuple[str, float, float]]]:
    pieces = [np.zeros(int(0.35 * SR), dtype=np.float32)]
    truth: list[tuple[str, float, float]] = []
    t = 0.35
    for speaker, text in SCRIPT:
        audio = synth(VOICES[speaker], text)
        truth.append((speaker, t, t + len(audio) / SR))
        pieces.append(audio)
        t += len(audio) / SR
        gap = np.zeros(int(0.55 * SR), dtype=np.float32)
        pieces.append(gap)
        t += 0.55
    return np.concatenate(pieces), truth


def main() -> int:
    if not EMBEDDER.available:
        print("FAIL — speaker model unavailable:", EMBEDDER.status["error"])
        return 1

    audio, truth = build()
    print(f"audio {len(audio) / SR:.1f}s · {len(SCRIPT)} utterances · "
          f"vad spans {len(vad_segments(audio, SR))}")

    spans = diarize_audio(audio, SR)
    label_of: dict[str, set[int]] = {}
    for speaker, ts, te in truth:
        votes: dict[int, float] = {}
        for s, e, lab in spans:
            overlap = min(e, te) - max(s, ts)
            if overlap > 0:
                votes[lab] = votes.get(lab, 0) + overlap
        got = max(votes, key=votes.get) if votes else -1
        label_of.setdefault(speaker, set()).add(got)
        print(f"  {speaker} {ts:5.2f}-{te:5.2f} -> label {got}")

    consistent = sum(1 for v in label_of.values() if len(v) == 1)
    distinct = {next(iter(v)) for v in label_of.values() if len(v) == 1}

    # Turn rendering smoke test.
    turns = group_turns([{"speaker": 0, "text": "hello", "start": 0, "end": 1},
                         {"speaker": 0, "text": "again", "start": 1, "end": 2},
                         {"speaker": 1, "text": "bye", "start": 2, "end": 3}])
    rendered = render_turns(turns)
    print("\nrendered:\n" + rendered)
    print("turns:", json.dumps(turns))

    ok = consistent == len(label_of) and len(distinct) == 3 and len(turns) == 2
    print("\nPASS" if ok else "\nFAIL",
          f"— {consistent}/{len(label_of)} speakers kept one label, "
          f"distinct labels {sorted(distinct)} (expect 3), turns {len(turns)} (expect 2)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
