"""Advanced diarisation regression test.

Three properties the first test suite cannot see, each of which produced a real
transcript bug:

  1. **Fast turn-taking** — two people alternating with a 60 ms gap land in ONE
     VAD span. Without boundary refinement that span gets one label, so half the
     words are attributed to the wrong person.
  2. **No phantom speakers** — an ambiguous match in live dictation used to start
     a new speaker instead of accepting the closest familiar voice.
  3. **Talk time** — stats must add up, and must be labelled *estimated* when the
     turns carry no clock (live dictation).

Run from the repo root:  .venv/bin/python scripts/test_diarization_advanced.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moscribe.diarization import (  # noqa: E402
    HOP,
    SpeakerCluster,
    adaptive_threshold,
    diarize_audio_detailed,
    frame_energy,
    refine_speaker_bounds,
    speaker_stats,
    vad_segments,
)
from moscribe.spk_embed import EMBEDDER  # noqa: E402

SR = 24000
FAILS: list[str] = []

VOICES = {"A": "Samantha (English (US))", "B": "Daniel (English (UK))"}

# Short, back-to-back turns: this is the case VAD cannot separate.
SCRIPT = [
    ("A", "So the migration plan is ready"),
    ("B", "Right, and the load tests are still running"),
    ("A", "Which means we cannot ship before Friday"),
    ("B", "Then we push it to the following week"),
]


def synth(voice: str, text: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "u.wav"
        subprocess.run(["say", "-v", voice, "-o", str(out),
                        "--data-format=LEI16@24000", text], check=True)
        with wave.open(str(out)) as w:
            raw = w.readframes(w.getnframes())
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def trim_voiced(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """`say` pads each file with silence; without trimming, the gaps are fake."""
    energy = frame_energy(x)
    if energy.size == 0:
        return x
    thr = max(float(np.percentile(energy, 95)) * 0.15, 0.004)
    idx = np.where(energy > thr)[0]
    if idx.size == 0:
        return x
    a = max(0, int(idx[0]) * HOP - int(0.05 * sr))
    b = min(x.size, (int(idx[-1]) + 1) * HOP + int(0.05 * sr))
    return x[a:b]


def build_conversation(gap_sec: float) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
    """Concatenate the script with `gap_sec` of silence between turns."""
    chunks: list[np.ndarray] = [np.zeros(int(0.30 * SR), dtype=np.float32)]
    spans: list[tuple[str, float, float]] = []
    pos = 0.30
    for speaker, text in SCRIPT:
        clip = trim_voiced(synth(VOICES[speaker], text))
        spans.append((speaker, pos, pos + len(clip) / SR))
        chunks.append(clip)
        pos += len(clip) / SR
        gap = np.zeros(int(gap_sec * SR), dtype=np.float32)
        chunks.append(gap)
        pos += len(gap) / SR
    return np.concatenate(chunks), spans


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'ok ' if ok else '✗  '}{name}" + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


def test_fast_turns() -> None:
    print("fast turn-taking (60 ms gaps, VAD merges them)")
    audio, truth = build_conversation(0.06)
    coarse = vad_segments(audio, SR)
    check("VAD merges the turns into fewer spans than utterances",
          len(coarse) < len(truth), f"{len(coarse)} spans for {len(truth)} utterances")

    refined = refine_speaker_bounds(audio, coarse, SR)
    check("boundary refinement splits the merged span",
          len(refined) > len(coarse), f"{len(coarse)} -> {len(refined)}")

    detail = diarize_audio_detailed(audio, SR)
    speakers = {d["speaker"] for d in detail if d["speaker"] >= 0}
    check("two speakers found (not one)", len(speakers) == 2,
          f"speakers={sorted(speakers)}")

    # Every ground-truth utterance must map to one label, and the labels must
    # alternate A,B,A,B — the point of the split is that B is not called A.
    assigned: list[tuple[str, int]] = []
    for spk, start, end in truth:
        mid = (start + end) / 2
        hit = [d for d in detail if d["start"] <= mid <= d["end"] and d["speaker"] >= 0]
        if hit:
            assigned.append((spk, hit[0]["speaker"]))
    by_speaker: dict[str, set[int]] = {}
    for spk, lab in assigned:
        by_speaker.setdefault(spk, set()).add(lab)
    check("each voice keeps exactly one label",
          all(len(v) == 1 for v in by_speaker.values()) and len(assigned) == len(truth),
          f"mapping={assigned}")
    distinct = {next(iter(v)) for v in by_speaker.values() if v}
    check("the two voices get different labels",
          len(distinct) == 2, f"labels={sorted(distinct)}")

    conf = [d["confidence"] for d in detail if d["speaker"] >= 0]
    check("confidence is reported per span",
          bool(conf) and all(0.0 <= c <= 1.0 for c in conf),
          f"range={min(conf):.2f}-{max(conf):.2f}" if conf else "none")
    starts = [d["start"] for d in detail]
    ends = [d["end"] for d in detail]
    check("spans do not overlap (VAD chops with a 0.2 s overlap)",
          all(ends[i] <= starts[i + 1] + 1e-6 for i in range(len(starts) - 1)),
          f"spans={list(zip(starts, ends))}")
    thr = adaptive_threshold(
        EMBEDDER.embed_many([audio[int(d["start"] * SR):int(d["end"] * SR)] for d in detail], SR),
        [d["duration"] for d in detail])
    print(f"      adaptive threshold on this take: {thr:.3f}")


def test_phantom_speakers() -> None:
    print("\nlive clustering (ambiguous match must not invent a speaker)")
    rng = np.random.default_rng(7)

    def unit(v: np.ndarray) -> np.ndarray:
        return (v / np.linalg.norm(v)).astype(np.float32)

    base = unit(rng.normal(size=256))

    def orth(*refs: np.ndarray) -> np.ndarray:
        v = rng.normal(size=256).astype(np.float32)
        for r in refs:
            v = v - float(v @ r) * r
        return unit(v)

    other = orth(base)
    spare = orth(base, other)

    c = SpeakerCluster(threshold=0.78, max_speakers=6)
    c.add(base, 4.0)
    c.add(other, 4.0)
    check("two voices registered", c.speaker_count == 2)

    # A span 0.50 from voice A and 0.30 from voice B: familiar, but not confident
    # enough to be a match and not unlike enough to be a new person.
    amb = unit(0.50 * base + 0.30 * other + 0.806 * spare)
    before = float(c.centroids[0] @ base)
    lab = c.add(amb, 2.0)
    after = float(c.centroids[0] @ base)
    check("ambiguous span keeps the familiar speaker",
          lab == 0 and c.speaker_count == 2, f"label={lab} speakers={c.speaker_count}")
    check("ambiguous span leaves the centroid alone", abs(after - before) < 1e-6,
          f"{before:.4f} -> {after:.4f}")

    lab2 = c.add(orth(base, other, spare), 3.0)
    check("a genuinely new voice still creates a speaker",
          lab2 == 2 and c.speaker_count == 3, f"label={lab2} speakers={c.speaker_count}")

    short = SpeakerCluster(threshold=0.78, max_speakers=6)
    short.add(base, 4.0)
    lab3 = short.add(orth(base), 0.4)
    check("a 0.4 s fragment never creates a speaker",
          lab3 == 0 and short.speaker_count == 1, f"label={lab3}")


def test_speaker_stats() -> None:
    print("\ntalk-time statistics")
    turns = [
        {"speaker": 0, "text": "one two three four", "start": 0.0, "end": 4.0},
        {"speaker": 1, "text": "five six", "start": 4.0, "end": 6.0},
        {"speaker": 0, "text": "seven eight nine", "start": 6.0, "end": 9.0},
    ]
    rows = speaker_stats(turns)
    check("two speakers reported", len(rows) == 2)
    check("first speaker has the most time", rows[0]["speaker"] == 0 and rows[0]["seconds"] == 7.0,
          f"{rows}")
    check("shares add up to one",
          abs(sum(r["share"] for r in rows) - 1.0) < 0.01, f"{[r['share'] for r in rows]}")
    check("wall-clock data is not marked estimated",
          not any(r["estimated"] for r in rows))

    live = speaker_stats([{"speaker": 0, "text": "a b c d e f g h i j", "start": 1.0, "end": None}])
    check("live turns fall back to an estimate and say so",
          live[0]["estimated"] and live[0]["seconds"] > 0, f"{live}")


def main() -> int:
    test_fast_turns()
    test_phantom_speakers()
    test_speaker_stats()
    print("\nPASS" if not FAILS else f"\nFAIL ({len(FAILS)}): {', '.join(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
