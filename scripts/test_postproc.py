"""Post-processing and audio-conditioning regression test — no engine, no network.

Covers the three things that silently wreck a transcript without raising:
decoder loops, hallucinated caption boilerplate, and windows too quiet to hold
speech. Also checks the stitcher still behaves after `tidy()` got stricter.

Run from the repo root:  .venv/bin/python scripts/test_postproc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moscribe.asr_engine import (  # noqa: E402
    collapse_repeats,
    condition_audio,
    is_silent,
    stitch_overlap,
    strip_artifacts,
    tidy,
)

SR = 24000
FAILS: list[str] = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(f"  {'ok ' if ok else '✗  '}{name}")
    if not ok:
        print(f"      got : {got!r}\n      want: {want!r}")
        FAILS.append(name)


def test_tidy() -> None:
    print("tidy / loops / artifacts")
    check("ties sentence punctuation",
          tidy("we ship in November.Before that, the tests run."),
          "we ship in November. Before that, the tests run.")
    check("collapses a word loop",
          tidy("so we need to check the the the the numbers again"),
          "so we need to check the numbers again")
    check("keeps rhetorical repetition of three",
          collapse_repeats("no no no"), "no no no")
    check("collapses a long word loop",
          collapse_repeats("no no no no no no"), "no")
    check("collapses a phrase loop",
          collapse_repeats("we need to check we need to check we need to check the logs"),
          "we need to check the logs")
    check("does not touch a single occurrence",
          collapse_repeats("we need to check the logs tomorrow morning"),
          "we need to check the logs tomorrow morning")
    check("drops a caption credit",
          strip_artifacts("Subtitles by the Amara.org community"), "")
    check("drops a subscribe outro",
          strip_artifacts("Please like and subscribe to the channel"), "")
    check("drops a music marker", strip_artifacts("[Music]"), "")
    check("keeps the sentence around a marker",
          strip_artifacts("So [Music] where were we"), "So where were we")
    check("keeps [inaudible] — it carries meaning",
          strip_artifacts("the number is [inaudible] unfortunately"),
          "the number is [inaudible] unfortunately")
    check("empty stays empty", tidy("   "), "")


def test_stitch() -> None:
    print("\nstitch_overlap")
    tail = "Perfect. The main question is when I"
    new = "Perfect. The main question is whether we ship in November."
    check("drops the re-said overlap", stitch_overlap(tail, new),
          "whether we ship in November.")
    check("keeps genuinely new text",
          stitch_overlap("We agreed on the budget.", "The vendor needs a signature."),
          "The vendor needs a signature.")


def test_conditioning() -> None:
    print("\ncondition_audio / is_silent")
    rng = np.random.default_rng(0)
    t = np.arange(SR * 2) / SR
    speech = (0.05 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)))
    speech += rng.normal(0, 0.002, speech.size).astype(np.float32)

    quiet_src = speech / max(float(np.max(np.abs(speech))), 1e-9) * 0.03  # -30 dBFS
    quiet = condition_audio(quiet_src.astype(np.float32), SR)
    check("amplifies a quiet recording", bool(np.max(np.abs(quiet)) > 0.5), True)
    check("does not clip", bool(np.max(np.abs(quiet)) <= 1.0), True)

    offset = (speech + 0.3).astype(np.float32)
    fixed = condition_audio(offset, SR)
    check("removes DC offset", bool(abs(float(np.mean(fixed))) < 0.01), True)

    rumble = (0.3 * np.sin(2 * np.pi * 45 * t)).astype(np.float32)
    # Neutralise the level stage so this measures the filter alone.
    hp = condition_audio(rumble, SR, target_peak=float(np.max(np.abs(rumble))))
    check("attenuates sub-70 Hz rumble",
          bool(np.sqrt(np.mean(hp ** 2)) < 0.35 * np.sqrt(0.5)), True)

    check("silence is detected", is_silent(np.zeros(SR, dtype=np.float32)), True)
    check("room tone is detected", is_silent(rng.normal(0, 0.0008, SR).astype(np.float32)), True)
    check("speech is not called silence", is_silent(speech.astype(np.float32)), False)


def main() -> int:
    test_tidy()
    test_stitch()
    test_conditioning()
    print("\nPASS" if not FAILS else f"\nFAIL ({len(FAILS)}): {', '.join(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    raise SystemExit(main())
