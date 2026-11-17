"""Speaker separation for VibeScribe.

Pipeline (no torch, no pyannote, no HF token):

  1. **VAD** — short-time energy with an adaptive noise floor splits the audio
     into utterances separated by pauses.
  2. **Embedding** — `spk_embed.SpeakerEmbedder` (WeSpeaker ResNet34, ONNX)
     turns each utterance into an L2-normalised 256-d voiceprint.
  3. **Clustering** — average-linkage agglomerative clustering on cosine
     similarity. Offline for uploads (globally consistent labels), incremental
     with re-clustering for live dictation (labels stay stable as history grows).
  4. **Alignment** — every transcript window is split word-proportionally
     across the speaker spans it overlaps, so a 12 s chunk holding two voices
     becomes two turns instead of one mislabelled blob.

The MFCC heuristic from the first version is gone: it could not separate two
voices unless they were obviously different, which is why speaker tags looked
random.
"""

from __future__ import annotations

import re

import numpy as np

from .spk_embed import EMBEDDER

SR = 24000
FRAME = 480          # 20 ms analysis window (energy only)
HOP = 240            # 10 ms hop

DEFAULT_THRESHOLD = 0.78     # cosine similarity needed to call it the same voice
MIN_EMBED_SEC = 0.45         # shorter spans are unreliable -> inherit neighbour

# Speaker-change detection inside one merged span. VAD splits at pauses; two people
# taking turns with less than `min_silence` between them land in a single span, and
# one label then covers both. Probe windows find where the voice actually changes.
PROBE_WIN = 0.9              # seconds per probe window
PROBE_HOP = 0.3              # probe stride
CHANGE_SIM = 0.55            # adjacent probes below this = a change point candidate
SAME_VOICE_SIM = 0.55        # halves of a genuine single-speaker span score above this
LOCAL_PROBES = 3             # probes each side used to confirm a cut
MAX_CUTS = 8                 # hard ceiling: no span holds more turns than this
REFINE_MIN_SEC = 2.6         # shorter spans cannot hold two turns
REFINE_MIN_PIECE = 1.0       # never emit a sliver around a cut


# --------------------------------------------------------------------- VAD

def _frames(audio: np.ndarray) -> np.ndarray:
    if audio.size < FRAME:
        return np.zeros((0, FRAME), dtype=np.float32)
    n = 1 + (audio.size - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return audio[idx]


def frame_energy(audio: np.ndarray) -> np.ndarray:
    f = _frames(audio)
    if f.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.sqrt(np.mean(f.astype(np.float32) ** 2, axis=1))


def vad_segments(audio: np.ndarray, sr: int = SR, min_speech: float = 0.35,
                 min_silence: float = 0.25, max_len: float = 12.0) -> list[tuple[float, float]]:
    """Speech spans in seconds, split at pauses (and inside long monologues)."""
    energy = frame_energy(audio)
    if energy.size == 0:
        return []
    hop_s = HOP / sr
    noise = float(np.percentile(energy, 10))
    peak = float(np.percentile(energy, 95))
    thresh = max(noise * 2.2, peak * 0.10, 0.0035)
    voiced = energy > thresh

    spans: list[list[int]] = []
    run = None
    for i, v in enumerate(voiced):
        if v and run is None:
            run = i
        elif not v and run is not None:
            spans.append([run, i])
            run = None
    if run is not None:
        spans.append([run, len(voiced)])

    merged: list[list[int]] = []
    for s, e in spans:
        if merged and (s - merged[-1][1]) * hop_s < min_silence:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    out: list[tuple[float, float]] = []
    total = audio.size / sr
    for s, e in merged:
        if (e - s) * hop_s < min_speech:
            continue
        start_s = max(0.0, s * hop_s - 0.10)
        end_s = min(total, e * hop_s + 0.10)
        while end_s - start_s > max_len:           # chop monologues at a hard cap
            out.append((start_s, start_s + max_len))
            start_s += max_len - 0.2
        out.append((start_s, end_s))
    return out


# ---------------------------------------------------------------- clustering

def _cosine_matrix(embs: list[np.ndarray]) -> np.ndarray:
    m = np.vstack(embs)
    return np.clip(m @ m.T, -1.0, 1.0)


def _refine(vecs: np.ndarray, labels: list[int], iters: int = 4) -> list[int]:
    """Spherical k-means refinement on top of the agglomerative init.

    Agglomerative clustering alone produces a few stray clusters when short
    spans land between two voices; a couple of centroid reassignment rounds
    pulls them back to the dominant speakers.
    """
    lab = np.asarray(labels)
    uniq = [u for u in sorted(set(labels)) if u >= 0]
    if len(uniq) < 2:
        return labels
    for _ in range(iters):
        cents = []
        for u in uniq:
            c = vecs[lab == u].mean(axis=0)
            n = np.linalg.norm(c)
            cents.append(c / n if n > 1e-9 else c)
        cents_a = np.vstack(cents)
        sims = vecs @ cents_a.T
        new = np.asarray(uniq)[sims.argmax(axis=1)]
        new[labels == -1] = -1
        if np.array_equal(new, lab):
            break
        lab = new
    return [int(x) for x in lab]


def _renumber(labels: list[int]) -> list[int]:
    """Relabel by first appearance so 'Speaker 1' is whoever spoke first."""
    order: dict[int, int] = {}
    out = []
    for lab in labels:
        if lab < 0:
            out.append(-1)
            continue
        if lab not in order:
            order[lab] = len(order)
        out.append(order[lab])
    return out


def _unit_centroids(vecs: np.ndarray, labels, weights=None) -> dict[int, np.ndarray]:
    lab = np.asarray(labels)
    w = np.ones(len(lab)) if weights is None else np.asarray(weights, dtype=float)
    out: dict[int, np.ndarray] = {}
    for u in sorted({int(x) for x in lab if x >= 0}):
        m = lab == u
        if not m.any():
            continue
        c = (vecs[m] * w[m][:, None]).sum(axis=0) / max(float(w[m].sum()), 1e-9)
        n = float(np.linalg.norm(c))
        out[u] = c / n if n > 1e-9 else c
    return out


def _reassign(vecs: np.ndarray, labels: list[int], weights: np.ndarray,
              iters: int = 3, margin: float = 0.04) -> list[int]:
    """Duration-weighted centroid reassignment that abstains on a coin flip.

    Plain argmax flips a span between two speakers whose centroids differ by 0.02
    cosine, and each flip is a wrong label in the transcript. A span only moves
    when the new centroid beats its current one by `margin` — otherwise staying put
    is the safer answer. Long spans also carry more weight when centroids are
    rebuilt, so a 12 s monologue defines its speaker instead of averaging with a
    dozen clipped "mm-hm"s.
    """
    lab = np.asarray(labels)
    cents = _unit_centroids(vecs, lab, weights)
    if len(cents) < 2:
        return [int(x) for x in lab]
    keys = sorted(cents)
    A = np.vstack([cents[k] for k in keys])
    pos = {k: i for i, k in enumerate(keys)}

    for _ in range(iters):
        sims = vecs @ A.T
        best = sims.argmax(axis=1)
        new = np.asarray([keys[b] for b in best])
        old_sim = np.array([sims[i, pos[int(lab[i])]] if int(lab[i]) in pos else -1.0
                            for i in range(len(sims))])
        # Only switch on a clear win; an unknown/absent old label may be overwritten.
        clear = (old_sim < 0) | ((sims.max(axis=1) - old_sim) >= margin)
        new[~clear] = lab[~clear]
        new[lab == -1] = -1
        if np.array_equal(new, lab):
            break
        lab = new
        cents = _unit_centroids(vecs, lab, weights)
        if len(cents) < 2:
            break
        keys = sorted(cents)
        A = np.vstack([cents[k] for k in keys])
        pos = {k: i for i, k in enumerate(keys)}
    return [int(x) for x in lab]


def cluster_with_confidence(embs: list[np.ndarray | None], durations: list[float] | None = None,
                            threshold: float = DEFAULT_THRESHOLD, max_speakers: int = 6,
                            prune_below: float = 2.0) -> tuple[list[int], list[float]]:
    """Cluster embeddings; return labels plus a per-unit confidence in [0, 1].

    ``None`` embeddings inherit the temporally nearest labelled span. Short spans
    are attached by similarity when that is decisive and by adjacency otherwise;
    clusters holding less than ``prune_below`` seconds of speech are folded into
    the nearest real cluster — that is what stops coughs and door slams from
    inventing speakers.
    """
    idx = [i for i, e in enumerate(embs) if e is not None]
    labels = [-1] * len(embs)
    conf = [0.0] * len(embs)
    if not idx:
        return labels, conf

    # Short utterances ("yes", "mm-hm") carry almost no speaker information and
    # used to drag whole neighbours into the wrong cluster. Cluster on the
    # reliable spans only, then attach the short ones.
    RELIABLE = 0.8
    if durations is not None:
        reliable = [i for i in idx if durations[i] >= RELIABLE]
        short = [i for i in idx if durations[i] < RELIABLE]
    else:
        reliable, short = idx, []
    if not reliable:
        reliable, short = idx, []

    vecs = np.vstack([embs[i] for i in reliable])
    sim = np.clip(vecs @ vecs.T, -1.0, 1.0)
    n = len(vecs)

    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > 1:
        best = (-2.0, -1, -1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = float(sim[np.ix_(clusters[i], clusters[j])].mean())
                if s > best[0]:
                    best = (s, i, j)
        if best[0] < threshold:
            break
        _, i, j = best
        clusters[i] = clusters[i] + clusters[j]
        clusters.pop(j)
        if len(clusters) <= 1:
            break

    # Hard cap on speaker count: merge the closest pair until we fit.
    while len(clusters) > max_speakers:
        best = (-2.0, 0, 1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = float(sim[np.ix_(clusters[i], clusters[j])].mean())
                if s > best[0]:
                    best = (s, i, j)
        _, i, j = best
        clusters[i] = clusters[i] + clusters[j]
        clusters.pop(j)

    clusters.sort(key=lambda c: min(c))
    raw = [-1] * n
    for label, members in enumerate(clusters):
        for k in members:
            raw[k] = label
    raw = _refine(vecs, raw)

    # Fold clusters that hold too little speech.
    if durations is not None:
        dur = np.asarray([durations[reliable[k]] for k in range(n)], dtype=float)
        while True:
            uniq = sorted({lab for lab in raw if lab >= 0})
            if len(uniq) <= 1:
                break
            weights = {u: float(dur[np.asarray(raw) == u].sum()) for u in uniq}
            weak = min(uniq, key=lambda u: weights[u])
            if weights[weak] >= prune_below:
                break
            cents = {}
            for u in uniq:
                c = vecs[np.asarray(raw) == u].mean(axis=0)
                cents[u] = c / max(1e-9, np.linalg.norm(c))
            target, best_sim = None, -2.0
            for u in uniq:
                if u == weak:
                    continue
                s = float(np.dot(cents[weak], cents[u]))
                if s > best_sim:
                    target, best_sim = u, s
            if target is None:
                break
            raw = [target if lab == weak else lab for lab in raw]

    # Duration-weighted reassignment with an abstain rule, then renumber so
    # "Speaker 1" is whoever spoke first.
    weights = (np.asarray([durations[reliable[k]] for k in range(n)], dtype=float)
               if durations is not None else np.ones(n))
    raw = _reassign(vecs, [int(x) for x in raw], weights)
    raw = _renumber(raw)

    cents = _unit_centroids(vecs, raw, weights)
    keys = sorted(cents)
    if len(keys) > 1:
        A = np.vstack([cents[k] for k in keys])
        sims = vecs @ A.T
        order = np.argsort(-sims, axis=1)
        for pos, i in enumerate(reliable):
            best_i = int(order[pos, 0])
            second_sim = float(sims[pos, order[pos, 1]]) if A.shape[0] > 1 else 0.0
            best_sim = float(sims[pos, best_i])
            strength = min(1.0, max(0.0, (best_sim - 0.35) / 0.45))
            margin = min(1.0, max(0.0, (best_sim - second_sim) / 0.20))
            conf[i] = round(0.5 * strength + 0.5 * margin, 3)
    else:
        for i in reliable:
            conf[i] = 1.0

    for pos, i in enumerate(reliable):
        labels[i] = raw[pos]

    # Attach the short spans.
    #
    # A 0.7 s fragment carries too little signal to identify anyone with an
    # absolute threshold, but when its embedding is *decisively* closer to one
    # centroid than the others that is better evidence than adjacency. Only fall
    # back to the neighbour-in-time heuristic (next speaker usually owns an onset)
    # when the similarity is ambiguous.
    for i in short:
        emb = embs[i]
        if emb is not None and cents:
            ranks = sorted(((float(np.dot(emb, c)), u) for u, c in cents.items()), reverse=True)
            best_sim, best_u = ranks[0]
            second_sim = ranks[1][0] if len(ranks) > 1 else -1.0
            if best_sim >= 0.50 and (best_sim - second_sim) >= 0.10:
                labels[i] = best_u
                conf[i] = round(min(1.0, 0.4 + (best_sim - second_sim)), 3)
                continue
        prev_lab = next_lab = None
        for off in range(1, len(labels) + 1):
            if i - off >= 0 and labels[i - off] >= 0:
                prev_lab = labels[i - off]
                break
        for off in range(1, len(labels) + 1):
            if i + off < len(labels) and labels[i + off] >= 0:
                next_lab = labels[i + off]
                break
        labels[i] = next_lab if next_lab is not None else (prev_lab if prev_lab is not None else -1)
        conf[i] = 0.35
    # Short/unembedded spans inherit the nearest labelled neighbour.
    for i, e in enumerate(labels):
        if e >= 0:
            continue
        for off in range(1, len(labels)):
            prev = i - off
            nxt = i + off
            if prev >= 0 and labels[prev] >= 0:
                labels[i] = labels[prev]
                conf[i] = 0.3
                break
            if nxt < len(labels) and labels[nxt] >= 0:
                labels[i] = labels[nxt]
                conf[i] = 0.3
                break
    return labels, conf


def cluster_embeddings(embs: list[np.ndarray | None], durations: list[float] | None = None,
                       threshold: float = DEFAULT_THRESHOLD,
                       max_speakers: int = 6,
                       prune_below: float = 2.0) -> list[int]:
    """Average-linkage agglomerative clustering + refinement (labels only)."""
    labels, _conf = cluster_with_confidence(embs, durations, threshold, max_speakers, prune_below)
    return labels


def centroid_of(embs: list[np.ndarray], labels: list[int], label: int) -> np.ndarray | None:
    picked = [e for e, lab in zip(embs, labels) if lab == label and e is not None]
    if not picked:
        return None
    c = np.mean(picked, axis=0)
    n = float(np.linalg.norm(c))
    return c / n if n > 1e-9 else None


class SpeakerCluster:
    """Stable online speaker tracker for live audio.

    Re-clustering the accumulated embeddings after every window looked tidy but
    was actively harmful: labels get renumbered as history grows, so text that
    was already streamed to the UI keeps its old number while later text uses
    the new one — the same voice appears as two speakers mid-sentence.

    Instead each voice keeps a persistent centroid. A span joins the closest
    centroid when cosine similarity clears the threshold; otherwise it starts a
    new speaker (creation order = label order, forever stable). Clips shorter
    than ``min_new`` never invent a speaker — they inherit the nearest centroid,
    because 0.5 s of audio cannot identify anyone. A long span that sits between
    ``floor`` and the threshold is treated as a familiar voice rather than a new
    one, and it does not update the centroid.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, max_speakers: int = 6,
                 min_new: float = 0.9, floor: float = 0.45):
        self.threshold = threshold
        self.max_speakers = max(1, int(max_speakers))
        self.min_new = min_new
        # Below this cosine the span is unlike every known voice; between `floor`
        # and `threshold` it is a familiar voice we are merely unsure about.
        self.floor = floor
        self.centroids: list[np.ndarray] = []
        self.counts: list[int] = []
        self.last = -1

    @property
    def speaker_count(self) -> int:
        return len(self.centroids)

    def add(self, emb: np.ndarray | None, duration: float) -> int:
        if emb is None:
            return self.last
        if not self.centroids:
            self.centroids.append(emb)
            self.counts.append(1)
            self.last = 0
            return 0

        sims = [float(np.dot(emb, c)) for c in self.centroids]
        order = sorted(range(len(sims)), key=lambda k: sims[k], reverse=True)
        best = order[0]
        runner_up = sims[order[1]] if len(order) > 1 else -1.0
        # Short, edge-clipped spans (a 2 s window fragment) never reach the same
        # cosine as a clean 5 s utterance — measured 0.65-0.72 for the *same*
        # voice vs 0.14-0.37 for others. So also accept a clear winner by margin,
        # which is what actually separates voices when absolute scores sag.
        decisive = sims[best] >= 0.55 and (sims[best] - runner_up) >= 0.18
        confident = sims[best] >= self.threshold or decisive

        if confident:
            # Only a decisive span is allowed to move a centroid: a fragment that
            # barely matched would otherwise drag the voiceprint toward whatever
            # the microphone heard at the edge of the window.
            if duration >= 0.9 or (sims[best] - runner_up) >= 0.25:
                n = self.counts[best]
                c = (self.centroids[best] * n + emb) / (n + 1)
                norm = float(np.linalg.norm(c))
                self.centroids[best] = c / norm if norm > 1e-9 else self.centroids[best]
                self.counts[best] = n + 1
            self.last = best
            return best

        # A familiar-but-ambiguous span is *not* a new person. Creating a speaker
        # on a weak match is how one voice ended up as three speakers in a long
        # dictation: accept the best centroid, leave the centroid untouched.
        if sims[best] >= self.floor:
            self.last = best
            return best

        # Clearly unlike anyone so far: that is a new voice, if the span is long
        # enough to be evidence and there is room for it.
        if duration >= self.min_new and len(self.centroids) < self.max_speakers:
            self.centroids.append(emb)
            self.counts.append(1)
            self.last = len(self.centroids) - 1
            return self.last

        self.last = best
        return best


# --------------------------------------------------------------- alignment

def split_text_by_speaker(text: str, start: float, end: float,
                          spans: list[tuple[float, float, int]]) -> list[dict]:
    """Split one transcript window into per-speaker pieces.

    Words are distributed proportionally to how much speech time each speaker
    contributed inside ``[start, end]`` — no forced alignment needed, and it
    reads correctly because people speak in order.
    """
    text = (text or "").strip()
    words = text.split()
    if not words:
        return []

    overlaps: list[tuple[float, float, int]] = []
    for s, e, spk in spans:
        if spk < 0:
            continue
        o = min(e, end) - max(s, start)
        if o > 0.05:
            overlaps.append((max(s, start), min(e, end), spk))
    if not overlaps:
        return [{"speaker": -1, "text": text, "start": round(start, 2), "end": round(end, 2)}]

    overlaps.sort(key=lambda t: t[0])
    total = sum(e - s for s, e, _ in overlaps)
    if total <= 0:
        return [{"speaker": overlaps[0][2], "text": text,
                 "start": round(start, 2), "end": round(end, 2)}]

    # Proportional word budget, largest-remainder rounding, order preserved.
    raw = [len(words) * (e - s) / total for s, e, _ in overlaps]
    counts = [int(np.floor(r)) for r in raw]
    remainder = len(words) - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:max(0, remainder)]:
        counts[i] += 1
    for i in range(len(counts) - 1, -1, -1):        # guarantee every speaker gets >= 1 word
        if counts[i] == 0 and len(words) >= len(counts):
            donor = max(range(len(counts)), key=lambda j: counts[j])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[i] += 1

    pieces: list[dict] = []
    cursor = 0
    for (s, e, spk), count in zip(overlaps, counts):
        if count <= 0:
            continue
        chunk = " ".join(words[cursor:cursor + count]).strip()
        cursor += count
        if not chunk:
            continue
        if pieces and pieces[-1]["speaker"] == spk:      # merge neighbours
            pieces[-1]["text"] = f"{pieces[-1]['text']} {chunk}".strip()
            pieces[-1]["end"] = round(e, 2)
        else:
            pieces.append({"speaker": spk, "text": chunk,
                           "start": round(s, 2), "end": round(e, 2)})
    if cursor < len(words) and pieces:                   # rounding leftovers
        pieces[-1]["text"] = f"{pieces[-1]['text']} {' '.join(words[cursor:])}".strip()
    return pieces


# --------------------------------------------------------------- presentation

_TIGHT = re.compile(r"([.!?])([A-Z])")


def tidy(text: str) -> str:
    """Repair seams where the model emitted 'notice.No problem' without a space."""
    return _TIGHT.sub(r"\1 \2", (text or "").strip())


def group_turns(segments: list[dict]) -> list[dict]:
    """Merge consecutive same-speaker pieces into dialogue turns."""
    turns: list[dict] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        spk = seg.get("speaker")
        if turns and turns[-1].get("speaker") == spk:
            turns[-1]["text"] = tidy(f"{turns[-1]['text']} {text}")
            if seg.get("end") is not None:
                turns[-1]["end"] = seg["end"]
            if seg.get("lang"):
                turns[-1]["lang"] = seg["lang"]      # newest window wins
        else:
            turns.append({"speaker": spk, "text": tidy(text),
                          "start": seg.get("start"), "end": seg.get("end"),
                          **({"lang": seg["lang"]} if seg.get("lang") else {})})
    return turns


def speaker_name(spk) -> str:
    return f"Speaker {int(spk) + 1}" if isinstance(spk, (int, np.integer)) and spk >= 0 else "Speaker ?"


def render_turns(turns: list[dict], labels: bool = True) -> str:
    if not turns:
        return ""
    if not labels or all(t.get("speaker") in (None, -1) for t in turns):
        return " ".join(t["text"] for t in turns).strip()
    return "\n".join(f"{speaker_name(t.get('speaker'))}: {t['text']}" for t in turns)


# ------------------------------------------------------------------ façade

def _group_spans(spans: list[tuple[float, float]],
                 max_gap: float = 0.45, min_len: float = 1.6,
                 max_len: float = 8.0) -> tuple[list[tuple[float, float]], list[int]]:
    """Merge neighbouring VAD spans into units long enough to embed reliably.

    Returns the merged groups plus a mapping span-index -> group-index. Clustering
    on 0.9 s fragments is guesswork; clustering on the 4 s they belong to is not.
    Boundaries still come from the original spans, so nothing is lost downstream.
    """
    groups: list[list[float]] = []
    for s, e in spans:
        if groups and s - groups[-1][1] <= max_gap and (groups[-1][1] - groups[-1][0]) < max_len:
            groups[-1][1] = e
        else:
            groups.append([float(s), float(e)])

    mapping: list[int] = [0] * len(spans)
    gi = 0
    for si, (s, e) in enumerate(spans):
        while gi + 1 < len(groups) and s >= groups[gi][1] - 1e-6:
            gi += 1
        mapping[si] = gi

    # A group too short to embed grows into the silence beside it — never into a
    # neighbour's speech, or the groups would overlap and a span could be scored
    # as two people at once.
    for i, g in enumerate(groups):
        if g[1] - g[0] >= min_len:
            continue
        if i + 1 < len(groups) and groups[i + 1][0] > g[1]:
            g[1] = groups[i + 1][0]
        elif i > 0 and groups[i - 1][1] < g[0]:
            g[0] = groups[i - 1][1]
    return [(g[0], g[1]) for g in groups], mapping


def adaptive_threshold(embs: list[np.ndarray | None],
                       durations: list[float] | None = None,
                       fallback: float = DEFAULT_THRESHOLD) -> float:
    """Pick a decision threshold from this recording's own similarity spread.

    Clean studio audio puts different voices at 0.1-0.3 and one voice at 0.85;
    a laptop mic in a room compresses that to 0.45/0.75. A fixed 0.78 is therefore
    wrong for half of all recordings. Two-cluster 1-D k-means over the pairwise
    similarities finds the natural split and returns its midpoint.
    """
    vals = [e for e, d in zip(embs, durations or [1.0] * len(embs))
            if e is not None and d >= 0.8]
    if len(vals) < 4:
        return fallback
    m = np.vstack(vals)
    sims = np.clip(m @ m.T, -1.0, 1.0)
    iu = np.triu_indices(len(vals), k=1)
    flat = sims[iu]
    if flat.size < 6:
        return fallback

    lo, hi = np.percentile(flat, 25), np.percentile(flat, 75)
    if hi - lo < 0.05:
        lo, hi = float(flat.min()), float(flat.max())
    for _ in range(25):
        low_mask = np.abs(flat - lo) <= np.abs(flat - hi)
        new_lo = float(flat[low_mask].mean()) if low_mask.any() else lo
        new_hi = float(flat[~low_mask].mean()) if (~low_mask).any() else hi
        if abs(new_lo - lo) < 1e-4 and abs(new_hi - hi) < 1e-4:
            break
        lo, hi = new_lo, new_hi

    # Only trust the split when it is actually bimodal.
    if hi - lo < 0.12 or low_mask.mean() < 0.15:
        return fallback

    # The decision point belongs on the *same-speaker* side of the gap: accepting
    # a wrong match merges two people irreversibly, while rejecting a good one
    # only costs a label the user can rename. Anchor at one sigma below the
    # same-speaker mean rather than the midpoint of the two clusters.
    same = flat[~low_mask]
    spread = float(same.std()) if same.size > 1 else 0.0
    return float(min(0.90, max(0.60, hi - max(0.045, 1.2 * spread))))



def _probe_clips(audio: np.ndarray, s: float, e: float,
                 sr: int = SR) -> tuple[list[np.ndarray], list[float]]:
    """Voiced probe windows inside ``[s, e]`` plus their centre times.

    Unvoiced probes are dropped before embedding: a silent 0.9 s window yields an
    embedding that matches nothing, which would look exactly like a speaker
    change.
    """
    span = audio[int(s * sr):int(e * sr)]
    if span.size < int(PROBE_WIN * sr):
        return [], []
    energy = frame_energy(span)
    floor = max(0.004, float(np.percentile(energy, 60)) * 0.35) if energy.size else 0.004

    win = int(PROBE_WIN * sr)
    hop = int(PROBE_HOP * sr)
    clips: list[np.ndarray] = []
    times: list[float] = []
    for st in range(0, span.size - win + 1, hop):
        clip = span[st:st + win]
        if float(np.sqrt(np.mean(clip ** 2))) < floor:
            continue
        clips.append(clip)
        times.append((s * sr + st + win / 2) / sr)
    return clips, times


def _snap_to_quiet(audio: np.ndarray, t: float, sr: int, search: float = 0.35) -> float:
    """Move a cut to the quietest 20 ms frame nearby.

    A probe grid puts the boundary within ±0.45 s of the truth, and the truth is
    usually the pause between two utterances. Snapping there avoids slicing a word
    in half, which is what makes the speaker change look abrupt in the transcript.
    """
    lo = max(0, int((t - search) * sr))
    hi = min(audio.size, int((t + search) * sr))
    if hi - lo < int(0.04 * sr):
        return t
    seg = audio[lo:hi]
    energy = frame_energy(seg)
    if energy.size == 0:
        return t
    idx = int(np.argmin(energy))
    snapped = (lo + idx * HOP + FRAME / 2) / sr
    # Only accept the snap when it really is quieter than a voiced frame;
    # otherwise keep the probe-derived point.
    if float(energy[idx]) <= float(np.percentile(energy, 25)):
        return snapped
    return t


def _cuts_on_curve(s: float, e: float, times: list[float],
                   embs: list[np.ndarray | None],
                   audio: np.ndarray | None = None,
                   sr: int = SR) -> list[float]:
    """Cut points where the voice on the left stops matching the voice on the right.

    Comparing *adjacent* probes does not work: when a turn falls between two probe
    windows, both straddle the change and each still scores ~0.56-0.59, above any
    useful threshold. Comparing the mean of the ``LOCAL_PROBES`` windows on each
    side of every probe boundary is the statistic that separates them — measured on
    the synthetic take, the same-speaker floor is ~0.61 and a real change scores
    0.24-0.37 at two neighbouring positions.
    """
    n = len(embs)
    w = LOCAL_PROBES
    if n < 2 * w + 1:
        return []

    def local_mean(lo: int, hi: int) -> np.ndarray | None:
        picked: list[np.ndarray] = []
        for k in range(max(lo, 0), min(hi, n)):
            v = embs[k]
            if v is not None:
                picked.append(v)
        if len(picked) < 2:
            return None
        c = np.mean(np.vstack(picked), axis=0)
        norm = float(np.linalg.norm(c))
        return c / norm if norm > 1e-9 else None

    candidates: list[tuple[float, float]] = []
    for k in range(w, n - w + 1):
        left = local_mean(k - w, k)
        right = local_mean(k, k + w)
        if left is None or right is None:
            continue
        t = (times[k - 1] + times[k]) / 2
        if t - s < REFINE_MIN_PIECE or e - t < REFINE_MIN_PIECE:
            continue
        score = float(np.dot(left, right))
        if score < CHANGE_SIM:
            candidates.append((score, t))

    cuts: list[float] = []
    # How many turns fit in this span: a 15 s merged span can legitimately hold
    # five changes, a 3 s one cannot. The piece-length rule below still keeps
    # every surviving piece at least REFINE_MIN_PIECE long.
    limit = min(MAX_CUTS, max(2, int((e - s) / 2.0)))
    for _score, t in sorted(candidates):
        if any(abs(t - c) < REFINE_MIN_PIECE for c in cuts):
            continue
        cuts.append(_snap_to_quiet(audio, t, sr) if audio is not None else t)
        if len(cuts) >= limit:
            break
    return sorted(cuts)


def _dedupe_units(units: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Make spans monotonic and non-overlapping.

    VAD chops a long continuous speech run with a 0.2 s overlap, so two spans (and
    therefore two groups) can cover the same audio. Overlapping spans double-count
    speech: the same 0.2 s is scored twice and the text split across both windows
    gets a share it does not have. Clamping each start to the previous end costs
    nothing and keeps every downstream consumer honest.
    """
    out: list[tuple[float, float]] = []
    for s, e in sorted(units):
        if out and s < out[-1][1]:
            s = out[-1][1]
        if e - s >= 0.3:
            out.append((s, e))
    return out


def refine_speaker_bounds(audio: np.ndarray, groups: list[tuple[float, float]],
                          sr: int = SR) -> list[tuple[float, float]]:
    """Split merged spans where the speaker actually changes.

    Two stages, both batched, because a per-group ONNX call per probe would make
    diarisation of a long file slower than transcription:

      1. **Quarters gate** — four equal embeddings per group, cheap. A span holding
         one voice scores high on every adjacent pair; a span with a turn inside
         always has a low pair somewhere. (Two halves are not enough: in a span
         that switches A→B→A both halves are mixtures of the same two voices and
         score as "same person".)
      2. **Probe scan** — only for suspect spans: 0.9 s windows every 0.3 s, one
         batched embedding call, then cut at the confirmed similarity valleys.

    Over-splitting is harmless (the pieces are clustered again and usually get the
    same label, and same-label neighbours are re-merged when windows are planned);
    under-splitting is not, which is why the gate is generous.
    """
    out: list[tuple[float, float]] = []
    candidates: list[tuple[tuple[float, float], int]] = []
    quarter_clips: list[np.ndarray] = []
    QUARTERS = 4

    for s, e in groups:
        if e - s < REFINE_MIN_SEC:
            out.append((s, e))
            continue
        step = (e - s) / QUARTERS
        if step < 0.6:
            out.append((s, e))
            continue
        candidates.append(((s, e), len(quarter_clips)))
        for k in range(QUARTERS):
            quarter_clips.append(audio[int((s + k * step) * sr):int((s + (k + 1) * step) * sr)])

    if not candidates:
        return out

    quarters = EMBEDDER.embed_many(quarter_clips, sr) if quarter_clips else []
    suspects: list[tuple[float, float]] = []
    for (s, e), base_i in candidates:
        vecs = quarters[base_i:base_i + QUARTERS]
        sims: list[float] = []
        for k in range(QUARTERS - 1):
            a, b = vecs[k], vecs[k + 1]
            if a is None or b is None:
                continue
            sims.append(float(np.dot(a, b)))
        if sims and min(sims) < CHANGE_SIM:
            suspects.append((s, e))
        else:
            out.append((s, e))

    if not suspects:
        return sorted(out)

    probe_clips: list[np.ndarray] = []
    probe_owner: list[int] = []
    probe_times: list[list[float]] = []
    for gi, (s, e) in enumerate(suspects):
        clips, times = _probe_clips(audio, s, e, sr)
        probe_clips.extend(clips)
        probe_owner.extend([gi] * len(clips))
        probe_times.append(times)

    if not probe_clips:
        return sorted(out + suspects)

    embs = EMBEDDER.embed_many(probe_clips, sr)
    for gi, (s, e) in enumerate(suspects):
        idx = [k for k in range(len(probe_owner)) if probe_owner[k] == gi]
        cuts = _cuts_on_curve(s, e, [probe_times[gi][j] for j in range(len(idx))],
                              [embs[k] for k in idx], audio, sr) if idx else []
        edges = [s, *cuts, e]
        for a, b in zip(edges, edges[1:]):
            out.append((a, b))
    return sorted(out)


def diarize_audio_detailed(audio: np.ndarray, sr: int = SR,
                           threshold: float = DEFAULT_THRESHOLD,
                           max_speakers: int = 6) -> list[dict]:
    """Full-audio diarisation with per-span confidence.

    Pipeline: VAD spans → merged groups → **boundary refinement** (a merged span
    holding two voices is the one error clustering cannot repair) → embeddings →
    clustering with a margin-aware reassignment pass.
    """
    spans = vad_segments(audio, sr)
    if not spans:
        return []
    groups, _mapping = _group_spans(spans)
    units = _dedupe_units(refine_speaker_bounds(audio, groups, sr) or groups)

    clips = [audio[int(s * sr):int(e * sr)] for s, e in units]
    embs = EMBEDDER.embed_many(clips, sr)
    durations = [e - s for s, e in units]

    auto = adaptive_threshold(embs, durations)
    effective = float(min(0.95, max(0.55, auto + (threshold - DEFAULT_THRESHOLD))))
    labels, conf = cluster_with_confidence(embs, durations, effective, max_speakers)
    return [{"start": round(s, 2), "end": round(e, 2), "speaker": labels[i],
             "confidence": round(conf[i], 3), "duration": round(e - s, 2)}
            for i, (s, e) in enumerate(units)]


def diarize_audio(audio: np.ndarray, sr: int = SR,
                  threshold: float = DEFAULT_THRESHOLD,
                  max_speakers: int = 6) -> list[tuple[float, float, int]]:
    """Diarisation as plain ``[(start, end, speaker_label)]`` spans."""
    return [(d["start"], d["end"], d["speaker"])
            for d in diarize_audio_detailed(audio, sr, threshold, max_speakers)]


def speaker_stats(turns: list[dict], fallback_sec_per_word: float = 0.34) -> list[dict]:
    """Talk time, turn count and share of speech per speaker.

    Wall-clock time is used where the turns carry it (diarised uploads); live
    dictation has no per-turn clock, so words are converted with a speaking-rate
    constant rather than reporting a fake zero.
    """
    acc: dict[int, dict] = {}
    for t in turns or []:
        spk = t.get("speaker")
        spk = -1 if spk is None else int(spk)
        row = acc.setdefault(spk, {"speaker": spk, "seconds": 0.0, "turns": 0,
                                   "words": 0, "estimated": False})
        row["turns"] += 1
        words = len(str(t.get("text") or "").split())
        row["words"] += words
        start, end = t.get("start"), t.get("end")
        if start is not None and end is not None and end > start:
            row["seconds"] += float(end - start)
        else:
            row["seconds"] += words * fallback_sec_per_word
            row["estimated"] = True
    total = sum(r["seconds"] for r in acc.values()) or 1.0
    rows = sorted(acc.values(), key=lambda r: r["seconds"], reverse=True)
    for r in rows:
        r["seconds"] = round(r["seconds"], 1)
        r["share"] = round(r["seconds"] / total, 3)
    return rows


def describe_status() -> dict:
    return EMBEDDER.status
