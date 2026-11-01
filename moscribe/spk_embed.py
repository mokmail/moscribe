"""WeSpeaker ResNet34 speaker embeddings (ONNX, CPU).

The VibeVoice-ASR-BitNet 1.5B build cannot emit the model's own
``Start/End/Speaker/Content`` JSON, so speaker identity is recovered with a
dedicated speaker-verification network: `wespeaker-voxceleb-resnet34-LM`,
converted to ONNX (25 MB, no torch, no HF token, no network at runtime once
cached).

Frontend: Kaldi-compatible 80-bin log-mel fbank (16 kHz, 25 ms / 10 ms,
hamming, snip-edges). Per-utterance mean normalisation (CMVN without variance)
— measured on four synthetic voices, that turns a -0.01 cosine margin into
+0.11, i.e. clean separation.

Model file: ``models/spk/wespeaker-resnet34.onnx`` (downloaded on first use).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

MODEL_SR = 16000
FRAME_LEN = 400        # 25 ms @ 16 kHz
FRAME_SHIFT = 160      # 10 ms
N_FFT = 512
N_MELS = 80
FMIN = 20.0
PREEMPH = 0.97
MIN_FRAMES = 25        # ~0.25 s of audio before an embedding is meaningful
CROP_SEC = 3.0         # longest window the net sees at once
MAX_CROPS = 4          # crops averaged per clip

_MODEL_URL = ("https://huggingface.co/onnx-community/"
              "wespeaker-voxceleb-resnet34-LM/resolve/main/onnx/model.onnx")
from .paths import DATA_DIR, models_dir  # noqa: E402  (local module)

_DEFAULT_MODEL = models_dir() / "spk" / "wespeaker-resnet34.onnx"
_FALLBACK_MODEL = DATA_DIR / "models" / "spk" / "wespeaker-resnet34.onnx"


# ------------------------------------------------------------------ frontend

def _mel_matrix() -> np.ndarray:
    """Kaldi-style triangular mel filterbank, 20 Hz .. Nyquist."""
    n_bins = N_FFT // 2 + 1
    freqs = np.arange(n_bins, dtype=np.float64) * MODEL_SR / N_FFT
    mel = lambda f: 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)  # noqa: E731
    edges = 700.0 * (np.exp(np.linspace(mel(FMIN), mel(MODEL_SR / 2), N_MELS + 2) / 1127.0) - 1.0)
    fb = np.zeros((N_MELS, n_bins), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        left, centre, right = edges[m - 1], edges[m], edges[m + 1]
        up = (freqs - left) / max(1e-9, centre - left)
        down = (right - freqs) / max(1e-9, right - centre)
        fb[m - 1] = np.clip(np.minimum(up, down), 0.0, None)
    return fb


_FB = _mel_matrix()
_WINDOW = np.hamming(FRAME_LEN).astype(np.float32)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Band-limited (FFT) resampling — good enough for 24k→16k, no scipy needed."""
    if sr_from == sr_to or x.size == 0:
        return x.astype(np.float32, copy=False)
    n_out = max(1, int(round(x.size * sr_to / sr_from)))
    if n_out == x.size:
        return x.astype(np.float32, copy=False)
    spec = np.fft.rfft(x)
    m = n_out // 2 + 1
    if m <= spec.size:
        spec = spec[:m]
    else:
        spec = np.pad(spec, (0, m - spec.size))
    return (np.fft.irfft(spec, n_out) * (n_out / x.size)).astype(np.float32)


def fbank(x: np.ndarray) -> np.ndarray | None:
    """(frames, 80) log-mel features for 16 kHz mono float audio."""
    if x.size < FRAME_LEN:
        return None
    x = x.astype(np.float32, copy=True)
    x -= x.mean()                                    # remove_dc_offset
    x = np.append(x[0], x[1:] - PREEMPH * x[:-1])    # pre-emphasis
    n = 1 + (x.size - FRAME_LEN) // FRAME_SHIFT
    if n < 1:
        return None
    idx = np.arange(FRAME_LEN)[None, :] + FRAME_SHIFT * np.arange(n)[:, None]
    frames = x[idx] * _WINDOW
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2
    mel = power @ _FB.T
    return np.log(np.maximum(mel, 1e-10)).astype(np.float32)


# ------------------------------------------------------------------ embedder

class SpeakerEmbedder:
    """Thin, thread-safe wrapper around the ONNX session."""

    def __init__(self, model_path: str | Path | None = None, auto_download: bool = True):
        chosen = model_path or os.environ.get("VIBESCRIBE_SPK_MODEL")
        if not chosen:
            frozen = bool(getattr(sys, "frozen", False))
            if _DEFAULT_MODEL.exists():
                chosen = _DEFAULT_MODEL
            elif frozen:
                chosen = _FALLBACK_MODEL        # the bundle is read-only: download into data
            else:
                chosen = _DEFAULT_MODEL
        self.model_path = Path(chosen)
        self.auto_download = auto_download
        self._session = None
        self._lock = threading.Lock()
        self._error: str | None = None
        self._tried = False

    # -- lifecycle ---------------------------------------------------------
    def _download(self) -> bool:
        if not self.auto_download:
            return False
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.model_path.with_suffix(".part")
        try:
            subprocess.run(["curl", "-sL", "--fail", "-o", str(tmp), _MODEL_URL],
                           check=True, timeout=600)
            tmp.replace(self.model_path)
            return True
        except Exception as exc:  # noqa: BLE001
            self._error = f"model download failed: {exc}"
            tmp.unlink(missing_ok=True)
            return False

    def _load(self) -> bool:
        if self._session is not None:
            return True
        if self._tried:
            return False
        self._tried = True
        try:
            if not self.model_path.exists() and not self._download():
                return False
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 4))
            opts.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=opts,
                providers=["CPUExecutionProvider"])
            return True
        except Exception as exc:  # noqa: BLE001
            self._error = f"{type(exc).__name__}: {exc}"
            return False

    @property
    def available(self) -> bool:
        return self._load()

    @property
    def status(self) -> dict:
        ok = self.available
        return {"ok": ok, "model": str(self.model_path),
                "size_mb": round(self.model_path.stat().st_size / 1e6, 1) if self.model_path.exists() else 0,
                "error": self._error}

    # -- inference ---------------------------------------------------------
    def embed(self, audio: np.ndarray, sr: int = 24000) -> np.ndarray | None:
        """L2-normalised 256-d embedding, or None when the clip is too short/silent."""
        vecs = self.embed_many([audio], sr)
        return vecs[0]

    def embed_many(self, clips: list[np.ndarray], sr: int = 24000) -> list[np.ndarray | None]:
        """Batch several clips; one ONNX run for all crops of all clips.

        Long clips are scored as several overlapping crops whose embeddings are
        averaged. One embedding of a 12 s monologue is dominated by whichever
        second happened to be loudest; averaging crops roughly halves the
        variance, which is what makes close voices separable at all.
        """
        if not clips:
            return []
        if not self._load():
            return [None] * len(clips)

        feats: list[np.ndarray] = []
        owner: list[int] = []
        for i, clip in enumerate(clips):
            x = resample(np.asarray(clip, dtype=np.float32), sr, MODEL_SR)
            if x.size < MIN_FRAMES * FRAME_SHIFT or float(np.sqrt(np.mean(x ** 2))) < 1e-4:
                continue
            for crop in _crops(x):
                fb = fbank(crop)
                if fb is None or fb.shape[0] < MIN_FRAMES:
                    continue
                feats.append(fb - fb.mean(axis=0, keepdims=True))   # per-utterance CMVN
                owner.append(i)
        if not feats:
            return [None] * len(clips)

        max_t = max(f.shape[0] for f in feats)
        batch = np.zeros((len(feats), max_t, N_MELS), dtype=np.float32)
        for row, f in enumerate(feats):
            batch[row, :f.shape[0]] = f
            if f.shape[0] < max_t:                    # repeat-last pad keeps features in-distribution
                batch[row, f.shape[0]:] = f[-1]
        with self._lock:
            out = self._session.run(None, {"input_features": batch})[0]

        sums: list[np.ndarray | None] = [None] * len(clips)
        for row, idx in enumerate(owner):
            v = np.asarray(out[row], dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n <= 1e-9:
                continue
            v = v / n
            sums[idx] = v if sums[idx] is None else sums[idx] + v

        vecs: list[np.ndarray | None] = []
        for acc in sums:
            if acc is None:
                vecs.append(None)
                continue
            n = float(np.linalg.norm(acc))
            vecs.append(acc / n if n > 1e-9 else None)
        return vecs


def _crops(x: np.ndarray, crop_sec: float = CROP_SEC, max_crops: int = MAX_CROPS) -> list[np.ndarray]:
    """Evenly spaced overlapping crops covering the clip."""
    span = int(crop_sec * MODEL_SR)
    if x.size <= span * 1.4:
        return [x]
    k = min(max_crops, max(2, x.size // max(1, span)))
    step = (x.size - span) / (k - 1)
    return [x[int(round(i * step)):int(round(i * step)) + span] for i in range(k)]


EMBEDDER = SpeakerEmbedder()
