# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
Beat classifier trained on real ECGs (MIT-BIH Arrhythmia Database).

Every QRS complex is labelled as one of the AAMI EC57 beat classes:

    N  normal or bundle-branch-block beat
    S  supraventricular ectopic beat (premature atrial / junctional)
    V  ventricular ectopic beat (PVC)
    F  fusion of a ventricular and a normal beat

The network is a small 1-D CNN over a 0.6 s window around the R peak plus
five timing features (how early the beat came, how long the pause after it
was, and the local rhythm around it). It was trained with train_beats.py on the
training patients of the standard inter-patient split, and the weights in
ai/beat_cnn.npz are what the monitor runs.

Inference is plain NumPy, so the monitor doesn't need a deep-learning library.
The same preprocessing code is used for training and for the live monitor.
"""

import json
import os

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

CLASSES = ["N", "S", "V", "F"]
NAMES = {"N": "Normal", "S": "Supraventricular ectopic", "V": "Ventricular ectopic (PVC)", "F": "Fusion"}
FS_MODEL = 360                      # MIT-BIH sampling rate
PRE, POST = 90, 126                 # samples before / after the R peak (0.25 s, 0.35 s)
WIN = PRE + POST

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "beat_cnn.npz")
METRICS = os.path.join(HERE, "beat_cnn_metrics.json")


# ── preprocessing (shared by training and the monitor) ──────────────────────

def to_model_rate(sig, fs):
    """Resample to 360 Hz (exact ratio for 500 Hz) and band-pass 0.5-40 Hz."""
    sig = np.asarray(sig, dtype=float)
    if fs != FS_MODEL:
        g = np.gcd(int(fs), FS_MODEL)
        sig = resample_poly(sig, FS_MODEL // g, int(fs) // g)
    b, a = butter(2, [0.5 / (FS_MODEL / 2), 40 / (FS_MODEL / 2)], btype="band")
    return filtfilt(b, a, sig, padtype="even", padlen=min(len(sig) - 1, FS_MODEL))


def beat_windows(sig360, peaks360):
    """Windows around each R peak, normalised per beat, plus timing features.
    Beats too close to either end, or without a beat on both sides, are skipped.
    Returns (X [n, WIN], T [n, 5], kept_index)."""
    peaks = np.asarray(peaks360, dtype=int)
    X, T, keep = [], [], []
    rr = np.diff(peaks) / FS_MODEL
    for i in range(1, len(peaks) - 1):
        p = peaks[i]
        if p - PRE < 0 or p + POST > len(sig360):
            continue
        w = sig360[p - PRE:p + POST]
        w = (w - np.median(w)) / (np.std(w) + 1e-6)
        pre, post = rr[i - 1], rr[i]
        local = np.mean(rr[max(0, i - 10):i]) if i > 1 else pre
        T.append([pre / local, post / local, pre, pre / post, np.median(rr[max(0, i - 10):i + 10]) / local])
        X.append(w)
        keep.append(i)
    if not X:
        return np.zeros((0, WIN)), np.zeros((0, 5)), np.zeros(0, dtype=int)
    return np.asarray(X, np.float32), np.asarray(T, np.float32), np.asarray(keep)


# ── the network (NumPy forward pass; train_beats.py has the same in JAX) ────

def _conv1d(x, w, b):
    """x [n, L, cin], w [k, cin, cout] -> 'valid' convolution [n, L-k+1, cout]."""
    k = w.shape[0]
    L = x.shape[1] - k + 1
    cols = np.stack([x[:, i:i + L, :] for i in range(k)], axis=2)      # [n, L, k, cin]
    return np.einsum("nlkc,kco->nlo", cols, w) + b


def _pool2(x):
    L = x.shape[1] // 2 * 2
    return x[:, :L].reshape(x.shape[0], L // 2, 2, x.shape[2]).max(axis=2)


def forward(params, X, T):
    h = X[:, :, None]
    for i in (1, 2, 3):
        h = np.maximum(_conv1d(h, params[f"w{i}"], params[f"b{i}"]), 0)
        if i < 3:
            h = _pool2(h)
    h = h.mean(axis=1)                                                     # global average pool
    t = (T - params["t_mean"]) / params["t_std"]
    h = np.concatenate([h, t], axis=1)
    h = np.maximum(h @ params["w4"] + params["b4"], 0)
    z = h @ params["w5"] + params["b5"]
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class BeatClassifier:
    """Loads ai/beat_cnn.npz. classify(sig, fs, r_peaks) labels every beat that
    has a neighbour on both sides."""

    def __init__(self, path=WEIGHTS):
        with np.load(path) as z:
            self.params = {k: z[k] for k in z.files}
        self.metrics = json.load(open(METRICS)) if os.path.exists(METRICS) else None

    @staticmethod
    def available(path=WEIGHTS):
        return os.path.exists(path)

    def classify(self, sig, fs, r_peaks):
        """Returns a list of (peak_index_in_sig, label, confidence)."""
        if len(r_peaks) < 3:
            return []
        s360 = to_model_rate(sig, fs)
        peaks360 = np.round(np.asarray(r_peaks) * FS_MODEL / fs).astype(int)
        X, T, keep = beat_windows(s360, peaks360)
        if not len(X):
            return []
        P = forward(self.params, X, T)
        lab = P.argmax(axis=1)
        return [(int(r_peaks[k]), CLASSES[l], float(P[j, l])) for j, (k, l) in enumerate(zip(keep, lab))]
