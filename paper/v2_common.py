# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Shared code for the second round of paper experiments (paper/v2_*.py).

Same data, network and training recipe as train_beats.py / ablate.py, with three additions:
  * a second timing-feature set that is normalised by the patient's own recent rhythm
    ("adaptive", see timing_features),
  * beats can come from annotated R peaks or from the monitor's own QRS detector,
  * training is a function, so it can be run inside a cross-validation loop on DS1.
The shipped weights (ai/beat_cnn.npz) are never touched.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from ai.beat_model import FS_MODEL, POST, PRE, WIN, to_model_rate          # noqa: E402
from ai.classifier import _bandpass, detect_r_peaks                        # noqa: E402
from train_beats import AAMI, DS1, DS2                                     # noqa: E402

CLASSES = ["N", "S", "V", "F"]
RUNS = os.path.join(REPO, "paper", "runs_v2")
CACHE = os.path.join(REPO, "paper", "v2_cache")
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")          # every beat annotation (for QRS detection scoring)
TOL = int(0.150 * FS_MODEL)                         # EC57 matching window: 150 ms


# ── records ─────────────────────────────────────────────────────────────────

def record(rec, data_dir):
    """Filtered MLII signal at 360 Hz, every beat annotation, AAMI labels (-1 = not N/S/V/F),
    and the rhythm annotations. Cached as .npz."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{rec}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    import wfdb
    r = wfdb.rdrecord(os.path.join(data_dir, rec))
    ann = wfdb.rdann(os.path.join(data_dir, rec), "atr")
    lead = r.sig_name.index("MLII") if "MLII" in r.sig_name else 0
    raw = r.p_signal[:, lead].astype(np.float64)
    sig = to_model_rate(raw, r.fs)
    beats = [(s, c) for s, c in zip(ann.sample, ann.symbol) if c in BEAT_SYMBOLS]
    peaks = np.array([b[0] for b in beats])
    labels = np.array([CLASSES.index(AAMI[c]) if c in AAMI else -1 for _, c in beats])
    rhythm = [(s, n.strip("\x00").strip()) for s, c, n in zip(ann.sample, ann.symbol, ann.aux_note)
              if c == "+" and n.startswith("(")]
    out = {"sig": sig.astype(np.float32), "raw": raw.astype(np.float32), "peaks": peaks, "labels": labels,
           "rhythm_at": np.array([s for s, _ in rhythm]), "rhythm": np.array([n for _, n in rhythm])}
    np.savez(path, **out)
    return out


# ── the monitor's QRS detector, run the way the monitor runs it ─────────────

def detect(sig, win_s=8.0, step_s=6.0):
    """The monitor analyses the last 8 s on screen, so the detector (ai/classifier.detect_r_peaks,
    unchanged) is run on 8-s windows that overlap by 2 s; peaks found twice are merged."""
    fs = FS_MODEL
    w, st = int(win_s * fs), int(step_s * fs)
    found = []
    for a in range(0, max(1, len(sig) - w + st), st):
        seg = sig[a:a + w]
        if len(seg) < fs:
            break
        pk = detect_r_peaks(_bandpass(seg.astype(float), fs), fs)
        # keep the central part of each window, except at the record's ends
        lo = 0 if a == 0 else int(1.0 * fs)
        hi = len(seg) if a + w >= len(sig) else len(seg) - int(1.0 * fs)
        found.extend(a + p for p in pk if lo <= p < hi)
    found = np.unique(np.asarray(found, dtype=int))
    if len(found) < 2:
        return found
    keep = [found[0]]
    for p in found[1:]:
        if p - keep[-1] > int(0.20 * fs):          # same refractory period as the detector
            keep.append(p)
    return np.asarray(keep)


def detect_xqrs(raw):
    """Reference detector: XQRS from the wfdb package (PhysioNet's own tools), on the raw 360 Hz
    signal. Used only to separate detector errors from classifier errors; the monitor doesn't use it."""
    import logging
    import wfdb.processing as wp
    logging.disable(logging.WARNING)
    x = wp.XQRS(sig=np.asarray(raw, dtype=float), fs=FS_MODEL)
    x.detect(verbose=False)
    return np.asarray(x.qrs_inds, dtype=int)


def match(true_peaks, det_peaks, tol=TOL):
    """Greedy nearest matching within tol. Returns idx_true_for_det (-1 if a false detection)."""
    out = np.full(len(det_peaks), -1)
    used = np.zeros(len(true_peaks), bool)
    j = 0
    for i, d in enumerate(det_peaks):
        while j < len(true_peaks) - 1 and true_peaks[j + 1] <= d:
            j += 1
        best, bd = -1, tol + 1
        for k in (j - 1, j, j + 1):
            if 0 <= k < len(true_peaks) and not used[k] and abs(true_peaks[k] - d) < bd:
                best, bd = k, abs(true_peaks[k] - d)
        if best >= 0:
            out[i] = best
            used[best] = True
    return out


# ── inputs ───────────────────────────────────────────────────────────────────

def timing_features(peaks, i, rr, kind):
    pre, post = rr[i - 1], rr[i]
    local = np.mean(rr[max(0, i - 10):i]) if i > 1 else pre
    t = [pre / local, post / local, pre, pre / post, np.median(rr[max(0, i - 10):i + 10]) / local]
    if kind == "adaptive":
        # The patient's own baseline: median RR over the preceding 5 minutes (causal; at the
        # start of a recording, whatever history there is).
        lo = np.searchsorted(peaks, peaks[i] - 300 * FS_MODEL)
        hist = rr[max(0, lo):i]
        base = np.median(hist) if len(hist) >= 3 else local
        t += [pre / base, post / base]
    return t


def windows(sig, peaks, kind="local"):
    """Same windows and normalisation as ai/beat_model.beat_windows; returns (X, T, kept index)."""
    peaks = np.asarray(peaks, dtype=int)
    rr = np.diff(peaks) / FS_MODEL
    X, T, keep = [], [], []
    for i in range(1, len(peaks) - 1):
        p = peaks[i]
        if p - PRE < 0 or p + POST > len(sig):
            continue
        w = sig[p - PRE:p + POST]
        X.append((w - np.median(w)) / (np.std(w) + 1e-6))
        T.append(timing_features(peaks, i, rr, kind))
        keep.append(i)
    return np.asarray(X, np.float32), np.asarray(T, np.float32), np.asarray(keep, dtype=int)


def dataset(records, data_dir, kind="local", shifts=(0,), source="annotated"):
    """Beats of the given records. source='annotated' uses the database's R peaks (beats that
    aren't N/S/V/F are dropped, as in train_beats.load). source='detected' uses the monitor's
    detector ('xqrs': the reference detector) and keeps every detection, labelled by the annotation it matches (-1 = none / other)."""
    X, T, Y, R = [], [], [], []
    for rec in records:
        d = record(rec, data_dir)
        if source == "annotated":
            m = d["labels"] >= 0
            pk, lab = d["peaks"][m], d["labels"][m]
            for sh in shifts:
                x, t, k = windows(d["sig"], pk + sh, kind)
                X.append(x); T.append(t); Y.append(lab[k]); R.append(np.full(len(k), int(rec)))
        else:
            det = detect(d["sig"]) if source == "detected" else detect_xqrs(d["raw"])
            mi = match(d["peaks"], det)
            lab = np.where(mi >= 0, d["labels"][np.maximum(mi, 0)], -1)
            x, t, k = windows(d["sig"], det, kind)
            X.append(x); T.append(t); Y.append(lab[k]); R.append(np.full(len(k), int(rec)))
    return np.concatenate(X), np.concatenate(T), np.concatenate(Y), np.concatenate(R)


# ── training (same recipe as train_beats.py) ─────────────────────────────────

def init_params(rng, n_t):
    def he(shape, fan_in):
        return (rng.standard_normal(shape) * np.sqrt(2 / fan_in)).astype(np.float32)
    return {"w1": he((7, 1, 16), 7), "b1": np.zeros(16, np.float32),
            "w2": he((5, 16, 32), 80), "b2": np.zeros(32, np.float32),
            "w3": he((5, 32, 32), 160), "b3": np.zeros(32, np.float32),
            "w4": he((32 + n_t, 32), 32 + n_t), "b4": np.zeros(32, np.float32),
            "w5": he((32, 4), 32), "b5": np.zeros(4, np.float32)}


def train(X, T, Y, seed=0, balance=0.5, epochs=14, noise_sd=0.05, log=None):
    import jax
    import jax.numpy as jnp
    import optax
    t_mean, t_std = T.mean(0), T.std(0) + 1e-6
    counts = np.bincount(Y, minlength=4)
    cw = (counts.sum() / np.maximum(counts, 1)) ** balance
    cw = jnp.asarray(cw / cw[Y].mean(), dtype=jnp.float32)

    def fwd(p, x, t):
        h = x[:, :, None]
        for i in (1, 2, 3):
            h = jax.lax.conv_general_dilated(h, p[f"w{i}"], (1,), "VALID",
                                             dimension_numbers=("NWC", "WIO", "NWC")) + p[f"b{i}"]
            h = jax.nn.relu(h)
            if i < 3:
                L = h.shape[1] // 2 * 2
                h = h[:, :L].reshape(h.shape[0], L // 2, 2, h.shape[2]).max(axis=2)
        h = jnp.concatenate([h.mean(axis=1), (t - t_mean) / t_std], axis=1)
        return jax.nn.relu(h @ p["w4"] + p["b4"]) @ p["w5"] + p["b5"]

    def loss_fn(p, x, t, y):
        return (optax.softmax_cross_entropy_with_integer_labels(fwd(p, x, t), y) * cw[y]).mean()

    rng = np.random.default_rng(seed)
    params = {k: jnp.asarray(v) for k, v in init_params(rng, T.shape[1]).items()}
    spe = len(Y) // 256
    opt = optax.adamw(optax.cosine_decay_schedule(2e-3, epochs * spe), weight_decay=1e-4)
    state = opt.init(params)

    @jax.jit
    def step(p, s, x, t, y):
        l, g = jax.value_and_grad(loss_fn)(p, x, t, y)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l

    for ep in range(epochs):
        order = rng.permutation(len(Y))
        tot = 0.0
        for b in range(spe):
            idx = order[b * 256:(b + 1) * 256]
            xb = X[idx] + rng.normal(0, noise_sd, (len(idx), WIN)).astype(np.float32)
            params, state, l = step(params, state, xb, T[idx], Y[idx])
            tot += float(l)
        if log:
            log(f"  epoch {ep + 1} loss {tot / spe:.4f}")
    out = {k: np.asarray(v) for k, v in params.items()}
    out["t_mean"], out["t_std"] = t_mean.astype(np.float32), t_std.astype(np.float32)
    return out


def predict(params, X, T):
    from ai.beat_model import forward          # the monitor's NumPy forward pass
    return np.concatenate([forward(params, X[i:i + 4096], T[i:i + 4096]).argmax(1)
                           for i in range(0, len(X), 4096)]) if len(X) else np.zeros(0, int)


# ── metrics ──────────────────────────────────────────────────────────────────

def confusion(y, pred):
    """4x5 matrix: rows true N,S,V,F; columns predicted N,S,V,F, plus column 4 = 'no beat there'.
    Predictions on detections that match no N/S/V/F beat (y = -1) go in a 5th row."""
    cm = np.zeros((5, 5), int)
    for a, b in zip(y, pred):
        cm[a if a >= 0 else 4, b] += 1
    return cm


def scores(cm, missed=None):
    """Per-class Se and +P (EC57). missed[c] = beats of class c the detector never found."""
    missed = np.zeros(4, int) if missed is None else np.asarray(missed)
    per = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        per[c] = {"beats": int(cm[i, :4].sum() + missed[i]),
                  "se": float(tp / max(1, cm[i, :4].sum() + missed[i])),
                  "ppv": float(tp / max(1, cm[:, i].sum()))}
    f1 = {c: 2 * v["se"] * v["ppv"] / max(1e-9, v["se"] + v["ppv"]) for c, v in per.items()}
    return per, f1
