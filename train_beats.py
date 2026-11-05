"""
Train the beat classifier on the MIT-BIH Arrhythmia Database.

    pip install wfdb jax optax
    python train_beats.py                       # downloads MIT-BIH from PhysioNet (~100 MB)
    python train_beats.py --data path/to/mitdb  # or use a local copy

Inter-patient evaluation (de Chazal et al., 2004): the model trains on the 22
DS1 patients and is tested on 22 different DS2 patients, so the scores say how
it does on someone it has never seen. Paced records are excluded, as is standard.
Beats are grouped into the AAMI classes N, S, V, F (the rare Q class is dropped).

Writes ai/beat_cnn.npz (weights the monitor loads with NumPy) and
ai/beat_cnn_metrics.json (test results shown on the monitor's Analysis page).
"""

import argparse
import json
import os
import time

import numpy as np

from ai.beat_model import CLASSES, FS_MODEL, METRICS, WEIGHTS, WIN, beat_windows, forward, to_model_rate

AAMI = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
        "A": "S", "a": "S", "J": "S", "S": "S",
        "V": "V", "E": "V",
        "F": "F"}
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]


def load(records, data_dir, shifts=(0,)):
    """Beat windows + timing features + labels for the given records (lead MLII)."""
    import wfdb
    X, T, Y = [], [], []
    for rec in records:
        r = wfdb.rdrecord(os.path.join(data_dir, rec))
        ann = wfdb.rdann(os.path.join(data_dir, rec), "atr")
        lead = r.sig_name.index("MLII") if "MLII" in r.sig_name else 0
        sig = to_model_rate(r.p_signal[:, lead], r.fs)          # already 360 Hz: filter only
        beats = [(s, AAMI[c]) for s, c in zip(ann.sample, ann.symbol) if c in AAMI]
        peaks = np.array([b[0] for b in beats])
        labels = np.array([CLASSES.index(b[1]) for b in beats])
        for sh in shifts:                                         # R-peak jitter, like a live detector
            x, t, keep = beat_windows(sig, peaks + sh)
            X.append(x); T.append(t); Y.append(labels[keep])
    return np.concatenate(X), np.concatenate(T), np.concatenate(Y)


def init_params(rng):
    def he(shape, fan_in):
        return (rng.standard_normal(shape) * np.sqrt(2 / fan_in)).astype(np.float32)
    return {
        "w1": he((7, 1, 16), 7), "b1": np.zeros(16, np.float32),
        "w2": he((5, 16, 32), 80), "b2": np.zeros(32, np.float32),
        "w3": he((5, 32, 32), 160), "b3": np.zeros(32, np.float32),
        "w4": he((37, 32), 37), "b4": np.zeros(32, np.float32),
        "w5": he((32, 4), 32), "b5": np.zeros(4, np.float32),
    }


def report(y, pred):
    cm = np.zeros((4, 4), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    per = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        per[c] = {"beats": int(cm[i].sum()),
                  "sensitivity": float(tp / max(1, cm[i].sum())),
                  "ppv": float(tp / max(1, cm[:, i].sum()))}
    return {"accuracy": float(np.trace(cm) / cm.sum()), "per_class": per, "confusion": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mitdb")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--balance", type=float, default=0.5, help="class weight = (1/frequency) ** balance")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.data, "100.hea")):
        import wfdb
        print(f"Downloading MIT-BIH into {args.data} ...")
        wfdb.dl_database("mitdb", args.data)

    import jax
    import jax.numpy as jnp
    import optax

    t0 = time.time()
    X, T, Y = load(DS1, args.data, shifts=(-4, 0, 4))
    Xte, Tte, Yte = load(DS2, args.data)
    print(f"train beats {len(Y):,} (with jitter copies)  test beats {len(Yte):,}  [{time.time() - t0:.0f}s]")
    print("train class counts", dict(zip(CLASSES, np.bincount(Y, minlength=4).tolist())))

    t_mean, t_std = T.mean(0), T.std(0) + 1e-6
    counts = np.bincount(Y, minlength=4)
    cw = (counts.sum() / np.maximum(counts, 1)) ** args.balance
    cw = jnp.asarray(cw / cw[Y].mean(), dtype=jnp.float32)

    def jforward(p, x, t):
        h = x[:, :, None]
        for i in (1, 2, 3):
            h = jax.lax.conv_general_dilated(h, p[f"w{i}"], (1,), "VALID", dimension_numbers=("NWC", "WIO", "NWC")) + p[f"b{i}"]
            h = jax.nn.relu(h)
            if i < 3:
                L = h.shape[1] // 2 * 2
                h = h[:, :L].reshape(h.shape[0], L // 2, 2, h.shape[2]).max(axis=2)
        h = h.mean(axis=1)
        h = jnp.concatenate([h, (t - t_mean) / t_std], axis=1)
        h = jax.nn.relu(h @ p["w4"] + p["b4"])
        return h @ p["w5"] + p["b5"]

    def loss_fn(p, x, t, y):
        logits = jforward(p, x, t)
        ll = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        return (ll * cw[y]).mean()

    rng = np.random.default_rng(args.seed)
    params = {k: jnp.asarray(v) for k, v in init_params(rng).items()}
    steps_per_epoch = len(Y) // 256
    sched = optax.cosine_decay_schedule(2e-3, args.epochs * steps_per_epoch)
    opt = optax.adamw(sched, weight_decay=1e-4)
    state = opt.init(params)

    @jax.jit
    def step(p, s, x, t, y):
        l, g = jax.value_and_grad(loss_fn)(p, x, t, y)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l

    for ep in range(args.epochs):
        order = rng.permutation(len(Y))
        tot = 0.0
        for b in range(steps_per_epoch):
            idx = order[b * 256:(b + 1) * 256]
            xb = X[idx] + rng.normal(0, 0.05, (len(idx), WIN)).astype(np.float32)   # a little noise
            params, state, l = step(params, state, xb, T[idx], Y[idx])
            tot += float(l)
        print(f"epoch {ep + 1:2d}  loss {tot / steps_per_epoch:.4f}  [{time.time() - t0:.0f}s]")

    out = {k: np.asarray(v) for k, v in params.items()}
    out["t_mean"], out["t_std"] = t_mean.astype(np.float32), t_std.astype(np.float32)
    np.savez_compressed(WEIGHTS, **out)

    # test with the NumPy forward pass the monitor uses
    pred = np.concatenate([forward(out, Xte[i:i + 4096], Tte[i:i + 4096]).argmax(1) for i in range(0, len(Yte), 4096)])
    res = report(Yte, pred)
    res.update({"split": "inter-patient: train DS1 (22 patients), test DS2 (22 other patients)",
                "classes": CLASSES, "test_beats": int(len(Yte)), "epochs": args.epochs})
    json.dump(res, open(METRICS, "w"), indent=2)
    print(f"\nDS2 (unseen patients): accuracy {res['accuracy']:.3f}")
    for c, v in res["per_class"].items():
        print(f"  {c}: sensitivity {v['sensitivity']:.3f}  PPV {v['ppv']:.3f}  ({v['beats']} beats)")
    print(f"saved {WEIGHTS} and {METRICS}")


if __name__ == "__main__":
    main()
