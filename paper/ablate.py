# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Seeds and ablations for the paper. Same data, split, network and training recipe as
train_beats.py; the shipped weights are not touched.

    python paper/ablate.py --data path/to/mitdb --variant full|morph|timing --seed N
writes paper/runs/<variant>_s<seed>.json  (seed 0 of "full" reproduces ai/beat_cnn_metrics.json)
"""
import argparse, json, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from ai.beat_model import CLASSES, WIN, forward
from train_beats import DS1, DS2, init_params, load, report

OUT = os.path.join(REPO, "paper", "runs")
CACHE = os.path.join(REPO, "paper", "beats_cache.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mitdb")
    ap.add_argument("--variant", default="full", choices=["full", "morph", "timing"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--balance", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    import jax, jax.numpy as jnp, optax
    t0 = time.time()
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        X, T, Y, Xte, Tte, Yte = (z[k] for k in ("X", "T", "Y", "Xte", "Tte", "Yte"))
    else:
        X, T, Y = load(DS1, a.data, shifts=(-4, 0, 4))
        Xte, Tte, Yte = load(DS2, a.data)
        np.savez(CACHE, X=X, T=T, Y=Y, Xte=Xte, Tte=Tte, Yte=Yte)

    # Ablations: remove one input stream by making it constant (the network shape is unchanged).
    if a.variant == "morph":
        T, Tte = np.zeros_like(T), np.zeros_like(Tte)
    if a.variant == "timing":
        X, Xte = np.zeros_like(X), np.zeros_like(Xte)
    noise_sd = 0.0 if a.variant == "timing" else 0.05

    t_mean, t_std = T.mean(0), T.std(0) + 1e-6
    counts = np.bincount(Y, minlength=4)
    cw = (counts.sum() / np.maximum(counts, 1)) ** a.balance
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
        ll = optax.softmax_cross_entropy_with_integer_labels(jforward(p, x, t), y)
        return (ll * cw[y]).mean()

    rng = np.random.default_rng(a.seed)
    params = {k: jnp.asarray(v) for k, v in init_params(rng).items()}
    spe = len(Y) // 256
    opt = optax.adamw(optax.cosine_decay_schedule(2e-3, a.epochs * spe), weight_decay=1e-4)
    state = opt.init(params)

    @jax.jit
    def step(p, s, x, t, y):
        l, g = jax.value_and_grad(loss_fn)(p, x, t, y)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l

    curve = []
    for ep in range(a.epochs):
        order = rng.permutation(len(Y))
        tot = 0.0
        for b in range(spe):
            idx = order[b * 256:(b + 1) * 256]
            xb = X[idx] + (rng.normal(0, noise_sd, (len(idx), WIN)).astype(np.float32) if noise_sd else 0)
            params, state, l = step(params, state, xb, T[idx], Y[idx])
            tot += float(l)
        curve.append(tot / spe)
        print(f"{a.variant} s{a.seed} epoch {ep + 1} loss {curve[-1]:.4f} [{time.time() - t0:.0f}s]", flush=True)

    out = {k: np.asarray(v) for k, v in params.items()}
    out["t_mean"], out["t_std"] = t_mean.astype(np.float32), t_std.astype(np.float32)
    pred = np.concatenate([forward(out, Xte[i:i + 4096], Tte[i:i + 4096]).argmax(1) for i in range(0, len(Yte), 4096)])
    res = report(Yte, pred)
    res.update(variant=a.variant, seed=a.seed, epochs=a.epochs, train_beats=int(len(Y)), test_beats=int(len(Yte)),
               train_counts=np.bincount(Y, minlength=4).tolist(), loss_curve=curve, seconds=round(time.time() - t0))
    json.dump(res, open(f"{OUT}/{a.variant}_s{a.seed}.json", "w"), indent=2)
    print(a.variant, a.seed, "acc", round(res["accuracy"], 4), {c: (round(v["sensitivity"], 3), round(v["ppv"], 3)) for c, v in res["per_class"].items()})


if __name__ == "__main__":
    main()
