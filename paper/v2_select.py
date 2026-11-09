# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Blind model selection: every choice is made on DS1 only, DS2 is not touched.

The first paper compared a few timing-feature and class-weight variants on DS2 (the test
patients), which it flagged as a weakness. Here the two choices are made by 4-fold
cross-validation over DS1 *patients* (records), so each validation fold is patients the
model hasn't seen, like DS2 will be:

  * class-weight exponent b in {0.25, 0.5, 0.75, 1.0}   (weight = (1/frequency) ** b)
  * timing features: "local" (the shipped 5) or "adaptive" (+ 2 normalised by the patient's
    own 5-minute rhythm)

Selection rule, fixed before running: highest macro F1 over N, S and V on the pooled
out-of-fold predictions (F is left out: 414 beats in DS1, too few to steer on).

    python paper/v2_select.py --data path/to/mitdb   -> paper/runs_v2/select.json
"""
import argparse
import json
import os
import time

import numpy as np

from v2_common import DS1, RUNS, confusion, dataset, predict, scores, train

FOLD_SEED = 7
BALANCES = [0.25, 0.5, 0.75, 1.0]
KINDS = ["local", "adaptive"]


def folds():
    recs = list(np.random.default_rng(FOLD_SEED).permutation(DS1))
    return [sorted(recs[i::4]) for i in range(4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mitdb")
    a = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    out_path = os.path.join(RUNS, "select.json")
    done = json.load(open(out_path)) if os.path.exists(out_path) else {"folds": folds(), "runs": {}}
    t0 = time.time()
    for kind in KINDS:
        data = {}
        for k, val in enumerate(done["folds"]):
            tr = [r for r in DS1 if r not in val]
            data[k] = (dataset(tr, a.data, kind, shifts=(-4, 0, 4)), dataset(val, a.data, kind))
        for b in BALANCES:
            key = f"{kind}_b{b}"
            if key in done["runs"]:
                continue
            cm = np.zeros((5, 5), int)
            for k in range(4):
                (X, T, Y, _), (Xv, Tv, Yv, _) = data[k]
                p = train(X, T, Y, seed=0, balance=b)
                cm += confusion(Yv, predict(p, Xv, Tv))
                print(f"{key} fold {k} done [{time.time() - t0:.0f}s]", flush=True)
            per, f1 = scores(cm)
            done["runs"][key] = {"kind": kind, "balance": b, "per_class": per, "f1": f1,
                                 "macro_f1_NSV": float(np.mean([f1["N"], f1["S"], f1["V"]])),
                                 "confusion": cm[:4, :4].tolist()}
            print(key, round(done["runs"][key]["macro_f1_NSV"], 4),
                  {c: (round(v["se"], 3), round(v["ppv"], 3)) for c, v in per.items()}, flush=True)
            json.dump(done, open(out_path, "w"), indent=2)
    best = max(done["runs"].values(), key=lambda r: r["macro_f1_NSV"])
    done["selected"] = {"kind": best["kind"], "balance": best["balance"], "macro_f1_NSV": best["macro_f1_NSV"]}
    json.dump(done, open(out_path, "w"), indent=2)
    print("selected", done["selected"])


if __name__ == "__main__":
    main()
