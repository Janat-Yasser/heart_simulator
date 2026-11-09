# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Final models: trained on all of DS1, tested once on DS2.

Configurations (5 seeds each):
  * "shipped recipe": local timing features, class-weight exponent 0.5 (what ai/beat_cnn.npz used),
    re-run here so every number in the comparison comes from the same software versions;
  * the configuration chosen blind on DS1 by v2_select.py;
  * that configuration with the other timing-feature set, to isolate the adaptive features.

Each model is tested two ways on DS2:
  * annotated: on the database's R peaks (as in the first paper);
  * detected: on the beats found by the monitor's own QRS detector (paper/v2_common.detect).
    Beats the detector misses count as missed for their class, and labels given to detections
    that match no beat count against +P.

    python paper/v2_final.py --data path/to/mitdb   -> paper/runs_v2/final_<config>_s<seed>.json
"""
import argparse
import json
import os
import time

import numpy as np

from v2_common import DS1, DS2, RUNS, confusion, dataset, match, predict, record, scores, train, detect

SEEDS = [0, 1, 2, 3, 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mitdb")
    a = ap.parse_args()
    sel = json.load(open(os.path.join(RUNS, "select.json")))["selected"]
    other = "local" if sel["kind"] == "adaptive" else "adaptive"
    configs = [("local", 0.5), (sel["kind"], sel["balance"]), (other, sel["balance"])]
    configs = list(dict.fromkeys(configs))
    t0 = time.time()

    # beats of each DS2 class the detector never finds (they can't be classified at all)
    missed = np.zeros(4, int)
    for rec in DS2:
        d = record(rec, a.data)
        m = match(d["peaks"], detect(d["sig"]))
        found = np.zeros(len(d["peaks"]), bool)
        found[m[m >= 0]] = True
        for c in range(4):
            missed[c] += int(((d["labels"] == c) & ~found).sum())

    for kind, b in configs:
        X, T, Y, _ = dataset(DS1, a.data, kind, shifts=(-4, 0, 4))
        Xa, Ta, Ya, Ra = dataset(DS2, a.data, kind, source="annotated")
        Xd, Td, Yd, Rd = dataset(DS2, a.data, kind, source="detected")
        for seed in SEEDS:
            name = f"final_{kind}_b{b}_s{seed}"
            path = os.path.join(RUNS, name + ".json")
            if os.path.exists(path):
                continue
            p = train(X, T, Y, seed=seed, balance=b)
            pa, pd_ = predict(p, Xa, Ta), predict(p, Xd, Td)
            cma, cmd = confusion(Ya, pa), confusion(Yd, pd_)
            per_a, f1_a = scores(cma)
            per_d, f1_d = scores(cmd, missed)
            json.dump({"kind": kind, "balance": b, "seed": seed,
                       "annotated": {"per_class": per_a, "f1": f1_a, "accuracy": float(np.trace(cma[:4, :4]) / cma[:4, :4].sum()),
                                     "confusion": cma[:4, :4].tolist()},
                       "detected": {"per_class": per_d, "f1": f1_d, "confusion": cmd.tolist(), "missed": missed.tolist()}},
                      open(path, "w"), indent=2)
            np.savez_compressed(os.path.join(RUNS, name + "_pred.npz"), ya=Ya, pa=pa, ra=Ra, yd=Yd, pd=pd_, rd=Rd)
            print(name, f"[{time.time() - t0:.0f}s]",
                  {c: (round(v["se"], 3), round(v["ppv"], 3)) for c, v in per_a.items()}, "| detected",
                  {c: (round(v["se"], 3), round(v["ppv"], 3)) for c, v in per_d.items()}, flush=True)


if __name__ == "__main__":
    main()
