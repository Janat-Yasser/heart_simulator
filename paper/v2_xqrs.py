# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Is it the detector or the classifier? The same models, fed beats from a reference detector.

v2_final.py found that with the monitor's own QRS detector, ventricular beats are mostly lost.
To separate the two error sources, this script retrains the shipped recipe (local timing
features, class-weight exponent 0.5) and the blind-selected configuration with the same 5 seeds,
and tests each model on beats from (a) the database annotations, (b) the monitor's detector and
(c) XQRS, PhysioNet's reference detector from the wfdb package. Seeds are the same as in
v2_final.py, so (a) and (b) also check that training is reproducible on this machine.

    python paper/v2_xqrs.py --data path/to/mitdb   -> paper/runs_v2/xqrs.json
"""
import argparse
import json
import os

import numpy as np

from v2_common import CLASSES, DS2, RUNS, confusion, dataset, detect, detect_xqrs, match, predict, record, scores, train
from v2_common import DS1

SEEDS = [0, 1, 2, 3, 4]


def missed_beats(source, data):
    missed = np.zeros(4, int)
    tot = {"beats": 0, "detections": 0, "tp": 0}
    for rec in DS2:
        d = record(rec, data)
        det = detect(d["sig"]) if source == "detected" else detect_xqrs(d["raw"])
        m = match(d["peaks"], det)
        found = np.zeros(len(d["peaks"]), bool)
        found[m[m >= 0]] = True
        for c in range(4):
            missed[c] += int(((d["labels"] == c) & ~found).sum())
        tot["beats"] += len(d["peaks"]); tot["detections"] += len(det); tot["tp"] += int(found.sum())
    return missed, {"se": tot["tp"] / tot["beats"], "ppv": tot["tp"] / tot["detections"], **tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mitdb")
    a = ap.parse_args()
    sel = json.load(open(os.path.join(RUNS, "select.json")))["selected"]
    configs = list(dict.fromkeys([("local", 0.5), (sel["kind"], sel["balance"])]))
    out = {"detectors": {}, "configs": {}}
    miss = {}
    for src in ("detected", "xqrs"):
        miss[src], out["detectors"][src] = missed_beats(src, a.data)
        print(src, out["detectors"][src], "missed per class", miss[src].tolist(), flush=True)
    for kind, b in configs:
        X, T, Y, _ = dataset(DS1, a.data, kind, shifts=(-4, 0, 4))
        sets = {"annotated": dataset(DS2, a.data, kind, source="annotated"),
                "detected": dataset(DS2, a.data, kind, source="detected"),
                "xqrs": dataset(DS2, a.data, kind, source="xqrs")}
        res = {k: [] for k in sets}
        for seed in SEEDS:
            p = train(X, T, Y, seed=seed, balance=b)
            np.savez_compressed(os.path.join(RUNS, f"params_{kind}_b{b}_s{seed}.npz"), **p)
            for k, (Xs, Ts, Ys, _) in sets.items():
                cm = confusion(Ys, predict(p, Xs, Ts))
                per, _ = scores(cm, None if k == "annotated" else miss[k])
                res[k].append({"per_class": per, "confusion": cm.tolist()})
            print(kind, b, seed, {k: {c: (round(v[-1]["per_class"][c]["se"], 3), round(v[-1]["per_class"][c]["ppv"], 3))
                                     for c in ("S", "V")} for k, v in res.items()}, flush=True)
        summ = {}
        for k, runs in res.items():
            summ[k] = {c: {"se_mean": float(np.mean([r["per_class"][c]["se"] for r in runs])),
                           "se_sd": float(np.std([r["per_class"][c]["se"] for r in runs], ddof=1)),
                           "ppv_mean": float(np.mean([r["per_class"][c]["ppv"] for r in runs])),
                           "ppv_sd": float(np.std([r["per_class"][c]["ppv"] for r in runs], ddof=1))} for c in CLASSES}
            cm = np.array([r["confusion"] for r in runs]).mean(0)
            summ[k]["confusion_mean"] = cm.round(1).tolist()
        out["configs"][f"{kind}_b{b}"] = {"summary": summ, "runs": res}
        out["missed"] = {k: v.tolist() for k, v in miss.items()}
        json.dump(out, open(os.path.join(RUNS, "xqrs.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
