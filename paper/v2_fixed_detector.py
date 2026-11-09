# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""After the detector fix (developed while looking at DS2, so it is also checked on DS1, which played
no part in developing it).

 (ai/classifier.detect_r_peaks now also finds downward complexes):
QRS detection on DS2, and the five original-recipe models end to end, reusing the weights
saved by v2_xqrs.py (no retraining). Results with the original detector are in final_*.json,
xqrs.json, rhythm.json and detector_variants.json, produced at commit 51f6d16.

    python paper/v2_fixed_detector.py --data path/to/mitdb   -> paper/runs_v2/fixed_detector.json
"""
import argparse
import json
import os

import numpy as np

from v2_common import CLASSES, DS2, RUNS, confusion, dataset, detect, match, predict, record, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mitdb")
    a = ap.parse_args()
    n = {"beats": 0, "det": 0, "tp": 0}
    missed = np.zeros(4, int)
    found_c = np.zeros(4, int)
    total_c = np.zeros(4, int)
    for rec in DS2:
        d = record(rec, a.data)
        det = detect(d["sig"])
        m = match(d["peaks"], det)
        found = np.zeros(len(d["peaks"]), bool)
        found[m[m >= 0]] = True
        n["beats"] += len(d["peaks"]); n["det"] += len(det); n["tp"] += int(found.sum())
        for c in range(4):
            missed[c] += int(((d["labels"] == c) & ~found).sum())
            found_c[c] += int(((d["labels"] == c) & found).sum())
            total_c[c] += int((d["labels"] == c).sum())
    out = {"qrs_se": n["tp"] / n["beats"], "qrs_ppv": n["tp"] / n["det"], **n,
           "found_by_class": {c: float(found_c[i] / total_c[i]) for i, c in enumerate(CLASSES)}, "missed": missed.tolist()}
    # the same detection scoring on DS1, for the fixed detector and the original one
    from v2_common import DS1
    from v2_detector_check import detect_variant
    for name, fn in (("fixed", lambda d: detect(d["sig"])), ("original", lambda d: detect_variant(d["sig"]))):
        k = {"beats": 0, "det": 0, "tp": 0, "v": 0, "v_found": 0}
        for rec in DS1:
            d = record(rec, a.data)
            det = fn(d)
            m = match(d["peaks"], det)
            found = np.zeros(len(d["peaks"]), bool)
            found[m[m >= 0]] = True
            k["beats"] += len(d["peaks"]); k["det"] += len(det); k["tp"] += int(found.sum())
            k["v"] += int((d["labels"] == 2).sum()); k["v_found"] += int(((d["labels"] == 2) & found).sum())
        out[f"ds1_{name}"] = {"qrs_se": k["tp"] / k["beats"], "qrs_ppv": k["tp"] / k["det"], "v_found": k["v_found"] / k["v"], **{f"n_{x}": y for x, y in k.items()}}
        print("DS1", name, {x: round(y, 4) for x, y in out[f"ds1_{name}"].items() if isinstance(y, float)})
    X, T, Y, _ = dataset(DS2, a.data, "local", source="detected")
    runs = []
    for seed in range(5):
        z = np.load(os.path.join(RUNS, f"params_local_b0.5_s{seed}.npz"))
        p = {k: z[k] for k in z.files}
        cm = confusion(Y, predict(p, X, T))
        per, _ = scores(cm, missed)
        runs.append({"per_class": per, "confusion": cm.tolist()})
    out["end_to_end"] = {c: {"se_mean": float(np.mean([r["per_class"][c]["se"] for r in runs])),
                             "se_sd": float(np.std([r["per_class"][c]["se"] for r in runs], ddof=1)),
                             "ppv_mean": float(np.mean([r["per_class"][c]["ppv"] for r in runs])),
                             "ppv_sd": float(np.std([r["per_class"][c]["ppv"] for r in runs], ddof=1))} for c in CLASSES}
    out["runs"] = runs
    json.dump(out, open(os.path.join(RUNS, "fixed_detector.json"), "w"), indent=2)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items() if k not in ("runs", "end_to_end")})
    print({c: (round(v["se_mean"], 3), round(v["ppv_mean"], 3)) for c, v in out["end_to_end"].items()})


if __name__ == "__main__":
    main()
