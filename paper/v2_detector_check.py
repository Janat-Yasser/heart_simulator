# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Why does the monitor's QRS detector lose ventricular beats on real ECG? (post hoc)

ai/classifier.detect_r_peaks looks for *positive* peaks and then drops peaks shorter than 0.6x
the median peak height (meant to reject tall T waves). Many real PVCs in lead II point
downward. This re-runs the detector, in the same 8-s windows, with one change at a time:
  * shipped:            unchanged
  * no height filter:   without the 0.6x-median rule
  * absolute value:     peaks searched on |signal| (both polarities), everything else unchanged
The shipped monitor code is not modified.

    python paper/v2_detector_check.py --data path/to/mitdb   -> paper/runs_v2/detector_variants.json
"""
import argparse
import json
import os

import numpy as np
from scipy.signal import find_peaks

from v2_common import DS2, FS_MODEL, RUNS, match, record
from ai.classifier import _bandpass


def detect_variant(sig, absval=False, height_filter=True):
    fs = FS_MODEL
    w, st = 8 * fs, 6 * fs
    found = []
    for a in range(0, max(1, len(sig) - w + st), st):
        seg = sig[a:a + w]
        if len(seg) < fs:
            break
        c = _bandpass(seg.astype(float), fs)
        c = np.abs(c) if absval else c
        amp = np.percentile(np.abs(c), 99.5)
        if amp < 0.2:
            continue
        pk, pr = find_peaks(c, height=0.45 * amp, distance=int(0.2 * fs), prominence=0.35 * amp)
        if height_filter and len(pk) >= 3:
            pk = pk[pr["peak_heights"] >= 0.6 * np.median(pr["peak_heights"])]
        lo = 0 if a == 0 else fs
        hi = len(seg) if a + w >= len(sig) else len(seg) - fs
        found += [a + p for p in pk if lo <= p < hi]
    f = np.unique(found)
    keep = [f[0]]
    for p in f[1:]:
        if p - keep[-1] > int(0.2 * fs):
            keep.append(p)
    return np.array(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mitdb")
    a = ap.parse_args()
    out = {}
    for name, kw in [("shipped", {}), ("no height filter", {"height_filter": False}), ("absolute value", {"absval": True})]:
        n = {"beats": 0, "det": 0, "tp": 0, "v": 0, "v_found": 0, "s": 0, "s_found": 0}
        for rec in DS2:
            d = record(rec, a.data)
            det = detect_variant(d["sig"], **kw)
            m = match(d["peaks"], det)
            found = np.zeros(len(d["peaks"]), bool)
            found[m[m >= 0]] = True
            n["beats"] += len(d["peaks"]); n["det"] += len(det); n["tp"] += int(found.sum())
            n["v"] += int((d["labels"] == 2).sum()); n["v_found"] += int(((d["labels"] == 2) & found).sum())
            n["s"] += int((d["labels"] == 1).sum()); n["s_found"] += int(((d["labels"] == 1) & found).sum())
        out[name] = {**{f"n_{k}": v for k, v in n.items()},
                     "qrs_se": n["tp"] / n["beats"], "qrs_ppv": n["tp"] / n["det"],
                     "v_found": n["v_found"] / n["v"], "s_found": n["s_found"] / n["s"]}
        print(name, {k: round(v, 4) for k, v in out[name].items() if isinstance(v, float)})
    json.dump(out, open(os.path.join(RUNS, "detector_variants.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
