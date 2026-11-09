# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""The monitor's rule-based rhythm analyser (ai/classifier.classify, unchanged) on real ECG.

So far it had only been tested on the simulator's own strips. Here it reads 10-second strips
of the 44 non-paced MIT-BIH records, lead MLII, and its output is compared with the
cardiologists' rhythm annotations and with the heart rate from the annotated beats.
Only strips with one rhythm annotation for the whole 10 s are used.

    python paper/v2_rhythm.py --data path/to/mitdb   -> paper/runs_v2/rhythm.json
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

from v2_common import DS1, DS2, FS_MODEL, RUNS, record
from ai.classifier import classify

STRIP = 10
NAMES = {"(N": "Sinus rhythm", "(AFIB": "Atrial fibrillation", "(AFL": "Atrial flutter",
         "(B": "Ventricular bigeminy", "(T": "Ventricular trigeminy", "(VT": "Ventricular tachycardia",
         "(SBR": "Sinus bradycardia", "(SVTA": "Supraventricular tachyarrhythmia", "(NOD": "Nodal rhythm",
         "(IVR": "Idioventricular rhythm", "(BII": "2nd-degree block", "(PREX": "Pre-excitation (WPW)",
         "(VFL": "Ventricular flutter", "(AB": "Atrial bigeminy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mitdb")
    ap.add_argument("--tag", default="", help="suffix for the output files, e.g. _fixed")
    a = ap.parse_args()
    n = STRIP * FS_MODEL
    rows = []
    for rec in DS1 + DS2:
        d = record(rec, a.data)
        at, rh = d["rhythm_at"], d["rhythm"]
        for s in range(0, len(d["raw"]) - n + 1, n):
            k = np.searchsorted(at, s, side="right") - 1
            if k < 0 or (k + 1 < len(at) and at[k + 1] < s + n):
                continue                                   # no rhythm yet, or it changes inside the strip
            beats = d["peaks"][(d["peaks"] >= s) & (d["peaks"] < s + n)]
            if len(beats) < 3:
                continue
            ref_hr = 60.0 / np.mean(np.diff(beats) / FS_MODEL)
            res = classify(d["raw"][s:s + n].astype(float), FS_MODEL)
            rows.append({"rec": rec, "start_s": s / FS_MODEL, "rhythm": str(rh[k]), "ref_hr": float(ref_hr),
                         "hr": float(res.hr_bpm), "label": res.label})
    hr_err = np.array([r["hr"] - r["ref_hr"] for r in rows if r["label"] != "No Signal"])
    by = defaultdict(Counter)
    for r in rows:
        by[r["rhythm"]][r["label"]] += 1
    af_true = np.array([r["rhythm"] == "(AFIB" for r in rows])
    af_call = np.array([r["label"] == "Atrial Fibrillation" for r in rows])
    vt_true = np.array([r["rhythm"] == "(VT" for r in rows])
    vt_call = np.array([r["label"] in ("Ventricular Tachycardia", "Wide-Complex Tachycardia") for r in rows])
    sinus = np.array([r["rhythm"] == "(N" for r in rows])
    abs_err = np.abs(hr_err)
    sinus_err = np.array([r["hr"] - r["ref_hr"] for r in rows if r["rhythm"] == "(N" and r["label"] != "No Signal"])
    out = {
        "strips": len(rows), "records": len({r["rec"] for r in rows}),
        "hr": {"mae": float(abs_err.mean()), "median_abs": float(np.median(abs_err)),
               "within_5bpm": float((abs_err <= 5).mean()), "within_10bpm": float((abs_err <= 10).mean()),
               "mean_signed": float(hr_err.mean())},
        "hr_sinus_only": {"mae": float(np.abs(sinus_err).mean()), "within_5bpm": float((np.abs(sinus_err) <= 5).mean())},
        "no_signal_strips": int(sum(r["label"] == "No Signal" for r in rows)),
        "af": {"strips": int(af_true.sum()), "sensitivity": float((af_call & af_true).sum() / max(1, af_true.sum())),
               "specificity": float((~af_call & ~af_true).sum() / max(1, (~af_true).sum())),
               "ppv": float((af_call & af_true).sum() / max(1, af_call.sum())),
               "af_records": sorted({r["rec"] for r in rows if r["rhythm"] == "(AFIB"})},
        "vt": {"strips": int(vt_true.sum()), "sensitivity": float((vt_call & vt_true).sum() / max(1, vt_true.sum())),
               "false_calls_elsewhere": int((vt_call & ~vt_true).sum())},
        "sinus": {"strips": int(sinus.sum()),
                  "called_normal_sinus": float(np.mean([r["label"] == "Normal Sinus Rhythm" for r in rows if r["rhythm"] == "(N"])),
                  "called_af": float(np.mean([r["label"] == "Atrial Fibrillation" for r in rows if r["rhythm"] == "(N"]))},
        "by_rhythm": {NAMES.get(k, k): dict(v.most_common()) for k, v in sorted(by.items(), key=lambda kv: -sum(kv[1].values()))},
    }
    os.makedirs(RUNS, exist_ok=True)
    json.dump(out, open(os.path.join(RUNS, f"rhythm{a.tag}.json"), "w"), indent=2)
    json.dump(rows, open(os.path.join(RUNS, f"rhythm_strips{a.tag}.json"), "w"))
    print(json.dumps({k: v for k, v in out.items() if k != "by_rhythm"}, indent=1))
    for k, v in out["by_rhythm"].items():
        print(k, sum(v.values()), v)


if __name__ == "__main__":
    main()
