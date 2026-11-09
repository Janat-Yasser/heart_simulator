# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Tables for the second version of the paper, from paper/runs_v2/ only.

  * mean ± SD over seeds for every configuration, on annotated and on detected beats;
  * patient-level bootstrap CIs (resampling the 22 DS2 patients, 2,000 times), because beats
    from one patient are not independent and DS2 has only 22 patients;
  * the paired difference adaptive − local at the selected class weight;
  * QRS detection Se / +P of the monitor's detector on DS2, per record.

    python paper/v2_report.py --data path/to/mitdb   -> paper/runs_v2/report.json
"""
import argparse
import glob
import json
import os

import numpy as np

from v2_common import CLASSES, DS2, RUNS, detect, match, record

B = 2000


def cm_by_record(y, p, r, recs):
    out = {}
    for rec in recs:
        m = r == int(rec)
        cm = np.zeros((5, 5), int)
        np.add.at(cm, (np.where(y[m] >= 0, y[m], 4), p[m]), 1)
        out[rec] = cm
    return out


def se_ppv(cm, c, missed=0):
    tp = cm[c, c]
    return tp / max(1, cm[c, :4].sum() + missed), tp / max(1, cm[:, c].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/mitdb")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    # detector on DS2, per record, and missed beats per class per record
    det_rows, missed = {}, {}
    for rec in DS2:
        d = record(rec, a.data)
        det = detect(d["sig"])
        m = match(d["peaks"], det)
        found = np.zeros(len(d["peaks"]), bool)
        found[m[m >= 0]] = True
        det_rows[rec] = {"beats": int(len(d["peaks"])), "detections": int(len(det)), "tp": int(found.sum())}
        missed[rec] = np.array([int(((d["labels"] == c) & ~found).sum()) for c in range(4)])
    tp = sum(v["tp"] for v in det_rows.values())
    detector = {"se": tp / sum(v["beats"] for v in det_rows.values()),
                "ppv": tp / sum(v["detections"] for v in det_rows.values()), "per_record": det_rows}

    configs = {}
    for f in sorted(glob.glob(os.path.join(RUNS, "final_*_s*_pred.npz"))):
        name = os.path.basename(f).rsplit("_s", 1)[0].replace("final_", "")
        configs.setdefault(name, []).append(np.load(f))

    report = {"detector": detector, "configs": {}}
    per_rec = {}
    for name, runs in configs.items():
        A = [cm_by_record(z["ya"], z["pa"], z["ra"], DS2) for z in runs]
        D = [cm_by_record(z["yd"], z["pd"], z["rd"], DS2) for z in runs]
        per_rec[name] = (A, D)

        def stat(cms, recs, use_missed):
            vals = {}
            for c, cls in enumerate(CLASSES):
                s, p_ = [], []
                for run in cms:
                    cm = sum(run[r] for r in recs)
                    mi = sum(missed[r][c] for r in recs) if use_missed else 0
                    x, y = se_ppv(cm, c, mi)
                    s.append(x); p_.append(y)
                vals[cls] = (np.array(s), np.array(p_))
            return vals

        entry = {"seeds": len(runs)}
        for label, cms, um in (("annotated", A, False), ("detected", D, True)):
            full = stat(cms, DS2, um)
            boots = {cls: [] for cls in CLASSES}
            for _ in range(B):
                recs = list(rng.choice(DS2, len(DS2), replace=True))
                bs = stat(cms, recs, um)
                for cls in CLASSES:
                    boots[cls].append((bs[cls][0].mean(), bs[cls][1].mean()))
            res = {}
            for cls in CLASSES:
                bb = np.array(boots[cls])
                res[cls] = {"se_mean": float(full[cls][0].mean()), "se_sd": float(full[cls][0].std(ddof=1)),
                            "ppv_mean": float(full[cls][1].mean()), "ppv_sd": float(full[cls][1].std(ddof=1)),
                            "se_ci": [float(np.percentile(bb[:, 0], 2.5)), float(np.percentile(bb[:, 0], 97.5))],
                            "ppv_ci": [float(np.percentile(bb[:, 1], 2.5)), float(np.percentile(bb[:, 1], 97.5))],
                            "per_seed_se": full[cls][0].round(4).tolist(), "per_seed_ppv": full[cls][1].round(4).tolist()}
            acc = []
            for run in cms:
                cm = sum(run[r] for r in DS2)
                acc.append(np.trace(cm[:4, :4]) / cm[:4, :4].sum())
            res["accuracy_mean"], res["accuracy_sd"] = float(np.mean(acc)), float(np.std(acc, ddof=1))
            entry[label] = res
        # mean row-normalised confusion (annotated)
        cms = np.array([sum(run[r] for r in DS2)[:4, :4] for run in A])
        entry["confusion_annotated_mean"] = cms.mean(0).round(1).tolist()
        report["configs"][name] = entry

    sel = json.load(open(os.path.join(RUNS, "select.json")))["selected"]
    b = sel["balance"]
    la, ad = f"local_b{b}", f"adaptive_b{b}"
    if la in per_rec and ad in per_rec:
        diffs = {}
        for c, cls in enumerate(CLASSES[:3]):
            def mean_se(name, recs, k):
                out = []
                for run in per_rec[name][0]:
                    cm = sum(run[r] for r in recs)
                    out.append(se_ppv(cm, c)[k])
                return np.mean(out)
            point = [mean_se(ad, DS2, k) - mean_se(la, DS2, k) for k in (0, 1)]
            bs = []
            for _ in range(B):
                recs = list(rng.choice(DS2, len(DS2), replace=True))
                bs.append([mean_se(ad, recs, k) - mean_se(la, recs, k) for k in (0, 1)])
            bs = np.array(bs)
            diffs[cls] = {"se_diff": float(point[0]), "se_ci": np.percentile(bs[:, 0], [2.5, 97.5]).tolist(),
                          "ppv_diff": float(point[1]), "ppv_ci": np.percentile(bs[:, 1], [2.5, 97.5]).tolist()}
        report["adaptive_minus_local"] = {"balance": b, "annotated": diffs}
        # per-patient S sensitivity, for the figure
        pp = {}
        for name in (la, ad):
            rows = {}
            for rec in DS2:
                cms = [run[rec] for run in per_rec[name][0]]
                n = cms[0][1, :4].sum()
                if n >= 20:
                    rows[rec] = {"s_beats": int(n), "s_se": float(np.mean([cm[1, 1] / n for cm in cms]))}
            pp[name] = rows
        report["per_patient_S"] = pp
    report["selected"] = sel
    json.dump(report, open(os.path.join(RUNS, "report.json"), "w"), indent=2)
    print("detector Se %.4f +P %.4f" % (detector["se"], detector["ppv"]))
    for name, e in report["configs"].items():
        for label in ("annotated", "detected"):
            r = e[label]
            print(name, label, "acc %.3f" % r["accuracy_mean"],
                  {c: "Se %.1f [%.1f-%.1f] +P %.1f [%.1f-%.1f]" % (100 * r[c]["se_mean"], 100 * r[c]["se_ci"][0], 100 * r[c]["se_ci"][1],
                                                               100 * r[c]["ppv_mean"], 100 * r[c]["ppv_ci"][0], 100 * r[c]["ppv_ci"][1]) for c in CLASSES})
    print(json.dumps(report.get("adaptive_minus_local"), indent=1))


if __name__ == "__main__":
    main()
