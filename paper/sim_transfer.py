# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Labels the classifier gives to the simulator's own rhythms (Table 4 of the paper).
    python paper/sim_transfer.py   -> paper/sim_transfer.json"""
import os
import json, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.beat_model import BeatClassifier
from ai.classifier import _bandpass, detect_r_peaks
from core.ecg_generator import ECGParameters, RhythmType, generate_ecg, get_rhythm_hr
FS = 500
clf = BeatClassifier()
out = {}
for rh in RhythmType:
    counts = {"N": 0, "S": 0, "V": 0, "F": 0}; tp = fp = fn = 0; total = 0; missed = 0
    for seed in range(5):
        for bpm in (60, 75, 90):
            hr = get_rhythm_hr(rh, bpm)
            p = ECGParameters(bpm=hr, rhythm=rh)
            _, s, true_peaks = generate_ecg(30, p, seed=seed)
            # ground truth: which true beats are ventricular
            ect = [(rh == RhythmType.VTACH) or (rh == RhythmType.PVC and i % 4 == 3) for i in range(len(true_peaks))]
            res = clf.classify(s, FS, detect_r_peaks(_bandpass(s, FS), FS))
            tp_arr = np.array(true_peaks)
            for idx, lab, conf in res:
                j = int(np.argmin(np.abs(tp_arr - idx)))
                if abs(tp_arr[j] - idx) > 0.075 * FS:
                    missed += 1; continue
                counts[lab] += 1; total += 1
                isv = ect[j]
                if lab == "V" and isv: tp += 1
                elif lab == "V" and not isv: fp += 1
                elif lab != "V" and isv: fn += 1
    out[rh.value] = {"beats": total, "fractions": {k: round(v / max(1, total), 3) for k, v in counts.items()},
                     "V_sens": round(tp / (tp + fn), 3) if tp + fn else None, "V_ppv": round(tp / (tp + fp), 3) if tp + fp else None,
                     "unmatched": missed}
    print(rh.value, out[rh.value])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_transfer.json"), "w"), indent=2)
