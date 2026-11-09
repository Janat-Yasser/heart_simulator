# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Post hoc: how much of the DS2 S-beat result is one patient (record 232)?
    python paper/v2_s232.py   -> paper/runs_v2/s_by_232.json"""
import glob
import json
import os

import numpy as np

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs_v2")
out = {}
for cfg in ("local_b0.5", "adaptive_b0.25", "local_b0.25"):
    runs = [np.load(f) for f in sorted(glob.glob(os.path.join(RUNS, f"final_{cfg}_s*_pred.npz")))]
    s232, sother, false_s = [], [], []
    for z in runs:
        y, p, r = z["ya"], z["pa"], z["ra"]
        m = (y == 1) & (r == 232)
        s232.append(float((p[m] == 1).mean()))
        m = (y == 1) & (r != 232)
        sother.append(float((p[m] == 1).mean()))
        false_s.append(int(((p == 1) & (y != 1)).sum()))
    out[cfg.replace(".", "_")] = {"se_232": float(np.mean(s232)), "se_other": float(np.mean(sother)),
                                  "se_other_sd": float(np.std(sother, ddof=1)), "runs": len(runs),
                                  "false_S_per_run": float(np.mean(false_s))}
z = runs[0]
n_s = int((z["ya"] == 1).sum()); n_232 = int(((z["ya"] == 1) & (z["ra"] == 232)).sum())
out.update(n_S=n_s, n_232=n_232, n_other=n_s - n_232, share_of_S=n_232 / n_s)
json.dump(out, open(os.path.join(RUNS, "s_by_232.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
