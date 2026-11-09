# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Figures for the second version of the paper, from paper/runs_v2/ only.
    python paper/v2_figures.py   -> paper/fig/pipeline.png, paper/fig/patients_s.png"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS, FIG = os.path.join(HERE, "runs_v2"), os.path.join(HERE, "fig")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
INK, MUTED, GRID = "#2b2b2a", "#8a8980", "#e4e3dd"
COL = {"annotated": "#2a78d6", "xqrs": "#1baf7a", "detected": "#eb6834"}
LAB = {"annotated": "annotated R peaks", "xqrs": "reference detector (XQRS)", "detected": "monitor's original detector"}


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


x = json.load(open(os.path.join(RUNS, "xqrs.json")))["configs"]["local_b0.5"]["summary"]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9))
for ax, cls in zip(axs, ("V", "S")):
    for k, src in enumerate(("annotated", "xqrs", "detected")):
        vals = [x[src][cls]["se_mean"], x[src][cls]["ppv_mean"]]
        errs = [x[src][cls]["se_sd"], x[src][cls]["ppv_sd"]]
        pos = np.arange(2) + (k - 1) * 0.26
        ax.bar(pos, vals, width=0.24, color=COL[src], label=LAB[src], yerr=errs, error_kw={"lw": 0.8, "ecolor": INK, "capsize": 2})
        for p, v, e in zip(pos, vals, errs):
            ax.text(p, v + e + 0.03, f"{100 * v:.0f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks([0, 1], ["sensitivity", "positive predictivity"], fontsize=8, color=INK)
    ax.set_title({"V": "Ventricular ectopic beats (V)", "S": "Supraventricular ectopic beats (S)"}[cls],
                 fontsize=9, color=INK, loc="left")
    ax.set_ylim(0, 1.15); ax.set_yticks([0, .25, .5, .75, 1])
    style(ax)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower center", ncol=3)
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(os.path.join(FIG, "pipeline.png"), dpi=220)

# per-patient S sensitivity
from v2_common import DS2, record  # noqa: E402
runs = [np.load(f) for f in sorted(glob.glob(os.path.join(RUNS, "final_local_b0.5_s*_pred.npz")))]
rows = []
for rec in DS2:
    n = int(sum((runs[0]["ra"] == int(rec)) & (runs[0]["ya"] == 1)))
    if n >= 30:
        se = np.mean([((z["pa"][(z["ra"] == int(rec)) & (z["ya"] == 1)]) == 1).mean() for z in runs])
        rows.append((rec, n, se))
rows.sort(key=lambda r: -r[1])
fig, ax = plt.subplots(figsize=(5.2, 2.4))
y = np.arange(len(rows))[::-1]
ax.barh(y, [r[2] for r in rows], height=0.6, color=["#eb6834" if r[0] == "232" else "#2a78d6" for r in rows])
for yy, r in zip(y, rows):
    ax.text(r[2] + 0.015, yy, f"{100 * r[2]:.0f}%", va="center", fontsize=7.5, color=INK)
ax.set_yticks(y, [f"record {r[0]}  ({r[1]:,} S beats)" for r in rows], fontsize=8, color=INK)
ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.tick_params(axis="x", colors=MUTED, labelsize=8)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_xlabel("S beats detected (mean of 5 runs)", fontsize=8, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "patients_s.png"), dpi=220)
print("written")
