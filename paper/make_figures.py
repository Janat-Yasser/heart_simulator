# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""Figures for the paper: architecture, class waveforms, confusion matrix.
Run after paper/ablate.py (needs paper/beats_cache.npz and paper/runs/full_s0.json)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
C = ["N", "S", "V", "F"]

# architecture
fig, ax = plt.subplots(figsize=(7.4, 2.3)); ax.set_xlim(-1.5, 101.5); ax.set_ylim(-1, 32); ax.axis("off")
def box(x, y, w, h, t, fc="#EEF1F6"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1", fc=fc, ec="#7A869A", lw=0.9))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.6)
def arr(a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color="#555", lw=0.9, shrinkA=0, shrinkB=0))
box(0, 11, 10, 8, "ECG\n(MLII lead)"); box(13, 11, 12, 8, "Resample\n360 Hz, band-\npass 0.5–40 Hz")
box(28, 19, 11, 8, "0.6 s window\nnormalised\nper beat", "#E8F0FB"); box(42, 19, 10, 8, "Conv 7×16\nReLU, pool 2", "#E8F0FB")
box(55, 19, 10, 8, "Conv 5×32\nReLU, pool 2", "#E8F0FB"); box(68, 19, 9, 8, "Conv 5×32\nReLU\n+ avg pool", "#E8F0FB")
box(28, 3, 24, 8, "5 RR timing features:\nprematurity, pause,\nlocal rhythm", "#FBF0E6")
box(80, 11, 8, 8, "Concat\n37", "#EEEEEE"); box(91, 11, 9, 8, "Dense 32\n→ softmax\nN, S, V, F", "#EEEEEE")
for a, b in [((10.3, 15), (12.7, 15)), ((25.3, 16.5), (27.7, 23)), ((25.3, 13.5), (27.7, 7)), ((39.3, 23), (41.7, 23)),
             ((52.3, 23), (54.7, 23)), ((65.3, 23), (67.7, 23)), ((77.3, 23), (79.7, 17)), ((52.3, 7), (79.7, 13)), ((88.3, 15), (90.7, 15))]:
    arr(a, b)
ax.text(52, 30.2, "morphology branch (convolutional)", ha="center", fontsize=7.5, color="#3A5A8C")
ax.text(40, -0.6, "timing branch", ha="center", fontsize=7.5, color="#9A5A1C")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "arch.png"), dpi=220)

# class waveforms
z = np.load(os.path.join(HERE, "beats_cache.npz")); Xte, Yte = z["Xte"], z["Yte"]
names = {"N": "Normal (N)", "S": "Supraventricular (S)", "V": "Ventricular (V)", "F": "Fusion (F)"}
cols = {"N": "#4C72B0", "S": "#55A868", "V": "#C44E52", "F": "#8172B2"}
t = (np.arange(Xte.shape[1]) - 90) / 360 * 1000
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.1), sharey=True)
for k, c in enumerate(C):
    X = Xte[Yte == k]; m, s = X.mean(0), X.std(0)
    axs[k].fill_between(t, m - s, m + s, color=cols[c], alpha=.18, lw=0); axs[k].plot(t, m, color=cols[c], lw=1.4)
    axs[k].set_title(f"{names[c]}\nn = {len(X):,}", fontsize=8.5); axs[k].set_xlabel("ms from R peak", fontsize=8)
    axs[k].axvline(0, color="#999", lw=.6, ls=":"); axs[k].spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("normalised amplitude", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "beats.png"), dpi=220)

# confusion matrix of the shipped (seed 0) model
cm = np.array(json.load(open(os.path.join(HERE, "runs", "full_s0.json")))["confusion"], float)
cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{100 * cmn[i, j]:.1f}%\n({int(cm[i, j]):,})", ha="center", va="center", fontsize=7.2,
                color="white" if cmn[i, j] > .6 else "#222")
ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(C); ax.set_yticklabels(C)
ax.set_xlabel("Predicted"); ax.set_ylabel("True (annotation)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "confusion.png"), dpi=220)
print("wrote paper/fig/arch.png, beats.png, confusion.png")
