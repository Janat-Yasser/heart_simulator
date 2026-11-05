# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
Rhythm quiz: the monitor hides the rhythm name, plays a random strip, and asks
you to name it. The explanation afterwards uses what was measured on that strip.
"""

from dataclasses import dataclass

import numpy as np

from .ecg_generator import RhythmType

NAMES = {
    RhythmType.NORMAL_SINUS: "Normal sinus rhythm",
    RhythmType.BRADYCARDIA:  "Sinus bradycardia",
    RhythmType.TACHYCARDIA:  "Sinus tachycardia",
    RhythmType.AFIB:         "Atrial fibrillation",
    RhythmType.PVC:          "Sinus rhythm with PVCs",
    RhythmType.HEART_BLOCK:  "1st-degree AV block",
    RhythmType.VTACH:        "Ventricular tachycardia",
    RhythmType.STEMI:        "ST elevation (STEMI)",
}

# a believable rate range for each rhythm (bpm)
RATES = {
    RhythmType.NORMAL_SINUS: (62, 92),
    RhythmType.BRADYCARDIA:  (38, 48),
    RhythmType.TACHYCARDIA:  (125, 145),
    RhythmType.AFIB:         (110, 135),
    RhythmType.PVC:          (64, 88),
    RhythmType.HEART_BLOCK:  (58, 82),
    RhythmType.VTACH:        (155, 190),
    RhythmType.STEMI:        (72, 98),
}

LOOK_FOR = {
    RhythmType.NORMAL_SINUS: "a P before every QRS, regular RR, rate 60-100.",
    RhythmType.BRADYCARDIA:  "normal-looking beats, just under 50 a minute.",
    RhythmType.TACHYCARDIA:  "narrow QRS, P waves present, over 100 a minute.",
    RhythmType.AFIB:         "RR intervals that never repeat and a wavy baseline with no P waves.",
    RhythmType.PVC:          "an early, wide, bizarre beat with no P wave, then a pause.",
    RhythmType.HEART_BLOCK:  "a long, constant gap between each P and its QRS (over 200 ms).",
    RhythmType.VTACH:        "fast, wide, identical complexes with no P waves.",
    RhythmType.STEMI:        "the segment after the QRS sitting above the baseline.",
}


@dataclass
class Question:
    rhythm: RhythmType
    bpm: int
    options: list          # RhythmType, shuffled, includes the answer


def make_question(rng=None, previous=None) -> Question:
    rng = rng or np.random.default_rng()
    pool = [r for r in RhythmType if r != previous]
    rhythm = pool[int(rng.integers(len(pool)))]
    lo, hi = RATES[rhythm]
    bpm = int(rng.integers(lo, hi + 1))
    others = [r for r in RhythmType if r != rhythm]
    picks = [others[i] for i in rng.choice(len(others), 3, replace=False)]
    options = picks + [rhythm]
    rng.shuffle(options)
    return Question(rhythm, bpm, options)


def explain(rhythm: RhythmType, f: dict) -> str:
    """Why this strip is `rhythm`, in terms of what was measured on it."""
    if not f or not f.get('valid'):
        return "Look for " + LOOK_FOR[rhythm]
    hr, cv, qrs, pr = f['hr'], f['rr_cv'] * 100, f['qrs_ms'], f['pr_ms']
    why = {
        RhythmType.NORMAL_SINUS: f"Rate {hr:.0f}, RR varies only {cv:.0f}%, and every QRS has a P wave {pr:.0f} ms ahead of it.",
        RhythmType.BRADYCARDIA:  f"Rate {hr:.0f} bpm is under 50, but each beat still has a P wave and a narrow QRS ({qrs:.0f} ms).",
        RhythmType.TACHYCARDIA:  f"Rate {hr:.0f} bpm is over 100 with a narrow QRS ({qrs:.0f} ms) and P waves in front.",
        RhythmType.AFIB:         f"RR varies {cv:.0f}% from beat to beat with no steady P waves: irregularly irregular.",
        RhythmType.PVC:          f"{f['premature_wide']} beats on this strip arrive early and wide, with no P wave before them.",
        RhythmType.HEART_BLOCK:  f"Every P conducts, but the PR interval is {pr:.0f} ms (normal is 120-200).",
        RhythmType.VTACH:        f"Rate {hr:.0f} with a QRS of {qrs:.0f} ms in {f['wide_fraction'] * 100:.0f}% of beats: fast and wide.",
        RhythmType.STEMI:        f"The ST segment sits {f['st_level']:+.2f} mV off the baseline after each QRS.",
    }[rhythm]
    return why + " Look for " + LOOK_FOR[rhythm]
