# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
ECG Signal Generator
Gaussian-basis synthesis of P, QRS and T waves, timed in real seconds around each
R peak (QT follows Bazett, QRS width doesn't scale with rate). Supports 8 rhythms.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class RhythmType(Enum):
    NORMAL_SINUS   = "Normal Sinus Rhythm"
    BRADYCARDIA    = "Bradycardia"
    TACHYCARDIA    = "Tachycardia"
    AFIB           = "Atrial Fibrillation"
    PVC            = "Premature Ventricular Contraction"
    HEART_BLOCK    = "1st Degree Heart Block"
    VTACH          = "Ventricular Tachycardia"
    STEMI          = "ST Elevation (STEMI)"


RHYTHM_INFO = {
    RhythmType.NORMAL_SINUS:  ("Normal",       "#00e676", "Regular rhythm, 60–100 bpm"),
    RhythmType.BRADYCARDIA:   ("Bradycardia",   "#4fc3f7", "Slow heart rate < 50 bpm"),
    RhythmType.TACHYCARDIA:   ("Tachycardia",   "#ffb74d", "Fast heart rate > 100 bpm"),
    RhythmType.AFIB:          ("AFib",          "#ef5350", "Irregular rhythm, no P waves"),
    RhythmType.PVC:           ("PVC",           "#ef5350", "Early ectopic beats every 4th"),
    RhythmType.HEART_BLOCK:   ("1° Block",      "#ce93d8", "Prolonged PR interval > 200 ms"),
    RhythmType.VTACH:         ("V-Tach",        "#ff1744", "Wide QRS > 100 bpm — DANGER"),
    RhythmType.STEMI:         ("STEMI",         "#ff6d00", "ST elevation — possible MI"),
}


@dataclass
class ECGParameters:
    bpm:          float = 72.0
    sample_rate:  int   = 500
    rhythm: RhythmType  = RhythmType.NORMAL_SINUS

    # Wave amplitudes (mV)
    p_amp:  float = 0.15
    q_amp:  float = 0.10
    r_amp:  float = 1.00
    s_amp:  float = 0.25
    t_amp:  float = 0.30
    st_elevation: float = 0.0   # mV ST shift (STEMI)

    # Intervals in SECONDS (real ECG intervals don't scale with heart rate the way
    # a fraction of the RR interval would: a ventricular beat is wide at any rate).
    pr_interval: float = 0.16   # P onset -> QRS onset (1st-degree block: > 0.20 s)
    qrs_sigma:   float = 0.010  # R wave width (normal QRS ~ 80-100 ms)
    qtc:         float = 0.40   # Bazett-corrected QT; QT = QTc * sqrt(RR)

    # Noise
    baseline_noise: float = 0.012
    muscle_noise:   float = 0.006


def _add_gauss(sig: np.ndarray, t: np.ndarray, centre: float, sigma: float, amp: float, fs: int) -> None:
    """Add a Gaussian wave (in place), only computing the samples that matter (+-5 sigma)."""
    lo = max(0, int((centre - 5 * sigma) * fs))
    hi = min(len(t), int((centre + 5 * sigma) * fs) + 1)
    if hi > lo:
        sig[lo:hi] += amp * np.exp(-0.5 * ((t[lo:hi] - centre) / sigma) ** 2)


def _add_plateau(sig, t, start, end, level, fs, edge=0.015):
    """Smooth step up at `start`, down at `end` (used for ST elevation)."""
    lo = max(0, int((start - 4 * edge) * fs)); hi = min(len(t), int((end + 4 * edge) * fs) + 1)
    if hi > lo:
        x = t[lo:hi]
        sig[lo:hi] += level * 0.25 * (1 + np.tanh((x - start) / edge)) * (1 - np.tanh((x - end) / edge))


def _beat(sig, t, r_time, rr, p: ECGParameters, fs, ectopic=False, p_wave=True):
    """Draw one beat centred on its R peak at `r_time` (seconds)."""
    qt = p.qtc * np.sqrt(rr)                       # Bazett: QT shortens at faster rates
    if not ectopic:
        qrs_on = r_time - 0.045
        if p_wave:
            # P wave peaks ~50 ms after P onset; PR = P onset -> QRS onset
            _add_gauss(sig, t, qrs_on - p.pr_interval + 0.05, 0.022, p.p_amp, fs)
        _add_gauss(sig, t, r_time - 0.025, 0.008, -p.q_amp, fs)
        _add_gauss(sig, t, r_time, p.qrs_sigma, p.r_amp, fs)
        _add_gauss(sig, t, r_time + 0.025, 0.008, -p.s_amp, fs)
        # T peak placed so the tangent-method T end (peak + 2 sigma) lands at qrs_on + QT
        t_peak = qrs_on + qt - 0.08
        _add_gauss(sig, t, t_peak, 0.04, p.t_amp, fs)
        if p.st_elevation:
            _add_plateau(sig, t, r_time + 0.05, t_peak - 0.03, p.st_elevation, fs)
    else:
        # Ventricular (ectopic) beat: no P wave, wide bizarre QRS (~160 ms), discordant T
        _add_gauss(sig, t, r_time, 0.028, 1.45 * p.r_amp, fs)
        _add_gauss(sig, t, r_time + 0.065, 0.028, -0.80 * p.r_amp, fs)
        _add_gauss(sig, t, r_time - 0.08 + qt + 0.02, 0.06, -0.35, fs)


def generate_ecg(duration_sec: float, params: ECGParameters,
                 seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Returns (time, signal, r_peak_indices).
    r_peak_indices are sample indices of each R-peak, used for audio sync.
    """
    fs = params.sample_rate
    n  = int(duration_sec * fs)
    t  = np.arange(n) / fs
    s  = np.zeros(n)
    rng = np.random.default_rng(seed)

    p = params
    if p.rhythm == RhythmType.HEART_BLOCK:
        p = ECGParameters(**{**params.__dict__, 'pr_interval': max(params.pr_interval, 0.28)})
    elif p.rhythm == RhythmType.STEMI and params.st_elevation < 0.10:
        # the STEMI rhythm should show ST elevation even with the ST slider at zero
        p = ECGParameters(**{**params.__dict__, 'st_elevation': 0.25})

    rr_list = _build_rr(p, duration_sec, rng)
    r_peaks = []
    r_time = 0.35
    for i, rr in enumerate(rr_list):
        if i > 0:
            r_time += rr
        if r_time >= duration_sec:
            break
        ectopic = (p.rhythm == RhythmType.VTACH) or (p.rhythm == RhythmType.PVC and i % 4 == 3)
        p_wave = p.rhythm != RhythmType.AFIB
        _beat(s, t, r_time, rr if i else rr_list[0], p, fs, ectopic=ectopic, p_wave=p_wave)
        r_peaks.append(int(round(r_time * fs)))

    if p.rhythm == RhythmType.AFIB:
        # Fibrillatory "f waves": chaotic 4-9 Hz atrial activity replacing P waves
        for _ in range(4):
            f = rng.uniform(4.0, 9.0)
            s += rng.uniform(0.015, 0.035) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))

    # Physiological artefacts
    s += 0.04 * np.sin(2 * np.pi * 0.15 * t)          # baseline wander (breathing)
    s += rng.normal(0, p.baseline_noise, n)
    s += rng.normal(0, p.muscle_noise, n) * np.sin(50 * np.pi * t)
    return t, s, r_peaks


# ── RR generators ───────────────────────────────────────────────────────────

def _build_rr(p: ECGParameters, dur: float, rng) -> np.ndarray:
    r = p.rhythm
    if r == RhythmType.NORMAL_SINUS:
        return _rr_normal(p.bpm, dur, rng, 0.020)
    elif r == RhythmType.BRADYCARDIA:
        return _rr_normal(min(p.bpm, 48), dur, rng, 0.030)
    elif r == RhythmType.TACHYCARDIA:
        return _rr_normal(max(p.bpm, 125), dur, rng, 0.010)
    elif r == RhythmType.AFIB:
        base = 60.0 / max(p.bpm, 110)
        n = int(dur / base) + 4
        return base * (0.55 + rng.exponential(0.45, n))
    elif r == RhythmType.PVC:
        base = 60.0 / p.bpm
        n = int(dur / base) + 4
        rr = base + rng.normal(0, 0.015 * base, n)
        for i in range(3, n - 1, 4):
            rr[i]   = base * 0.70
            rr[i+1] = base * 1.30
        return rr
    elif r == RhythmType.HEART_BLOCK:
        # 1st-degree block keeps a normal rate; the PR interval is what's prolonged
        return _rr_normal(p.bpm, dur, rng, 0.010)
    elif r == RhythmType.VTACH:
        return _rr_normal(max(p.bpm, 150), dur, rng, 0.005)
    elif r == RhythmType.STEMI:
        return _rr_normal(p.bpm, dur, rng, 0.020)
    return _rr_normal(p.bpm, dur, rng)


def _rr_normal(bpm, dur, rng, jitter=0.02):
    base = 60.0 / max(bpm, 1)
    n = int(dur / base) + 4
    return base + rng.normal(0, jitter * base, n)


def get_rhythm_hr(rhythm: RhythmType, user_bpm: float) -> float:
    if rhythm == RhythmType.BRADYCARDIA: return min(user_bpm, 48)
    if rhythm == RhythmType.TACHYCARDIA: return max(user_bpm, 125)
    if rhythm == RhythmType.VTACH:       return max(user_bpm, 150)
    if rhythm == RhythmType.AFIB:        return max(user_bpm, 110)
    return user_bpm
