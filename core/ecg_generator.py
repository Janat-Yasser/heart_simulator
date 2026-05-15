"""
ECG Signal Generator — Full Physics Model
Gaussian basis synthesis of P, QRS, T waves.
Supports 8 rhythm types with physiologically accurate parameters.
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
    q_amp:  float = 0.06
    r_amp:  float = 1.00
    s_amp:  float = 0.20
    t_amp:  float = 0.28
    st_elevation: float = 0.0   # mV ST shift (STEMI)

    # Wave widths (fraction of RR interval)
    p_width: float = 0.040
    q_width: float = 0.014
    r_width: float = 0.016
    s_width: float = 0.014
    t_width: float = 0.065

    # Wave positions (fraction of RR)
    p_pos:  float = 0.160
    q_pos:  float = 0.260
    r_pos:  float = 0.300
    s_pos:  float = 0.340
    t_pos:  float = 0.500

    # Noise
    baseline_noise: float = 0.012
    muscle_noise:   float = 0.006


def _gauss(x, c, w, a):
    return a * np.exp(-0.5 * ((x - c) / max(w, 1e-6)) ** 2)


def _single_beat(phase: np.ndarray, p: ECGParameters, ectopic=False) -> np.ndarray:
    sig = np.zeros_like(phase)
    if not ectopic:
        sig += _gauss(phase, p.p_pos,  p.p_width,  p.p_amp)
        sig -= _gauss(phase, p.q_pos,  p.q_width,  p.q_amp)
        sig += _gauss(phase, p.r_pos,  p.r_width,  p.r_amp)
        sig -= _gauss(phase, p.s_pos,  p.s_width,  p.s_amp)
        sig += _gauss(phase, p.t_pos,  p.t_width,  p.t_amp)
        # ST elevation plateau
        if p.st_elevation != 0:
            st_s = p.s_pos + 0.02
            st_e = p.t_pos - 0.02
            mask = (phase >= st_s) & (phase <= st_e)
            sig[mask] += p.st_elevation
    else:
        # Wide, bizarre ectopic beat (PVC / VTach)
        sig += _gauss(phase, 0.30, 0.035, 1.45)
        sig -= _gauss(phase, 0.38, 0.030, 0.80)
        sig += _gauss(phase, 0.52, 0.070, -0.32)  # inverted T
    return sig


def generate_ecg(duration_sec: float, params: ECGParameters,
                 seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Returns (time, signal, r_peak_indices).
    r_peak_indices are sample indices of each R-peak — used for audio sync.
    """
    n  = int(duration_sec * params.sample_rate)
    t  = np.linspace(0, duration_sec, n, endpoint=False)
    s  = np.zeros(n)
    rng = np.random.default_rng(seed)

    rr_list = _build_rr(params, duration_sec, rng)

    r_peaks = []
    beat_start = 0.0

    for i, rr in enumerate(rr_list):
        beat_end = beat_start + rr
        mask = (t >= beat_start) & (t < beat_end)
        if not np.any(mask):
            beat_start = beat_end
            continue

        phase = (t[mask] - beat_start) / rr
        ectopic = False

        if params.rhythm == RhythmType.AFIB:
            # Suppress P, add fibrillatory baseline
            pm = ECGParameters(**{**params.__dict__, 'p_amp': 0.0})
            pm.rhythm = params.rhythm
            s[mask] += _single_beat(phase, pm)
            fib = rng.normal(0, 0.06, mask.sum()) * np.sin(2 * np.pi * 350 * t[mask])
            s[mask] += fib
        elif params.rhythm == RhythmType.PVC and i % 4 == 3:
            ectopic = True
            s[mask] += _single_beat(phase, params, ectopic=True)
        elif params.rhythm == RhythmType.VTACH:
            s[mask] += _single_beat(phase, params, ectopic=True)
        else:
            s[mask] += _single_beat(phase, params)

        # R-peak index: find max in QRS zone
        qrs_mask  = mask & (t >= beat_start + rr * 0.25) & (t <= beat_start + rr * 0.40)
        if np.any(qrs_mask):
            r_peaks.append(int(np.argmax(s * qrs_mask)))

        beat_start = beat_end

    # Physiological artefacts
    s += 0.04 * np.sin(2 * np.pi * 0.15 * t)          # baseline wander
    s += rng.normal(0, params.baseline_noise, n)
    s += rng.normal(0, params.muscle_noise, n) * np.sin(50 * np.pi * t)

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
        return _rr_normal(p.bpm, dur, rng, 0.010) * 1.08
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
