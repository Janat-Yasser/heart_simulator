# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
Companion vital signs, driven by the simulated ECG.

- Plethysmograph (SpO2 pulse wave): one arterial pulse per QRS, ~250 ms after it.
  Its height follows how much blood the beat ejects: a premature ventricular beat
  arrives before the ventricle has filled, so its pulse is weak or missing (the
  "pulse deficit"); AFib beats vary with the preceding interval; VT barely perfuses.
- Respiration (impedance) wave.
- Simulated SpO2, respiratory rate and blood pressure per rhythm.

Everything here is simulated for teaching; none of it is derived from a real patient.
"""

from dataclasses import dataclass

import numpy as np

from .ecg_generator import RhythmType


@dataclass
class SimVitals:
    spo2: int
    resp_rate: int
    sys: int
    dia: int

    @property
    def map(self) -> int:
        return round(self.dia + (self.sys - self.dia) / 3)


VITALS = {
    RhythmType.NORMAL_SINUS: SimVitals(98, 14, 118, 76),
    RhythmType.BRADYCARDIA:  SimVitals(96, 12, 102, 64),
    RhythmType.TACHYCARDIA:  SimVitals(97, 20, 128, 82),
    RhythmType.AFIB:         SimVitals(94, 18, 108, 70),
    RhythmType.PVC:          SimVitals(97, 15, 116, 74),
    RhythmType.HEART_BLOCK:  SimVitals(97, 14, 120, 78),
    RhythmType.VTACH:        SimVitals(86, 28, 76, 44),
    RhythmType.STEMI:        SimVitals(93, 22, 96, 60),
}


def _pulse_shape(t):
    """One arterial pulse (t in seconds from its foot): fast upstroke, dicrotic notch."""
    sys_peak = np.exp(-0.5 * ((t - 0.12) / 0.055) ** 2)
    dicrotic = 0.38 * np.exp(-0.5 * ((t - 0.33) / 0.07) ** 2)
    tail = np.where(t > 0.12, 0.25 * np.exp(-(t - 0.12) / 0.35), 0.0)
    y = sys_peak + dicrotic + tail
    return np.where(t >= 0, y, 0.0)


def generate_pleth(n, fs, r_peaks, rhythm, seed=7):
    """Pleth wave aligned to the ECG's R peaks (sample indices).
    Returns (wave, amps): amps[i] is how strong beat i's pulse is (1 = normal)."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    t = np.arange(n) / fs
    peaks = list(r_peaks)
    amps = []
    for i, pk in enumerate(peaks):
        rr_before = (pk - peaks[i - 1]) / fs if i else 0.8
        amp = 1.0
        if rhythm == RhythmType.VTACH:
            amp = 0.22 + 0.05 * rng.random()
        elif rhythm == RhythmType.PVC and i % 4 == 3:
            amp = 0.15                       # premature beat: barely any pulse
        elif rhythm == RhythmType.PVC and i % 4 == 0 and i:
            amp = 1.25                       # post-extrasystolic potentiation
        elif rhythm == RhythmType.AFIB:
            amp = float(np.clip(0.35 + 0.75 * (rr_before / 0.8), 0.3, 1.25))
        elif rhythm == RhythmType.STEMI:
            amp = 0.7
        amps.append(amp)
        foot = pk / fs + 0.20
        lo, hi = int(foot * fs), min(n, int((foot + 0.9) * fs))
        if lo < n:
            y[lo:hi] += amp * _pulse_shape(t[lo:hi] - foot)
    return y + rng.normal(0, 0.008, n), np.array(amps)


def pulse_rate(r_peaks, amps, lo, hi, fs, min_amp=0.3):
    """Pulse rate the oximeter would count between samples lo..hi: only beats
    strong enough to make a pulse count, so it can be lower than the ECG rate."""
    pk = np.asarray(r_peaks); am = np.asarray(amps)
    sel = (pk >= lo) & (pk < hi) & (am >= min_amp)
    beats = pk[sel]
    if len(beats) < 2:
        return None
    return 60.0 * (len(beats) - 1) / ((beats[-1] - beats[0]) / fs)


def generate_resp(n, fs, rate_per_min, seed=11):
    """Impedance respiration wave with natural breath-to-breath variation."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    f = rate_per_min / 60.0
    phase = 2 * np.pi * f * t + 0.4 * np.sin(2 * np.pi * 0.05 * t + rng.random() * 6)
    wave = np.sin(phase) + 0.18 * np.sin(2 * phase + 0.6)
    return wave + rng.normal(0, 0.01, n)
