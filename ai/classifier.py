# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
Arrhythmia classifier (rule-based, interpretable)

Measures the same features a clinician reads off a rhythm strip (rate, RR
regularity, P waves, PR interval, QRS width, ST level, QT/QTc) and applies
textbook criteria. Every number shown in the UI is measured from the signal.

"Confidence" is how far the measured features sit from the decision thresholds
(e.g. a QRS of 180 ms is far more clearly "wide" than one of 125 ms). It is NOT
a calibrated probability.

This is an educational tool, not a medical device.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClassificationResult:
    label:       str
    confidence:  float          # 0-1, margin-based (see module docstring)
    hr_bpm:      float
    rr_mean_ms:  float
    rr_std_ms:   float
    qt_ms:       float          # Bazett-corrected QTc (ms), measured
    st_level:    float          # ST deviation at J+60 ms (mV)
    p_present:   bool
    qrs_wide:    bool           # QRS >= 120 ms
    alerts:      list = field(default_factory=list)
    color:       str = "#90a4ae"
    qrs_ms:      float = 0.0
    pr_ms:       float = 0.0
    qt_raw_ms:   float = 0.0


# ── Signal helpers ───────────────────────────────────────────────────────────

def _bandpass(signal, fs, lo=0.5, hi=40.0):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(3, [lo, hi], btype='bandpass', fs=fs, output='sos')
    # long padding keeps the filter's start-up transient from looking like a beat
    return sosfiltfilt(sos, signal, padtype='even', padlen=min(len(signal) - 1, fs))


def detect_r_peaks(clean, fs):
    """R peaks with a 200 ms refractory period (works up to ~300 bpm).

    Complexes are found on the absolute signal, because many real PVCs point down in lead II.
    Each peak is then moved to the upward R wave if one at least half as tall lies within 60 ms,
    so upright beats are located on their R wave as before."""
    from scipy.signal import find_peaks
    if len(clean) < fs // 2:
        return np.array([], dtype=int)
    mag = np.abs(clean)
    amp = np.percentile(mag, 99.5)
    if amp < 0.2:
        return np.array([], dtype=int)
    peaks, props = find_peaks(mag, height=0.45 * amp, distance=int(0.20 * fs), prominence=0.35 * amp)
    if len(peaks) >= 3:
        # Drop tall T waves: real QRS complexes are similar in size to each other.
        keep = props['peak_heights'] >= 0.6 * np.median(props['peak_heights'])
        peaks = peaks[keep]
    # T-wave check (as in Pan-Tompkins): a peak soon after a beat whose steepest slope is under
    # half that beat's is a T wave, not a QRS.
    if len(peaks) >= 2:
        d = np.abs(np.gradient(clean))
        sw = int(0.05 * fs)
        slope = np.array([d[max(0, p - sw):p + sw].max() for p in peaks])
        kept = [0]
        for i in range(1, len(peaks)):
            j = kept[-1]
            if peaks[i] - peaks[j] < int(0.36 * fs) and slope[i] < 0.5 * slope[j]:
                continue
            kept.append(i)
        peaks = peaks[kept]
    w = int(0.06 * fs)
    out = []
    for p in peaks:
        lo, hi = max(0, p - w), min(len(clean), p + w + 1)
        q = lo + int(np.argmax(clean[lo:hi]))
        out.append(q if clean[q] >= 0.5 * mag[p] else p)
    return np.unique(np.asarray(out, dtype=int))


def _qrs_bounds(clean, pk, fs):
    """QRS onset/offset around an R peak: where the slope falls below 12% of its local max."""
    w = int(0.15 * fs)
    lo, hi = max(1, pk - w), min(len(clean) - 1, pk + w)
    d = np.abs(np.diff(clean[lo:hi]))
    if not len(d):
        return pk, pk
    thr = 0.12 * d.max()
    c = pk - lo
    on = c
    while on > 0 and (d[on - 1] > thr or d[max(0, on - 6):on].max() > thr):
        on -= 1
    off = min(c, len(d) - 1)
    while off < len(d) - 1 and (d[off] > thr or d[off:off + 6].max() > thr):
        off += 1
    return lo + on, lo + off


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_features(signal: np.ndarray, sample_rate: int = 500) -> dict:
    fs = sample_rate
    clean = _bandpass(np.asarray(signal, dtype=float), fs)
    peaks = detect_r_peaks(clean, fs)
    if len(peaks) < 3:
        return {'valid': False}

    rr = np.diff(peaks) / fs                     # seconds
    rr_med = float(np.median(rr))
    hr = 60.0 / float(np.mean(rr))

    qrs_ms, pr_ms, st, qt_s, p_amps, premature, wide_flags = [], [], [], [], [], [], []
    for i, pk in enumerate(peaks[1:-1], start=1):          # skip edge beats (may be clipped)
        on, off = _qrs_bounds(clean, pk, fs)
        width = (off - on) / fs * 1000
        qrs_ms.append(width)
        wide_flags.append(width >= 120)
        premature.append(rr[i - 1] < 0.85 * rr_med)

        # Isoelectric baseline just before QRS onset
        b0 = max(0, on - int(0.05 * fs))
        base = float(np.median(clean[b0:on])) if on > b0 else 0.0
        r_amp = float(clean[pk] - base)

        # P wave: biggest bump up to 350 ms before QRS onset (but after the previous T)
        p_lo = max(peaks[i - 1] + int(0.45 * rr[i - 1] * fs), on - int(0.35 * fs))
        p_hi = on - int(0.035 * fs)
        if p_hi - p_lo > int(0.04 * fs):
            seg = clean[p_lo:p_hi] - base
            j = int(np.argmax(seg))
            p_amps.append(float(seg[j]) / max(r_amp, 1e-6))
            pr_ms.append((on - (p_lo + j - int(0.05 * fs))) / fs * 1000)   # P onset ~50 ms before its peak
        else:
            p_amps.append(0.0)

        # ST level: J+60 ms (J+40 ms above 100 bpm, before the T wave starts).
        # Use the lower of J+20 and that point so a T upstroke isn't mistaken for ST elevation.
        jx = off + int((0.04 if rr[i] < 0.6 else 0.06) * fs)
        j20 = off + int(0.02 * fs)
        if jx < len(clean):
            st.append(float(min(clean[j20], clean[jx]) - base))

        # QT: QRS onset -> T end (tangent method on the T wave's down-slope)
        t_lo, t_hi = off + int(0.08 * fs), min(len(clean) - 1, pk + int(0.65 * rr[i] * fs))
        if t_hi - t_lo > int(0.05 * fs):
            seg = clean[t_lo:t_hi] - base
            tp = int(np.argmax(np.abs(seg)))
            sign = np.sign(seg[tp]) or 1.0
            tail = sign * seg[tp:]
            if len(tail) > 3:
                slope = np.diff(tail)
                k = int(np.argmin(slope))                   # steepest return toward baseline
                if slope[k] < 0:
                    t_end = tp + k + tail[k] / -slope[k]    # where the tangent hits baseline
                    qt_s.append((t_lo + t_end - on) / fs)

    def med(x, d=0.0):
        return float(np.median(x)) if len(x) else d

    qt = med(qt_s)
    return {
        'valid': True,
        'hr': hr,
        'rr_mean': float(np.mean(rr) * 1000),
        'rr_std': float(np.std(rr) * 1000),
        'rr_cv': float(np.std(rr) / np.mean(rr)),
        'qrs_ms': med(qrs_ms, 80.0),
        'wide_fraction': float(np.mean(wide_flags)) if wide_flags else 0.0,
        'premature_wide': int(sum(p and w for p, w in zip(premature, wide_flags))),
        'p_ratio': med(p_amps),
        'p_fraction': float(np.mean([a > 0.08 for a in p_amps])) if p_amps else 0.0,
        'pr_ms': med(pr_ms),
        'st_level': med(st),
        'qt_ms': qt * 1000,
        'qtc_ms': (qt / np.sqrt(rr_med)) * 1000 if qt else 0.0,
        'n_peaks': int(len(peaks)),
    }


# ── Rules ────────────────────────────────────────────────────────────────────

def _margin(value, threshold, scale):
    """Map distance past a threshold to 0.55-0.98."""
    return float(np.clip(0.55 + 0.43 * (abs(value - threshold) / scale), 0.55, 0.98))


def _classify_rhythm(f):
    hr, qrs, cv = f['hr'], f['qrs_ms'], f['rr_cv']
    p_ok = f['p_fraction'] >= 0.6

    if f['wide_fraction'] >= 0.6 and hr >= 100:
        return "Ventricular Tachycardia", _margin(qrs, 120, 60), "#ff1744"
    if f['premature_wide'] >= 1 and f['wide_fraction'] < 0.6:
        return "Premature Ventricular Contractions", _margin(f['wide_fraction'], 0, 0.3), "#ef5350"
    # Irregularly irregular rhythm: AFib (fibrillatory waves can fool the P-wave detector)
    if cv > 0.20 or (cv > 0.10 and not p_ok):
        return "Atrial Fibrillation", _margin(cv, 0.10, 0.25), "#ef5350"
    # Above ~150 bpm the T wave runs into the ST segment, so ST can't be judged
    # (bedside monitors suspend ST analysis at these rates too).
    if hr <= 150 and f['st_level'] >= 0.10:
        return "ST Elevation", _margin(f['st_level'], 0.10, 0.15), "#ff6d00"
    # PR can only be read when the P wave isn't buried in the previous T (<= 150 bpm)
    if p_ok and hr <= 150 and f['pr_ms'] > 200:
        return "1st Degree AV Block", _margin(f['pr_ms'], 200, 80), "#ce93d8"
    if hr < 50:
        return ("Sinus Bradycardia" if p_ok else "Junctional Bradycardia"), _margin(hr, 50, 15), "#4fc3f7"
    if hr > 100:
        if qrs >= 120:
            return "Wide-Complex Tachycardia", _margin(qrs, 120, 60), "#ffb74d"
        return "Sinus Tachycardia", _margin(hr, 100, 40), "#ffb74d"
    if p_ok and cv < 0.10:
        return "Normal Sinus Rhythm", _margin(cv, 0.10, 0.10), "#00e676"
    if p_ok:
        return "Sinus Arrhythmia", _margin(cv, 0.10, 0.10), "#a5d6a7"
    return "Unclassified Rhythm", 0.5, "#90a4ae"


def classify(signal: np.ndarray, sample_rate: int = 500) -> ClassificationResult:
    f = extract_features(signal, sample_rate)
    if not f.get('valid'):
        return ClassificationResult(
            label='No Signal', confidence=0.0, hr_bpm=0, rr_mean_ms=0, rr_std_ms=0,
            qt_ms=0, st_level=0, p_present=False, qrs_wide=False,
            alerts=['Cannot detect QRS complexes'], color='#607d8b')

    label, conf, color = _classify_rhythm(f)
    alerts = []
    # QTc isn't meaningful when the QRS itself is wide (the QT then includes the wide QRS)
    if f['qtc_ms'] > 500 and f['qrs_ms'] < 120:
        alerts.append(f"Prolonged QTc {f['qtc_ms']:.0f} ms — torsades risk")
    st_ok = f['hr'] <= 150
    if st_ok and f['st_level'] >= 0.10:
        alerts.append(f"ST elevation {f['st_level'] * 1000:.0f} µV — ischemia?")
    if st_ok and f['st_level'] <= -0.10:
        alerts.append(f"ST depression {abs(f['st_level']) * 1000:.0f} µV")
    if f['hr'] < 40 or f['hr'] > 180:
        alerts.append(f"Extreme heart rate: {f['hr']:.0f} bpm")
    if f['wide_fraction'] >= 0.6 and f['hr'] >= 100:
        alerts.append("Wide-complex tachycardia — treat as VT until proven otherwise")

    return ClassificationResult(
        label=label, confidence=conf, hr_bpm=f['hr'], rr_mean_ms=f['rr_mean'],
        rr_std_ms=f['rr_std'], qt_ms=f['qtc_ms'], st_level=f['st_level'],
        p_present=f['p_fraction'] >= 0.6, qrs_wide=f['qrs_ms'] >= 120,
        alerts=alerts, color=color, qrs_ms=f['qrs_ms'], pr_ms=f['pr_ms'], qt_raw_ms=f['qt_ms'])
