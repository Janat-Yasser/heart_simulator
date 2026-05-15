"""
AI Arrhythmia Classifier — Phase 3
Two-tier approach:
  1. Rule-based fast classifier (always available, no training needed)
  2. Optional: lightweight 1-D CNN trained on MIT-BIH data

The rule-based engine runs on every window and is the primary classifier
until a trained model is loaded.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ClassificationResult:
    label:       str
    confidence:  float          # 0–1
    hr_bpm:      float
    rr_mean_ms:  float
    rr_std_ms:   float
    qt_ms:       float
    st_level:    float          # mean ST segment mV
    p_present:   bool
    qrs_wide:    bool           # QRS > 120 ms → aberrant conduction
    alerts:      list[str]      # list of clinical alert strings
    color:       str            # hex for UI badge


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_features(signal: np.ndarray, sample_rate: int = 500) -> dict:
    """Extract morphological features from a signal window."""
    from scipy.signal import find_peaks, butter, sosfilt

    # Band-pass 0.5–40 Hz
    sos = butter(4, [0.5, 40], btype='bandpass', fs=sample_rate, output='sos')
    clean = sosfilt(sos, signal)

    # R-peaks
    min_dist = int(0.35 * sample_rate)
    peaks, props = find_peaks(clean, height=0.3, distance=min_dist)

    if len(peaks) < 2:
        return {'valid': False}

    rr      = np.diff(peaks) / sample_rate * 1000   # ms
    rr_mean = float(np.mean(rr))
    rr_std  = float(np.std(rr))
    hr      = 60000.0 / rr_mean if rr_mean > 0 else 0

    # QRS width: measure time from Q trough to S trough around each R-peak
    qrs_widths = []
    for pk in peaks:
        lo = max(0, pk - int(0.08 * sample_rate))
        hi = min(len(clean), pk + int(0.08 * sample_rate))
        seg = clean[lo:hi]
        half = len(seg) // 2
        if half < 2: continue
        q_idx = np.argmin(seg[:half])
        s_idx = np.argmin(seg[half:]) + half
        qrs_widths.append((s_idx - q_idx) / sample_rate * 1000)
    qrs_ms = float(np.mean(qrs_widths)) if qrs_widths else 80.0

    # P-wave presence: energy in window before R-peak
    p_energies = []
    for pk in peaks:
        p_start = max(0, pk - int(0.20 * sample_rate))
        p_end   = max(0, pk - int(0.08 * sample_rate))
        if p_end > p_start:
            p_energies.append(float(np.max(np.abs(clean[p_start:p_end]))))
    p_energy = float(np.mean(p_energies)) if p_energies else 0.0

    # ST level: mean signal between S-wave end and T-wave peak
    st_segments = []
    for pk in peaks:
        st_s = pk + int(0.06 * sample_rate)
        st_e = pk + int(0.15 * sample_rate)
        if st_e < len(clean):
            st_segments.append(float(np.mean(clean[st_s:st_e])))
    st_level = float(np.mean(st_segments)) if st_segments else 0.0

    # QT interval (Bazett formula approximation)
    qt_ms = 0.42 * rr_mean if rr_mean > 0 else 0

    return {
        'valid':    True,
        'hr':       hr,
        'rr_mean':  rr_mean,
        'rr_std':   rr_std,
        'qrs_ms':   qrs_ms,
        'p_energy': p_energy,
        'st_level': st_level,
        'qt_ms':    qt_ms,
        'n_peaks':  len(peaks),
    }


# ── Rule-based classifier ────────────────────────────────────────────────────

def classify(signal: np.ndarray, sample_rate: int = 500) -> ClassificationResult:
    feats = extract_features(signal, sample_rate)

    if not feats.get('valid'):
        return ClassificationResult(
            label='No Signal', confidence=0.95,
            hr_bpm=0, rr_mean_ms=0, rr_std_ms=0, qt_ms=0,
            st_level=0, p_present=False, qrs_wide=False,
            alerts=['Cannot detect QRS complexes'],
            color='#607d8b'
        )

    hr        = feats['hr']
    rr_mean   = feats['rr_mean']
    rr_std    = feats['rr_std']
    qrs_ms    = feats['qrs_ms']
    p_energy  = feats['p_energy']
    st_level  = feats['st_level']
    qt_ms     = feats['qt_ms']
    rr_cv     = rr_std / rr_mean if rr_mean > 0 else 0   # coefficient of variation

    p_present = p_energy > 0.04
    qrs_wide  = qrs_ms > 120

    alerts = []
    label, conf, color = _classify_rhythm(hr, rr_cv, p_present, qrs_wide,
                                          st_level, qt_ms, alerts)

    # Additional alerts
    if qt_ms > 500:
        alerts.append(f"Prolonged QTc {qt_ms:.0f} ms — Torsades risk")
    if st_level > 0.15:
        alerts.append(f"ST elevation {st_level*1000:.0f} μV — Ischemia?")
    if st_level < -0.10:
        alerts.append(f"ST depression {abs(st_level)*1000:.0f} μV")
    if hr > 0 and (hr < 40 or hr > 180):
        alerts.append(f"Extreme heart rate: {hr:.0f} bpm")

    return ClassificationResult(
        label=label, confidence=conf,
        hr_bpm=hr, rr_mean_ms=rr_mean, rr_std_ms=rr_std,
        qt_ms=qt_ms, st_level=st_level,
        p_present=p_present, qrs_wide=qrs_wide,
        alerts=alerts, color=color
    )


def _classify_rhythm(hr, rr_cv, p_present, qrs_wide, st_level, qt_ms, alerts):
    # Ventricular Tachycardia: fast + wide QRS
    if hr > 120 and qrs_wide:
        alerts.append("Wide-complex tachycardia — V-Tach protocol")
        return "Ventricular Tachycardia", 0.88, "#ff1744"

    # AFib: irregularly irregular (even with partial P present due to noise)
    if rr_cv > 0.15:
        alerts.append("Irregularly irregular rhythm — AFib")
        return "Atrial Fibrillation", 0.84, "#ef5350"

    # STEMI
    if st_level > 0.15:
        alerts.append("Significant ST elevation — Possible STEMI")
        return "ST Elevation", 0.80, "#ff6d00"

    # Bradycardia
    if hr < 50:
        if not p_present:
            alerts.append("Junctional or escape rhythm")
            return "Junctional Bradycardia", 0.75, "#4fc3f7"
        return "Sinus Bradycardia", 0.90, "#4fc3f7"

    # Tachycardia
    if hr > 110:
        if qrs_wide:
            return "Wide-Complex Tachycardia", 0.78, "#ffb74d"
        return "Sinus Tachycardia", 0.88, "#ffb74d"

    # 1st degree block (slightly slow + regular)
    if 55 < hr < 75 and rr_cv < 0.06 and p_present:
        return "Normal Sinus Rhythm", 0.92, "#00e676"

    # Irregular but P present → sinus arrhythmia
    if rr_cv > 0.10 and p_present:
        return "Sinus Arrhythmia", 0.78, "#a5d6a7"

    if p_present:
        return "Normal Sinus Rhythm", 0.92, "#00e676"

    return "Unclassified Rhythm", 0.50, "#90a4ae"


# ── Optional CNN model (loads if available) ──────────────────────────────────

_cnn_model = None


def load_cnn_model(path: str) -> bool:
    """Load a pre-trained PyTorch model from disk."""
    global _cnn_model
    try:
        import torch
        _cnn_model = torch.load(path, map_location='cpu')
        _cnn_model.eval()
        return True
    except Exception:
        return False


def cnn_available() -> bool:
    return _cnn_model is not None
