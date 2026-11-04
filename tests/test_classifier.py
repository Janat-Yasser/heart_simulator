"""The analyser must recognise every rhythm the simulator can generate, measure the
heart rate correctly up to 220 bpm, and report the intervals it claims to measure."""
import itertools

import numpy as np
import pytest

from ai.classifier import classify, extract_features
from core.ecg_generator import ECGParameters, RhythmType as R, generate_ecg, get_rhythm_hr

EXPECTED = {
    R.NORMAL_SINUS: 'Normal Sinus Rhythm',
    R.BRADYCARDIA: 'Sinus Bradycardia',
    R.TACHYCARDIA: 'Sinus Tachycardia',
    R.AFIB: 'Atrial Fibrillation',
    R.PVC: 'Premature Ventricular Contractions',
    R.HEART_BLOCK: '1st Degree AV Block',
    R.VTACH: 'Ventricular Tachycardia',
    R.STEMI: 'ST Elevation',
}
RATES = {R.NORMAL_SINUS: [60, 90], R.BRADYCARDIA: [40], R.TACHYCARDIA: [110, 180, 220],
         R.AFIB: [110, 140], R.PVC: [60, 90], R.HEART_BLOCK: [60, 90], R.VTACH: [150, 190],
         R.STEMI: [60, 80]}
NOISE = [(0.012, 0.006), (0.03, 0.015)]   # default and "noisy" slider settings
WINDOW_S = 10                              # the monitor analyses a 10 s buffer


def strip(rhythm, bpm, noise, seed):
    p = ECGParameters(bpm=bpm, rhythm=rhythm, baseline_noise=noise[0], muscle_noise=noise[1],
                      st_elevation=0.25 if rhythm == R.STEMI else 0.0)
    _, s, _ = generate_ecg(WINDOW_S + 2, p, seed=seed)
    return s[-WINDOW_S * 500:]


CASES = [(r, b, n, seed) for r, rates in RATES.items() for b in rates for n in NOISE for seed in (0, 1)]


@pytest.mark.parametrize("rhythm,bpm,noise,seed", CASES,
                         ids=[f"{r.name}-{b}bpm-n{n[0]}-s{s}" for r, b, n, s in CASES])
def test_recognises_every_simulated_rhythm(rhythm, bpm, noise, seed):
    assert classify(strip(rhythm, bpm, noise, seed)).label == EXPECTED[rhythm]


@pytest.mark.parametrize("bpm", [40, 72, 120, 180, 220])
def test_heart_rate_is_measured_up_to_220_bpm(bpm):
    r = classify(strip(R.NORMAL_SINUS, bpm, NOISE[0], 3))
    assert abs(r.hr_bpm - bpm) < 3


def test_intervals_are_measured_not_derived():
    f = extract_features(strip(R.NORMAL_SINUS, 72, NOISE[0], 0))
    assert 340 <= f['qtc_ms'] <= 440          # generator uses QTc 400 ms
    assert 120 <= f['pr_ms'] <= 200           # normal PR
    assert f['qrs_ms'] < 120                  # narrow QRS
    block = extract_features(strip(R.HEART_BLOCK, 72, NOISE[0], 0))
    assert block['pr_ms'] > 200               # prolonged PR is what defines 1st-degree block
    vt = extract_features(strip(R.VTACH, 160, NOISE[0], 0))
    assert vt['qrs_ms'] >= 120                # ventricular beats are wide at any rate


def test_confidence_reflects_margin():
    mild = classify(strip(R.TACHYCARDIA, 110, NOISE[0], 0)).confidence
    marked = classify(strip(R.TACHYCARDIA, 180, NOISE[0], 0)).confidence
    assert 0.5 <= mild < marked <= 0.98


def test_flat_line_is_reported_as_no_signal():
    r = classify(np.zeros(WINDOW_S * 500))
    assert r.label == 'No Signal' and r.confidence == 0.0
