"""Pleth, alarms and quiz logic behind the monitor (no GUI needed)."""
import numpy as np

from ai.classifier import classify, extract_features
from core import alarms as AL
from core import quiz as QZ
from core.ecg_generator import ECGParameters, RhythmType, generate_ecg
from core.vitals import generate_pleth, pulse_rate

FS = 500


def strip(rhythm, bpm=72, dur=10):
    _, s, peaks = generate_ecg(dur, ECGParameters(bpm=bpm, rhythm=rhythm))
    return s, peaks


def test_pvc_beats_make_weak_pulses_so_pulse_rate_is_below_heart_rate():
    s, peaks = strip(RhythmType.PVC, 72, 20)
    _, amps = generate_pleth(len(s), FS, peaks, RhythmType.PVC)
    pr = pulse_rate(peaks, amps, 0, len(s), FS)
    hr = 60 * (len(peaks) - 1) / ((peaks[-1] - peaks[0]) / FS)
    assert pr < hr * 0.85


def test_normal_rhythm_pulse_rate_matches_heart_rate():
    s, peaks = strip(RhythmType.NORMAL_SINUS, 80, 20)
    _, amps = generate_pleth(len(s), FS, peaks, RhythmType.NORMAL_SINUS)
    pr = pulse_rate(peaks, amps, 0, len(s), FS)
    assert abs(pr - 80) < 4


def test_vt_raises_a_red_alarm_and_normal_raises_none():
    s, _ = strip(RhythmType.VTACH, 170)
    vt = AL.evaluate(classify(s, FS), spo2=95)
    assert vt and vt[0].priority == AL.RED and vt[0].text == "V-Tach"
    s, _ = strip(RhythmType.NORMAL_SINUS, 75)
    assert AL.evaluate(classify(s, FS), spo2=98) == []


def test_alarms_are_sorted_most_urgent_first():
    s, _ = strip(RhythmType.AFIB, 120)
    out = AL.evaluate(classify(s, FS), spo2=84)
    assert [a.priority for a in out] == sorted([a.priority for a in out], reverse=True)
    assert out[0].priority == AL.RED


def test_quiz_questions_are_valid_and_explained():
    rng = np.random.default_rng(3)
    for _ in range(30):
        q = QZ.make_question(rng)
        assert q.rhythm in q.options and len(set(q.options)) == 4
        lo, hi = QZ.RATES[q.rhythm]
        assert lo <= q.bpm <= hi
    s, _ = strip(RhythmType.HEART_BLOCK, 70)
    text = QZ.explain(RhythmType.HEART_BLOCK, extract_features(s, FS))
    assert "PR interval" in text and "ms" in text


def test_stemi_rhythm_shows_st_elevation_without_the_slider():
    s, _ = strip(RhythmType.STEMI, 80)
    assert extract_features(s, FS)['st_level'] >= 0.10
