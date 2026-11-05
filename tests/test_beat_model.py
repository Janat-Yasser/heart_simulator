"""The MIT-BIH-trained beat classifier, applied to the simulator's own ECG."""
from collections import Counter

import pytest

from ai.beat_model import BeatClassifier
from ai.classifier import _bandpass, detect_r_peaks
from core.ecg_generator import ECGParameters, RhythmType, generate_ecg

pytestmark = pytest.mark.skipif(not BeatClassifier.available(), reason="no trained weights")
FS = 500


def calls(rhythm, bpm):
    _, s, _ = generate_ecg(12, ECGParameters(bpm=bpm, rhythm=rhythm))
    return Counter(l for _, l, _ in BeatClassifier().classify(s, FS, detect_r_peaks(_bandpass(s, FS), FS)))


@pytest.mark.parametrize("bpm", [60, 90])
def test_sinus_beats_are_normal(bpm):
    c = calls(RhythmType.NORMAL_SINUS, bpm)
    assert c["N"] >= 0.9 * sum(c.values())


@pytest.mark.parametrize("bpm", [60, 80])
def test_every_fourth_beat_of_the_pvc_rhythm_is_ventricular(bpm):
    c = calls(RhythmType.PVC, bpm)
    total = sum(c.values())
    assert abs(c["V"] - total / 4) <= 1


def test_ventricular_tachycardia_is_all_ventricular_beats():
    c = calls(RhythmType.VTACH, 170)
    assert c["V"] >= 0.9 * sum(c.values())


def test_reported_test_scores_are_inter_patient():
    m = BeatClassifier().metrics
    assert "inter-patient" in m["split"] and m["per_class"]["V"]["sensitivity"] > 0.8
