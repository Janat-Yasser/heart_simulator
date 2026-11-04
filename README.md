# CardioSim: Virtual Heart Simulator

An educational cardiac monitor for the desktop: a real-time synthetic ECG, a rhythm analyser that measures what a clinician reads off a strip, a conduction animation, a 3D heart and synchronised audio.

![Python](https://img.shields.io/badge/Python-PyQt5-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-72%20passing-brightgreen)

> Educational simulator, **not a medical device**. Don't use it for diagnosis.

---


## Quick Start

```bash
cd heart_simulator
python -m venv venv
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
python main.py
```

---

## Sound Files (optional)

The app synthesises all sounds automatically. To use real recorded sounds,
download free `.wav` files and place them in `assets/sounds/`:

| File | Where to get it |
|------|----------------|
| `heartbeat.wav` | [freesound.org/s/47935](https://freesound.org/s/47935) |
| `beep.wav`      | [freesound.org/s/542074](https://freesound.org/s/542074) |
| `alarm.wav`     | [freesound.org/s/366102](https://freesound.org/s/366102) |

All three are Creative Commons / CC0 — free to use.

See `assets/sounds/README.txt` for more options.

---

## Pages

| Page | Description |
|------|-------------|
| **Monitor** | Scrolling ECG trace, rhythm strip, live stats: HR, RR and QTc (measured from the trace), SpO₂ (simulated) |
| **Controls** | BPM, rhythm type, noise, ST elevation, audio mode |
| **Rhythm Analysis** | Rule-based analyser: measured features, the rhythm call, how clearly it passed its threshold, and alerts |
| **Conduction** | Animated SA→AV→His→LBB/RBB pathway |
| **3D Heart** | Rotating 3D model (OpenGL) or anatomical 2D cross-section |

---

## Rhythm Modes

| Rhythm | Key Features |
|--------|-------------|
| Normal Sinus | Regular, 60–100 bpm |
| Bradycardia | < 50 bpm, lower pitch beep |
| Tachycardia | > 125 bpm (up to 220 bpm) |
| Atrial Fibrillation | Irregular RR, absent P waves, fibrillatory baseline, alarm |
| PVC | Early ectopic beat every 4th, compensatory pause |
| 1st Degree Heart Block | PR interval prolonged to ~300 ms at a normal rate |
| Ventricular Tachycardia | Wide QRS, > 150 bpm, alarm |
| STEMI | Elevated ST segment |

---

## Audio Modes

| Mode | Description |
|------|-------------|
| Beep | Classic monitor beep, pitch varies by rhythm |
| Heart sounds | S1/S2 lub-dub (synthesised or from heartbeat.wav) |
| Both | Beep + heart sounds together |
| Mute | Silent |

---

## 3D Heart

- Requires `PyOpenGL` and `PyOpenGL_accelerate`
- **Drag** to rotate, **scroll** to zoom
- Auto-rotates; click to take manual control
- Heart pulses in sync with each ECG beat
- Falls back to a 2D anatomical cross-section if OpenGL is unavailable

---

## How the analysis works

`ai/classifier.py` measures each beat the way a rhythm strip is read:

| Feature | How it's measured |
|---|---|
| Heart rate / RR | R-peak detection with a 200 ms refractory period (works to ~300 bpm) |
| QRS width | QRS onset to offset, where the slope drops below 12% of its peak |
| P wave and PR | Largest deflection up to 350 ms before the QRS, then P onset to QRS onset |
| ST level | Relative to the PR baseline at J+60 ms (J+40 ms above 100 bpm). Not assessed above 150 bpm, where the T wave merges into the ST segment |
| QT / QTc | QRS onset to T end (tangent method), corrected with Bazett's formula |

Textbook criteria then decide the rhythm (VT, PVCs, AFib, ST elevation, 1st-degree block, brady/tachycardia, sinus). The percentage shown is a **rule margin**, meaning how far the features sit past the deciding threshold. It isn't a calibrated probability.

The ECG generator (`core/ecg_generator.py`) times every wave in real seconds around each R peak: QT follows Bazett's formula, and ventricular beats stay wide (~160 ms) at any rate.

## Tests

```bash
pip install pytest
pytest
```

72 tests: every simulated rhythm across several heart rates, two noise levels and random seeds, plus heart rate accuracy up to 220 bpm and interval measurements (QTc, PR, QRS).

## CNN training on MIT-BIH (optional, experimental)

```bash
pip install torch wfdb scikit-learn
python train_cnn.py
```

This downloads the MIT-BIH Arrhythmia Database and trains a 1-D CNN on the 5 AAMI beat classes (N, S, V, F, Q). It uses the standard **inter-patient** split (de Chazal 2004): training patients (DS1) and test patients (DS2) never overlap, so the scores are honest. The weights are saved to `ai/ecg_cnn.pt`. The live monitor doesn't use the CNN yet; that's on the roadmap.

---

## Structure

```
heart_simulator/
├── main.py                    ← Entry point
├── requirements.txt
├── train_cnn.py               ← Optional CNN trainer (MIT-BIH, inter-patient)
├── tests/                     ← pytest suite
├── core/
│   └── ecg_generator.py       ← ECG generator (8 rhythms, real-time intervals)
├── ai/
│   └── classifier.py          ← Rule-based rhythm analyser
├── audio/
│   └── sound_engine.py        ← Synthesised + file-based audio
├── ui/
│   ├── widgets.py             ← Reusable UI components
│   └── heart_page.py          ← 3D/2D heart model page
└── assets/
    └── sounds/
        ├── README.txt         ← Sound file download instructions
        ├── heartbeat.wav      ← (place here)
        ├── beep.wav           ← (place here)
        └── alarm.wav          ← (place here)
```

---

## Roadmap

- [ ] Run the MIT-BIH CNN inside the monitor (resample 500 → 360 Hz, per-beat voting)
- [ ] Validate the rule-based analyser on real MIT-BIH strips, not just simulated ones
- [ ] 2nd- and 3rd-degree AV block, atrial flutter
- [ ] ECG quiz mode for students

