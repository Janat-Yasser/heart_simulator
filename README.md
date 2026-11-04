# CardioSim — Virtual Heart Simulator v3

A clinical-grade cardiac monitor desktop application.
White + crimson design. Real-time ECG, AI classification, 3D heart model, and audio.

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
| **Monitor** | Scrolling ECG trace, rhythm strip, live stats (HR, RR, QTc, SpO₂) |
| **Controls** | BPM, rhythm type, noise, ST elevation, audio mode |
| **AI Analysis** | Morphological classifier with confidence, feature readout, alerts |
| **Conduction** | Animated SA→AV→His→LBB/RBB pathway |
| **3D Heart** | Rotating 3D model (OpenGL) or anatomical 2D cross-section |

---

## Rhythm Modes

| Rhythm | Key Features |
|--------|-------------|
| Normal Sinus | Regular, 60–100 bpm |
| Bradycardia | < 50 bpm, lower pitch beep |
| Tachycardia | > 125 bpm |
| Atrial Fibrillation | Irregular RR, absent P waves, fibrillatory baseline, alarm |
| PVC | Early ectopic beat every 4th, compensatory pause |
| 1st Degree Heart Block | Prolonged PR interval |
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

## Phase 4 — AI Training on MIT-BIH (optional)

```bash
pip install torch wfdb scikit-learn
python train_cnn.py
```

Downloads the MIT-BIH Arrhythmia Database, trains a 1-D CNN,
saves `ai/ecg_cnn.pt`. Restart the app to load the trained model.

---

## Structure

```
heartsim_v3/
├── main.py                    ← Entry point
├── requirements.txt
├── train_cnn.py               ← Phase 4 (optional)
├── core/
│   └── ecg_generator.py       ← ECG physics engine (8 rhythms)
├── ai/
│   └── classifier.py          ← Rule-based + optional CNN
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
