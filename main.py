"""
CardioSim: a bedside patient monitor you can teach with.

A simulated patient on a monitor modelled on real ICU monitors: lead II ECG,
SpO2 pleth and respiration sweeping across a black screen, numerics measured
from the waveform, alarms ranked red / yellow / cyan, a manual blood-pressure
cuff, printable rhythm strips and a rhythm quiz.

Run:
    python main.py

Install:
    pip install -r requirements.txt

Keys: F freeze, Space pause alarm audio, N start NIBP, P print strip, Q quiz.
Everything shown is simulated for teaching; it is not a medical device.
"""

import os
import sys
import time

import numpy as np
from PyQt5.QtCore import Qt, QTimer, QSettings
from PyQt5.QtGui import QColor, QPalette, QPixmap, QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSlider, QPushButton, QFrame, QStackedWidget, QCheckBox,
    QDialog, QScrollArea, QShortcut, QGridLayout, QSizePolicy,
)

from core.ecg_generator import ECGParameters, RhythmType, generate_ecg, get_rhythm_hr
from core.vitals import VITALS, generate_pleth, generate_resp, pulse_rate
from core import alarms as AL
from core import quiz as QZ
from ai.classifier import classify, extract_features, _bandpass, detect_r_peaks
from ai.beat_model import BeatClassifier, CLASSES as BEAT_CLASSES
import audio.sound_engine as snd
from ui.monitor import (
    SweepChannel, ParamBox, AlarmBar, font, family,
    SCREEN, BEZEL, RULE, KEY, KEY_HI, KEY_TXT, DIM,
    ECG, SPO2, RESP, NIBP, ALARM_RED, ALARM_YEL,
)
from ui.widgets import AIPanel, ConductionDiagram
from ui.heart_page import HeartPage
from ui.strip import render_strip

FS = 500
SWEEP_S = 8            # seconds across the screen
GEN_S = 90.0           # seconds generated per rhythm, then looped
HERE = os.path.dirname(os.path.abspath(__file__))


def lbl(text, size=13, color=KEY_TXT, kind="ui", bold=False, wrap=False):
    w = QLabel(text)
    w.setWordWrap(wrap)
    w.setFont(font(size, kind, bold))
    w.setStyleSheet(f"color:{color}; background:transparent;")
    return w


# ── teaching pages ───────────────────────────────────────────────────────────

class AnalysisPage(QWidget):
    def __init__(self):
        super().__init__()
        lo = QHBoxLayout(self); lo.setContentsMargins(22, 18, 22, 18); lo.setSpacing(22)
        left = QVBoxLayout(); left.setSpacing(10)
        left.addWidget(lbl("Rhythm analysis", 24, NIBP, "num", True))
        left.addWidget(lbl("A rule-based reader looks at the same 8 seconds you see: it finds each QRS, "
                           "then measures rate, regularity, P waves, PR, QRS width, ST and QT.", 13, DIM, wrap=True))
        self.panel = AIPanel()
        left.addWidget(self.panel)
        left.addWidget(lbl("Beat classifier, trained on real ECGs", 20, NIBP, "num", True))
        self.beats = lbl("", 13, KEY_TXT, wrap=True)
        self.beats.setTextFormat(Qt.RichText)
        left.addWidget(self.beats)
        left.addStretch()
        right = QVBoxLayout(); right.setSpacing(8)
        right.addWidget(lbl("Conduction", 24, NIBP, "num", True))
        right.addWidget(lbl("Each beat travels SA node, AV node, His bundle, bundle branches.", 13, DIM, wrap=True))
        self.cond = ConductionDiagram()
        right.addWidget(self.cond, 1)
        for name, desc, col in [
            ("SA node", "the natural pacemaker, 60 to 100 a minute", "#FF5A5A"),
            ("AV node", "holds the impulse back about 120 ms: that pause is the PR interval", SPO2),
            ("Bundle of His", "carries it into the ventricles, then splits left and right", ALARM_YEL),
            ("Bundle branches", "fast Purkinje fibres: a narrow QRS means they did the work", ECG),
        ]:
            r = QLabel(f"<span style='color:{col}'>●</span>&nbsp; <b style='color:{NIBP}'>{name}</b>"
                       f"&nbsp; <span style='color:{DIM}'>{desc}</span>")
            r.setFont(font(13, "ui")); r.setWordWrap(True)
            right.addWidget(r)
        lo.addLayout(left, 5); lo.addLayout(right, 4)


class SetupPage(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        p = win.params
        lo = QVBoxLayout(self); lo.setContentsMargins(26, 20, 26, 20); lo.setSpacing(14)
        lo.addWidget(lbl("Setup", 24, NIBP, "num", True))
        grid = QGridLayout(); grid.setHorizontalSpacing(18); grid.setVerticalSpacing(16)
        lo.addLayout(grid)
        row = [0]

        def slider(name, lo_, hi_, val, fmt, on):
            s = QSlider(Qt.Horizontal); s.setRange(lo_, hi_); s.setValue(val)
            v = lbl(fmt(val), 16, ECG, "num", True); v.setMinimumWidth(80)
            def cb(x): v.setText(fmt(x)); on(x)
            s.valueChanged.connect(cb)
            grid.addWidget(lbl(name, 14), row[0], 0); grid.addWidget(s, row[0], 1); grid.addWidget(v, row[0], 2)
            row[0] += 1
            return s

        def regen(): win.schedule_regen()
        slider("R-wave amplitude", 30, 200, int(p.r_amp * 100), lambda x: f"{x/100:.1f} mV",
               lambda x: (setattr(p, 'r_amp', x / 100), regen()))
        slider("ST shift", -20, 40, int(p.st_elevation * 100), lambda x: f"{x/100:+.2f} mV",
               lambda x: (setattr(p, 'st_elevation', x / 100), regen()))
        def noise(x):
            p.baseline_noise = x / 1000; p.muscle_noise = x / 2000; regen()
        slider("Signal noise", 0, 80, int(p.baseline_noise * 1000), lambda x: f"{x}", noise)
        slider("HR alarm low", 30, 70, win.hr_low, lambda x: f"{x} bpm", lambda x: win.set_limits(low=x))
        slider("HR alarm high", 90, 180, win.hr_high, lambda x: f"{x} bpm", lambda x: win.set_limits(high=x))
        slider("Volume", 0, 100, 70, lambda x: f"{x}%", lambda x: snd.set_volume(x / 100))

        mode = QComboBox()
        for m, d in [("beep", "QRS tone"), ("heart", "Heart sounds"), ("both", "Tone and heart sounds"), ("mute", "Silent")]:
            mode.addItem(d, m)
        mode.currentIndexChanged.connect(lambda i: snd.set_mode(mode.itemData(i)))
        grid.addWidget(lbl("Beat sound", 14), row[0], 0); grid.addWidget(mode, row[0], 1); row[0] += 1
        self.pitch = QCheckBox("Tone pitch follows SpO₂ (falls as saturation drops)")
        self.pitch.setChecked(True)
        grid.addWidget(self.pitch, row[0], 1); row[0] += 1
        lo.addWidget(lbl("Sound files: put heartbeat.wav, beep.wav or alarm.wav in assets/sounds/ to replace "
                         "the synthesised ones.", 12, DIM, wrap=True))
        lo.addStretch()


class StripDialog(QDialog):
    def __init__(self, img, path, parent):
        super().__init__(parent)
        self.setWindowTitle("Rhythm strip")
        lo = QVBoxLayout(self)
        pic = QLabel(); pic.setPixmap(QPixmap.fromImage(img).scaledToWidth(min(img.width(), 1180), Qt.SmoothTransformation))
        lo.addWidget(pic)
        row = QHBoxLayout()
        row.addWidget(lbl(f"Saved to {path}", 12, DIM)); row.addStretch()
        b = QPushButton("Close"); b.setObjectName("Key"); b.clicked.connect(self.accept)
        row.addWidget(b); lo.addLayout(row)
        self.setStyleSheet(f"background:{BEZEL};")


# ── quiz strip ───────────────────────────────────────────────────────────────

class QuizBar(QFrame):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self.setObjectName("Quiz")
        lo = QVBoxLayout(self); lo.setContentsMargins(14, 10, 14, 12); lo.setSpacing(8)
        top = QHBoxLayout()
        self.q = lbl("Name this rhythm", 18, NIBP, "num", True)
        self.score = lbl("", 14, DIM)
        top.addWidget(self.q); top.addStretch(); top.addWidget(self.score)
        lo.addLayout(top)
        self.row = QHBoxLayout(); self.row.setSpacing(8)
        self.btns = []
        for i in range(4):
            b = QPushButton(""); b.setObjectName("Answer"); b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _, i=i: win.quiz_answer(i))
            self.row.addWidget(b); self.btns.append(b)
        self.next = QPushButton("Next strip"); self.next.setObjectName("KeyOn")
        self.next.clicked.connect(win.quiz_next)
        self.row.addWidget(self.next)
        lo.addLayout(self.row)
        self.why = lbl("", 14, KEY_TXT); self.why.setWordWrap(True)
        lo.addWidget(self.why)


# ── main window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CardioSim")
        self.setMinimumSize(1280, 800)
        self.settings = QSettings("CardioSim", "CardioSim")

        self.params = ECGParameters()
        self.hr_low, self.hr_high = AL.HR_LOW, AL.HR_HIGH
        self._sig = self._pleth = self._resp = None
        self._peaks, self._amps = [], []
        self._pos = 0; self._peak_i = 0
        self._abuf = np.zeros(FS * SWEEP_S)    # what's on screen, for analysis
        self._filled = 0
        self._spo2 = 98.0
        self.result = None; self.features = {}
        self.alarms = []
        self.beatnet = BeatClassifier() if BeatClassifier.available() else None
        self.beat_calls = []
        self._alarm_sound = None
        self.quiz = None; self._quiz_saved = None
        self.streak = 0; self.best = int(self.settings.value("best_streak", 0))
        self._nibp = {"state": "idle", "p": 0.0, "value": None, "at": ""}

        snd.init_audio()
        self._build()
        self._style()
        self._regen_timer = QTimer(self); self._regen_timer.setSingleShot(True)
        self._regen_timer.timeout.connect(self.regenerate)
        self.regenerate()

        self._t = QTimer(self); self._t.timeout.connect(self._tick); self._t.start(33)
        self._a = QTimer(self); self._a.timeout.connect(self._analyse); self._a.start(1000)
        self._f = QTimer(self); self._f.timeout.connect(self._flash); self._f.start(330)
        self._n = QTimer(self); self._n.timeout.connect(self._nibp_tick); self._n.start(100)

        for key, fn in [("F", self.toggle_freeze), ("Space", self.pause_alarms), ("N", self.start_nibp),
                        ("P", self.print_strip), ("Q", lambda: self.k_quiz.click())]:
            QShortcut(QKeySequence(key), self, activated=fn)

    # ── layout ───────────────────────────────────────────────────────────────
    def _build(self):
        root = QWidget(); root.setObjectName("Bezel")
        self.setCentralWidget(root)
        rl = QVBoxLayout(root); rl.setContentsMargins(14, 14, 14, 10); rl.setSpacing(8)

        screen = QFrame(); screen.setObjectName("Screen")
        sl = QVBoxLayout(screen); sl.setContentsMargins(0, 0, 0, 0); sl.setSpacing(0)
        self.bar = AlarmBar(); sl.addWidget(self.bar)
        self.pages = QStackedWidget(); sl.addWidget(self.pages, 1)
        rl.addWidget(screen, 1)

        # page 0: the monitor
        mon = QWidget(); ml = QHBoxLayout(mon); ml.setContentsMargins(0, 0, 0, 0); ml.setSpacing(0)
        waves = QVBoxLayout(); waves.setSpacing(0)
        self.ch_ecg = SweepChannel("II", ECG, FS, SWEEP_S, (-0.9, 1.7), "x1   25 mm/s", width=1.9)
        self.ch_pl = SweepChannel("Pleth", SPO2, FS, SWEEP_S, (-0.15, 1.45), "")
        self.ch_rs = SweepChannel("Resp", RESP, FS, SWEEP_S, (-1.5, 1.5), "x1")
        waves.addWidget(self.ch_ecg, 5); waves.addWidget(self.ch_pl, 3); waves.addWidget(self.ch_rs, 2)
        self.quizbar = QuizBar(self); self.quizbar.hide()
        waves.addWidget(self.quizbar)
        ml.addLayout(waves, 7)
        vr = QFrame(); vr.setFixedWidth(1); vr.setStyleSheet(f"background:{RULE};"); ml.addWidget(vr)

        nums = QVBoxLayout(); nums.setSpacing(0)
        self.b_hr = ParamBox("HR", ECG, "bpm", (self.hr_high, self.hr_low), big=1.15, heart=True)
        self.b_sp = ParamBox("SpO₂", SPO2, "%  (sim)", (100, AL.SPO2_LOW))
        self.b_bp = ParamBox("NIBP", NIBP, "mmHg  (sim)", big=0.8)
        low = QHBoxLayout(); low.setSpacing(0)
        self.b_rr = ParamBox("RESP", RESP, "rpm", big=0.9)
        self.b_st = ParamBox("QTc", ECG, "ms", big=0.75)
        low.addWidget(self.b_rr); low.addWidget(self.b_st)
        nums.addWidget(self.b_hr, 4); nums.addWidget(self.b_sp, 3); nums.addWidget(self.b_bp, 3)
        lw = QWidget(); lw.setLayout(low); nums.addWidget(lw, 2)
        nw = QWidget(); nw.setLayout(nums); nw.setMinimumWidth(360)
        ml.addWidget(nw, 3)
        self.pages.addWidget(mon)

        self.analysis = AnalysisPage(); self.pages.addWidget(self.analysis)
        self.heart = HeartPage(); self.pages.addWidget(self.heart)
        self.setup = SetupPage(self)
        sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QFrame.NoFrame); sc.setWidget(self.setup)
        self.pages.addWidget(sc)

        # softkeys
        keys = QHBoxLayout(); keys.setSpacing(6)
        self.rhythm = QComboBox(); self.rhythm.setObjectName("Rhythm")
        for r in RhythmType:
            self.rhythm.addItem(QZ.NAMES[r], r)
        self.rhythm.currentIndexChanged.connect(self._rhythm_changed)
        keys.addWidget(self.rhythm)
        minus = QPushButton("−"); plus = QPushButton("+")
        for b, d in ((minus, -2), (plus, 2)):
            b.setObjectName("Key"); b.setAutoRepeat(True); b.setFixedWidth(42)
            b.clicked.connect(lambda _, d=d: self.nudge_rate(d))
        self.rate = lbl("72 bpm", 15, KEY_TXT, "num", True); self.rate.setAlignment(Qt.AlignCenter)
        self.rate.setFixedWidth(76)
        keys.addWidget(minus); keys.addWidget(self.rate); keys.addWidget(plus)
        self.quiz_hides = [self.rhythm, minus, self.rate, plus]
        keys.addSpacing(14)

        def key(text, fn, checkable=False):
            b = QPushButton(text); b.setObjectName("Key"); b.setCheckable(checkable)
            b.setCursor(Qt.PointingHandCursor); b.clicked.connect(fn); keys.addWidget(b); return b
        self.k_freeze = key("Freeze", self.toggle_freeze, True)
        self.k_pause = key("Pause alarms", self.pause_alarms)
        key("Start NIBP", self.start_nibp)
        key("Print strip", self.print_strip)
        self.k_quiz = key("Rhythm quiz", self.toggle_quiz, True)
        keys.addStretch()
        self.navs = []
        for i, t in enumerate(["Main screen", "Analysis", "Anatomy", "Setup"]):
            b = key(t, lambda _, i=i: self.show_page(i), True); b.setObjectName("Nav"); self.navs.append(b)
        self.navs[0].setChecked(True)
        rl.addLayout(keys)

    def _style(self):
        f_ui, f_num = family("ui"), family("num")
        self.setStyleSheet(f"""
            QWidget {{ background:{SCREEN}; color:{KEY_TXT}; }}
            #Bezel {{ background:{BEZEL}; }}
            #Screen {{ background:{SCREEN}; border:1px solid #0A0B0D; border-radius:6px; }}
            QPushButton#Key, QPushButton#Nav, QPushButton#KeyOn, QComboBox#Rhythm {{
                background:{KEY}; color:{KEY_TXT}; border:1px solid #2B3139; border-bottom:2px solid #0B0D10;
                border-radius:5px; padding:9px 14px; font-size:13px; }}
            QPushButton#Key:hover, QPushButton#Nav:hover, QComboBox#Rhythm:hover {{ background:{KEY_HI}; }}
            QPushButton#Key:checked {{ background:{NIBP}; color:#000; }}
            QPushButton#Nav:checked {{ background:{KEY_HI}; color:{NIBP}; border-bottom:2px solid {ECG}; }}
            QPushButton#KeyOn {{ background:{NIBP}; color:#000; font-weight:600; }}
            QComboBox#Rhythm {{ min-width:210px; font-family:'{f_num}'; font-size:15px; }}
            QComboBox::drop-down {{ border:none; width:22px; }}
            QComboBox QAbstractItemView {{ background:{KEY}; color:{KEY_TXT}; selection-background-color:{KEY_HI};
                selection-color:{ECG}; border:1px solid #2B3139; }}
            #Quiz {{ background:#07090C; border-top:1px solid {RULE}; }}
            QPushButton#Answer {{ background:{KEY}; color:{NIBP}; border:1px solid #2B3139; border-radius:5px;
                padding:10px 12px; font-family:'{f_num}'; font-size:15px; }}
            QPushButton#Answer:hover {{ border-color:{ECG}; }}
            QPushButton#Answer[state="right"] {{ background:{ECG}; color:#000; border-color:{ECG}; }}
            QPushButton#Answer[state="wrong"] {{ background:#3A0C0C; color:{ALARM_RED}; border-color:{ALARM_RED}; }}
            QSlider::groove:horizontal {{ background:#1E242B; height:4px; border-radius:2px; }}
            QSlider::sub-page:horizontal {{ background:{ECG}; border-radius:2px; }}
            QSlider::handle:horizontal {{ background:{NIBP}; width:14px; height:14px; margin:-5px 0; border-radius:7px; }}
            QComboBox {{ background:{KEY}; border:1px solid #2B3139; border-radius:5px; padding:6px 10px; }}
            QCheckBox::indicator {{ width:15px; height:15px; border:1px solid #3A424C; border-radius:3px; background:{KEY}; }}
            QCheckBox::indicator:checked {{ background:{ECG}; border-color:{ECG}; }}
            QFrame#Card {{ background:#080A0D; border:1px solid {RULE}; border-radius:6px; }}
            #FeatureCell {{ background:#0F1318; border:1px solid {RULE}; border-radius:4px; }}
            QProgressBar {{ background:#1E242B; border:none; border-radius:2px; }}
            QProgressBar::chunk {{ background:{ECG}; border-radius:2px; }}
            QScrollArea {{ border:none; }}
            QScrollBar:vertical {{ background:{SCREEN}; width:8px; }}
            QScrollBar::handle:vertical {{ background:#2B3139; border-radius:4px; min-height:24px; }}
            QDialog {{ background:{BEZEL}; }}
        """)

    # ── simulation ───────────────────────────────────────────────────────────
    def schedule_regen(self, ms=350):
        self._regen_timer.start(ms)

    def regenerate(self):
        p = self.params
        _, sig, peaks = generate_ecg(GEN_S, p)
        n = len(sig)
        self._sig, self._peaks = sig, list(peaks)
        self._pleth, self._amps = generate_pleth(n, FS, peaks, p.rhythm)
        v = VITALS[p.rhythm]
        self._resp = generate_resp(n, FS, v.resp_rate)
        self._pos = 0; self._peak_i = 0
        self.rate.setText(f"{get_rhythm_hr(p.rhythm, p.bpm):.0f} bpm")

    def _rhythm_changed(self, i):
        if self.quiz:
            return
        self.params.rhythm = self.rhythm.itemData(i)
        self.schedule_regen(50)

    def nudge_rate(self, d):
        self.params.bpm = float(np.clip(self.params.bpm + d, 30, 220))
        self.rate.setText(f"{get_rhythm_hr(self.params.rhythm, self.params.bpm):.0f} bpm")
        self.schedule_regen()

    def set_limits(self, low=None, high=None):
        if low is not None: self.hr_low = low
        if high is not None: self.hr_high = high
        self.b_hr.limits = (self.hr_high, self.hr_low); self.b_hr.update()

    def _tick(self):
        if self._sig is None:
            return
        # advance by real elapsed time, so playback stays real-time even if a frame is late
        now = time.monotonic()
        self._carry = getattr(self, "_carry", 0.0) + (now - getattr(self, "_last", now)) * FS
        self._last = now
        k = int(min(self._carry, FS // 4))
        self._carry -= k
        if k <= 0:
            return
        end = self._pos + k
        if end > len(self._sig):
            self._pos, self._peak_i, end = 0, 0, k
        a, b = self._pos, end
        ecg = self._sig[a:b]
        self.ch_ecg.push(ecg); self.ch_pl.push(self._pleth[a:b]); self.ch_rs.push(self._resp[a:b])
        self._abuf = np.roll(self._abuf, -len(ecg)); self._abuf[-len(ecg):] = ecg
        self._filled = min(len(self._abuf), self._filled + len(ecg))
        self._pos = end
        while self._peak_i < len(self._peaks) and self._peaks[self._peak_i] < self._pos:
            self._beat(); self._peak_i += 1

    def _beat(self):
        self.b_hr.beat()
        self.analysis.cond.trigger()
        self.heart.trigger_pulse()
        snd.beat(self.params.rhythm, int(round(self._spo2)) if self.setup.pitch.isChecked() else None)

    # ── analysis, numerics, alarms ───────────────────────────────────────────
    def _analyse(self):
        if self._filled < FS * 4:
            return
        buf = self._abuf[-self._filled:]
        f = extract_features(buf, FS)
        r = classify(buf, FS)
        self.features, self.result = f, r
        v = VITALS[self.params.rhythm]

        # SpO2 drifts toward the rhythm's simulated value
        self._spo2 += np.clip(v.spo2 - self._spo2, -1.0, 1.0) + np.random.uniform(-0.25, 0.25)
        spo2 = int(round(self._spo2))

        if f.get('valid'):
            self.b_hr.set(value=f"{f['hr']:.0f}")
            qtc = f['qtc_ms']
            narrow = f['qrs_ms'] < 120
            self.b_st.set(value=f"{qtc:.0f}" if qtc and narrow else "---",
                          foot=f"ST II {f['st_level']:+.2f}" if f['hr'] <= 150 and narrow else "ST II not measured")
        else:
            self.b_hr.set(value="---")
        lo = max(0, self._pos - FS * SWEEP_S)
        pr = pulse_rate(self._peaks, self._amps, lo, self._pos, FS)
        self.b_sp.set(value=str(spo2), foot=f"Pulse {pr:.0f}" if pr else "Pulse ---")
        self.b_rr.set(value=str(v.resp_rate))

        self.ch_ecg.set_status("Rhythm hidden" if self.quiz else "")
        self.b_hr.set(sub="" if self.quiz else short_call(r.label))
        self.analysis.panel.update(r)
        pvc_rate = self._run_beatnet(buf)

        self.alarms = [] if self.quiz else AL.evaluate(r, spo2, self.hr_low, self.hr_high)
        if self.beatnet and not self.quiz and pvc_rate >= 6 and r.label != "Ventricular Tachycardia" \
                and not any(a.text == "PVCs" for a in self.alarms):
            self.alarms.append(AL.Alarm(AL.YELLOW, f"PVCs {pvc_rate:.0f}/min"))
        self.alarms = [AL.Alarm(a.priority, f"PVCs {pvc_rate:.0f}/min") if a.text == "PVCs" and self.beatnet else a for a in self.alarms]
        self.alarms.sort(key=lambda a: -a.priority)
        self.bar.set_alarms(self.alarms)
        self.b_hr.alarm = alarm_color(self.alarms, ("V-Tach", "Extreme", "HR "))
        self.b_sp.alarm = alarm_color(self.alarms, ("Desat", "SpO"))
        self.b_st.alarm = alarm_color(self.alarms, ("QTc", "ST II"))
        self._sound_alarms()

    def _run_beatnet(self, buf):
        """Label every beat on screen with the MIT-BIH-trained classifier, draw the
        non-normal ones over the trace and return ventricular beats per minute."""
        if not self.beatnet:
            self.analysis.beats.setText("Not trained yet: run <b>python train_beats.py</b> to train it on MIT-BIH.")
            return 0
        peaks = detect_r_peaks(_bandpass(buf, FS), FS)
        calls = self.beatnet.classify(buf, FS, peaks)
        self.beat_calls = calls
        colours = {"V": ALARM_YEL, "S": SPO2, "F": "#C79BFF"}
        if not self.quiz:
            for pk, lab, conf in calls:
                if lab != "N":
                    self.ch_ecg.mark(len(buf) - pk, lab, colours[lab])
        counts = {c: sum(1 for _, l, _ in calls if l == c) for c in BEAT_CLASSES}
        secs = len(buf) / FS
        pvc_rate = counts["V"] * 60 / secs
        m = self.beatnet.metrics or {}
        per = m.get("per_class", {})
        rows = "".join(
            f"<tr><td style='padding:2px 14px 2px 0'><b>{c}</b> {BEAT_NAMES[c]}</td>"
            f"<td style='padding:2px 14px'>{counts[c]}</td>"
            f"<td style='padding:2px 14px'>{per.get(c, {}).get('sensitivity', 0) * 100:.0f}%</td>"
            f"<td style='padding:2px 0 2px 14px'>{per.get(c, {}).get('ppv', 0) * 100:.0f}%</td></tr>" for c in BEAT_CLASSES)
        hidden = "<i>Hidden during the quiz.</i><br>" if self.quiz else ""
        self.analysis.beats.setText(
            f"{hidden}A 1-D CNN trained on 22 patients of the MIT-BIH Arrhythmia Database labels every beat on screen. "
            f"Ectopic beats are marked over the trace (V amber, S cyan). The last two columns are its scores on "
            f"22 <i>other</i> patients it never saw."
            f"<table style='margin-top:8px'><tr style='color:{DIM}'><td>Beat class</td><td style='padding:0 14px'>On screen</td>"
            f"<td style='padding:0 14px'>Catches</td><td style='padding-left:14px'>When it says so, right</td></tr>{rows}</table>"
            f"<span style='color:{DIM}'>Supraventricular (S) and fusion (F) beats are hard to tell apart from normal ones "
            f"on an unseen patient; normal (N) and ventricular (V) are reliable.</span>")
        if not self.quiz and counts["V"]:
            self.b_hr.set(foot=f"PVC {pvc_rate:.0f}/min")
        else:
            self.b_hr.set(foot="")
        return pvc_rate

    def _sound_alarms(self):
        want = None
        if self.bar.paused_left() == 0:
            top = max((a.priority for a in self.alarms if a.sounds), default=None)
            want = {AL.RED: 'high', AL.YELLOW: 'medium'}.get(top)
        if want != self._alarm_sound:
            if want: snd.start_alarm(want)
            else: snd.stop_alarm()
            self._alarm_sound = want

    def _flash(self):
        # red flashes about 1.5 Hz; yellow holds steady in the bar but pulses slowly on the box
        self._flash_n = getattr(self, "_flash_n", 0) + 1
        on = self._flash_n % 2 == 0
        self.bar.flash_on = on; self.bar.update()
        for b in (self.b_hr, self.b_sp, self.b_st):
            b.flash_on = on if b.alarm == ALARM_RED else (self._flash_n % 6 < 4)
            b.update()

    def pause_alarms(self):
        self.bar.paused_until = 0 if self.bar.paused_left() else time.time() + 120
        self.k_pause.setText("Resume alarms" if self.bar.paused_left() else "Pause alarms")
        self.bar.update(); self._sound_alarms()

    def toggle_freeze(self):
        on = not self.ch_ecg.frozen
        for c in (self.ch_ecg, self.ch_pl, self.ch_rs):
            c.frozen = on; c.update()
        self.k_freeze.setChecked(on)

    # ── NIBP ─────────────────────────────────────────────────────────────────
    def start_nibp(self):
        if self._nibp["state"] in ("inflating", "deflating"):
            self._nibp["state"] = "idle"
            self.b_bp.set(foot="Stopped")
            return
        v = VITALS[self.params.rhythm]
        self._nibp.update(state="inflating", p=0.0, target=v.sys + 30)

    def _nibp_tick(self):
        n = self._nibp
        if n["state"] == "inflating":
            n["p"] += n["target"] / 25
            if n["p"] >= n["target"]:
                n["state"] = "deflating"
            self.b_bp.set(foot=f"Inflating {n['p']:.0f}")
        elif n["state"] == "deflating":
            n["p"] -= 3.2
            v = VITALS[self.params.rhythm]
            self.b_bp.set(foot=f"Measuring {n['p']:.0f}")
            if n["p"] <= v.dia - 8:
                j = np.random.randint(-3, 4)
                sys_, dia = v.sys + j, v.dia + j // 2
                n.update(state="done", at=time.strftime("%H:%M"))
                self.b_bp.set(value=f"{sys_}/{dia}", small=f"({round(dia + (sys_ - dia) / 3)})",
                              foot=f"Manual {n['at']}")
        elif n["state"] == "idle" and self.b_bp.value == "---":
            self.b_bp.set(value="-?-/-?-", foot="Press Start NIBP")

    # ── strip ────────────────────────────────────────────────────────────────
    def print_strip(self):
        if self._filled < FS * 6:
            return
        f, r = self.features or {}, self.result
        title = "Quiz strip: name the rhythm" if self.quiz else (r.label if r else "Lead II")
        m = []
        if f.get('valid'):
            m = [("HR", f"{f['hr']:.0f} bpm"), ("PR", f"{f['pr_ms']:.0f} ms" if f['pr_ms'] else "---"),
                 ("QRS", f"{f['qrs_ms']:.0f} ms"),
                 ("QT / QTc", f"{f['qt_ms']:.0f} / {f['qtc_ms']:.0f}" if f['qtc_ms'] else "---"),
                 ("ST", f"{f['st_level']:+.2f} mV"), ("RR varies", f"{f['rr_cv'] * 100:.0f}%")]
        img = render_strip(self._abuf[-FS * 6:], FS, title, m)
        folder = os.path.join(HERE, "strips"); os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, time.strftime("strip-%Y%m%d-%H%M%S.png"))
        img.save(path)
        self.last_strip = path
        if not os.environ.get("CARDIOSIM_HEADLESS"):
            StripDialog(img, os.path.relpath(path, HERE), self).exec_()

    # ── quiz ─────────────────────────────────────────────────────────────────
    def toggle_quiz(self):
        if self.quiz:
            self.quiz = None
            self.quizbar.hide()
            for w in self.quiz_hides: w.show()
            self.params.rhythm, self.params.bpm = self._quiz_saved
            self.rhythm.setCurrentIndex(list(RhythmType).index(self.params.rhythm))
            self.bar.message = ""
            self.regenerate()
            self.k_quiz.setChecked(False)
            return
        self._quiz_saved = (self.params.rhythm, self.params.bpm)
        self.show_page(0)
        self.quizbar.show()
        for w in self.quiz_hides: w.hide()
        self.k_quiz.setChecked(True)
        self.quiz_next()

    def quiz_next(self):
        prev = self.quiz.rhythm if self.quiz else None
        self.quiz = QZ.make_question(previous=prev)
        self.quiz_answered = False
        self.params.rhythm, self.params.bpm = self.quiz.rhythm, float(self.quiz.bpm)
        self.regenerate()
        for c in (self.ch_ecg, self.ch_pl, self.ch_rs):
            c.clear()
        self._abuf[:] = 0; self._filled = 0
        self.bar.message = "Rhythm quiz: alarms and rhythm calls are hidden"
        self.bar.set_alarms([])
        qb = self.quizbar
        qb.q.setText("Name this rhythm")
        for b, r in zip(qb.btns, self.quiz.options):
            b.setText(QZ.NAMES[r]); b.setProperty("state", ""); b.setEnabled(True); restyle(b)
        qb.next.hide(); qb.why.setText("Watch a few beats. Rate is in the HR box; pulse strength is in the pleth.")
        self._score_text()

    def quiz_answer(self, i):
        if not self.quiz or self.quiz_answered:
            return
        self.quiz_answered = True
        pick = self.quiz.options[i]; right = pick == self.quiz.rhythm
        f = extract_features(self._abuf[-self._filled:], FS) if self._filled > FS * 3 else {}
        qb = self.quizbar
        for b, r in zip(qb.btns, self.quiz.options):
            b.setProperty("state", "right" if r == self.quiz.rhythm else ("wrong" if r == pick else ""))
            restyle(b)
        self.streak = self.streak + 1 if right else 0
        if self.streak > self.best:
            self.best = self.streak; self.settings.setValue("best_streak", self.best)
        qb.q.setText(("Correct: " if right else "Not quite. It's ") + QZ.NAMES[self.quiz.rhythm])
        qb.why.setText(QZ.explain(self.quiz.rhythm, f))
        qb.next.show()
        self._score_text()

    def _score_text(self):
        self.quizbar.score.setText(f"Streak {self.streak}     Best {self.best}")

    # ── pages ────────────────────────────────────────────────────────────────
    def show_page(self, i):
        self.pages.setCurrentIndex(i)
        for j, b in enumerate(self.navs):
            b.setChecked(i == j)

    def closeEvent(self, e):
        snd.stop_alarm(); snd.shutdown(); super().closeEvent(e)


def restyle(w):
    w.style().unpolish(w); w.style().polish(w)


BEAT_NAMES = {"N": "normal", "S": "supraventricular", "V": "ventricular", "F": "fusion"}


def short_call(label):
    return {"Normal Sinus Rhythm": "Sinus rhythm", "Premature Ventricular Contractions": "Sinus with PVCs",
            "Ventricular Tachycardia": "V-Tach", "1st Degree AV Block": "1° AV block"}.get(label, label)


def alarm_color(alarms, prefixes):
    for a in alarms:
        if a.priority > AL.ADVISORY and a.text.startswith(prefixes):
            return ALARM_RED if a.priority == AL.RED else ALARM_YEL
    return None


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CardioSim")
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in [(QPalette.Window, BEZEL), (QPalette.WindowText, KEY_TXT), (QPalette.Base, KEY),
                      (QPalette.Text, KEY_TXT), (QPalette.Button, KEY), (QPalette.ButtonText, KEY_TXT),
                      (QPalette.Highlight, KEY_HI), (QPalette.HighlightedText, ECG)]:
        pal.setColor(role, QColor(col))
    app.setPalette(pal)
    app.setFont(font(13, "ui"))
    win = MainWindow()
    win.resize(1440, 880)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
