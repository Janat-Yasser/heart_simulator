"""
Virtual Heart Simulator v3 — main.py
White + crimson clinical monitor. Full cardiac simulation desktop app.

Run:
    python main.py

Install:
    pip install PyQt5 pyqtgraph numpy scipy pygame
    pip install PyOpenGL PyOpenGL_accelerate   # optional, for 3D heart
"""

import sys, os, math
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QPushButton, QFrame, QGridLayout,
    QSplitter, QStackedWidget, QSizePolicy, QScrollArea,
    QCheckBox, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer, QObject, QThread, pyqtSignal
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QPainter, QPen, QBrush, QLinearGradient,
    QIcon, QPainterPath, QPixmap
)
import pyqtgraph as pg

pg.setConfigOption('background', '#FFFFFF')
pg.setConfigOption('foreground', '#2D3142')

from core.ecg_generator import (
    ECGParameters, RhythmType, RHYTHM_INFO, generate_ecg
)
from ai.classifier import classify, extract_features
import audio.sound_engine as snd
from ui.widgets import (
    label, hdiv, spacer,
    PulseDot, StatTile, AIPanel, ConductionDiagram
)
from ui.heart_page import HeartPage

# ── Tokens ────────────────────────────────────────────────────────────────────
BG = "#FAFBFC"; BG2 = "#F0F2F5"; BG3 = "#E8ECF0"
WHITE = "#FFFFFF"; BORDER = "#E2E6EA"; BORDER2 = "#CDD3DA"
CRIM = "#C41E3A"; CRIM2 = "#A01830"; CRIM_L = "#FDEEF1"
ECG_RED = "#E8192C"; ECG_RED2 = "#FF6375"
INK = "#2D3142"; INK2 = "#4A5568"; INK3 = "#718096"; MUTED = "#A0AABB"
OK = "#0BAB60"; WARN = "#E67E22"; DANGER = "#C41E3A"; INFO = "#2563EB"
FNT_UI = "'SF Pro Display', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
FNT_MONO = "'JetBrains Mono', 'Fira Code', Consolas, monospace"
FNT_SER = "Georgia, 'Times New Roman', serif"


# ── Regen worker ──────────────────────────────────────────────────────────────
class RegenWorker(QObject):
    finished = pyqtSignal(object, object, object)
    def __init__(self, params, dur): super().__init__(); self.p = params; self.d = dur
    def run(self): self.finished.emit(*generate_ecg(self.d, self.p))


# ── Nav button ────────────────────────────────────────────────────────────────
class NavBtn(QPushButton):
    def __init__(self, icon_char, text, idx, parent_fn):
        super().__init__()
        self._icon = icon_char; self._text = text; self._idx = idx; self._sel = False
        self.clicked.connect(lambda: parent_fn(idx))
        self.setFixedHeight(44)
        self.setCursor(Qt.PointingHandCursor)

    def set_selected(self, v):
        self._sel = v; self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        if self._sel:
            bg = QColor(CRIM_L)
            p.setBrush(QBrush(bg)); p.setPen(Qt.NoPen)
            p.drawRoundedRect(6, 4, W - 12, H - 8, 8, 8)
        icon_col = QColor(CRIM if self._sel else INK3)
        text_col = QColor(CRIM if self._sel else INK2)
        p.setPen(QPen(icon_col)); p.setFont(QFont(FNT_UI, 14))
        p.drawText(14, 0, 24, H, Qt.AlignVCenter, self._icon)
        p.setPen(QPen(text_col)); p.setFont(QFont(FNT_UI, 10, QFont.Bold if self._sel else QFont.Normal))
        p.drawText(42, 0, W - 48, H, Qt.AlignVCenter, self._text)


# ── Sidebar ───────────────────────────────────────────────────────────────────
class Sidebar(QFrame):
    page_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setObjectName("Sidebar")
        self.setFixedWidth(200)
        lo = QVBoxLayout(self)
        lo.setContentsMargins(10, 20, 10, 20)
        lo.setSpacing(4)

        # Logo
        logo = QLabel("♥")
        logo.setFont(QFont(FNT_SER, 28))
        logo.setStyleSheet(f"color:{CRIM}; background:transparent;")
        logo.setAlignment(Qt.AlignCenter)
        app_name = QLabel("CardioSim")
        app_name.setFont(QFont(FNT_SER, 15, QFont.Bold))
        app_name.setStyleSheet(f"color:{INK}; background:transparent;")
        app_name.setAlignment(Qt.AlignCenter)
        version = QLabel("v3.1 · Educational simulator")
        version.setFont(QFont(FNT_UI, 8))
        version.setStyleSheet(f"color:{MUTED}; background:transparent;")
        version.setAlignment(Qt.AlignCenter)
        lo.addWidget(logo); lo.addWidget(app_name); lo.addWidget(version)
        lo.addSpacing(16)
        lo.addWidget(self._hdiv())

        # Nav items
        pages = [
            ("♥", "Monitor",    0),
            ("⚡", "Controls",  1),
            ("🤖", "Rhythm Analysis",2),
            ("⚙", "Conduction", 3),
            ("◎", "3D Heart",   4),
        ]
        lo.addSpacing(8)
        self._btns = []
        for icon, txt, idx in pages:
            btn = NavBtn(icon, txt, idx, self._on_click)
            lo.addWidget(btn)
            self._btns.append(btn)

        lo.addStretch()
        lo.addWidget(self._hdiv())
        lo.addSpacing(8)

        # Status dot
        self._status_row = QHBoxLayout()
        self._sdot = QLabel("●")
        self._sdot.setStyleSheet(f"color:{OK}; font-size:10px;")
        self._slbl = QLabel("Signal active")
        self._slbl.setFont(QFont(FNT_UI, 8)); self._slbl.setStyleSheet(f"color:{INK3};")
        self._status_row.addWidget(self._sdot); self._status_row.addWidget(self._slbl); self._status_row.addStretch()
        lo.addLayout(self._status_row)

        self._select(0)

    def _hdiv(self):
        f = QFrame(); f.setFrameShape(QFrame.HLine)
        f.setStyleSheet(f"background:{BORDER}; max-height:1px;")
        return f

    def _on_click(self, idx):
        self._select(idx)
        self.page_changed.emit(idx)

    def _select(self, idx):
        for i, btn in enumerate(self._btns):
            btn.set_selected(i == idx)

    def set_status(self, active: bool, text: str):
        col = OK if active else DANGER
        self._sdot.setStyleSheet(f"color:{col}; font-size:10px;")
        self._slbl.setText(text)


# ── Monitor page ──────────────────────────────────────────────────────────────
class MonitorPage(QWidget):
    def __init__(self):
        super().__init__()
        lo = QVBoxLayout(self)
        lo.setContentsMargins(20, 16, 20, 16)
        lo.setSpacing(12)

        # ── Top status bar ─────────────────────────────────────────────────
        top = QHBoxLayout(); top.setSpacing(12)

        # Rhythm badge
        self.rhythm_badge = QLabel("● NORMAL SINUS RHYTHM")
        self.rhythm_badge.setFont(QFont(FNT_UI, 11, QFont.Bold))
        self.rhythm_badge.setAlignment(Qt.AlignCenter)
        self.rhythm_badge.setStyleSheet(f"""
            color:{OK}; background:{OK}15;
            border:1.5px solid {OK}55; border-radius:10px; padding:8px 20px;
        """)

        # Beat indicator + HR
        self.beat_dot = PulseDot(22)
        self.hr_big   = QLabel("—")
        self.hr_big.setFont(QFont(FNT_MONO, 32, QFont.Bold))
        self.hr_big.setStyleSheet(f"color:{INK}; background:transparent;")
        self.hr_unit  = QLabel("bpm")
        self.hr_unit.setFont(QFont(FNT_UI, 11))
        self.hr_unit.setStyleSheet(f"color:{MUTED}; background:transparent;")
        self.hr_unit.setAlignment(Qt.AlignBottom)

        hr_row = QHBoxLayout()
        hr_row.addWidget(self.beat_dot, alignment=Qt.AlignVCenter)
        hr_row.addSpacing(4)
        hr_row.addWidget(self.hr_big, alignment=Qt.AlignVCenter)
        hr_row.addWidget(self.hr_unit, alignment=Qt.AlignBottom)

        top.addWidget(self.rhythm_badge)
        top.addStretch()
        top.addLayout(hr_row)
        lo.addLayout(top)

        # ── Stat tiles ─────────────────────────────────────────────────────
        tile_row = QHBoxLayout(); tile_row.setSpacing(10)
        self.t_rr  = StatTile("RR Interval", "ms",   CRIM)
        self.t_qt  = StatTile("QTc",          "ms",   INFO)
        self.t_amp = StatTile("R Amplitude",  "mV",   OK)
        self.t_spo = StatTile("SpO₂ (sim)",   "%",    WARN)
        self.t_spo.set_value("99", OK)
        for t in (self.t_rr, self.t_qt, self.t_amp, self.t_spo):
            tile_row.addWidget(t)
        lo.addLayout(tile_row)

        # ── ECG plots ──────────────────────────────────────────────────────
        ecg_card = QFrame(); ecg_card.setObjectName("ECGCard")
        ecl = QVBoxLayout(ecg_card); ecl.setContentsMargins(16, 12, 16, 12); ecl.setSpacing(6)

        # Label bar
        lbar = QHBoxLayout()
        lbar.addWidget(label("LEAD II", 8, color=MUTED))
        lbar.addWidget(label("500 Hz  ·  25 mm/s  ·  10 mm/mV", 8, color=MUTED))
        lbar.addStretch()
        self._speed_lbl = label("Real-time", 8, color=MUTED)
        lbar.addWidget(self._speed_lbl)
        ecl.addLayout(lbar)

        # Main plot
        self._sr     = 500
        self._disp   = 10
        self._bufn   = self._sr * self._disp
        self._buf    = np.zeros(self._bufn)
        self._tx     = np.linspace(0, self._disp, self._bufn)

        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.showGrid(x=True, y=True, alpha=0.07)
        self.plot.setYRange(-0.6, 1.6)
        self.plot.setXRange(0, self._disp)
        self.plot.getPlotItem().getViewBox().setMouseEnabled(False, False)
        for ax in ('left', 'bottom'):
            self.plot.getAxis(ax).setStyle(tickFont=QFont(FNT_MONO, 8))
        self.plot.setLabel('left',   'mV',  color=MUTED, size='8pt')
        self.plot.setLabel('bottom', 'sec', color=MUTED, size='8pt')
        # Draw grid lines in crimson tint
        self.plot.getAxis('left').setPen(pg.mkPen(color=BORDER))
        self.plot.getAxis('bottom').setPen(pg.mkPen(color=BORDER))

        # ECG curve — bold crimson
        self._curve = self.plot.plot(
            self._tx, self._buf,
            pen=pg.mkPen(color=ECG_RED, width=2.2)
        )
        # Scan line
        self._scan = pg.InfiniteLine(
            pos=self._disp,
            pen=pg.mkPen(color=BORDER2, width=1, style=Qt.SolidLine)
        )
        self.plot.addItem(self._scan)
        ecl.addWidget(self.plot, stretch=3)

        # Rhythm strip
        ecl.addWidget(label("RHYTHM STRIP", 7, color=MUTED))
        self._strip4  = self._sr * 4
        self._strip   = np.zeros(self._strip4)
        self._stx     = np.linspace(0, 4, self._strip4)
        self.plot2    = pg.PlotWidget()
        self.plot2.setMenuEnabled(False); self.plot2.setFixedHeight(70)
        self.plot2.showGrid(x=False, y=False)
        self.plot2.setYRange(-0.5, 1.3); self.plot2.setXRange(0, 4)
        self.plot2.getPlotItem().getViewBox().setMouseEnabled(False, False)
        self.plot2.getAxis('left').setStyle(showValues=False)
        self.plot2.getAxis('bottom').setStyle(tickFont=QFont(FNT_MONO, 7))
        self._curve2 = self.plot2.plot(
            self._stx, self._strip,
            pen=pg.mkPen(color=ECG_RED2, width=1.4)
        )
        ecl.addWidget(self.plot2)
        lo.addWidget(ecg_card, stretch=1)


# ── Controls page ─────────────────────────────────────────────────────────────
class ControlsPage(QWidget):
    apply_requested = pyqtSignal()

    def __init__(self, params: ECGParameters):
        super().__init__()
        self.params = params
        lo = QVBoxLayout(self)
        lo.setContentsMargins(24, 20, 24, 20)
        lo.setSpacing(14)
        lo.addWidget(label("Controls", 18, bold=True, color=INK, font=FNT_SER))
        lo.addWidget(label("Adjust simulation parameters and apply changes.", 10, color=INK3))
        lo.addWidget(hdiv())

        def section(t): lo.addWidget(label(t.upper(), 8, color=MUTED))
        def row(): r = QHBoxLayout(); r.setSpacing(10); return r

        # BPM
        section("Target Heart Rate")
        r1 = row()
        self.bpm_sl = QSlider(Qt.Horizontal)
        self.bpm_sl.setRange(30, 200); self.bpm_sl.setValue(72)
        self._bpm_v = label("72", 13, bold=True, color=CRIM, font=FNT_MONO)
        self._bpm_v.setFixedWidth(44)
        lo.addWidget(label("30", 9, color=MUTED))  # replaced below in proper row
        r1.addWidget(self.bpm_sl); r1.addWidget(self._bpm_v)
        lo.addLayout(r1)
        self.bpm_sl.valueChanged.connect(lambda v: (
            setattr(params, 'bpm', float(v)), self._bpm_v.setText(str(v))
        ))

        # Rhythm
        section("Rhythm Type")
        self.rhy_cb = QComboBox()
        for r in RhythmType:
            _, col, desc = RHYTHM_INFO[r]
            self.rhy_cb.addItem(r.value, r)
        lo.addWidget(self.rhy_cb)
        self.rhy_cb.currentIndexChanged.connect(
            lambda i: setattr(params, 'rhythm', self.rhy_cb.itemData(i))
        )

        # R amplitude
        section("R-Wave Amplitude")
        r2 = row()
        self.amp_sl = QSlider(Qt.Horizontal); self.amp_sl.setRange(20, 200); self.amp_sl.setValue(100)
        self._amp_v = label("1.0", 13, bold=True, color=CRIM, font=FNT_MONO); self._amp_v.setFixedWidth(44)
        r2.addWidget(self.amp_sl); r2.addWidget(self._amp_v)
        lo.addLayout(r2)
        self.amp_sl.valueChanged.connect(lambda v: (
            setattr(params, 'r_amp', v / 100.0), self._amp_v.setText(f"{v/100:.1f}")
        ))

        # Noise
        section("Noise Level")
        r3 = row()
        self.noi_sl = QSlider(Qt.Horizontal); self.noi_sl.setRange(0, 80); self.noi_sl.setValue(12)
        self._noi_v = label("12", 13, bold=True, color=CRIM, font=FNT_MONO); self._noi_v.setFixedWidth(44)
        r3.addWidget(self.noi_sl); r3.addWidget(self._noi_v)
        lo.addLayout(r3)
        def on_n(v):
            params.baseline_noise = v / 1000.0; params.muscle_noise = v / 2000.0
            self._noi_v.setText(str(v))
        self.noi_sl.valueChanged.connect(on_n)

        # ST elevation
        section("ST Elevation / Depression")
        r4 = row()
        self.st_sl = QSlider(Qt.Horizontal); self.st_sl.setRange(-20, 40); self.st_sl.setValue(0)
        self._st_v = label("+0.00", 13, bold=True, color=CRIM, font=FNT_MONO); self._st_v.setFixedWidth(60)
        r4.addWidget(self.st_sl); r4.addWidget(self._st_v)
        lo.addLayout(r4)
        self.st_sl.valueChanged.connect(lambda v: (
            setattr(params, 'st_elevation', v / 100.0), self._st_v.setText(f"{v/100:+.2f}")
        ))

        lo.addWidget(hdiv())

        # Audio
        lo.addWidget(label("Audio", 18, bold=True, color=INK, font=FNT_SER))
        ar = row()
        self.au_cb = QCheckBox("Sound enabled"); self.au_cb.setChecked(True)
        self.au_cb.stateChanged.connect(lambda s: snd.set_enabled(bool(s)))
        self.mo_cb = QComboBox()
        for m, d in [("beep","Monitor beep"),("heart","Heart sounds"),("both","Beep + Heart"),("mute","Mute")]:
            self.mo_cb.addItem(d, m)
        self.mo_cb.currentIndexChanged.connect(lambda i: snd.set_mode(self.mo_cb.itemData(i)))
        ar.addWidget(self.au_cb); ar.addWidget(self.mo_cb)
        lo.addLayout(ar)

        # Volume
        vr = row()
        vr.addWidget(label("Volume", 9, color=INK3))
        self.vol_sl = QSlider(Qt.Horizontal); self.vol_sl.setRange(0, 100); self.vol_sl.setValue(70)
        self.vol_sl.valueChanged.connect(lambda v: snd.set_volume(v / 100.0))
        vr.addWidget(self.vol_sl)
        lo.addLayout(vr)

        # Sound file hint
        hint = QLabel(
            "💡 To use real sounds: create <b>assets/sounds/</b> and place<br>"
            "<code>heartbeat.wav</code>, <code>beep.wav</code>, <code>alarm.wav</code> inside."
        )
        hint.setFont(QFont(FNT_UI, 9)); hint.setWordWrap(True)
        hint.setStyleSheet(f"""
            color:{INK3}; background:{BG2};
            border:1px solid {BORDER}; border-radius:8px; padding:10px;
        """)
        lo.addWidget(hint)

        lo.addWidget(hdiv())
        # Apply
        self.apply_btn = QPushButton("▶  Apply & Regenerate Signal")
        self.apply_btn.setObjectName("ApplyBtn")
        self.apply_btn.clicked.connect(self.apply_requested.emit)
        self.apply_btn.setCursor(Qt.PointingHandCursor)
        lo.addWidget(self.apply_btn)
        lo.addStretch()


# ── Main window ───────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CardioSim  —  Virtual Heart Simulator")
        self.setMinimumSize(1300, 820)

        self.params   = ECGParameters()
        self._sr      = 500
        self._disp    = 10
        self._bufn    = self._sr * self._disp
        self._chunk   = self._sr // 30
        self._gen_dur = 90.0

        self._gen_sig   = None
        self._gen_peaks = []
        self._play_pos  = 0
        self._peak_ptr  = 0

        self._beat_cnt  = 0
        self._ai_cnt    = 0
        self._ai_ivl    = self._sr * 4
        self._elapsed   = 0

        self._audio_ok = snd.init_audio()

        self._build_ui()
        self._apply_style()
        self._regenerate()

        self._tick_timer = QTimer()
        self._tick_timer.timeout.connect(self._tick)
        self._tick_timer.start(33)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        rl = QHBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        # Sidebar
        self._sidebar = Sidebar()
        self._sidebar.page_changed.connect(self._switch_page)
        rl.addWidget(self._sidebar)

        # Separator
        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"background:{BORDER}; max-width:1px;")
        rl.addWidget(sep)

        # Page stack
        self._stack = QStackedWidget()
        rl.addWidget(self._stack, stretch=1)

        # ── Page 0: Monitor ───────────────────────────────────────────────
        self._monitor = MonitorPage()
        self._stack.addWidget(self._monitor)

        # ── Page 1: Controls ─────────────────────────────────────────────
        self._ctrl_page = ControlsPage(self.params)
        self._ctrl_page.apply_requested.connect(self._regenerate)
        scroll1 = QScrollArea()
        scroll1.setWidgetResizable(True); scroll1.setFrameShape(QFrame.NoFrame)
        scroll1.setWidget(self._ctrl_page)
        self._stack.addWidget(scroll1)

        # ── Page 2: AI ────────────────────────────────────────────────────
        ai_outer = QWidget()
        ai_l = QVBoxLayout(ai_outer); ai_l.setContentsMargins(24, 20, 24, 20)
        ai_l.addWidget(label("Rhythm Analysis", 18, bold=True, color=INK, font="Georgia, serif"))
        ai_l.addWidget(label("Rule-based morphological classifier · updates every 4 seconds", 10, color=INK3))
        ai_l.addWidget(hdiv())
        self._ai_panel = AIPanel()
        ai_l.addWidget(self._ai_panel)
        ai_l.addStretch()
        self._stack.addWidget(ai_outer)

        # ── Page 3: Conduction ────────────────────────────────────────────
        cond_outer = QWidget()
        co_l = QVBoxLayout(cond_outer); co_l.setContentsMargins(24, 20, 24, 20)
        co_l.addWidget(label("Conduction System", 18, bold=True, color=INK, font="Georgia, serif"))
        co_l.addWidget(label("Cardiac impulse pathway animated on each beat.", 10, color=INK3))
        co_l.addWidget(hdiv())
        self._cond = ConductionDiagram()
        co_l.addWidget(self._cond, stretch=1)
        # Legend
        nodes_info = [
            ("SA Node",         "Sinoatrial node — natural pacemaker. Fires 60–100 bpm.",    "#C41E3A"),
            ("AV Node",         "Atrioventricular node. Delays impulse ~120–200 ms.",         "#2563EB"),
            ("Bundle of His",   "Common bundle — splits into left and right branches.",       "#E67E22"),
            ("LBB / RBB",       "Left and right bundle branches → Purkinje fibres.",          "#0BAB60"),
        ]
        for name, desc, col in nodes_info:
            r = QHBoxLayout()
            d = QLabel("●"); d.setStyleSheet(f"color:{col}; font-size:10px;"); d.setFixedWidth(18)
            b = QLabel(f"<b style='color:{INK2}'>{name}</b>  <span style='color:{INK3}'>{desc}</span>")
            b.setFont(QFont("SF Pro Display, Segoe UI, Arial", 9)); b.setWordWrap(True)
            r.addWidget(d); r.addWidget(b); r.addStretch()
            co_l.addLayout(r)
        co_l.addStretch()
        self._stack.addWidget(cond_outer)

        # ── Page 4: 3D Heart ──────────────────────────────────────────────
        self._heart_page = HeartPage()
        self._stack.addWidget(self._heart_page)

    # ── Style ─────────────────────────────────────────────────────────────────

    def _apply_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background:{BG};
                color:{INK};
                font-family:{FNT_UI};
            }}
            #Sidebar {{
                background:{WHITE};
                border-right:1px solid {BORDER};
            }}
            #ECGCard {{
                background:{WHITE};
                border:1px solid {BORDER};
                border-radius:14px;
            }}
            QFrame#Card {{
                background:{WHITE};
                border:1px solid {BORDER};
                border-radius:14px;
            }}
            #StatTile {{
                background:{WHITE};
                border:1px solid {BORDER};
                border-radius:12px;
            }}
            #FeatureCell {{
                background:{BG2};
                border:1px solid {BORDER};
                border-radius:8px;
            }}
            QComboBox {{
                background:{WHITE};
                border:1.5px solid {BORDER2};
                border-radius:8px;
                padding:6px 12px;
                color:{INK};
                font-size:11px;
            }}
            QComboBox:hover {{ border-color:{CRIM}; }}
            QComboBox::drop-down {{ border:none; width:24px; }}
            QComboBox QAbstractItemView {{
                background:{WHITE}; color:{INK};
                border:1px solid {BORDER2};
                selection-background-color:{CRIM_L};
                selection-color:{CRIM};
            }}
            QSlider::groove:horizontal {{
                background:{BG3}; height:5px; border-radius:3px;
            }}
            QSlider::handle:horizontal {{
                background:{CRIM}; width:16px; height:16px;
                margin:-6px 0; border-radius:8px;
            }}
            QSlider::sub-page:horizontal {{
                background:{CRIM}; opacity:0.5; border-radius:3px;
            }}
            QCheckBox {{ color:{INK2}; font-size:11px; }}
            QCheckBox::indicator {{
                width:16px; height:16px;
                border:1.5px solid {BORDER2};
                border-radius:4px; background:{WHITE};
            }}
            QCheckBox::indicator:checked {{
                background:{CRIM}; border-color:{CRIM};
            }}
            QPushButton {{
                background:{WHITE}; border:1.5px solid {BORDER2};
                border-radius:8px; padding:8px 16px;
                color:{INK}; font-size:11px;
            }}
            QPushButton:hover  {{ background:{BG2}; border-color:{BORDER}; }}
            QPushButton:pressed {{ background:{BG3}; }}
            #ApplyBtn {{
                background:{CRIM}; border:none; color:{WHITE};
                font-weight:bold; font-size:12px; padding:12px;
                border-radius:10px;
            }}
            #ApplyBtn:hover {{ background:{CRIM2}; }}
            QScrollBar:vertical {{ background:{BG}; width:8px; border-radius:4px; }}
            QScrollBar::handle:vertical {{ background:{BORDER2}; border-radius:4px; min-height:24px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0px; }}
            QScrollArea {{ background:{BG}; border:none; }}
            QProgressBar {{ background:{BG3}; border-radius:3px; border:none; }}
            QProgressBar::chunk {{ background:{CRIM}; border-radius:3px; }}
        """)

    # ── Page switch ───────────────────────────────────────────────────────────

    def _switch_page(self, idx):
        self._stack.setCurrentIndex(idx)
        self._sidebar._select(idx)

    # ── Signal generation ─────────────────────────────────────────────────────

    def _regenerate(self):
        self._sidebar.set_status(False, "Generating…")
        t, s, peaks = generate_ecg(self._gen_dur, self.params)
        self._gen_sig   = s
        self._gen_peaks = peaks
        self._play_pos  = 0
        self._peak_ptr  = 0
        self._monitor._buf[:] = 0
        self._monitor._strip[:] = 0

        # Rhythm badge
        info = RHYTHM_INFO[self.params.rhythm]
        col  = info[1]
        self._monitor.rhythm_badge.setText(f"● {info[0].upper()}")
        self._monitor.rhythm_badge.setStyleSheet(f"""
            color:{col}; background:{col}18;
            border:1.5px solid {col}55;
            border-radius:10px; padding:8px 20px;
        """)

        if self.params.rhythm in (RhythmType.VTACH, RhythmType.AFIB):
            snd.start_alarm()
        else:
            snd.stop_alarm()

        self._sidebar.set_status(True, "Signal active")

    # ── Main tick ─────────────────────────────────────────────────────────────

    def _tick(self):
        if self._gen_sig is None:
            return

        end = self._play_pos + self._chunk
        if end >= len(self._gen_sig):
            self._play_pos = 0; self._peak_ptr = 0; end = self._chunk

        chunk = self._gen_sig[self._play_pos:end]
        self._play_pos = end

        # Scroll buffers
        n = len(chunk)
        self._monitor._buf   = np.roll(self._monitor._buf, -n); self._monitor._buf[-n:]   = chunk
        self._monitor._strip = np.roll(self._monitor._strip, -n); self._monitor._strip[-n:] = chunk

        # Update plots (only if visible)
        if self._stack.currentIndex() == 0:
            self._monitor._curve.setData(self._monitor._tx,   self._monitor._buf)
            self._monitor._curve2.setData(self._monitor._stx, self._monitor._strip)

        # Beat events
        while (self._peak_ptr < len(self._gen_peaks) and
               self._gen_peaks[self._peak_ptr] < self._play_pos):
            self._on_beat()
            self._peak_ptr += 1

        # Stats every 0.5s
        self._beat_cnt += n
        if self._beat_cnt >= self._sr // 2:
            self._beat_cnt = 0; self._update_stats()

        # AI every 4s
        self._ai_cnt += n
        if self._ai_cnt >= self._ai_ivl:
            self._ai_cnt = 0; self._run_ai()

        # Elapsed
        self._elapsed += n

    def _on_beat(self):
        self._monitor.beat_dot.pulse()
        self._cond.trigger()
        self._heart_page.trigger_pulse()
        snd.beat(self.params.rhythm)

    def _update_stats(self):
        # Every number here is measured from the signal on screen.
        buf = self._monitor._buf
        f = extract_features(buf, self._sr)
        if not f.get('valid'):
            return
        rr_m  = f['rr_mean']
        hr    = int(round(f['hr']))
        qt    = int(round(f['qtc_ms'])) if f['qtc_ms'] else '—'
        ramp  = round(float(np.percentile(buf, 99.5)), 2)

        hr_col = OK if 50 <= hr <= 110 else (INFO if hr < 50 else DANGER)
        self._monitor.hr_big.setText(str(hr))
        self._monitor.hr_big.setStyleSheet(f"color:{hr_col}; background:transparent; font-size:32px; font-weight:bold;")
        self._monitor.t_rr.set_value(f"{rr_m:.0f}")
        self._monitor.t_qt.set_value(qt)
        self._monitor.t_amp.set_value(ramp)

        # SpO2 can't be derived from an ECG; it's a simulated value that dips
        # with poor-output rhythms, and is labelled "(sim)" on the monitor.
        if self.params.rhythm in (RhythmType.VTACH, RhythmType.AFIB):
            self._monitor.t_spo.set_value("91", WARN)
        else:
            self._monitor.t_spo.set_value("99", OK)

    def _run_ai(self):
        result = classify(self._monitor._buf, self._sr)
        self._ai_panel.update(result)

    def closeEvent(self, e):
        snd.stop_alarm(); snd.shutdown(); super().closeEvent(e)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CardioSim")
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window,        QColor("#FAFBFC"))
    pal.setColor(QPalette.WindowText,    QColor(INK))
    pal.setColor(QPalette.Base,          QColor(WHITE))
    pal.setColor(QPalette.Text,          QColor(INK))
    pal.setColor(QPalette.Button,        QColor(WHITE))
    pal.setColor(QPalette.ButtonText,    QColor(INK))
    pal.setColor(QPalette.Highlight,     QColor(CRIM))
    pal.setColor(QPalette.HighlightedText, QColor(WHITE))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
