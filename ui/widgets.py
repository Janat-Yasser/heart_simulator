"""
ui/widgets.py — Reusable UI components for CardioSim v3
"""

import math
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QProgressBar, QSizePolicy, QScrollArea, QGridLayout
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import (
    QFont, QColor, QPainter, QPen, QBrush
)

# ── Design tokens ─────────────────────────────────────────────────────────────
BG    = "#FAFBFC"; BG2 = "#F0F2F5"; BG3 = "#E8ECF0"
WHITE = "#FFFFFF"; BORDER = "#E2E6EA"; BORDER2 = "#CDD3DA"
CRIM  = "#C41E3A"; CRIM2  = "#A01830"
CRIM_L = "#FDEEF1"; CRIM_M = "#F5C2CA"
INK   = "#2D3142"; INK2 = "#4A5568"; INK3 = "#718096"; MUTED = "#A0AABB"
OK    = "#0BAB60"; WARN  = "#E67E22"; DANGER = "#C41E3A"; INFO  = "#2563EB"
FNT_UI   = "'SF Pro Display', 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
FNT_MONO = "'JetBrains Mono', 'Fira Code', Consolas, monospace"
FNT_SER  = "Georgia, 'Times New Roman', serif"


# ── Helpers ───────────────────────────────────────────────────────────────────

def label(text, size=10, bold=False, color=INK2, font=FNT_UI, wrap=False):
    l = QLabel(text)
    l.setFont(QFont(font, size, QFont.Bold if bold else QFont.Normal))
    l.setStyleSheet(f"color:{color}; background:transparent;")
    if wrap:
        l.setWordWrap(True)
    return l


def hdiv():
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"background:{BORDER}; max-height:1px; border:none;")
    return f


def spacer(h=8):
    s = QWidget()
    s.setFixedHeight(h)
    return s


# ── Pulsing beat dot ──────────────────────────────────────────────────────────

class PulseDot(QWidget):
    def __init__(self, size=20):
        super().__init__()
        self.setFixedSize(size, size)
        self._r = 0.0
        t = QTimer(self)
        t.timeout.connect(self._decay)
        t.start(16)

    def pulse(self):
        self._r = 1.0

    def _decay(self):
        if self._r > 0:
            self._r = max(0.0, self._r - 0.07)
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        v  = self._r
        cx = self.width()  // 2
        cy = self.height() // 2
        r_outer = int(4 + v * 8)

        glow = QColor(CRIM)
        glow.setAlphaF(0.18 * v)
        p.setBrush(QBrush(glow))
        p.setPen(Qt.NoPen)
        p.drawEllipse(cx - r_outer - 4, cy - r_outer - 4,
                      (r_outer + 4) * 2, (r_outer + 4) * 2)

        core = QColor(CRIM)
        core.setAlphaF(0.25 + v * 0.75)
        p.setBrush(QBrush(core))
        p.drawEllipse(cx - r_outer, cy - r_outer, r_outer * 2, r_outer * 2)


# ── Stat tile ─────────────────────────────────────────────────────────────────

class StatTile(QFrame):
    def __init__(self, lbl, unit="", accent=CRIM):
        super().__init__()
        self.setObjectName("StatTile")
        self.accent = accent
        lo = QVBoxLayout(self)
        lo.setContentsMargins(16, 12, 16, 12)
        lo.setSpacing(2)

        self._lbl = QLabel(lbl.upper())
        self._lbl.setFont(QFont(FNT_UI, 8))
        self._lbl.setStyleSheet(
            f"color:{MUTED}; letter-spacing:1.2px; background:transparent;"
        )

        self._val = QLabel("—")
        self._val.setFont(QFont(FNT_MONO, 28, QFont.Bold))
        self._val.setStyleSheet(f"color:{INK}; background:transparent;")

        self._unit = QLabel(unit)
        self._unit.setFont(QFont(FNT_UI, 9))
        self._unit.setStyleSheet(f"color:{MUTED}; background:transparent;")

        # left accent bar (positioned in resizeEvent)
        self._bar = QFrame(self)
        self._bar.setFixedWidth(3)
        self._bar.setStyleSheet(
            f"background:{accent}; border-radius:2px;"
        )

        lo.addWidget(self._lbl)
        lo.addWidget(self._val)
        lo.addWidget(self._unit)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._bar.setGeometry(0, 12, 3, self.height() - 24)

    def set_value(self, v, color=None):
        self._val.setText(str(v))
        c = color or INK
        self._val.setStyleSheet(
            f"color:{c}; background:transparent; "
            f"font-size:28px; font-weight:bold;"
        )


# ── AI result panel ───────────────────────────────────────────────────────────

class AIPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("Card")
        lo = QVBoxLayout(self)
        lo.setContentsMargins(20, 16, 20, 16)
        lo.setSpacing(12)

        lo.addWidget(label("RHYTHM ANALYSIS (RULE-BASED)", 8, color=MUTED))

        self._badge = QLabel("Initialising…")
        self._badge.setFont(QFont(FNT_UI, 14, QFont.Bold))
        self._badge.setAlignment(Qt.AlignCenter)
        self._badge.setStyleSheet(
            f"color:{OK}; background:{OK}18; "
            f"border:1.5px solid {OK}55; "
            f"border-radius:10px; padding:10px 20px;"
        )
        lo.addWidget(self._badge)

        self._conf_bar = QProgressBar()
        self._conf_bar.setRange(0, 100)
        self._conf_bar.setValue(0)
        self._conf_bar.setFixedHeight(5)
        self._conf_bar.setTextVisible(False)
        self._conf_bar.setStyleSheet(
            f"QProgressBar {{ background:{BG3}; border-radius:3px; border:none; }}"
            f"QProgressBar::chunk {{ background:{CRIM}; border-radius:3px; }}"
        )
        self._conf_lbl = label("—", 8, color=MUTED)
        lo.addWidget(self._conf_bar)
        lo.addWidget(self._conf_lbl)

        lo.addWidget(hdiv())

        # Feature grid
        grid = QGridLayout()
        grid.setSpacing(8)
        self._fv = {}
        items = [
            ("Heart Rate",  "bpm", 0, 0),
            ("RR Interval", "ms",  0, 1),
            ("QTc",         "ms",  1, 0),
            ("ST Level",    "mV",  1, 1),
            ("P Wave",      "",    2, 0),
            ("QRS Width",   "",    2, 1),
        ]
        for name, unit, r, c in items:
            cell = QFrame()
            cell.setObjectName("FeatureCell")
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(10, 8, 10, 8)
            cl.setSpacing(1)
            cl.addWidget(label(name, 8, color=MUTED))
            vl = label(
                f"—  {unit}" if unit else "—",
                12, bold=True, color=INK, font=FNT_MONO
            )
            cl.addWidget(vl)
            grid.addWidget(cell, r, c)
            self._fv[name] = vl
        lo.addLayout(grid)

        lo.addWidget(hdiv())

        self._alert = QLabel("No active alerts")
        self._alert.setFont(QFont(FNT_UI, 9))
        self._alert.setWordWrap(True)
        self._alert.setStyleSheet(f"color:{INK3}; background:transparent;")
        lo.addWidget(self._alert)

    def update(self, r):
        self._badge.setText(r.label)
        self._badge.setStyleSheet(
            f"color:{r.color}; background:{r.color}15; "
            f"border:1.5px solid {r.color}60; "
            f"border-radius:10px; padding:10px 20px; "
            f"font-size:14px; font-weight:700;"
        )
        pct = int(r.confidence * 100)
        self._conf_bar.setValue(pct)
        self._conf_lbl.setText(f"{pct}% rule margin" if pct else "—")

        self._fv["Heart Rate"].setText(f"{r.hr_bpm:.0f}  bpm")
        self._fv["RR Interval"].setText(f"{r.rr_mean_ms:.0f}  ms")
        self._fv["QTc"].setText(f"{r.qt_ms:.0f}  ms")
        self._fv["ST Level"].setText(f"{r.st_level:+.3f}  mV")
        self._fv["P Wave"].setText("Present ✓" if r.p_present else "Absent ✗")
        self._fv["QRS Width"].setText(f"{r.qrs_ms:.0f} ms · " + ("wide" if r.qrs_wide else "normal") if r.qrs_ms else "—")

        if r.alerts:
            self._alert.setText("⚠  " + "   ⚠  ".join(r.alerts))
            self._alert.setStyleSheet(
                f"color:{DANGER}; background:{CRIM_L}; "
                f"border-radius:6px; padding:8px; "
                f"border:1px solid {CRIM_M};"
            )
        else:
            self._alert.setText("No active alerts")
            self._alert.setStyleSheet(
                f"color:{INK3}; background:transparent;"
            )


# ── Conduction system diagram (animated) ──────────────────────────────────────

class ConductionDiagram(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(220)
        self._active = -1
        self._glow   = 0.0

        t1 = QTimer(self); t1.timeout.connect(self._step); t1.start(110)
        t2 = QTimer(self); t2.timeout.connect(self._fade); t2.start(30)

        self._NODES = [
            ("SA Node",       0.50, 0.07, CRIM, "60–100 bpm initiator"),
            ("AV Node",       0.50, 0.35, INFO, "~120 ms delay"),
            ("Bundle of His", 0.50, 0.57, WARN, "splits L / R"),
            ("LBB",           0.28, 0.80, OK,   "left branch"),
            ("RBB",           0.72, 0.80, OK,   "right branch"),
        ]

    def trigger(self):
        self._active = 0
        self._glow   = 1.0

    def _step(self):
        if self._active >= 0:
            self._active += 1
            if self._active >= len(self._NODES):
                self._active = -1
            self.update()

    def _fade(self):
        if self._glow > 0:
            self._glow = max(0.0, self._glow - 0.04)
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()

        def xy(i):
            n = self._NODES[i]
            return int(n[1] * W), int(n[2] * H)

        # Draw edges
        for a, b, dashed in [(0,1,False),(1,2,False),(2,3,True),(2,4,True)]:
            x1, y1 = xy(a); x2, y2 = xy(b)
            p.setPen(QPen(QColor(BORDER2), 1,
                         Qt.DashLine if dashed else Qt.SolidLine))
            p.drawLine(x1, y1, x2, y2)

        # Draw nodes
        for i, (name, nx, ny, col, sub) in enumerate(self._NODES):
            x, y   = int(nx * W), int(ny * H)
            active = (i == self._active)
            r      = 18 if active else 13

            # glow ring
            if active:
                gc = QColor(col); gc.setAlphaF(0.18 * self._glow)
                p.setBrush(QBrush(gc)); p.setPen(Qt.NoPen)
                p.drawEllipse(x-r-7, y-r-7, (r+7)*2, (r+7)*2)

            fc = QColor(col if active else BORDER2)
            fc.setAlphaF(1.0 if active else 0.7)
            p.setBrush(QBrush(fc))
            p.setPen(QPen(QColor(col), 1))
            p.drawEllipse(x-r, y-r, r*2, r*2)

            p.setPen(QPen(QColor(INK if active else INK3)))
            p.setFont(QFont(FNT_UI, 7,
                            QFont.Bold if active else QFont.Normal))
            p.drawText(x-40, y+r+3, 80, 14, Qt.AlignCenter, name)

            if active:
                p.setFont(QFont(FNT_UI, 7))
                p.setPen(QPen(QColor(MUTED)))
                p.drawText(x-55, y+r+16, 110, 14, Qt.AlignCenter, sub)
