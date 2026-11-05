# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
Bedside-monitor widgets: sweep waveform channels, parameter boxes and the alarm bar.

The look follows real patient monitors: a black screen, one colour per
parameter (ECG green, SpO2 cyan, respiration yellow, blood pressure white),
waveforms that sweep left to right over the old trace instead of scrolling,
and alarms ranked red / yellow / cyan with *** / ** / * markers.
"""

import math
import time

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QTimer, QRectF, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QPainter, QPen, QBrush
from PyQt5.QtWidgets import QWidget, QSizePolicy

# ── palette ──────────────────────────────────────────────────────────────────
SCREEN = "#000000"
BEZEL = "#1C1F23"
RULE = "#23282E"           # hairlines between boxes
KEY = "#15191E"; KEY_HI = "#232931"; KEY_TXT = "#C9D1D9"
DIM = "#6B7580"

ECG = "#39FF7A"
SPO2 = "#3FD8FF"
RESP = "#FFE14D"
NIBP = "#E8EEF2"
ALARM_RED = "#FF2E2E"
ALARM_YEL = "#FFC400"
ALARM_CYAN = "#3FD8FF"

_FAMILY = {}


def family(kind="num"):
    """Condensed numerics (Bahnschrift on Windows) and a plain UI face, with fallbacks."""
    if kind in _FAMILY:
        return _FAMILY[kind]
    have = set(QFontDatabase().families())
    prefs = {
        "num": ["Bahnschrift", "DIN Alternate", "Roboto Condensed", "Arial Narrow",
                "DejaVu Sans Condensed", "Segoe UI"],
        "ui": ["Segoe UI", "SF Pro Text", "Helvetica Neue", "Roboto", "DejaVu Sans"],
    }[kind]
    _FAMILY[kind] = next((f for f in prefs if f in have), "Sans Serif")
    return _FAMILY[kind]


def font(size, kind="num", bold=False, weight=None):
    f = QFont(family(kind))
    f.setPixelSize(max(6, int(size)))
    if weight is not None:
        f.setWeight(weight)
    elif bold:
        f.setWeight(QFont.DemiBold)
    if kind == "num" and family("num") == "DejaVu Sans Condensed":
        f.setStretch(QFont.SemiCondensed)
    return f


# ── sweep channel ────────────────────────────────────────────────────────────

class SweepChannel(QWidget):
    """A waveform that is drawn left to right over its own previous sweep,
    with a short blank gap just ahead of the write point."""

    def __init__(self, name, color, fs, seconds, y_range, note="", gap_s=0.28, width=1.7):
        super().__init__()
        self.name, self.color, self.note = name, QColor(color), note
        self.n = int(fs * seconds)
        self.gap = int(fs * gap_s)
        self.y0, self.y1 = y_range
        self.ring = np.full(self.n, np.nan)
        self.w = 0
        self.frozen = False
        self.pen_w = width
        self.status = ""          # small text after the label (rhythm call etc.)
        self.written = 0          # samples written so far
        self.marks = {}           # sample number -> (text, colour): beat labels drawn over the trace
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(70)

    def push(self, samples):
        if self.frozen or not len(samples):
            return
        k = len(samples)
        idx = (self.w + np.arange(k)) % self.n
        self.ring[idx] = samples
        self.w = (self.w + k) % self.n
        self.written += k
        self.ring[(self.w + np.arange(self.gap)) % self.n] = np.nan
        old = self.written - (self.n - self.gap)
        for key in [m for m in self.marks if m < old]:
            del self.marks[key]
        self.update()

    def mark(self, age, text, color):
        """Label the sample written `age` samples ago (e.g. a beat call above its QRS)."""
        if not self.frozen and 0 <= age < self.n - self.gap:
            self.marks[self.written - age] = (text, QColor(color))

    def clear(self):
        self.ring[:] = np.nan
        self.w = 0
        self.marks.clear()
        self.update()

    def set_status(self, text):
        if text != self.status:
            self.status = text
            self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor(SCREEN))

        # label row
        p.setPen(self.color)
        p.setFont(font(15, "num", bold=True))
        p.drawText(12, 4, 200, 20, Qt.AlignLeft | Qt.AlignVCenter, self.name)
        fm_w = p.fontMetrics().horizontalAdvance(self.name)
        p.setFont(font(12, "ui"))
        p.setPen(QColor(DIM))
        p.drawText(20 + fm_w, 4, 400, 20, Qt.AlignLeft | Qt.AlignVCenter, self.note)
        if self.status:
            p.setPen(self.color)
            p.setFont(font(13, "ui"))
            p.drawText(0, 4, W - 12, 20, Qt.AlignRight | Qt.AlignVCenter, self.status)
        if self.frozen:
            p.setPen(QColor(NIBP))
            p.setFont(font(12, "ui", bold=True))
            p.drawText(0, H - 22, W - 12, 18, Qt.AlignRight | Qt.AlignVCenter, "Frozen")

        top, bot = 24, H - 6
        x = np.linspace(8, W - 8, self.n)
        y = top + (self.y1 - self.ring) / (self.y1 - self.y0) * (bot - top)
        path = pg.arrayToQPath(x, y, connect="finite")
        pen = QPen(self.color, self.pen_w)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.drawPath(path)

        # beat labels from the trained classifier, just under the label row
        if self.marks:
            p.setFont(font(13, "num", bold=True))
            for wrote, (text, col) in self.marks.items():
                idx = (self.w - (self.written - wrote)) % self.n
                xx = 8 + idx / (self.n - 1) * (W - 16)
                p.setPen(col)
                p.drawText(int(xx - 10), top, 20, 16, Qt.AlignCenter, text)


# ── parameter box ────────────────────────────────────────────────────────────

class ParamBox(QWidget):
    """One numeric parameter: label and unit top-left, alarm limits top-right,
    a big value, and an optional second line (e.g. pulse rate under SpO2)."""

    def __init__(self, name, color, unit="", limits=None, big=1.0, heart=False):
        super().__init__()
        self.name, self.color, self.unit = name, QColor(color), unit
        self.limits = limits
        self.value = "---"
        self.small = ""            # text after the big value, e.g. "(90)"
        self.sub = ""              # line under the label
        self.foot = ""             # bottom-left line
        self.big = big
        self.heart = heart
        self._beat = 0.0
        self.alarm = None          # None | ALARM_RED | ALARM_YEL
        self.flash_on = True
        self.setMinimumHeight(90)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if heart:
            t = QTimer(self); t.timeout.connect(self._decay); t.start(30)

    def beat(self):
        self._beat = 1.0
        self.update()

    def _decay(self):
        if self._beat > 0:
            self._beat = max(0.0, self._beat - 0.09)
            self.update()

    def set(self, value=None, small=None, sub=None, foot=None):
        for k, v in (("value", value), ("small", small), ("sub", sub), ("foot", foot)):
            if v is not None:
                setattr(self, k, str(v))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        W, H = self.width(), self.height()
        inverted = self.alarm is not None and self.flash_on
        bg = QColor(self.alarm) if inverted else QColor(SCREEN)
        fg = QColor("#000000") if inverted else self.color
        p.fillRect(0, 0, W, H, bg)
        p.fillRect(0, 0, W, 1, QColor(RULE))

        p.setPen(fg)
        p.setFont(font(17, "num", bold=True))
        p.drawText(14, 8, W, 22, Qt.AlignLeft | Qt.AlignVCenter, self.name)
        nw = p.fontMetrics().horizontalAdvance(self.name)
        p.setFont(font(12, "ui"))
        p.drawText(22 + nw, 10, 120, 20, Qt.AlignLeft | Qt.AlignVCenter, self.unit)

        if self.heart:
            c = QColor(fg)
            c.setAlphaF(0.25 + 0.75 * self._beat)
            p.setPen(c)
            p.setFont(font(16, "ui"))
            uw = p.fontMetrics().horizontalAdvance(self.unit) if self.unit else 0
            p.drawText(34 + nw + uw, 9, 30, 22, Qt.AlignLeft | Qt.AlignVCenter, "♥")

        if self.limits:
            p.setPen(fg if inverted else QColor(DIM))
            p.setFont(font(12, "num"))
            hi, lo = self.limits
            p.drawText(0, 8, W - 12, 15, Qt.AlignRight, str(hi))
            p.drawText(0, 23, W - 12, 15, Qt.AlignRight, str(lo))

        if self.sub:
            p.setPen(fg)
            p.setFont(font(13, "ui"))
            p.drawText(14, 32, W - 28, 18, Qt.AlignLeft | Qt.AlignVCenter, self.sub)

        # big value, sized to the box
        size = min(H * 0.62 * self.big, W * 0.42 * self.big)
        p.setFont(font(size, "num", weight=QFont.Medium))
        fm = p.fontMetrics()
        small_w = 0
        if self.small:
            sf = font(size * 0.34, "num")
            small_w = QFontMetricsSafe(sf).w(self.small) + 8
        vw = fm.horizontalAdvance(self.value)
        x = W - 14 - small_w - vw
        base = H - 12 - (16 if self.foot else 0)
        p.setPen(fg)
        p.drawText(int(x), int(base - fm.ascent()), int(vw + 4), int(fm.ascent() + 4),
                   Qt.AlignLeft | Qt.AlignBottom, self.value)
        if self.small:
            p.setFont(font(size * 0.34, "num"))
            p.drawText(int(W - 10 - small_w), int(base - size * 0.5), int(small_w), int(size * 0.5),
                       Qt.AlignLeft | Qt.AlignBottom, self.small)
        if self.foot:
            p.setPen(fg if inverted else QColor(DIM))
            p.setFont(font(12, "ui"))
            p.drawText(14, H - 22, W - 28, 16, Qt.AlignLeft | Qt.AlignVCenter, self.foot)


class QFontMetricsSafe:
    def __init__(self, f):
        from PyQt5.QtGui import QFontMetrics
        self.m = QFontMetrics(f)

    def w(self, s):
        return self.m.horizontalAdvance(s)


# ── alarm bar ────────────────────────────────────────────────────────────────

MARK = {2: "***", 1: "**", 0: "*"}
COLOR = {2: ALARM_RED, 1: ALARM_YEL, 0: ALARM_CYAN}


class AlarmBar(QWidget):
    """Top strip: the most urgent alarm on the left (red flashes, yellow holds),
    bed label, pause countdown and clock on the right."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(46)
        self.alarms = []
        self.paused_until = 0.0
        self.message = ""        # shown when there's no alarm (quiz, frozen...)
        self.flash_on = True
        self._rot = 0
        t = QTimer(self); t.timeout.connect(self._rotate); t.start(2200)
        c = QTimer(self); c.timeout.connect(self.update); c.start(1000)

    def _rotate(self):
        self._rot += 1
        self.update()

    def set_alarms(self, alarms):
        self.alarms = list(alarms)
        self.update()

    def paused_left(self):
        return max(0, int(self.paused_until - time.time()))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor(SCREEN))
        p.fillRect(0, H - 1, W, 1, QColor(RULE))

        box = QRectF(6, 6, W * 0.46, H - 12)
        if self.alarms:
            top = self.alarms[0]
            # the most urgent alarm owns the box; lower ones rotate through only at the same priority
            same = [x for x in self.alarms if x.priority == top.priority]
            a = same[self._rot % len(same)]
            col = QColor(COLOR[a.priority])
            solid = a.priority == 1 or self.flash_on or a.priority == 0
            if solid:
                p.setBrush(QBrush(col)); p.setPen(Qt.NoPen)
                p.drawRoundedRect(box, 3, 3)
                p.setPen(QColor("#000000"))
            else:
                p.setBrush(Qt.NoBrush); p.setPen(QPen(col, 2))
                p.drawRoundedRect(box, 3, 3)
                p.setPen(col)
            p.setFont(font(21, "num", bold=True))
            p.drawText(box.adjusted(14, 0, -14, 0), Qt.AlignLeft | Qt.AlignVCenter,
                       f"{MARK[a.priority]} {a.text}")
            extra = len(self.alarms) - 1
            if extra > 0:
                p.setFont(font(13, "ui"))
                p.drawText(box.adjusted(14, 0, -14, 0), Qt.AlignRight | Qt.AlignVCenter, f"+{extra} more")
        elif self.message:
            p.setPen(QColor(NIBP))
            p.setFont(font(17, "ui"))
            p.drawText(box.adjusted(8, 0, 0, 0), Qt.AlignLeft | Qt.AlignVCenter, self.message)

        # right side: pause, bed, clock
        right = W - 14
        clock = time.strftime("%H:%M")
        p.setPen(QColor(NIBP))
        p.setFont(font(20, "num", bold=True))
        cw = p.fontMetrics().horizontalAdvance(clock)
        p.drawText(right - cw, 0, cw, H, Qt.AlignVCenter, clock)
        right -= cw + 24
        p.setFont(font(14, "ui"))
        bed = "Bed 4   Adult"
        bw = p.fontMetrics().horizontalAdvance(bed)
        p.setPen(QColor(DIM))
        p.drawText(right - bw, 0, bw, H, Qt.AlignVCenter, bed)
        right -= bw + 28
        left = self.paused_left()
        if left:
            txt = f"Alarm audio paused {left // 60}:{left % 60:02d}"
            p.setPen(QColor(ALARM_YEL))
            p.setFont(font(14, "ui", bold=True))
            tw = p.fontMetrics().horizontalAdvance(txt)
            p.drawText(right - tw, 0, tw, H, Qt.AlignVCenter, txt)
            # crossed bell
            bx = right - tw - 24
            p.setPen(QPen(QColor(ALARM_YEL), 1.6))
            p.drawArc(QRectF(bx, H / 2 - 8, 14, 14), 0, 180 * 16)
            p.drawLine(int(bx), int(H / 2 - 1), int(bx), int(H / 2 + 5))
            p.drawLine(int(bx + 14), int(H / 2 - 1), int(bx + 14), int(H / 2 + 5))
            p.drawLine(int(bx - 2), int(H / 2 + 5), int(bx + 16), int(H / 2 + 5))
            p.drawLine(int(bx - 3), int(H / 2 - 10), int(bx + 17), int(H / 2 + 9))

