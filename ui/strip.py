# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
"Print strip": renders the last few seconds of lead II the way a monitor's
recorder prints it, on ECG paper at 25 mm/s and 10 mm/mV, with a 1 mV
calibration pulse, the measured RR interval over each beat and the
measurements underneath.
"""

import time

import numpy as np
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPolygonF

from ai.classifier import _bandpass, detect_r_peaks
from ui.monitor import font

PAPER = QColor("#FFF3F0")
MINOR = QColor("#F7CFC8")
MAJOR = QColor("#E99A8F")
TRACE = QColor("#1B1B1F")
INK = QColor("#3A2A28")
INK_SOFT = QColor("#8C5E57")

PX_MM = 7                     # pixels per millimetre
MM_S, MM_MV = 25, 10          # paper speed, gain


def render_strip(sig, fs, title, measures, seconds=6.0, mv_range=(-1.0, 2.0)):
    """sig: lead II samples (mV). measures: list of (label, value) pairs.
    Returns a QImage."""
    sig = np.asarray(sig, dtype=float)[-int(seconds * fs):]
    cal_mm = 8
    paper_w = int((seconds * MM_S + cal_mm) * PX_MM)
    paper_h = int((mv_range[1] - mv_range[0]) * MM_MV * PX_MM)
    head, foot, pad = 64, 58, 24
    W, H = paper_w + 2 * pad, head + paper_h + foot

    img = QImage(W, H, QImage.Format_RGB32)
    img.fill(PAPER)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    x0, y0 = pad, head

    # grid: 1 mm minor, 5 mm major
    for mm in range(0, int(paper_w / PX_MM) + 1):
        p.setPen(QPen(MAJOR if mm % 5 == 0 else MINOR, 1.2 if mm % 5 == 0 else 0.8))
        p.drawLine(x0 + mm * PX_MM, y0, x0 + mm * PX_MM, y0 + paper_h)
    for mm in range(0, int(paper_h / PX_MM) + 1):
        p.setPen(QPen(MAJOR if mm % 5 == 0 else MINOR, 1.2 if mm % 5 == 0 else 0.8))
        p.drawLine(x0, y0 + mm * PX_MM, x0 + paper_w, y0 + mm * PX_MM)

    def ypix(mv):
        return y0 + (mv_range[1] - mv) * MM_MV * PX_MM

    # calibration pulse: 1 mV tall, 200 ms wide
    base = ypix(0)
    cx = x0 + 1 * PX_MM
    pen = QPen(TRACE, 1.8); pen.setJoinStyle(Qt.MiterJoin)
    p.setPen(pen)
    p.drawPolyline(QPolygonF([QPointF(cx - PX_MM, base), QPointF(cx, base), QPointF(cx, ypix(1)),
                              QPointF(cx + 5 * PX_MM, ypix(1)), QPointF(cx + 5 * PX_MM, base),
                              QPointF(cx + 7 * PX_MM, base)]))

    # trace
    tx0 = x0 + cal_mm * PX_MM
    xs = tx0 + np.arange(len(sig)) / fs * MM_S * PX_MM
    ys = ypix(np.clip(sig, mv_range[0], mv_range[1]))
    p.setPen(QPen(TRACE, 1.6))
    p.drawPolyline(QPolygonF([QPointF(float(a), float(b)) for a, b in zip(xs, ys)]))

    # calipers: RR (ms) between consecutive beats
    try:
        peaks = detect_r_peaks(_bandpass(sig, fs), fs)
    except Exception:
        peaks = []
    p.setFont(font(12, "num"))
    for a, b in zip(peaks[:-1], peaks[1:]):
        xa, xb = tx0 + a / fs * MM_S * PX_MM, tx0 + b / fs * MM_S * PX_MM
        y = y0 + 10
        p.setPen(QPen(INK_SOFT, 1))
        p.drawLine(int(xa), y, int(xb), y)
        p.drawLine(int(xa), y - 4, int(xa), y + 4)
        p.drawLine(int(xb), y - 4, int(xb), y + 4)
        p.setPen(INK)
        p.drawText(QRectF(xa, y + 3, xb - xa, 16), Qt.AlignCenter, f"{(b - a) / fs * 1000:.0f}")

    # header
    p.setPen(INK)
    p.setFont(font(22, "num", bold=True))
    p.drawText(QRectF(pad, 10, W, 30), Qt.AlignLeft | Qt.AlignVCenter, title)
    p.setFont(font(13, "ui"))
    p.setPen(INK_SOFT)
    meta = f"Lead II     {MM_S} mm/s     {MM_MV} mm/mV     {time.strftime('%d %b %Y  %H:%M:%S')}"
    p.drawText(QRectF(pad, 38, W, 20), Qt.AlignLeft | Qt.AlignVCenter, meta)
    p.drawText(QRectF(0, 38, W - pad, 20), Qt.AlignRight | Qt.AlignVCenter, "CardioSim, simulated patient")

    # measurements
    x = pad
    fy = head + paper_h + 12
    for lab, val in measures:
        p.setPen(INK_SOFT); p.setFont(font(12, "ui"))
        p.drawText(QRectF(x, fy, 200, 16), Qt.AlignLeft, lab)
        p.setPen(INK); p.setFont(font(20, "num", bold=True))
        p.drawText(QRectF(x, fy + 15, 200, 26), Qt.AlignLeft, val)
        x += 150
    p.end()
    return img
