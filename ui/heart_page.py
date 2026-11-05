# Copyright (c) 2026 Janat Yasser. All rights reserved.
"""
3D Heart Model Page
Renders an anatomically-inspired 3D heart using PyOpenGL + Qt.
Falls back to a beautiful SVG anatomical diagram if OpenGL is unavailable.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QSlider, QSizePolicy, QScrollArea
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush, QPainterPath
import math

# ── Design tokens (duplicated for standalone import) ─────────────────────────
CRIM   = "#E0475A"; CRIM2 = "#8E1C2C"; BG = "#000000"; INK = "#E8EEF2"
INK2   = "#C3CBD3"; INK3 = "#8A949E"; MUTED = "#66707A"
BORDER = "#23282E"; WHITE = "#080A0D"; OK = "#39FF7A"
INFO   = "#3FD8FF"; WARN  = "#FFC400"; BG2 = "#0F1318"
FNT_UI = "'SF Pro Display', 'Segoe UI', Helvetica, Arial, sans-serif"
FNT_MONO = "'JetBrains Mono', Consolas, monospace"

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from PyQt5.QtWidgets import QOpenGLWidget
    from PyQt5.QtGui import QSurfaceFormat
    OPENGL_OK = True
except ImportError:
    OPENGL_OK = False


# ── OpenGL 3D heart ───────────────────────────────────────────────────────────

if OPENGL_OK:
    class HeartGL(QOpenGLWidget):
        def __init__(self):
            super().__init__()
            self._rot_x   = 20.0
            self._rot_y   = 0.0
            self._rot_z   = 0.0
            self._auto    = True
            self._zoom    = 1.0
            self._last_pos = None
            self._pulse    = 0.0
            self._pulse_dir = 1

            t = QTimer(self)
            t.timeout.connect(self._animate)
            t.start(16)

        def _animate(self):
            if self._auto:
                self._rot_y += 0.4
            self._pulse += self._pulse_dir * 0.03
            if self._pulse >= 1.0: self._pulse_dir = -1
            if self._pulse <= 0.0: self._pulse_dir =  1
            self.update()

        def trigger_pulse(self):
            self._pulse = 1.0; self._pulse_dir = -1

        def initializeGL(self):
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_LIGHT1)
            glEnable(GL_COLOR_MATERIAL)
            glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
            glShadeModel(GL_SMOOTH)
            glEnable(GL_NORMALIZE)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Light 0 – warm key light
            glLightfv(GL_LIGHT0, GL_POSITION, [2.0, 3.0, 4.0, 1.0])
            glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 0.9, 0.9, 1.0])
            glLightfv(GL_LIGHT0, GL_SPECULAR, [0.5, 0.3, 0.3, 1.0])
            glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.25, 0.1, 0.1, 1.0])

            # Light 1 – cool fill
            glLightfv(GL_LIGHT1, GL_POSITION, [-3.0, -1.0, 2.0, 1.0])
            glLightfv(GL_LIGHT1, GL_DIFFUSE,  [0.5, 0.5, 0.8, 1.0])
            glLightfv(GL_LIGHT1, GL_AMBIENT,  [0.0, 0.0, 0.0, 1.0])

            glMaterialfv(GL_FRONT, GL_SPECULAR, [0.8, 0.2, 0.2, 1.0])
            glMaterialf(GL_FRONT, GL_SHININESS, 60)

            glClearColor(0.0, 0.0, 0.0, 1.0)

        def resizeGL(self, w, h):
            glViewport(0, 0, w, max(h, 1))
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(40.0, w / max(h, 1), 0.1, 100.0)
            glMatrixMode(GL_MODELVIEW)

        def paintGL(self):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            gluLookAt(0, 0, 6.5 / self._zoom, 0, 0, 0, 0, 1, 0)

            glRotatef(self._rot_x, 1, 0, 0)
            glRotatef(self._rot_y, 0, 1, 0)

            pulse_s = 1.0 + self._pulse * 0.05
            glScalef(pulse_s, pulse_s, pulse_s)

            self._draw_heart()

        def _draw_heart(self):
            """
            Build heart from anatomical geometry primitives:
            - Two atria (upper chambers)
            - Two ventricles (lower chambers, left larger)
            - Aorta and pulmonary artery
            - Superior/inferior vena cava
            """
            q = gluNewQuadric()
            gluQuadricNormals(q, GLU_SMOOTH)

            # ── Left ventricle (large, lower left) ─────────────────────────
            glPushMatrix()
            glTranslatef(-0.25, -0.55, 0)
            glScalef(0.82, 1.05, 0.78)
            glColor4f(0.76, 0.10, 0.18, 1.0)
            gluSphere(q, 0.72, 32, 24)
            glPopMatrix()

            # ── Right ventricle (smaller, lower right) ─────────────────────
            glPushMatrix()
            glTranslatef(0.55, -0.35, 0.05)
            glScalef(0.60, 0.80, 0.60)
            glColor4f(0.72, 0.15, 0.22, 1.0)
            gluSphere(q, 0.62, 28, 20)
            glPopMatrix()

            # ── Left atrium (upper left, posterior) ────────────────────────
            glPushMatrix()
            glTranslatef(-0.35, 0.55, -0.10)
            glScalef(0.72, 0.58, 0.65)
            glColor4f(0.82, 0.18, 0.24, 1.0)
            gluSphere(q, 0.58, 24, 18)
            glPopMatrix()

            # ── Right atrium (upper right) ─────────────────────────────────
            glPushMatrix()
            glTranslatef(0.52, 0.45, 0.10)
            glScalef(0.62, 0.52, 0.58)
            glColor4f(0.80, 0.20, 0.26, 1.0)
            gluSphere(q, 0.54, 24, 18)
            glPopMatrix()

            # ── Aortic arch (top, curves left) ─────────────────────────────
            glPushMatrix()
            glTranslatef(-0.15, 1.05, 0.10)
            glScalef(0.28, 0.72, 0.28)
            glColor4f(0.55, 0.55, 0.72, 1.0)
            gluSphere(q, 0.60, 20, 16)
            glPopMatrix()

            glPushMatrix()
            glTranslatef(-0.38, 0.88, 0.10)
            glScalef(0.24, 0.36, 0.24)
            glColor4f(0.52, 0.52, 0.70, 1.0)
            gluCylinder(q, 0.55, 0.48, 1.2, 16, 6)
            glPopMatrix()

            # ── Pulmonary artery (top right) ───────────────────────────────
            glPushMatrix()
            glTranslatef(0.35, 0.95, 0.20)
            glScalef(0.22, 0.55, 0.22)
            glColor4f(0.55, 0.55, 0.80, 1.0)
            gluCylinder(q, 0.50, 0.42, 1.0, 14, 5)
            glPopMatrix()

            # ── Superior vena cava ─────────────────────────────────────────
            glPushMatrix()
            glTranslatef(0.62, 0.70, 0.05)
            glScalef(0.19, 0.50, 0.19)
            glColor4f(0.35, 0.45, 0.70, 1.0)
            gluCylinder(q, 0.45, 0.42, 0.9, 12, 4)
            glPopMatrix()

            # ── Coronary surface vessels (thin red lines on surface) ───────
            glLineWidth(2.0)
            glColor4f(0.60, 0.05, 0.10, 0.85)
            glDisable(GL_LIGHTING)
            glBegin(GL_LINE_STRIP)
            for i in range(30):
                a = i / 29 * math.pi
                x = -0.25 + 0.45 * math.sin(a) * math.cos(a * 2)
                y = -0.55 + 0.55 * math.cos(a)
                z = 0.72 + 0.04
                glVertex3f(x, y, z)
            glEnd()
            glBegin(GL_LINE_STRIP)
            for i in range(25):
                a = i / 24 * math.pi
                x = -0.25 + 0.30 * math.sin(a + 0.5)
                y = -0.55 + 0.40 * math.cos(a)
                z = 0.72 + 0.04
                glVertex3f(x, y, z)
            glEnd()
            glEnable(GL_LIGHTING)

            gluDeleteQuadric(q)

        def mousePressEvent(self, e):
            self._auto = False
            self._last_pos = e.pos()

        def mouseMoveEvent(self, e):
            if self._last_pos:
                dx = e.x() - self._last_pos.x()
                dy = e.y() - self._last_pos.y()
                self._rot_y += dx * 0.5
                self._rot_x += dy * 0.5
                self._last_pos = e.pos()
                self.update()

        def mouseReleaseEvent(self, _):
            self._last_pos = None

        def wheelEvent(self, e):
            self._zoom *= 1.1 if e.angleDelta().y() > 0 else 0.9
            self._zoom = max(0.3, min(3.0, self._zoom))
            self.update()


# ── SVG fallback heart ────────────────────────────────────────────────────────

class HeartDiagram2D(QWidget):
    """Anatomical heart cross-section drawn entirely in QPainter."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(400, 380)
        self._pulse = 0.0
        self._dir   = 1
        t = QTimer(self); t.timeout.connect(self._anim); t.start(20)

    def trigger_pulse(self):
        self._pulse = 1.0; self._dir = -1

    def _anim(self):
        self._pulse += self._dir * 0.025
        if self._pulse >= 1.0: self._dir = -1
        if self._pulse <= 0.0: self._dir =  1
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        cx, cy = W // 2, H // 2 + 10
        scale = min(W, H) / 420 * (1.0 + self._pulse * 0.018)

        def T(x, y):
            return int(cx + x * scale), int(cy + y * scale)

        def ellipse(x, y, rx, ry, col, alpha=255):
            qc = QColor(col); qc.setAlpha(alpha)
            p.setBrush(QBrush(qc))
            p.setPen(QPen(QColor(CRIM2), 1.2))
            ex, ey = T(x - rx, y - ry)
            p.drawEllipse(ex, ey, int(rx * 2 * scale), int(ry * 2 * scale))

        def vessel(x1, y1, x2, y2, w, col):
            qc = QColor(col)
            p.setPen(QPen(qc, w * scale, Qt.SolidLine, Qt.RoundCap))
            p.setBrush(Qt.NoBrush)
            p.drawLine(*T(x1, y1), *T(x2, y2))

        # Heart outline (two bezier lobes)
        path = QPainterPath()
        path.moveTo(*T(0, 160))
        # Left lobe
        path.cubicTo(*T(-80, 60), *T(-185, 30), *T(-130, -80))
        path.cubicTo(*T(-85, -175), *T(10, -140), *T(0, -80))
        # Right lobe
        path.cubicTo(*T(10, -140), *T(105, -175), *T(130, -80))
        path.cubicTo(*T(185, 30), *T(80, 60), *T(0, 160))

        outer = QColor(CRIM); outer.setAlpha(30)
        p.setBrush(QBrush(outer))
        p.setPen(QPen(QColor(CRIM), 2))
        p.drawPath(path)

        # Chambers
        ellipse(-50, 30,  72, 82, "#C82333", 220)   # Left ventricle
        ellipse( 62, 45,  52, 65, "#D63044", 200)   # Right ventricle
        ellipse(-55, -80, 58, 48, "#E05060", 200)   # Left atrium
        ellipse( 65, -60, 48, 42, "#E06070", 195)   # Right atrium

        # Septum (wall between chambers)
        p.setPen(QPen(QColor(CRIM2), 3 * scale))
        p.setBrush(Qt.NoBrush)
        p.drawLine(*T(8, 100), *T(8, -120))

        # Valves
        valve_col = "#8B1A2A"
        for vx, vy, vr, name in [
            (-48, -28, 16, "Mitral"),
            (62,  -15, 12, "Tricusp."),
            (-15, -105, 10, "Aortic"),
            (50,  -108, 9,  "Pulm."),
        ]:
            qc = QColor(valve_col); qc.setAlpha(200)
            p.setBrush(QBrush(qc))
            p.setPen(QPen(QColor(CRIM2), 1))
            ex, ey = T(vx - vr, vy - vr//2)
            p.drawEllipse(ex, ey, int(vr * 2 * scale), int(vr * scale))
            p.setFont(QFont(FNT_UI, 6))
            p.setPen(QPen(QColor(INK2)))
            lx, ly = T(vx + vr + 4, vy)
            p.drawText(lx, ly, name)

        # Great vessels
        vessel(-15, -130, -30, -185, 14, "#6C7DB0")  # Aorta
        vessel( 45, -118,  60, -178, 11, "#6C7DB0")  # Pulm artery
        vessel( 78,  -75,  90, -150,  9, "#7080A0")  # SVC
        vessel(-70,  100, -85,  160,  9, "#7080A0")  # IVC

        # Labels with leader lines
        labels_data = [
            (-130, 40,  -195, 40,   "Left Ventricle",   CRIM),
            ( 120, 55,   178, 55,   "Right Ventricle",  "#D44"),
            (-130, -88, -200, -88,  "Left Atrium",      "#C33"),
            ( 120, -62,  192, -62,  "Right Atrium",     "#D33"),
            ( -25, -178, -80, -195, "Aorta",            "#4466AA"),
            (  62, -168,  115, -185,"Pulm. Artery",     "#4466AA"),
        ]
        for x1, y1, x2, y2, txt, col in labels_data:
            p.setPen(QPen(QColor(col), 0.8))
            p.setBrush(Qt.NoBrush)
            p.drawLine(*T(x1, y1), *T(x2, y2))
            # dot
            dx, dy = T(x1, y1)
            dc = QColor(col); dc.setAlpha(160)
            p.setBrush(QBrush(dc)); p.setPen(Qt.NoPen)
            p.drawEllipse(dx - 3, dy - 3, 6, 6)
            # text
            tx, ty = T(x2, y2)
            p.setPen(QPen(QColor(col)))
            p.setFont(QFont(FNT_UI, 7, QFont.Bold))
            align = Qt.AlignLeft if x2 > 0 else Qt.AlignRight
            p.drawText(tx + (5 if x2 > 0 else -85), ty - 8, 80, 16, align, txt)


# ── Heart page (wrapper with controls) ───────────────────────────────────────

class HeartPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lo = QHBoxLayout(self)
        lo.setContentsMargins(24, 20, 24, 20)
        lo.setSpacing(20)

        # Left: 3D / 2D view
        view_frame = QFrame()
        view_frame.setObjectName("Card")
        vl = QVBoxLayout(view_frame)
        vl.setContentsMargins(0, 0, 0, 0)

        # Title bar
        title_bar = QHBoxLayout()
        title_bar.setContentsMargins(20, 16, 20, 8)
        tl = QLabel("3D heart" if OPENGL_OK else "Heart, cross-section")
        tl.setFont(QFont(FNT_UI, 14, QFont.Bold))
        tl.setStyleSheet(f"color:{INK};")
        sub = QLabel("Drag to rotate, scroll to zoom" if OPENGL_OK else "Pulses with every beat")
        sub.setFont(QFont(FNT_UI, 9))
        sub.setStyleSheet(f"color:{MUTED};")
        title_bar.addWidget(tl)
        title_bar.addStretch()
        title_bar.addWidget(sub)
        vl.addLayout(title_bar)

        if OPENGL_OK:
            self._view = HeartGL()
        else:
            self._view = HeartDiagram2D()

        self._view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vl.addWidget(self._view, stretch=1)

        lo.addWidget(view_frame, stretch=3)

        # Right: anatomy info panel
        info = QFrame()
        info.setObjectName("Card")
        info.setFixedWidth(300)
        il = QVBoxLayout(info)
        il.setContentsMargins(20, 16, 20, 16)
        il.setSpacing(10)

        _al = QLabel("Cardiac Anatomy")
        _al.setFont(QFont(FNT_UI, 13, QFont.Bold))
        _al.setStyleSheet(f"color:{INK};")
        il.addWidget(_al)

        structures = [
            ("Left Ventricle",   CRIM,    "Main pumping chamber. Ejects blood into aorta at ~120 mmHg systolic pressure."),
            ("Right Ventricle",  "#D44",  "Pumps deoxygenated blood to lungs via pulmonary artery at ~25 mmHg."),
            ("Left Atrium",      "#C44",  "Receives oxygenated blood from pulmonary veins. Feeds left ventricle."),
            ("Right Atrium",     "#D55",  "Receives deoxygenated blood from vena cavae. Feeds right ventricle."),
            ("Aorta",            "#6688CC","Largest artery. Carries oxygenated blood from left ventricle to body."),
            ("Pulmonary Artery", "#6688CC","Carries deoxygenated blood to the lungs for oxygenation."),
            ("Mitral Valve",     "#8B1A2A","Bicuspid valve between left atrium and left ventricle."),
            ("Aortic Valve",     "#8B1A2A","Semilunar valve. Prevents backflow into left ventricle."),
        ]

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        sw = QWidget()
        sw.setStyleSheet("background:transparent;")
        sl = QVBoxLayout(sw)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(8)

        for name, col, desc in structures:
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{ background:{BG2}; border:none; border-left:3px solid {col};
                border-radius:0px; padding:0px; }}
            """)
            cl = QVBoxLayout(card); cl.setContentsMargins(10, 8, 10, 8); cl.setSpacing(2)
            nl = QLabel(name); nl.setFont(QFont(FNT_UI, 9, QFont.Bold))
            nl.setStyleSheet(f"color:{col}; background:transparent;")
            dl = QLabel(desc); dl.setFont(QFont(FNT_UI, 8))
            dl.setWordWrap(True)
            dl.setStyleSheet(f"color:{INK3}; background:transparent;")
            cl.addWidget(nl); cl.addWidget(dl)
            sl.addWidget(card)

        sl.addStretch()
        scroll.setWidget(sw)
        il.addWidget(scroll)

        lo.addWidget(info)

    def trigger_pulse(self):
        if hasattr(self._view, 'trigger_pulse'):
            self._view.trigger_pulse()
