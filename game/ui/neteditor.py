"""«Свой кубик»: развёртка, которую раскрашивают мышью.

Развёртка - крестом: U сверху, в ряд L F R B, D снизу; наклейки в том же порядке, что в
модели (cube.py), поэтому раскраску можно сразу отдать окну. Цвет выбирается внизу,
нажатие или протяжка по наклейкам красит. Центры 3×3 не красятся: по ним кубик понимает,
где какая грань.
"""
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from game.core import cube as C
from game.ui.cubeview import STICKER_COLORS

FACE_AT = {C.U: (1, 0), C.L: (0, 1), C.F: (1, 1), C.R: (2, 1), C.B: (3, 1), C.D: (1, 2)}


class NetEditor(QWidget):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(360, 300)
        self.background = QColor("#1f242e")
        self.text = QColor("#e6e8ec")
        self.accent = QColor("#5fb4ff")
        self.brush = C.U
        self.painting = False
        self.load(3, C.Puzzle(3).solved())

    def load(self, n, state):
        self.n = n
        self.pz = C.Puzzle(n)
        self.state = state.copy()
        self.fixed = {f * n * n + (n * n) // 2 for f in range(6)} if n % 2 else set()
        self.update()

    def set_theme(self, background, text, accent):
        self.background, self.text, self.accent = QColor(background), QColor(text), QColor(accent)
        self.update()

    # --- раскладка ---

    def _cell(self):
        n = self.n
        w, h = self.width(), self.height()
        cell = min(w / (4 * n + 1), h / (3 * n + 3))
        ox = (w - cell * 4 * n) / 2
        oy = cell * 0.5
        return cell, ox, oy

    def sticker_rect(self, i):
        n = self.n
        cell, ox, oy = self._cell()
        f, rest = divmod(i, n * n)
        r, c = divmod(rest, n)
        fx, fy = FACE_AT[f]
        return QRectF(ox + (fx * n + c) * cell, oy + (fy * n + r) * cell, cell, cell)

    def swatch_rect(self, color):
        cell, ox, oy = self._cell()
        size = cell * 1.1
        total = 6 * size + 5 * size * 0.4
        x0 = (self.width() - total) / 2
        y = oy + 3 * self.n * cell + cell * 0.8
        return QRectF(x0 + color * size * 1.4, y, size, size)

    # --- рисование ---

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), self.background)
        for i in range(self.pz.size):
            r = self.sticker_rect(i).adjusted(2, 2, -2, -2)
            p.setPen(QPen(QColor("#0b0c0f"), 1))
            p.setBrush(QColor(STICKER_COLORS[int(self.state[i])]))
            p.drawRoundedRect(r, r.width() * 0.16, r.width() * 0.16)
            if i in self.fixed:
                p.setPen(QPen(QColor("#0b0c0f"), 2))
                c = r.center()
                p.drawEllipse(QPointF(c.x(), c.y()), r.width() * 0.12, r.width() * 0.12)
        for color in range(6):
            r = self.swatch_rect(color)
            p.setPen(QPen(self.accent if color == self.brush else QColor("#0b0c0f"), 3 if color == self.brush else 1))
            p.setBrush(QColor(STICKER_COLORS[color]))
            p.drawRoundedRect(r, r.width() * 0.2, r.width() * 0.2)
        p.end()

    # --- мышь ---

    def _paint_at(self, pos):
        for color in range(6):
            if self.swatch_rect(color).contains(pos):
                self.brush = color
                self.update()
                return
        for i in range(self.pz.size):
            if self.sticker_rect(i).contains(pos):
                if i not in self.fixed and int(self.state[i]) != self.brush:
                    self.state[i] = self.brush
                    self.update()
                    self.changed.emit()
                return

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.painting = True
            self._paint_at(e.position())

    def mouseMoveEvent(self, e):
        if self.painting:
            self._paint_at(e.position())

    def mouseReleaseEvent(self, _e):
        self.painting = False
