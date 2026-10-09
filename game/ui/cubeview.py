"""Объёмный кубик на QPainter: свои 3D-проекция и сортировка по глубине, без OpenGL.

Каждый кубик - коробка из шести граней: тёмный пластик и наклейка поверх. Кубики
рисуются от дальних к ближним (художник), грани к нам спиной не рисуются. Во время
поворота слоя кубики делятся на пласты вдоль оси поворота, и пласты тоже идут от
дальнего к ближнему - так вращающийся слой не протыкает соседний.

Мышь:
- тянуть за наклейку - повернуть слой в сторону движения;
- тянуть за фон - вращать весь кубик (только вид, состояние не меняется).
Курсор не захватывается и не прячется: обычные события нажатия и движения в пределах окна.
"""
import math

from PySide6.QtCore import QElapsedTimer, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget

from game.core import cube as C

STICKER_COLORS = {          # стандартная западная раскраска: белый сверху, зелёный спереди
    C.U: "#f4f4ef", C.R: "#c8262b", C.F: "#1f9a4a", C.D: "#f3d12b", C.L: "#f07d1a", C.B: "#2260c8",
}
BODY = QColor("#15171c")
BODY_EDGE = QColor("#0b0c0f")
LIGHT = (-0.35, 0.65, 0.68)
DRAG_START = 8              # пикселей до того, как движение мыши считается жестом
KEY_FACES = {Qt.Key.Key_U: "U", Qt.Key.Key_D: "D", Qt.Key.Key_L: "L", Qt.Key.Key_R: "R",
             Qt.Key.Key_F: "F", Qt.Key.Key_B: "B", Qt.Key.Key_M: "M", Qt.Key.Key_E: "E",
             Qt.Key.Key_S: "S", Qt.Key.Key_X: "x", Qt.Key.Key_Y: "y", Qt.Key.Key_Z: "z"}


def _rot_axis(v, axis, angle):
    """Поворот точки вокруг оси куба на угол (радианы), правый винт."""
    x, y, z = v
    c, s = math.cos(angle), math.sin(angle)
    if axis == 0:
        return (x, y * c - z * s, y * s + z * c)
    if axis == 1:
        return (x * c + z * s, y, -x * s + z * c)
    return (x * c - y * s, x * s + y * c, z)


class CubeView(QWidget):
    """Вид кубика. turn(move) ставит ход в очередь и плавно его показывает; по
    окончании - сигнал moveFinished(move, source). userTurn(move) - ход, который
    сделал человек мышью или клавишей (окно решает, что с ним делать)."""

    moveFinished = Signal(object, str)
    userTurn = Signal(object)
    viewChanged = Signal()

    def __init__(self, n=3, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(360, 360)
        self.yaw = -35.0
        self.pitch = 28.0
        self.turn_ms = 180
        self.interactive = True
        self.highlight = set()          # номера наклеек с обводкой
        self.background = QColor("#1f242e")
        self.accent = QColor("#5fb4ff")
        self.queue = []                 # (move, source)
        self.current = None             # (move, source) в анимации
        self.progress = 0.0
        self.clock = QElapsedTimer()
        self.timer = QTimer(self)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self._tick)
        self.hits = []                  # (многоугольник, номер наклейки) в порядке рисования
        self.press = None
        self.drag_mode = None
        self.set_puzzle(n)

    # --- состояние ---

    def set_puzzle(self, n, state=None):
        self.stop_animation()
        self.pz = C.Puzzle(n)
        self.n = n
        self.state = self.pz.solved() if state is None else state.copy()
        self._sticker_at = {}
        for i in range(self.pz.size):
            self._sticker_at[(tuple(int(v) for v in self.pz.cubie[i]),
                              tuple(int(v) for v in self.pz.normal[i]))] = i
        k = n - 1
        self.cubies = [(x, y, z) for x in range(-k, k + 1, 2) for y in range(-k, k + 1, 2)
                       for z in range(-k, k + 1, 2) if max(abs(x), abs(y), abs(z)) == k]
        self.highlight = set()
        self.update()

    def set_state(self, state):
        self.stop_animation()
        self.state = state.copy()
        self.update()

    def set_theme(self, background, accent):
        self.background = QColor(background)
        self.accent = QColor(accent)
        self.update()

    # --- ходы ---

    def turn(self, move, source="user", animate=True):
        if isinstance(move, str):
            move = self.pz.parse(move)
        if not animate or self.turn_ms <= 0:
            self.finish_all()
            self.state = self.pz.apply(self.state, [move])
            self.update()
            self.moveFinished.emit(move, source)
            return
        self.queue.append((move, source))
        if self.current is None:
            self._next()

    def animating(self):
        return self.current is not None or bool(self.queue)

    def stop_animation(self):
        """Бросить очередь: текущий ход не доделывается (для сброса и новой позиции)."""
        self.timer.stop()
        self.queue.clear()
        self.current = None
        self.progress = 0.0

    def finish_all(self):
        """Доделать все ходы из очереди сразу (сигналы идут как обычно)."""
        self.timer.stop()
        pending = ([self.current] if self.current else []) + self.queue
        self.queue = []
        self.current = None
        self.progress = 0.0
        for move, source in pending:
            self.state = self.pz.apply(self.state, [move])
            self.moveFinished.emit(move, source)
        self.update()

    def _duration(self, move):
        q = 2 if move.q == 2 else 1
        return self.turn_ms * (1.45 if q == 2 else 1.0)

    def _next(self):
        if not self.queue:
            self.current = None
            self.timer.stop()
            return
        self.current = self.queue.pop(0)
        self.progress = 0.0
        self.clock.start()
        self.timer.start()

    def _tick(self):
        if self.current is None:
            self.timer.stop()
            return
        move, source = self.current
        self.progress = min(1.0, self.clock.elapsed() / max(1.0, self._duration(move)))
        if self.progress >= 1.0:
            self.state = self.pz.apply(self.state, [move])
            self.current = None
            self.progress = 0.0
            self.update()
            self.moveFinished.emit(move, source)
            if self.current is None:
                self._next()
            return
        self.update()

    # --- геометрия ---

    def _angle(self):
        if self.current is None:
            return None
        move = self.current[0]
        t = self.progress
        t = t * t * (3 - 2 * t)                     # плавный разгон и торможение
        signed = {1: 90.0, 2: 180.0, 3: -90.0}[move.q]
        return move, math.radians(signed * t)

    def _view(self, p):
        """Точка куба (удвоенные координаты) в координаты камеры."""
        s = 1.0 / self.n
        x, y, z = p[0] * s, p[1] * s, p[2] * s
        a = math.radians(self.yaw)
        x, z = x * math.cos(a) + z * math.sin(a), -x * math.sin(a) + z * math.cos(a)
        b = math.radians(self.pitch)
        y, z = y * math.cos(b) - z * math.sin(b), y * math.sin(b) + z * math.cos(b)
        return (x, y, z)

    def _project(self, v):
        dist = 7.0
        scale = min(self.width(), self.height()) * 0.205 * dist
        k = scale / (dist - v[2])
        return QPointF(self.width() / 2 + v[0] * k, self.height() / 2 - v[1] * k)

    def _axis_toward_camera(self, axis):
        e = [0, 0, 0]
        e[axis] = 1
        return self._view(e)[2] - self._view((0, 0, 0))[2] > 0

    # --- рисование ---

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), self.background)
        self.hits = []
        anim = self._angle()
        groups = [self.cubies]
        if anim is not None:
            move, _ = anim
            lo, hi = min(move.layers), max(move.layers)
            layer = lambda c: (c[move.axis] + self.n - 1) // 2  # noqa: E731
            below = [c for c in self.cubies if layer(c) < lo]
            moving = [c for c in self.cubies if lo <= layer(c) <= hi]
            above = [c for c in self.cubies if layer(c) > hi]
            groups = [below, moving, above]
            if not self._axis_toward_camera(move.axis):
                groups.reverse()
        for group in groups:
            items = []
            for c in group:
                rot = None
                if anim is not None:
                    move, angle = anim
                    if (c[move.axis] + self.n - 1) // 2 in move.layers:
                        rot = (move.axis, angle)
                center = self._place(c, rot)
                items.append((self._view(center)[2], c, rot))
            items.sort(key=lambda it: it[0])
            for _, c, rot in items:
                self._draw_cubie(p, c, rot)
        p.end()

    def _place(self, point, rot):
        return _rot_axis(point, rot[0], rot[1]) if rot else point

    def _draw_cubie(self, p, c, rot):
        h = 0.97
        for axis in range(3):
            for sign in (-1, 1):
                normal = [0, 0, 0]
                normal[axis] = sign
                u_axis, v_axis = [a for a in range(3) if a != axis]
                corners = []
                for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    q = list(c)
                    q[axis] += sign * h
                    q[u_axis] += du * h
                    q[v_axis] += dv * h
                    corners.append(q)
                if sign * (1, -1, 1)[axis] < 0:          # обход против часовой, если смотреть снаружи
                    corners.reverse()
                world = [self._place(q, rot) for q in corners]
                cam = [self._view(q) for q in world]
                pts = [self._project(v) for v in cam]
                if not self._front(pts):
                    continue
                poly = QPolygonF(pts)
                p.setPen(QPen(BODY_EDGE, 1))
                p.setBrush(BODY)
                p.drawPolygon(poly)
                idx = self._sticker_at.get((c, tuple(normal)))
                if idx is None:
                    continue
                center = [sum(v[k] for v in world) / 4 for k in range(3)]
                inset = [[center[k] + 0.82 * (w[k] - center[k]) for k in range(3)] for w in world]
                spts = [self._project(self._view(v)) for v in inset]
                color = QColor(STICKER_COLORS[int(self.state[idx])])
                nrm = self._view(self._place(normal, (rot[0], rot[1])) if rot else normal)
                zero = self._view((0, 0, 0))
                nv = [nrm[k] - zero[k] for k in range(3)]
                light = sum(a * b for a, b in zip(nv, LIGHT)) / math.sqrt(sum(a * a for a in LIGHT))
                color = color.lighter(100 + int(12 * max(0.0, light))) if light > 0 else color.darker(
                    100 + int(25 * -light))
                color = color.darker(100 + int(18 * (1 - max(0.0, nv[2]))))
                path = _rounded(spts, 0.18)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(color)
                p.drawPath(path)
                if idx in self.highlight:
                    p.setPen(QPen(self.accent, 3))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawPath(path)
                self.hits.append((QPolygonF(spts), idx))

    @staticmethod
    def _front(pts):
        area = 0.0
        for i in range(4):
            a, b = pts[i], pts[(i + 1) % 4]
            area += a.x() * b.y() - b.x() * a.y()
        return area < 0          # экранная y вниз: передние грани идут по часовой

    # --- мышь и клавиши ---

    def sticker_at(self, pos):
        for poly, idx in reversed(self.hits):
            if poly.containsPoint(pos, Qt.FillRule.OddEvenFill):
                return idx
        return None

    def sticker_center(self, idx):
        """Центр наклейки на экране (для проверок и подсказок)."""
        for poly, i in self.hits:
            if i == idx:
                b = poly.boundingRect()
                return b.center()
        return None

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        self.setFocus()
        pos = e.position()
        idx = self.sticker_at(pos) if self.interactive else None
        self.press = (pos, idx, self.yaw, self.pitch)
        self.drag_mode = None

    def mouseMoveEvent(self, e):
        if self.press is None:
            return
        pos = e.position()
        start, idx, yaw0, pitch0 = self.press
        dx, dy = pos.x() - start.x(), pos.y() - start.y()
        if self.drag_mode is None:
            if math.hypot(dx, dy) < DRAG_START:
                return
            self.drag_mode = "orbit"
            if idx is not None:
                move = self.move_for_drag(idx, dx, dy)
                if move is not None:
                    self.drag_mode = "turn"
                    self.userTurn.emit(move)
        if self.drag_mode == "orbit":
            self.yaw = yaw0 + dx * 0.45
            self.pitch = max(-85.0, min(85.0, pitch0 + dy * 0.45))
            self.update()
            self.viewChanged.emit()

    def mouseReleaseEvent(self, _e):
        self.press = None
        self.drag_mode = None

    def move_for_drag(self, idx, dx, dy):
        """Какой слой и куда повернуть, если наклейку idx тянут на (dx, dy) пикселей."""
        normal = tuple(int(v) for v in self.pz.normal[idx])
        cub = tuple(int(v) for v in self.pz.cubie[idx])
        pos = tuple(int(v) for v in self.pz.pos[idx])
        here = self._project(self._view(pos))
        best = None
        for axis in range(3):
            if normal[axis] != 0:
                continue
            d = [0, 0, 0]
            d[axis] = 1
            there = self._project(self._view(tuple(pos[k] + d[k] for k in range(3))))
            sx, sy = there.x() - here.x(), there.y() - here.y()
            length = math.hypot(sx, sy)
            if length < 1e-6:
                continue
            score = (dx * sx + dy * sy) / length
            if best is None or abs(score) > abs(best[0]):
                best = (score, axis)
        if best is None:
            return None
        score, axis = best
        d = [0, 0, 0]
        d[axis] = 1 if score > 0 else -1
        r = (normal[1] * d[2] - normal[2] * d[1], normal[2] * d[0] - normal[0] * d[2],
             normal[0] * d[1] - normal[1] * d[0])      # ось поворота = нормаль × направление
        rot_axis = next(a for a in range(3) if r[a] != 0)
        q = 1 if r[rot_axis] > 0 else 3
        layer = (cub[rot_axis] + self.n - 1) // 2
        return C.Move(rot_axis, (layer,), q)

    def keyPressEvent(self, e):
        name = KEY_FACES.get(e.key())
        if name is None or not self.interactive:
            super().keyPressEvent(e)
            return
        if name in "MES" and self.n % 2 == 0:
            return
        if e.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            name += "'"
        self.userTurn.emit(self.pz.parse(name))


def _rounded(pts, r):
    """Четырёхугольник со скруглёнными углами (r - доля стороны)."""
    path = QPainterPath()
    n = len(pts)
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        p1 = QPointF(b.x() + (a.x() - b.x()) * r, b.y() + (a.y() - b.y()) * r)
        p2 = QPointF(b.x() + (c.x() - b.x()) * r, b.y() + (c.y() - b.y()) * r)
        if i == 0:
            path.moveTo(p1)
        else:
            path.lineTo(p1)
        path.quadTo(b, p2)
    path.closeSubpath()
    return path
