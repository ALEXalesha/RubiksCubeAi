"""Модель кубика N×N (2×2, 3×3, 4×4) из геометрии, без Qt.

Каждая наклейка - точка в пространстве и направление наружу. Координаты удвоены, чтобы
были целыми: грань лежит на ±N, центры кубиков - на нечётных (у 3×3: -2, 0, 2). Ход - это
поворот слоя на 90°: наклейки слоя поворачиваются как точки, и по новому месту ищется их
новый номер. Так перестановки ходов не набиваются руками, а выводятся, и одна и та же
модель работает для любого N.

Порядок наклеек - как у Kociemba (URFDLB, на каждой грани строки сверху вниз по развёртке):
U1..U9 = 0..8, R = 9..17, F = 18..26, D = 27..35, L = 36..44, B = 45..53 для 3×3.
Состояние - массив цветов uint8 длины 6·N², цвет - номер грани, на которой он в собранном
кубике. Ход: new = old[perm] (на место i встаёт наклейка, что была на perm[i]).

Запись ходов - обычная: U R F D L B, ' - против часовой, 2 - пол-оборота; Rw - два слоя,
2R - второй слой от R (для 4×4), M E S - средние слои 3×3, x y z - поворот всего куба.
"""
from dataclasses import dataclass
import random

import numpy as np

FACES = "URFDLB"
U, R, F, D, L, B = range(6)
AXIS_OF_FACE = (1, 0, 2, 1, 0, 2)          # U,D - ось y; R,L - x; F,B - z
SIGN_OF_FACE = (1, 1, 1, -1, -1, -1)
OPPOSITE = (D, L, B, U, R, F)

# Наружная нормаль грани и направления «вниз по строкам» и «вправо по столбцам» развёртки.
NORMAL = ((0, 1, 0), (1, 0, 0), (0, 0, 1), (0, -1, 0), (-1, 0, 0), (0, 0, -1))
ROW_DIR = ((0, 0, 1), (0, -1, 0), (0, -1, 0), (0, 0, -1), (0, -1, 0), (0, -1, 0))
COL_DIR = ((1, 0, 0), (0, 0, -1), (1, 0, 0), (1, 0, 0), (0, 0, 1), (-1, 0, 0))


def rotate(v, axis, q):
    """Поворот целочисленного вектора на q·90° против часовой вокруг +оси (правый винт)."""
    x, y, z = v
    for _ in range(q % 4):
        if axis == 0:
            x, y, z = x, -z, y
        elif axis == 1:
            x, y, z = z, y, -x
        else:
            x, y, z = -y, x, z
    return (x, y, z)


@dataclass(frozen=True)
class Move:
    """Поворот слоёв layers вокруг оси axis (0 x, 1 y, 2 z) на q·90° против часовой, если
    смотреть с конца оси. Слой 0 - со стороны L, D, B; слой N-1 - со стороны R, U, F."""
    axis: int
    layers: tuple
    q: int

    def inverse(self):
        return Move(self.axis, self.layers, (4 - self.q) % 4)


class NotationError(ValueError):
    pass


class Puzzle:
    """Кубик N×N: наклейки, перестановки ходов, разбор и запись ходов."""

    _cache = {}

    def __new__(cls, n):
        if n in cls._cache:
            return cls._cache[n]
        self = super().__new__(cls)
        cls._cache[n] = self
        self._init(n)
        return self

    def _init(self, n):
        if n < 2:
            raise ValueError("кубик от 2×2")
        self.n = n
        self.size = 6 * n * n
        pos = np.zeros((self.size, 3), np.int32)
        nor = np.zeros((self.size, 3), np.int32)
        for f in range(6):
            for r in range(n):
                for c in range(n):
                    i = f * n * n + r * n + c
                    nor[i] = NORMAL[f]
                    pos[i] = (n * np.array(NORMAL[f]) + (2 * r - (n - 1)) * np.array(ROW_DIR[f])
                              + (2 * c - (n - 1)) * np.array(COL_DIR[f]))
        self.pos = pos
        self.normal = nor
        self.cubie = pos - nor                   # центр кубика, на котором наклейка
        self.index = {tuple(p): i for i, p in enumerate(pos)}
        if len(self.index) != self.size:
            raise AssertionError("наклейки совпали местами")
        self._layer_perm = {}
        self._perm = {}

    # --- состояние ---

    def solved(self):
        return np.repeat(np.arange(6, dtype=np.uint8), self.n * self.n)

    def is_solved(self, state):
        """Собран: каждая грань одного цвета (как бы куб ни был повёрнут целиком)."""
        faces = np.asarray(state).reshape(6, self.n * self.n)
        return bool((faces == faces[:, :1]).all())

    def layer_of(self, i, axis):
        """Номер слоя наклейки i вдоль оси (0..N-1)."""
        return (int(self.cubie[i][axis]) + self.n - 1) // 2

    # --- ходы ---

    def layer_perm(self, axis, layer, q):
        key = (axis, layer, q % 4)
        p = self._layer_perm.get(key)
        if p is None:
            p = np.arange(self.size)
            coord = 2 * layer - (self.n - 1)
            for i in np.nonzero(self.cubie[:, axis] == coord)[0]:
                j = self.index[rotate(tuple(int(v) for v in self.pos[i]), axis, q)]
                p[j] = i
            p.setflags(write=False)
            self._layer_perm[key] = p
        return p

    def perm(self, move):
        if isinstance(move, str):
            move = self.parse(move)
        p = self._perm.get(move)
        if p is None:
            p = np.arange(self.size)
            for layer in move.layers:
                p = p[self.layer_perm(move.axis, layer, move.q)]
            p.setflags(write=False)
            self._perm[move] = p
        return p

    def apply(self, state, moves):
        """Ход или последовательность (строка «R U R'», список имён или Move)."""
        state = np.asarray(state)
        for m in self.moves(moves):
            state = state[self.perm(m)]
        return state

    def moves(self, seq):
        if isinstance(seq, Move):
            return [seq]
        if isinstance(seq, str):
            seq = seq.split()
        return [m if isinstance(m, Move) else self.parse(m) for m in seq]

    def sequence_perm(self, seq):
        p = np.arange(self.size)
        for m in self.moves(seq):
            p = p[self.perm(m)]
        return p

    # --- запись ---

    def face_move(self, face, depth=1, turns=1, wide=False):
        """Ход грани: turns - четвертей по часовой, если смотреть на грань; depth - какой
        слой от грани (или сколько слоёв, если wide)."""
        n = self.n
        if not 1 <= depth <= n:
            raise NotationError("нет такого слоя")
        depths = range(depth) if wide else (depth - 1,)
        sign = SIGN_OF_FACE[face]
        layers = tuple(sorted(n - 1 - d if sign > 0 else d for d in depths))
        q = (-turns) % 4 if sign > 0 else turns % 4
        return Move(AXIS_OF_FACE[face], layers, q)

    def parse(self, text):
        t = text.strip()
        if not t:
            raise NotationError("пустой ход")
        turns = 1
        if t.endswith("2'") or t.endswith("'2"):
            turns, t = 2, t[:-2]
        elif t.endswith("'"):
            turns, t = 3, t[:-1]
        elif t.endswith("2") and len(t) > 1 and not t[:-1].isdigit():
            turns, t = 2, t[:-1]
        digits = ""
        while t and t[0].isdigit():
            digits += t[0]
            t = t[1:]
        wide = t.endswith("w")
        if wide:
            t = t[:-1]
        if len(t) != 1:
            raise NotationError(f"не ход: {text!r}")
        letter = t
        n = self.n
        if letter in "xyz" and not digits and not wide:
            face = {"x": R, "y": U, "z": F}[letter]
            m = self.face_move(face, n, turns, wide=True)
            return m
        if letter in "MES" and not digits and not wide:
            if n % 2 == 0:
                raise NotationError("у чётного кубика нет среднего слоя")
            face = {"M": L, "E": D, "S": F}[letter]
            return self.face_move(face, n // 2 + 1, turns)
        if letter.upper() != letter or letter not in FACES:
            if letter in "urfdlb" and not digits and not wide:     # r = Rw
                return self.face_move(FACES.index(letter.upper()), 2, turns, wide=True)
            raise NotationError(f"не ход: {text!r}")
        face = FACES.index(letter)
        depth = int(digits) if digits else (2 if wide else 1)
        if wide and n < 3:
            raise NotationError("у 2×2 нет двойных слоёв")
        return self.face_move(face, depth, turns, wide=wide)

    def name(self, move):
        """Каноническая запись хода."""
        n = self.n
        layers = tuple(sorted(move.layers))
        k = len(layers)
        pos_face = {0: R, 1: U, 2: F}[move.axis]
        neg_face = OPPOSITE[pos_face]

        def suffix(face):
            turns = (-move.q) % 4 if SIGN_OF_FACE[face] > 0 else move.q % 4
            return {1: "", 2: "2", 3: "'"}[turns]

        if k == n:
            letter = {0: "x", 1: "y", 2: "z"}[move.axis]
            return letter + suffix(pos_face)
        if n == 3 and layers == (1,):
            face = {0: L, 1: D, 2: F}[move.axis]
            return {L: "M", D: "E", F: "S"}[face] + suffix(face)
        if layers == tuple(range(n - k, n)):
            face, depth, wide = pos_face, k, k > 1
        elif layers == tuple(range(k)):
            face, depth, wide = neg_face, k, k > 1
        elif k == 1:
            layer = layers[0]
            if layer >= n / 2:
                face, depth = pos_face, n - layer
            else:
                face, depth = neg_face, layer + 1
            return f"{depth}{FACES[face]}{suffix(face)}"
        else:
            raise NotationError(f"нет записи для {move}")
        if not wide:
            return FACES[face] + suffix(face)
        prefix = "" if depth == 2 else str(depth)
        return f"{prefix}{FACES[face]}w{suffix(face)}"

    def names(self, seq):
        return [self.name(m) for m in self.moves(seq)]

    def face_moves(self):
        """18 ходов граней: U U2 U' R R2 R' ... - так их считает Kociemba (по грани, затем
        четверти 1, 2, 3)."""
        return [self.face_move(f, 1, t) for f in range(6) for t in (1, 2, 3)]

    def scramble_moves(self):
        """Ходы для перемешивания: грани, у 4×4 и больше - ещё двойные слои."""
        out = self.face_moves()
        if self.n >= 4:
            out += [self.face_move(f, 2, t, wide=True) for f in (U, R, F) for t in (1, 2, 3)]
        return out

    def random_moves(self, length, rng=None):
        """Случайная запись без двух ходов одной оси подряд в одном слое."""
        rng = rng or random.Random()
        pool = self.scramble_moves()
        out = []
        while len(out) < length:
            m = rng.choice(pool)
            if out and out[-1].axis == m.axis and out[-1].layers == m.layers:
                continue
            if len(out) >= 2 and out[-1].axis == m.axis == out[-2].axis:
                continue
            out.append(m)
        return out


def simplify(moves):
    """Склеить соседние ходы одних слоёв (R R -> R2, R R' -> ничего). Ходы по одной оси
    в разных слоях переставимы, поэтому склеиваются и через них (R L R -> R2 L)."""
    out = []
    for m in moves:
        merged = False
        for j in range(len(out) - 1, -1, -1):
            o = out[j]
            if o.axis != m.axis:
                break
            if o.layers == m.layers:
                q = (o.q + m.q) % 4
                if q:
                    out[j] = Move(o.axis, o.layers, q)
                else:
                    del out[j]
                merged = True
                break
        if not merged:
            out.append(m)
    return out


def inverse(moves):
    return [m.inverse() for m in reversed(list(moves))]


def metric_htm(moves):
    """Длина в метрике HTM: любой поворот грани (и четверть, и пол-оборота) - один ход."""
    return len(moves)
