"""Кубик 3×3 на уровне кубиков: 8 углов и 12 рёбер, где какой стоит и как повёрнут.

Обозначения и порядок - как у Kociemba: углы URF UFL ULB UBR DFR DLF DBL DRB, рёбра UR UF UL
UB DR DF DL DB FR FL BL BR. cp[i] - какой угол стоит на месте i, co[i] - его поворот
(0 - наклейка U/D на грани U/D, 1 - по часовой, 2 - против), так же ep и eo для рёбер.
Ход - тоже CubieCube; состояние после хода M - это S * M.

Здесь же проверка, что раскраска вообще возможна: из развёртки нельзя собрать кубик, если
перепутаны наклейки угла, повёрнут один угол, перевёрнуто одно ребро или переставлены два
кубика. Это ловит check() и называет причину.
"""
from dataclasses import dataclass, field
import random

import numpy as np

from game.core import cube as C

U, R, F, D, L, B = range(6)

CORNER_NAMES = ("URF", "UFL", "ULB", "UBR", "DFR", "DLF", "DBL", "DRB")
EDGE_NAMES = ("UR", "UF", "UL", "UB", "DR", "DF", "DL", "DB", "FR", "FL", "BL", "BR")


def _f(face, k):
    """Номер наклейки по записи Kociemba: U1 -> _f(U, 1) = 0."""
    return face * 9 + k - 1


CORNER_FACELET = (
    (_f(U, 9), _f(R, 1), _f(F, 3)), (_f(U, 7), _f(F, 1), _f(L, 3)),
    (_f(U, 1), _f(L, 1), _f(B, 3)), (_f(U, 3), _f(B, 1), _f(R, 3)),
    (_f(D, 3), _f(F, 9), _f(R, 7)), (_f(D, 1), _f(L, 9), _f(F, 7)),
    (_f(D, 7), _f(B, 9), _f(L, 7)), (_f(D, 9), _f(R, 9), _f(B, 7)),
)
EDGE_FACELET = (
    (_f(U, 6), _f(R, 2)), (_f(U, 8), _f(F, 2)), (_f(U, 4), _f(L, 2)), (_f(U, 2), _f(B, 2)),
    (_f(D, 6), _f(R, 8)), (_f(D, 2), _f(F, 8)), (_f(D, 4), _f(L, 8)), (_f(D, 8), _f(B, 8)),
    (_f(F, 6), _f(R, 4)), (_f(F, 4), _f(L, 6)), (_f(B, 6), _f(L, 4)), (_f(B, 4), _f(R, 6)),
)
CORNER_COLOR = ((U, R, F), (U, F, L), (U, L, B), (U, B, R), (D, F, R), (D, L, F), (D, B, L), (D, R, B))
EDGE_COLOR = ((U, R), (U, F), (U, L), (U, B), (D, R), (D, F), (D, L), (D, B), (F, R), (F, L), (B, L), (B, R))
CENTERS = tuple(_f(f, 5) for f in range(6))


class InvalidCube(ValueError):
    """Раскраску нельзя получить поворотами. reason - код причины (для подсказки в окне):
    count, centers, corner, edge, duplicate, twist, flip, parity."""

    def __init__(self, reason, detail=""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def parity(perm):
    """Чётность перестановки: 0 - чётная, 1 - нечётная."""
    p = list(perm)
    seen = [False] * len(p)
    odd = 0
    for i in range(len(p)):
        if seen[i]:
            continue
        j, length = i, 0
        while not seen[j]:
            seen[j] = True
            j = p[j]
            length += 1
        odd ^= (length - 1) & 1
    return odd


@dataclass
class CubieCube:
    cp: list = field(default_factory=lambda: list(range(8)))
    co: list = field(default_factory=lambda: [0] * 8)
    ep: list = field(default_factory=lambda: list(range(12)))
    eo: list = field(default_factory=lambda: [0] * 12)

    def copy(self):
        return CubieCube(list(self.cp), list(self.co), list(self.ep), list(self.eo))

    def __mul__(self, other):
        """self * other: сначала self, потом other (other - обычно ход)."""
        cp = [self.cp[other.cp[i]] for i in range(8)]
        co = [(self.co[other.cp[i]] + other.co[i]) % 3 for i in range(8)]
        ep = [self.ep[other.ep[i]] for i in range(12)]
        eo = [(self.eo[other.ep[i]] + other.eo[i]) % 2 for i in range(12)]
        return CubieCube(cp, co, ep, eo)

    def inverse(self):
        cp = [0] * 8
        co = [0] * 8
        ep = [0] * 12
        eo = [0] * 12
        for i in range(8):
            cp[self.cp[i]] = i
        for i in range(8):
            co[i] = (-self.co[cp[i]]) % 3
        for i in range(12):
            ep[self.ep[i]] = i
        for i in range(12):
            eo[i] = self.eo[ep[i]]
        return CubieCube(cp, co, ep, eo)

    def is_solved(self):
        return self == CubieCube()

    def check(self):
        """InvalidCube, если такой кубик нельзя получить поворотами."""
        if sorted(self.cp) != list(range(8)) or sorted(self.ep) != list(range(12)):
            raise InvalidCube("duplicate")
        if sum(self.co) % 3:
            raise InvalidCube("twist")
        if sum(self.eo) % 2:
            raise InvalidCube("flip")
        if parity(self.cp) != parity(self.ep):
            raise InvalidCube("parity")

    def to_facelets(self):
        """Развёртка 54 цветов (номера граней URFDLB)."""
        f = np.repeat(np.arange(6, dtype=np.uint8), 9)
        for i in range(8):
            j, ori = self.cp[i], self.co[i]
            for k in range(3):
                f[CORNER_FACELET[i][(k + ori) % 3]] = CORNER_COLOR[j][k]
        for i in range(12):
            j, ori = self.ep[i], self.eo[i]
            for k in range(2):
                f[EDGE_FACELET[i][(k + ori) % 2]] = EDGE_COLOR[j][k]
        return f


def normalize(state):
    """Развёртку 3×3 с центрами где угодно (после M, E, S, x, y, z) повернуть целиком так,
    чтобы центры стояли как в собранном. Возвращает (развёртка, поворот) - поворот из
    ходов x y z; решение для повёрнутого кубика переводится обратно map_moves."""
    pz = C.Puzzle(3)
    state = np.asarray(state)
    if sorted(int(state[c]) for c in CENTERS) != list(range(6)):
        raise InvalidCube("centers")
    for rot in ROTATIONS:
        s = pz.apply(state, rot) if rot else state
        if all(int(s[c]) == f for f, c in enumerate(CENTERS)):
            return s, rot
    raise AssertionError("нет поворота к центрам")       # 24 поворота покрывают всё


def _rotations():
    """24 поворота куба целиком записями из x, y, z."""
    pz = C.Puzzle(3)
    out = []
    seen = set()
    for a in ("", "x", "x2", "x'", "z", "z'"):
        for b in ("", "y", "y2", "y'"):
            seq = " ".join(s for s in (a, b) if s)
            key = tuple(pz.sequence_perm(seq)) if seq else tuple(range(54))
            if key not in seen:
                seen.add(key)
                out.append(seq)
    return tuple(out)


ROTATIONS = _rotations()


def map_moves(moves, rotation):
    """Ходы, найденные для кубика после поворота rotation, - в ходы исходного кубика.
    Повёрнутый кубик S·rot решается ходами m1..mk; тогда исходный решается ходами
    rot·m1·rot⁻¹ ... - это те же повороты, но вокруг осей, которые были до поворота."""
    pz = C.Puzzle(3)
    if not rotation:
        return list(pz.moves(moves))
    rot = pz.moves(rotation)
    out = []
    for m in pz.moves(moves):
        # ось и слой хода в системе исходного кубика: повернуть нормаль грани назад
        face = _face_of(m)
        normal = C.NORMAL[face]
        for r in reversed(rot):                       # обратный поворот rot
            normal = C.rotate(normal, r.axis, -r.q)
        new_face = C.NORMAL.index(tuple(normal))
        turns = (-m.q) % 4 if C.SIGN_OF_FACE[face] > 0 else m.q % 4
        out.append(pz.face_move(new_face, 1, turns))
    return out


def _face_of(m):
    if len(m.layers) != 1 or m.layers[0] not in (0, 2):
        raise ValueError("переводятся только ходы граней")
    sign = 1 if m.layers[0] == 2 else -1
    return next(f for f in range(6) if C.AXIS_OF_FACE[f] == m.axis and C.SIGN_OF_FACE[f] == sign)


def from_facelets(state):
    """CubieCube по развёртке с центрами на месте. InvalidCube с причиной, если раскраска
    невозможна."""
    f = [int(v) for v in np.asarray(state)]
    if len(f) != 54:
        raise InvalidCube("count", "нужно 54 наклейки")
    counts = [f.count(c) for c in range(6)]
    if counts != [9] * 6:
        raise InvalidCube("count", str(counts))
    if [f[c] for c in CENTERS] != list(range(6)):
        raise InvalidCube("centers")
    cc = CubieCube()
    for i in range(8):
        cols = [f[k] for k in CORNER_FACELET[i]]
        ori = next((o for o in range(3) if cols[o] in (U, D)), None)
        if ori is None:
            raise InvalidCube("corner", CORNER_NAMES[i])
        c1, c2 = cols[(ori + 1) % 3], cols[(ori + 2) % 3]
        j = next((j for j in range(8) if CORNER_COLOR[j][1] == c1 and CORNER_COLOR[j][2] == c2
                  and CORNER_COLOR[j][0] == cols[ori]), None)
        if j is None:
            raise InvalidCube("corner", CORNER_NAMES[i])
        cc.cp[i], cc.co[i] = j, ori
    for i in range(12):
        a, b = f[EDGE_FACELET[i][0]], f[EDGE_FACELET[i][1]]
        for j in range(12):
            if EDGE_COLOR[j] == (a, b):
                cc.ep[i], cc.eo[i] = j, 0
                break
            if EDGE_COLOR[j] == (b, a):
                cc.ep[i], cc.eo[i] = j, 1
                break
        else:
            raise InvalidCube("edge", EDGE_NAMES[i])
    return cc


def check_facelets(state):
    """Причина, по которой раскраску 3×3 нельзя собрать, или None. Центры могут стоять
    повёрнутыми (после M, x и т.п.) - кубик сначала поворачивается целиком."""
    try:
        s, _ = normalize(state)
        from_facelets(s).check()
    except InvalidCube as e:
        return e.reason
    return None


def _move_cubes():
    """Ходы граней как CubieCube - из геометрии модели N×N (а не набитые руками)."""
    pz = C.Puzzle(3)
    out = []
    for m in pz.face_moves():
        out.append(from_facelets(pz.apply(pz.solved(), m)))
    return out


MOVE_CUBES = _move_cubes()            # 18: U1 U2 U3 R1 ... B3
MOVE_NAMES = tuple(f + s for f in C.FACES for s in ("", "2", "'"))


def random_cube(rng=None):
    """Равномерно случайный возможный кубик: случайные перестановки и повороты, затем
    чётность и сумма поворотов подгоняются (так получается ровно каждая возможная позиция
    с равной вероятностью)."""
    rng = rng or random.Random()
    cp = list(range(8))
    ep = list(range(12))
    rng.shuffle(cp)
    rng.shuffle(ep)
    if parity(cp) != parity(ep):
        ep[0], ep[1] = ep[1], ep[0]
    co = [rng.randrange(3) for _ in range(7)]
    co.append((-sum(co)) % 3)
    eo = [rng.randrange(2) for _ in range(11)]
    eo.append(sum(eo) % 2)
    return CubieCube(cp, co, ep, eo)
