"""Кубик 2×2: расстояние до сборки для каждой из 3 674 160 позиций (перебор в ширину).

У 2×2 нет центров, поэтому угол DBL считается неподвижным: кубик сначала поворачивается
целиком так, чтобы угол с цветами D, B, L стоял на месте DBL правильно, а решается он
ходами U, R, F (они DBL не трогают). Позиция - перестановка 7 остальных углов (5040) и
повороты шести из них (729, седьмой следует из суммы).

Таблица расстояний даёт оптимальное решение: из каждой позиции есть ход, после которого
до сборки на один ход меньше. Наибольшее расстояние - 11 ходов (HTM).
"""
import os

import numpy as np

from game.ai.kociemba import all_perms, perm_rank
from game.core import cube as C
from game.core import cubie as K

N_PERM, N_TWIST = 5040, 729
N_STATES = N_PERM * N_TWIST
FREE = (0, 1, 2, 3, 4, 5, 7)            # углы кроме DBL (6)
MOVE_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8)  # U U2 U' R R2 R' F F2 F' среди 18 ходов граней
MOVE_NAMES = tuple(K.MOVE_NAMES[m] for m in MOVE_IDS)
TABLE_VERSION = 1
# сколько позиций на каждом расстоянии (HTM) - известное распределение 2×2
DISTRIBUTION = (1, 9, 54, 321, 1847, 9992, 50136, 227536, 870072, 1887748, 623800, 2644)

SLOT = np.full(8, -1, np.int64)
SLOT[list(FREE)] = np.arange(7)


def encode(cp, co):
    """Номер позиции по углам 3×3 (cp, co) с DBL на месте."""
    perm = [int(SLOT[cp[i]]) for i in FREE]
    twist = 0
    for i in FREE[:6]:
        twist = twist * 3 + co[i]
    return int(perm_rank(perm)) * N_TWIST + twist


def _move_tables():
    perms = all_perms(7)                               # по номеру
    twists = np.zeros((N_TWIST, 8), np.int64)
    rest = np.arange(N_TWIST)
    for i in reversed(FREE[:6]):
        twists[:, i] = rest % 3
        rest = rest // 3
    twists[:, 7] = (-twists[:, list(FREE[:6])].sum(1)) % 3
    pm = np.zeros((N_PERM, 9), np.int64)
    tm = np.zeros((N_TWIST, 9), np.int64)
    for k, mid in enumerate(MOVE_IDS):
        m = K.MOVE_CUBES[mid]
        # новый cp[i] = старый cp[m.cp[i]]: по местам FREE берётся место-источник
        src = np.array([SLOT[m.cp[i]] for i in FREE])
        pm[:, k] = perm_rank(perms[:, src])
        co = (twists[:, np.array(m.cp)] + np.array(m.co)) % 3
        t = np.zeros(N_TWIST, np.int64)
        for i in FREE[:6]:
            t = t * 3 + co[:, i]
        tm[:, k] = t
    return pm, tm


def build(progress=None):
    say = progress or (lambda frac, text: None)
    say(0.0, "2x2")
    pm, tm = _move_tables()
    dist = np.full(N_STATES, 255, np.uint8)
    dist[0] = 0
    depth = 0
    while True:
        frontier = np.flatnonzero(dist == depth)
        if not frontier.size:
            break
        for lo in range(0, frontier.size, 200_000):
            part = frontier[lo:lo + 200_000]
            p, t = part // N_TWIST, part % N_TWIST
            cand = (pm[p] * N_TWIST + tm[t]).ravel()
            dist[cand[dist[cand] == 255]] = depth + 1
        depth += 1
        say(min(0.99, depth / 12), "2x2")
    if (dist == 255).any():
        raise AssertionError("часть позиций 2×2 недостижима - таблица ходов неверна")
    say(1.0, "2x2")
    return Table(dist, pm, tm)


class Table:
    def __init__(self, dist, pm, tm):
        self.dist = dist
        self.pm = pm
        self.tm = tm

    def save(self, path):
        path = os.fspath(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, version=np.array(TABLE_VERSION), dist=self.dist,
                     pm=self.pm.astype(np.uint16), tm=self.tm.astype(np.uint16))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        try:
            with np.load(os.fspath(path)) as d:
                if int(d["version"]) != TABLE_VERSION:
                    return None
                dist, pm, tm = d["dist"], d["pm"].astype(np.int64), d["tm"].astype(np.int64)
        except (OSError, ValueError, KeyError, EOFError):
            return None
        if dist.shape != (N_STATES,) or pm.shape != (N_PERM, 9) or tm.shape != (N_TWIST, 9):
            return None
        return cls(dist, pm, tm)

    @classmethod
    def load_or_build(cls, path, progress=None):
        t = cls.load(path) if path else None
        if t is None:
            t = build(progress)
            if path:
                try:
                    t.save(path)
                except OSError:
                    pass
        return t

    def distance(self, code):
        return int(self.dist[code])

    def solve_code(self, code):
        """Оптимальное решение: номера ходов 0..8 (U U2 U' R R2 R' F F2 F')."""
        out = []
        d = int(self.dist[code])
        while d > 0:
            p, t = code // N_TWIST, code % N_TWIST
            for k in range(9):
                nxt = int(self.pm[p, k]) * N_TWIST + int(self.tm[t, k])
                if self.dist[nxt] == d - 1:
                    out.append(k)
                    code, d = nxt, d - 1
                    break
            else:
                raise AssertionError("таблица расстояний противоречива")
        return out

    def distance_of_state(self, state2):
        s, _ = normalize2(state2)
        return self.distance(encode(*to_corners(s)))

    def solve_state(self, state2):
        """Оптимальное решение записью ходов для развёртки 2×2 как она лежит."""
        s, rot = normalize2(state2)
        moves = [MOVE_NAMES[k] for k in self.solve_code(encode(*to_corners(s)))]
        return [P2.name(m) for m in map_face_moves(P2, moves, rot)]


# --- развёртка 2×2 <-> углы ---

P2 = C.Puzzle(2)
P3 = C.Puzzle(3)
# наклейка 2×2 (грань, ряд, столбец) - та же, что угловая наклейка 3×3 (грань, 2·ряд, 2·столбец)
TO3 = np.array([f * 9 + (2 * r) * 3 + 2 * c for f in range(6) for r in range(2) for c in range(2)])


def _as3(state2):
    s3 = P3.solved().copy()
    s3[TO3] = np.asarray(state2)
    return [int(v) for v in s3]


def to_corners(state2):
    """(cp, co) по развёртке 2×2 с DBL на месте. InvalidCube, если так не бывает."""
    f = _as3(state2)
    cc = K.CubieCube()
    for i in range(8):
        cols = [f[k] for k in K.CORNER_FACELET[i]]
        ori = next((o for o in range(3) if cols[o] in (C.U, C.D)), None)
        if ori is None:
            raise K.InvalidCube("corner", K.CORNER_NAMES[i])
        key = (cols[ori], cols[(ori + 1) % 3], cols[(ori + 2) % 3])
        j = next((j for j in range(8) if K.CORNER_COLOR[j] == key), None)
        if j is None:
            raise K.InvalidCube("corner", K.CORNER_NAMES[i])
        cc.cp[i], cc.co[i] = j, ori
    if sorted(cc.cp) != list(range(8)):
        raise K.InvalidCube("duplicate")
    if sum(cc.co) % 3:
        raise K.InvalidCube("twist")
    return cc.cp, cc.co


def normalize2(state2):
    """Повернуть 2×2 целиком так, чтобы угол D-B-L стоял на месте DBL правильно.
    Возвращает (развёртка, поворот) или InvalidCube."""
    state2 = np.asarray(state2)
    if np.bincount(state2, minlength=6).tolist() != [4] * 6:
        raise K.InvalidCube("count")
    for rot in K.ROTATIONS:
        s = P2.apply(state2, rot) if rot else state2
        f = _as3(s)
        if [f[k] for k in K.CORNER_FACELET[6]] == [C.D, C.B, C.L]:
            return s, rot
    raise K.InvalidCube("corner")


def check_state(state2):
    """Причина, по которой 2×2 нельзя собрать, или None."""
    try:
        to_corners(normalize2(state2)[0])
    except K.InvalidCube as e:
        return e.reason
    return None


def map_face_moves(pz, moves, rotation):
    """Ходы граней, найденные для кубика после поворота rotation, - в ходы исходного
    кубика (та же грань, что до поворота стояла на этом месте)."""
    if not rotation:
        return list(pz.moves(moves))
    rot = pz.moves(rotation)
    out = []
    n = pz.n
    for m in pz.moves(moves):
        if len(m.layers) != 1 or m.layers[0] not in (0, n - 1):
            raise ValueError("переводятся только ходы граней")
        sign = 1 if m.layers[0] == n - 1 else -1
        face = next(f for f in range(6) if C.AXIS_OF_FACE[f] == m.axis and C.SIGN_OF_FACE[f] == sign)
        normal = C.NORMAL[face]
        for r in reversed(rot):
            normal = C.rotate(normal, r.axis, -r.q)
        new_face = C.NORMAL.index(tuple(normal))
        turns = (-m.q) % 4 if C.SIGN_OF_FACE[face] > 0 else m.q % 4
        out.append(pz.face_move(new_face, 1, turns))
    return out
