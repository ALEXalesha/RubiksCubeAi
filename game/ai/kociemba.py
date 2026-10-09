"""Двухфазный алгоритм Kociemba для 3×3.

Фаза 1 приводит кубик в подгруппу G1 = <U, D, R2, L2, F2, B2>: все углы и рёбра правильно
повёрнуты, а четыре ребра среднего слоя стоят в среднем слое. Фаза 2 собирает кубик
ходами только из G1. Обе фазы - поиск IDA* по координатам (номерам состояний) с таблицами
отсечения: таблица даёт точное число ходов до цели для пары координат, а максимум из
двух таблиц - нижнюю оценку для всего кубика.

Найдя первое решение, поиск продолжается: фаза 1 становится длиннее, фаза 2 короче, и
каждое новое решение короче прежнего. Остановка - по времени или когда короче не найти.

Таблицы ходов и отсечения считаются в numpy (векторно, по всем значениям координаты
сразу) и пишутся в файл .npz; при следующем запуске читаются из файла. Сам поиск - на
чистом Python: обращение к элементу array/bytes быстрее, чем к элементу numpy.
"""
from array import array
from itertools import permutations
import math
import os
import time

import numpy as np

from game.core import cubie as K

TABLES_VERSION = 1
N_TWIST, N_FLIP, N_SLICE, N_PERM8, N_PERM4 = 2187, 2048, 495, 40320, 24
N_MOVES = 18
# ходы фазы 2: U U2 U' R2 F2 D D2 D' L2 B2 (номера среди 18: грань*3 + четверти-1)
P2_MOVES = (0, 1, 2, 4, 7, 9, 10, 11, 13, 16)
P2_ONLY = frozenset(P2_MOVES)
# Фаза 2 не длиннее 12: длинные фазы 2 дороги в поиске, а решение с ними почти всегда
# короче находится через фазу 1 на ход-два длиннее (замер: 20 кубиков по 60 тыс. позиций -
# в среднем 22,85 хода без предела, 21,45 с пределом 12 за то же время).
P2_CAP = 12
MOVE_NAMES = K.MOVE_NAMES


# --- координаты (векторно: по строкам массива) ---


def twist_of(co):
    co = np.asarray(co)
    return (co[..., :7] * (3 ** np.arange(6, -1, -1))).sum(-1)


def twist_decode(t):
    t = np.asarray(t)
    co = np.zeros(t.shape + (8,), np.int64)
    rest = t.copy()
    for i in range(6, -1, -1):
        co[..., i] = rest % 3
        rest = rest // 3
    co[..., 7] = (-co[..., :7].sum(-1)) % 3
    return co


def flip_of(eo):
    eo = np.asarray(eo)
    return (eo[..., :11] * (2 ** np.arange(10, -1, -1))).sum(-1)


def flip_decode(f):
    f = np.asarray(f)
    eo = np.zeros(f.shape + (12,), np.int64)
    rest = f.copy()
    for i in range(10, -1, -1):
        eo[..., i] = rest % 2
        rest = rest // 2
    eo[..., 11] = eo[..., :11].sum(-1) % 2
    return eo


SLICE_MASKS = np.array([m for m in range(1 << 12) if bin(m).count("1") == 4], np.int64)
SLICE_INDEX = np.full(1 << 12, -1, np.int64)
SLICE_INDEX[SLICE_MASKS] = np.arange(len(SLICE_MASKS))
SLICE_SOLVED = int(SLICE_INDEX[0xF00])          # рёбра FR FL BL BR на местах 8..11


def slice_of(ep):
    """Где стоят четыре ребра среднего слоя (без порядка): номер из 495."""
    ep = np.asarray(ep)
    occ = (ep >= 8).astype(np.int64)
    return SLICE_INDEX[(occ << np.arange(12)).sum(-1)]


def slice_decode(s):
    m = SLICE_MASKS[np.asarray(s)]
    return ((m[..., None] >> np.arange(12)) & 1).astype(bool)


def perm_rank(p):
    """Номер перестановки (код Лемера) по строкам: (..., n) -> (...)."""
    p = np.asarray(p)
    n = p.shape[-1]
    less = (p[..., None, :] < p[..., :, None])          # [i, j]: p[j] < p[i]
    upper = np.triu(np.ones((n, n), bool), 1)            # только j > i
    counts = (less & upper).sum(-1)
    fact = np.array([math.factorial(n - 1 - i) for i in range(n)], np.int64)
    return (counts * fact).sum(-1)


def all_perms(n):
    """Все перестановки n в порядке их номеров perm_rank."""
    return np.array(list(permutations(range(n))), np.int64)


# --- таблицы ---


class Tables:
    """Таблицы ходов (twist, flip, slice, углы, рёбра U/D, рёбра среднего слоя) и
    отсечения (twist×slice, flip×slice - фаза 1; углы×слой, рёбра×слой - фаза 2)."""

    NAMES = ("twist_move", "flip_move", "slice_move", "cp_move", "ud_move", "slicep_move",
             "prune_twist_slice", "prune_flip_slice", "prune_cp_slicep", "prune_ud_slicep")

    def __init__(self, arrays):
        for k in self.NAMES:
            setattr(self, k, arrays[k])
        # для поиска - плоские array/bytes: доступ к элементу в разы быстрее numpy
        self.TM = array("H", self.twist_move.ravel().astype(np.uint16).tobytes())
        self.FM = array("H", self.flip_move.ravel().astype(np.uint16).tobytes())
        self.SM = array("H", self.slice_move.ravel().astype(np.uint16).tobytes())
        self.CM = array("H", self.cp_move.ravel().astype(np.uint16).tobytes())
        self.UM = array("H", self.ud_move.ravel().astype(np.uint16).tobytes())
        self.PM = array("H", self.slicep_move.ravel().astype(np.uint16).tobytes())
        self.P_TS = self.prune_twist_slice.astype(np.uint8).tobytes()
        self.P_FS = self.prune_flip_slice.astype(np.uint8).tobytes()
        self.P_CS = self.prune_cp_slicep.astype(np.uint8).tobytes()
        self.P_US = self.prune_ud_slicep.astype(np.uint8).tobytes()

    @classmethod
    def build(cls, progress=None):
        """Посчитать все таблицы. progress(доля 0..1, подпись) - для окна."""
        say = progress or (lambda frac, text: None)
        mc = K.MOVE_CUBES
        a = {}
        say(0.0, "twist")
        co = twist_decode(np.arange(N_TWIST))
        a["twist_move"] = np.stack([twist_of((co[:, m.cp] + m.co) % 3) for m in mc], 1)
        eo = flip_decode(np.arange(N_FLIP))
        a["flip_move"] = np.stack([flip_of((eo[:, m.ep] + m.eo) % 2) for m in mc], 1)
        occ = slice_decode(np.arange(N_SLICE))
        a["slice_move"] = np.stack([SLICE_INDEX[(occ[:, m.ep].astype(np.int64) << np.arange(12)).sum(-1)]
                                    for m in mc], 1)
        say(0.1, "permutations")
        p8 = all_perms(8)
        a["cp_move"] = np.stack([perm_rank(p8[:, m.cp]) for m in mc], 1)
        a["ud_move"] = np.stack([perm_rank(p8[:, np.array(mc[k].ep[:8])]) for k in P2_MOVES], 1)
        p4 = all_perms(4)
        a["slicep_move"] = np.stack([perm_rank(p4[:, np.array(mc[k].ep[8:]) - 8]) for k in P2_MOVES], 1)
        say(0.25, "phase 1")
        allm = np.arange(N_MOVES)
        a["prune_twist_slice"] = bfs(a["twist_move"], a["slice_move"], SLICE_SOLVED, allm)
        say(0.45, "phase 1")
        a["prune_flip_slice"] = bfs(a["flip_move"], a["slice_move"], SLICE_SOLVED, allm)
        say(0.65, "phase 2")
        p2 = np.arange(len(P2_MOVES))
        a["prune_cp_slicep"] = bfs(a["cp_move"][:, list(P2_MOVES)], a["slicep_move"], 0, p2)
        say(0.85, "phase 2")
        a["prune_ud_slicep"] = bfs(a["ud_move"], a["slicep_move"], 0, p2)
        say(1.0, "done")
        return cls(a)

    def save(self, path):
        path = os.fspath(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, version=np.array(TABLES_VERSION),
                     **{k: getattr(self, k).astype(np.uint16 if k.endswith("move") else np.uint8)
                        for k in self.NAMES})
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        """Таблицы из файла или None (нет файла, битый, старая версия)."""
        try:
            with np.load(os.fspath(path)) as data:
                if int(data["version"]) != TABLES_VERSION:
                    return None
                arrays = {k: data[k] for k in cls.NAMES}
        except (OSError, ValueError, KeyError, EOFError):
            return None
        expect = {"twist_move": (N_TWIST, 18), "flip_move": (N_FLIP, 18), "slice_move": (N_SLICE, 18),
                  "cp_move": (N_PERM8, 18), "ud_move": (N_PERM8, 10), "slicep_move": (N_PERM4, 10),
                  "prune_twist_slice": (N_TWIST * N_SLICE,), "prune_flip_slice": (N_FLIP * N_SLICE,),
                  "prune_cp_slicep": (N_PERM8 * N_PERM4,), "prune_ud_slicep": (N_PERM8 * N_PERM4,)}
        if any(arrays[k].shape != s for k, s in expect.items()):
            return None
        return cls(arrays)

    @classmethod
    def load_or_build(cls, path, progress=None):
        t = cls.load(path) if path else None
        if t is None:
            t = cls.build(progress)
            if path:
                try:
                    t.save(path)
                except OSError:
                    pass
        return t


def bfs(move1, move2, start2, moves):
    """Расстояния до цели (первая координата 0, вторая start2) для всех пар координат.
    move1[c, k], move2[c, k] - куда переходит координата от k-го хода из moves."""
    n2 = move2.shape[0]
    total = move1.shape[0] * n2
    dist = np.full(total, 255, np.uint8)
    dist[start2] = 0
    depth = 0
    while True:
        frontier = np.flatnonzero(dist == depth)
        if not frontier.size:
            break
        for lo in range(0, frontier.size, 100_000):
            part = frontier[lo:lo + 100_000]
            c1, c2 = part // n2, part % n2
            cand = (move1[c1][:, moves].astype(np.int64) * n2 + move2[c2][:, moves]).ravel()
            dist[cand[dist[cand] == 255]] = depth + 1
        depth += 1
    if (dist == 255).any():
        raise AssertionError("часть состояний недостижима - таблица ходов неверна")
    return dist


# --- поиск ---


class Result:
    """Решение и что было по пути: для «Мыслей ИИ» и паспорта."""

    def __init__(self, moves, phase1, nodes, seconds, solutions):
        self.moves = moves                # номера ходов 0..17
        self.phase1 = phase1              # сколько из них - фаза 1
        self.nodes = nodes                # сколько позиций просмотрено
        self.seconds = seconds
        self.solutions = solutions        # длины найденных по пути решений, по порядку

    @property
    def names(self):
        return [MOVE_NAMES[m] for m in self.moves]

    def __len__(self):
        return len(self.moves)


class _Stop(Exception):
    pass


class Solver:
    def __init__(self, tables):
        self.t = tables

    def coords(self, cc):
        return (int(twist_of(cc.co)), int(flip_of(cc.eo)), int(slice_of(cc.ep)), int(perm_rank(cc.cp)))

    def phase1_estimate(self, cc):
        tw, fl, sl, _ = self.coords(cc)
        return max(self.t.P_TS[tw * N_SLICE + sl], self.t.P_FS[fl * N_SLICE + sl])

    def solve(self, cc, time_limit=1.0, max_length=30, hard_limit=60.0, stop=None, target=0,
              node_limit=None):
        """Решение для CubieCube (центры на месте). Ищет до time_limit секунд, улучшая
        найденное; без решения ищет дальше, но не дольше hard_limit. node_limit - то же
        ограничение, но в просмотренных позициях: с ним итог не зависит от скорости
        машины (так считают тесты). target - хватит решения такой длины или короче.
        stop - threading.Event для отмены. None - не найдено (или отменено)."""
        cc.check()
        t = self.t
        TM, FM, SM, CM, UM, PM = t.TM, t.FM, t.SM, t.CM, t.UM, t.PM
        PTS, PFS, PCS, PUS = t.P_TS, t.P_FS, t.P_CS, t.P_US
        mcubes = K.MOVE_CUBES
        t0 = time.perf_counter()
        soft_end = t0 + time_limit
        hard_end = t0 + hard_limit
        best = [None]
        found = []
        nodes = [0]
        next_check = [256]
        path = []
        p2path = []
        tw0, fl0, sl0, cp0 = self.coords(cc)
        if cc.is_solved():
            return Result([], 0, 0, 0.0, [0])

        def check_time():
            now = time.perf_counter()
            if stop is not None and stop.is_set():
                raise _Stop
            if now > hard_end or (best[0] is not None and now > soft_end):
                raise _Stop
            if node_limit is not None and best[0] is not None and nodes[0] > node_limit:
                raise _Stop

        def search2(cp, ud, sp, togo, lastf):
            if togo == 0:
                return cp == 0 and ud == 0 and sp == 0
            nodes[0] += 1
            for k in range(10):
                m = P2_MOVES[k]
                f = m // 3
                if f == lastf or f == lastf - 3:
                    continue
                cp2 = CM[cp * 18 + m]
                ud2 = UM[ud * 10 + k]
                sp2 = PM[sp * 10 + k]
                if PCS[cp2 * 24 + sp2] >= togo or PUS[ud2 * 24 + sp2] >= togo:
                    continue
                p2path.append(m)
                if search2(cp2, ud2, sp2, togo - 1, f):
                    return True
                p2path.pop()
            return False

        def phase2(cp):
            depth1 = len(path)
            limit = min((len(best[0]) - 1 if best[0] is not None else max_length) - depth1, P2_CAP)
            if limit < 0:
                return
            # рёбра U/D и среднего слоя после фазы 1: ход за ходом по списку из 12
            ep = cc.ep
            for m in path:
                mep = mcubes[m].ep
                ep = [ep[mep[i]] for i in range(12)]
            ud = int(perm_rank(ep[:8]))
            sp = int(perm_rank([e - 8 for e in ep[8:]]))
            h = max(PCS[cp * 24 + sp], PUS[ud * 24 + sp])
            lastf = path[-1] // 3 if path else -9
            for d2 in range(h, limit + 1):
                del p2path[:]
                if search2(cp, ud, sp, d2, lastf):
                    best[0] = list(path) + list(p2path)
                    found.append(len(best[0]))
                    return
                check_time()

        def search1(tw, fl, sl, cp, togo, lastf):
            if togo == 0:
                if path and path[-1] in P2_ONLY:
                    return              # без последнего хода G1 был бы достигнут раньше
                phase2(cp)
                if best[0] is not None and len(best[0]) <= target:
                    raise _Stop
                return
            nodes[0] += 1
            if nodes[0] >= next_check[0]:
                next_check[0] = nodes[0] + 256
                check_time()
            for m in range(18):
                f = m // 3
                if f == lastf or f == lastf - 3:
                    continue
                tw2 = TM[tw * 18 + m]
                fl2 = FM[fl * 18 + m]
                sl2 = SM[sl * 18 + m]
                h = PTS[tw2 * N_SLICE + sl2]
                if h >= togo:
                    continue
                h2 = PFS[fl2 * N_SLICE + sl2]
                if h2 >= togo:
                    continue
                path.append(m)
                search1(tw2, fl2, sl2, CM[cp * 18 + m], togo - 1, f)
                path.pop()

        h1 = max(PTS[tw0 * N_SLICE + sl0], PFS[fl0 * N_SLICE + sl0])
        try:
            for depth1 in range(h1, max_length + 1):
                if best[0] is not None and depth1 >= len(best[0]):
                    break
                if depth1 == 0:
                    phase2(cp0)
                else:
                    search1(tw0, fl0, sl0, cp0, depth1, -9)
                check_time()
        except _Stop:
            if stop is not None and stop.is_set():
                return None
        if best[0] is None:
            return None
        moves = best[0]
        return Result(moves, _phase1_len(moves), nodes[0], time.perf_counter() - t0, found)


def _phase1_len(moves):
    """Длина фазы 1 - до последнего хода не из G1 включительно (для подписи в окне)."""
    last = 0
    for i, m in enumerate(moves):
        if m not in P2_ONLY:
            last = i + 1
    return last


def apply_moves(cc, moves):
    for m in moves:
        cc = cc * K.MOVE_CUBES[m]
    return cc
