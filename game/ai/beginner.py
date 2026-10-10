"""Послойный метод для начинающих - написанными правилами, для режима «Учусь».

Кубик мысленно переворачивается белым вниз (x2) - так учат почти везде: белый крест и
белые углы снизу, затем средний слой, затем жёлтая сторона сверху. Шаги:

1. белый крест - по одному ребру, короткий перебор (интуитивный шаг, ходов мало);
2. белые углы - угол над своим местом, «R U R' U'», пока не встанет;
3. средний слой - ребро сверху, алгоритм вправо «U R U' R' U' F' U F» или влево
   «U' L' U L U F U' F'»;
4. жёлтый крест - «F R U R' U' F'»;
5. жёлтые рёбра по местам - «R U R' U R U2 R' U»;
6. жёлтые углы по местам - «U R U' L' U R' U' L»;
7. повернуть жёлтые углы - «R' D' R D» дважды, пока угол не встанет, потом U к следующему
   (низ на время перепутается и вернётся сам после последнего угла).

Алгоритмы записаны для правого переднего места; для другого места кубик мысленно
поворачивается (y), и ходы переводятся в настоящие грани. Каждый шаг - запись ходов для
кубика как он лежит в окне и цвета кубиков, которые ставятся (их окно подсвечивает).
"""
import numpy as np

from game.ai.pocket import map_face_moves
from game.core import cube as C
from game.core import cubie as K

P3 = C.Puzzle(3)
IDX = {name: i for i, name in enumerate(K.MOVE_NAMES)}
Y_ROTS = ("", "y", "y2", "y'")
STEPS = ("cross", "corners", "middle", "ycross", "yedges", "ycorners", "yorient")
INSERT_CORNER = "R U R' U'"
POP_CORNER = "R U R'"
RIGHT = "U R U' R' U' F' U F"
LEFT = "U' L' U L U F U' F'"
YCROSS = "F R U R' U' F'"
YEDGES = "R U R' U R U2 R' U"
YCORNERS = "U R U' L' U R' U' L"
TWIST = "R' D' R D"


def apply(cc, names):
    for n in names:
        cc = cc * K.MOVE_CUBES[IDX[n]]
    return cc


def framed(alg, rot):
    """Алгоритм, записанный для кубика после поворота rot, - в настоящих гранях."""
    return [P3.name(m) for m in map_face_moves(P3, alg.split(), rot)]


def _slot_frames():
    """Для каждого места среднего и нижнего слоя - поворот y, в котором оно правое переднее."""
    want = {0: ("R", "F"), 1: ("F", "L"), 2: ("L", "B"), 3: ("B", "R")}
    out = {}
    for k, faces in want.items():
        out[k] = next(r for r in Y_ROTS if tuple(framed("R F", r)) == faces)
    return out


SLOT_FRAME = _slot_frames()           # 0 - FR/DFR, 1 - FL/DLF, 2 - BL/DBL, 3 - BR/DRB


def _edge_tables():
    """Где окажется ребро (место, поворот) после хода - для поиска креста."""
    nxt = {}
    for p in range(12):
        for o in range(2):
            row = []
            for m in K.MOVE_CUBES:
                p2 = m.ep.index(p)
                row.append((p2, (o + m.eo[p2]) % 2))
            nxt[(p, o)] = row
    return nxt


EDGE_NEXT = _edge_tables()


def _edge_dist(target):
    dist = {(target, 0): 0}
    frontier = [(target, 0)]
    while frontier:
        new = []
        for s in frontier:
            for t in EDGE_NEXT[s]:
                if t not in dist:
                    dist[t] = dist[s] + 1
                    new.append(t)
        frontier = new
    # ход обратим, поэтому расстояние «от цели» = расстояние «до цели»
    return dist


EDGE_DIST = {t: _edge_dist(t) for t in range(12)}


def _edge_state(cc, e):
    p = cc.ep.index(e)
    return (p, cc.eo[p])


def cross_edge(cc, edges):
    """Кратчайшие ходы, ставящие рёбра edges на места (поиск IDA* по их положениям)."""
    start = tuple(_edge_state(cc, e) for e in edges)

    def h(state):
        return max(EDGE_DIST[e][s] for e, s in zip(edges, state))

    path = []

    def dfs(state, togo, lastf):
        if togo == 0:
            return h(state) == 0
        for m in range(18):
            f = m // 3
            if f == lastf or f == lastf - 3:
                continue
            nxt = tuple(EDGE_NEXT[s][m] for s in state)
            if h(nxt) >= togo:
                continue
            path.append(m)
            if dfs(nxt, togo - 1, f):
                return True
            path.pop()
        return False

    for depth in range(h(start), 12):
        if dfs(start, depth, -9):
            return [K.MOVE_NAMES[m] for m in path]
    raise AssertionError("крест не найден")


def _solved_edge(cc, e):
    return cc.ep[e] == e and cc.eo[e] == 0


def _solved_corner(cc, c):
    return cc.cp[c] == c and cc.co[c] == 0


class Plan:
    """Шаги для кубика в окне: [(шаг, кусок или None, ходы)]."""

    def __init__(self, steps):
        self.steps = steps

    @property
    def moves(self):
        return [m for _, _, ms in self.steps for m in ms]


def white_down(state):
    """Поворот всего кубика (запись из x y z), после которого белый центр внизу."""
    _, rot1 = K.normalize(state)
    return " ".join(x for x in (rot1, "x2") if x)


def plan(state, as_is=False):
    """План сборки послойным методом для развёртки 3×3 как она лежит в окне. as_is -
    кубик уже повёрнут белым вниз (окно повернуло его само), ходы - в его же гранях."""
    if as_is:
        if K.check_facelets(state):
            raise K.InvalidCube(K.check_facelets(state))
        work, rot1 = np.asarray(state), None
    else:
        norm, rot1 = K.normalize(state)
        work = P3.apply(norm, "x2")
    relabel = np.zeros(6, np.int64)
    original = np.zeros(6, np.int64)
    for f in range(6):
        relabel[int(work[K.CENTERS[f]])] = f
        original[f] = int(work[K.CENTERS[f]])
    cc = K.from_facelets(relabel[work])
    cc.check()
    rotation = "" if as_is else " ".join(x for x in (rot1, "x2") if x)
    raw = []                                   # (шаг, кусок в рабочей системе, ходы)

    def emit(step, piece, names):
        nonlocal cc
        names = [n for n in names if n]
        cc = apply(cc, names)
        raw.append((step, piece, names))

    # 1. белый крест (рёбра DF DR DB DL - 5 4 7 6)
    done = []
    for e in (5, 4, 7, 6):
        done.append(e)
        if all(_solved_edge(cc, x) for x in done):
            continue
        emit("cross", ("edge", e), cross_edge(cc, done))

    # 2. белые углы (DFR DLF DBL DRB - 4 5 6 7)
    for c in (4, 5, 6, 7):
        if _solved_corner(cc, c):
            continue
        names = []
        pos = cc.cp.index(c)
        if pos >= 4:                                     # стоит внизу не так - поднять
            pop = framed(POP_CORNER, SLOT_FRAME[pos - 4])
            names += pop
            cc2 = apply(cc, pop)
        else:
            cc2 = cc
        above = c - 4
        k = next(k for k in range(4) if apply(cc2, ["U"] * k).cp[above] == c)
        names += ["U"] * k
        cc2 = apply(cc2, ["U"] * k)
        alg = framed(INSERT_CORNER, SLOT_FRAME[c - 4])
        for _ in range(6):
            if _solved_corner(cc2, c):
                break
            names += alg
            cc2 = apply(cc2, alg)
        emit("corners", ("corner", c), names)
    _check_layer(cc)

    # 3. средний слой (FR FL BL BR - 8 9 10 11)
    for e in (8, 9, 10, 11):
        if _solved_edge(cc, e):
            continue
        names = []
        pos = cc.ep.index(e)
        if pos >= 8:                                     # в среднем слое не так - вынуть
            names += framed(RIGHT, SLOT_FRAME[pos - 8])
        base = apply(cc, names)
        found = None
        for k in range(4):
            for rot in Y_ROTS:
                for alg in (RIGHT, LEFT):
                    seq = ["U"] * k + framed(alg, rot)
                    after = apply(base, seq)
                    if _solved_edge(after, e) and _kept(cc, after, e):
                        found = seq
                        break
                if found:
                    break
            if found:
                break
        if found is None:
            raise AssertionError("ребро среднего слоя не встало")
        emit("middle", ("edge", e), names + found)

    # 4-6. жёлтая сторона: перебор коротких цепочек «U^k + алгоритм»
    emit("ycross", None, _chain(cc, [["U"] * k + YCROSS.split() for k in range(4)],
                                lambda x: all(x.eo[i] == 0 for i in range(4))))
    seq = _chain(cc, [["U"] * k + YEDGES.split() for k in range(4)],
                 lambda x: any(all(_solved_edge(apply(x, ["U"] * j), i) for i in range(4)) for j in range(4)))
    after = apply(cc, seq)
    j = next(j for j in range(4) if all(_solved_edge(apply(after, ["U"] * j), i) for i in range(4)))
    emit("yedges", None, seq + ["U"] * j)
    emit("ycorners", None, _chain(cc, [framed(YCORNERS, r) for r in Y_ROTS],
                                  lambda x: all(x.cp[i] == i for i in range(4))))

    # 7. повернуть жёлтые углы: угол в URF, «R' D' R D» дважды, пока не встанет, затем U
    names = []
    cur = cc
    for _ in range(4):
        k = next((k for k in range(4) if apply(cur, ["U"] * k).co[0] != 0), None)
        if k is None:
            break
        names += ["U"] * k
        cur = apply(cur, ["U"] * k)
        while cur.co[0] != 0:                            # парами: пара поворачивает угол на треть
            names += TWIST.split() * 2
            cur = apply(cur, TWIST.split() * 2)
    k = next(k for k in range(4) if apply(cur, ["U"] * k).is_solved())
    emit("yorient", None, names + ["U"] * k)
    if not cc.is_solved():
        raise AssertionError("послойный метод не собрал кубик")

    steps = []
    for step, piece, names in raw:
        moves = C.simplify(map_face_moves(P3, names, rotation))
        colors = None
        if piece is not None:
            kind, j = piece
            table = K.EDGE_COLOR if kind == "edge" else K.CORNER_COLOR
            colors = tuple(int(original[c]) for c in table[j])
        steps.append((step, colors, [P3.name(m) for m in moves]))
    return Plan(steps)


def _check_layer(cc):
    if not (all(_solved_edge(cc, e) for e in (4, 5, 6, 7)) and all(_solved_corner(cc, c) for c in (4, 5, 6, 7))):
        raise AssertionError("первый слой не собран")


def _kept(before, after, new_edge):
    """Собранное раньше (первый слой и рёбра среднего слоя) не сломано."""
    _check_layer(after)
    return all(_solved_edge(after, e) for e in range(8, 12) if e != new_edge and _solved_edge(before, e))


def _chain(cc, macros, goal, depth=4):
    """Самая короткая цепочка макросов (в ширину), после которой goal и первые два слоя целы."""
    def fine(x):
        try:
            _check_layer(x)
        except AssertionError:
            return False
        return all(_solved_edge(x, e) for e in range(8, 12))

    if goal(cc):
        return []
    level = [([], cc)]
    for _ in range(depth):
        nxt = []
        for seq, x in level:
            for m in macros:
                y = apply(x, m)
                if fine(y) and goal(y):
                    return seq + m
                nxt.append((seq + m, y))
        level = nxt
    raise AssertionError("цепочка алгоритмов не найдена")


def piece_stickers(state, colors):
    """Наклейки кубика с такими цветами (в любом порядке) в развёртке state."""
    want = sorted(colors)
    state = np.asarray(state)
    by_cubie = {}
    for i in range(P3.size):
        by_cubie.setdefault(tuple(int(v) for v in P3.cubie[i]), []).append(i)
    for idx in by_cubie.values():
        if len(idx) == len(want) and sorted(int(state[i]) for i in idx) == want:
            return set(idx)
    return set()
