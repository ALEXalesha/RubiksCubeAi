"""Законы модели куба: ход 4 раза и ход с обратным - тождество, известные порядки,
коммутаторы, запись ходов, совпадение с таблицами Kociemba, проверка возможности."""
import random

import numpy as np
import pytest
from hypothesis import given, strategies as st

from game.core import cube as C
from game.core import cubie as K

P2, P3, P4 = C.Puzzle(2), C.Puzzle(3), C.Puzzle(4)


def all_single_moves(pz):
    out = []
    for axis in range(3):
        for layer in range(pz.n):
            for q in (1, 2, 3):
                out.append(C.Move(axis, (layer,), q))
    return out


def order(pz, seq):
    p = pz.sequence_perm(seq)
    cur = p.copy()
    k = 1
    ident = np.arange(pz.size)
    while not (cur == ident).all():
        cur = cur[p]
        k += 1
        assert k < 5000
    return k


@pytest.mark.parametrize("pz", [P2, P3, P4], ids=["2x2", "3x3", "4x4"])
def test_every_layer_move_is_a_permutation_of_order_dividing_four(pz):
    ident = np.arange(pz.size)
    for m in all_single_moves(pz):
        p = pz.perm(m)
        assert sorted(p) == list(range(pz.size))
        assert not (p == ident).all()
        s = pz.solved()
        assert (pz.apply(s, [m] * 4) == s).all()
        assert (pz.apply(s, [m, m.inverse()]) == s).all()
        if m.q == 2:
            assert (pz.apply(s, [m, m]) == s).all()
        else:
            assert order(pz, [m]) == 4


@pytest.mark.parametrize("pz", [P2, P3, P4], ids=["2x2", "3x3", "4x4"])
def test_quarter_turn_moves_exactly_one_layer(pz):
    """Четверть оборота слоя двигает 4N наклеек по бокам и, если слой крайний, ещё грань."""
    n = pz.n
    for m in all_single_moves(pz):
        moved = int((pz.perm(m) != np.arange(pz.size)).sum())
        outer = m.layers[0] in (0, n - 1)
        face = n * n - (1 if n % 2 else 0)
        assert moved == 4 * n + (face if outer else 0)


def test_known_orders_3x3():
    assert order(P3, "R U") == 105
    assert order(P3, "U R") == 105
    assert order(P3, "R U R' U'") == 6
    assert order(P3, "R2 U2") == 6
    assert order(P3, "R U2 D' B D'") == 1260           # самый большой порядок на 3×3
    assert order(P3, "R") == 4
    assert order(P3, "M") == 4


def test_commutators():
    s = P3.solved()
    for a, b in (("R", "L"), ("U", "D"), ("F", "B"), ("R", "M"), ("U", "E"), ("x", "R")):
        assert (P3.apply(s, f"{a} {b}") == P3.apply(s, f"{b} {a}")).all(), (a, b)
    # R и U не переставимы, а их коммутатор задевает только наклейки слоёв R и U
    comm = P3.sequence_perm("R U R' U'")
    moved = np.nonzero(comm != np.arange(54))[0]
    assert len(moved) > 0
    for i in moved:
        assert P3.cubie[i][0] == 2 or P3.cubie[i][1] == 2


def test_face_turn_directions():
    """U по часовой (смотрим сверху): передний ряд уходит налево."""
    s = P3.apply(P3.solved(), "U")
    assert list(s[36:39]) == [C.F] * 3            # верх L теперь зелёный (из F)
    assert list(s[18:21]) == [C.R] * 3            # верх F - из R
    s = P3.apply(P3.solved(), "R")
    assert [s[i] for i in (2, 5, 8)] == [C.F] * 3  # правый столбец U - из F
    s = P3.apply(P3.solved(), "F")
    assert [s[i] for i in (6, 7, 8)] == [C.L] * 3  # нижний ряд U - из L


@pytest.mark.parametrize("pz,names", [
    (P2, ["U", "R2", "F'", "D", "L'", "B2", "x", "y'", "z2"]),
    (P3, ["U", "R'", "F2", "M", "E'", "S2", "x'", "y", "z", "Rw", "Lw'", "D"]),
    (P4, ["U", "Rw", "Lw2", "2R", "2L'", "3Rw", "x", "Fw'", "2U2"]),
], ids=["2x2", "3x3", "4x4"])
def test_notation_round_trip(pz, names):
    for name in names:
        assert pz.name(pz.parse(name)) == name, name


def test_notation_equivalences():
    s = P3.solved()
    assert (P3.apply(s, "Rw") == P3.apply(s, "L x")).all()
    assert (P3.apply(s, "M") == P3.apply(s, "L' R x'")).all()
    assert (P3.apply(s, "U2") == P3.apply(s, "U U")).all()
    assert (P3.apply(s, "U'") == P3.apply(s, "U U U")).all()
    assert (P3.apply(s, "r") == P3.apply(s, "Rw")).all()
    assert (P4.apply(P4.solved(), "Rw") == P4.apply(P4.solved(), "R 2R")).all()


@pytest.mark.parametrize("bad", ["", "Q", "R3x", "M2w", "Uww"])
def test_bad_notation(bad):
    with pytest.raises(C.NotationError):
        P3.parse(bad)


def test_no_middle_layer_on_even_cube():
    with pytest.raises(C.NotationError):
        P4.parse("M")
    with pytest.raises(C.NotationError):
        P2.parse("Rw")


@given(st.integers(0, 10 ** 9), st.sampled_from([2, 3, 4]))
def test_scramble_then_inverse_solves(seed, n):
    pz = C.Puzzle(n)
    moves = pz.random_moves(25, random.Random(seed))
    s = pz.apply(pz.solved(), moves)
    assert pz.is_solved(pz.apply(s, C.inverse(moves)))
    assert np.bincount(s, minlength=6).tolist() == [n * n] * 6


@given(st.lists(st.sampled_from(["U", "U'", "U2", "R", "R'", "L", "L2", "D", "M"]), max_size=30))
def test_simplify_keeps_the_effect(names):
    moves = P3.moves(names)
    short = C.simplify(moves)
    assert len(short) <= len(moves)
    assert (P3.apply(P3.solved(), moves) == P3.apply(P3.solved(), short)).all()
    for a, b in zip(short, short[1:]):
        assert not (a.axis == b.axis and a.layers == b.layers)


def test_facelet_tables_match_geometry():
    for corner in K.CORNER_FACELET:
        assert len({tuple(P3.cubie[i]) for i in corner}) == 1
        assert len({tuple(P3.normal[i]) for i in corner}) == 3
    for edge in K.EDGE_FACELET:
        assert len({tuple(P3.cubie[i]) for i in edge}) == 1
    for f, c in enumerate(K.CENTERS):
        assert tuple(P3.normal[c]) == C.NORMAL[f] and tuple(P3.cubie[c]) == tuple(2 * v for v in C.NORMAL[f])


def test_move_cubes_match_kociemba():
    """Ходы, выведенные из геометрии, совпадают с таблицами Kociemba (cpU, cpR, ...)."""
    mU, mR, mF = K.MOVE_CUBES[0], K.MOVE_CUBES[3], K.MOVE_CUBES[6]
    assert mU.cp == [3, 0, 1, 2, 4, 5, 6, 7] and mU.co == [0] * 8
    assert mU.ep == [3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11] and mU.eo == [0] * 12
    assert mR.cp == [4, 1, 2, 0, 7, 5, 6, 3] and mR.co == [2, 0, 0, 1, 1, 0, 0, 2]
    assert mR.ep == [8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0] and mR.eo == [0] * 12
    assert mF.cp == [1, 5, 2, 3, 0, 4, 6, 7] and mF.co == [1, 2, 0, 0, 2, 1, 0, 0]
    assert mF.ep == [0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11]
    assert mF.eo == [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]


@given(st.lists(st.integers(0, 17), max_size=40))
def test_cubie_multiplication_matches_facelets(seq):
    s = P3.solved()
    cc = K.CubieCube()
    for m in seq:
        s = P3.apply(s, K.MOVE_NAMES[m])
        cc = cc * K.MOVE_CUBES[m]
    assert K.from_facelets(s) == cc
    assert (cc.to_facelets() == s).all()
    cc.check()
    assert (cc * cc.inverse()).is_solved()


@given(st.integers(0, 10 ** 9))
def test_random_cube_is_valid_and_round_trips(seed):
    cc = K.random_cube(random.Random(seed))
    cc.check()
    assert K.from_facelets(cc.to_facelets()) == cc
    assert K.check_facelets(cc.to_facelets()) is None


# --- невозможные кубики: каждый вид ловится и называется ---


def scrambled(seed=1):
    return P3.apply(P3.solved(), P3.random_moves(30, random.Random(seed)))


def test_twisted_corner_is_caught():
    s = scrambled()
    a, b, c = K.CORNER_FACELET[0]
    s[a], s[b], s[c] = s[b], s[c], s[a]
    assert K.check_facelets(s) == "twist"


def test_flipped_edge_is_caught():
    s = scrambled()
    a, b = K.EDGE_FACELET[5]
    s[a], s[b] = s[b], s[a]
    assert K.check_facelets(s) == "flip"


def test_two_swapped_edges_are_caught():
    s = scrambled()
    (a1, b1), (a2, b2) = K.EDGE_FACELET[0], K.EDGE_FACELET[7]
    s[a1], s[a2] = s[a2], s[a1]
    s[b1], s[b2] = s[b2], s[b1]
    assert K.check_facelets(s) == "parity"


def test_two_swapped_corners_are_caught():
    s = scrambled(2)
    for x, y in zip(K.CORNER_FACELET[2], K.CORNER_FACELET[6]):
        s[x], s[y] = s[y], s[x]
    assert K.check_facelets(s) in ("parity", "twist")


def test_impossible_corner_colors_are_caught():
    s = P3.solved()
    a, b, c = K.CORNER_FACELET[0]        # URF: белый, красный, зелёный
    o = K.CORNER_FACELET[6][0]           # наклейка D угла DBL
    s[b], s[o] = s[o], s[b]              # угол с двумя U/D цветами (белый и жёлтый)
    assert K.check_facelets(s) in ("corner", "edge", "duplicate")


def test_impossible_edge_colors_are_caught():
    s = P3.solved()
    a, b = K.EDGE_FACELET[0]             # UR
    c, d = K.EDGE_FACELET[4]             # DR
    s[b], s[c] = s[c], s[b]              # ребро «белое-жёлтое» не существует
    assert K.check_facelets(s) == "edge"


def test_duplicate_piece_is_caught():
    s = P3.solved()
    # два ребра UF: в UR ставим цвета UF, в UF - цвета UR (счёт цветов цел, но кусок
    # UF стоит дважды, а UR пропал)
    (a1, b1), (a2, b2) = K.EDGE_FACELET[0], K.EDGE_FACELET[1]
    s[b1] = C.F
    s[b2] = C.F
    # счёт поправить: одну зелёную наклейку в другом месте сделать красной
    s[K.EDGE_FACELET[8][0]] = C.R
    reason = K.check_facelets(s)
    assert reason in ("duplicate", "edge")


def test_wrong_color_count_and_centers_are_caught():
    s = P3.solved()
    s[0] = C.R
    assert K.check_facelets(s) == "count"
    s = P3.solved()
    s[K.CENTERS[0]], s[K.CENTERS[1]] = C.R, C.R
    assert K.check_facelets(s) == "centers"


@given(st.integers(0, 10 ** 9))
def test_any_reachable_state_is_valid_even_with_slices_and_rotations(seed):
    rng = random.Random(seed)
    names = [rng.choice(["U", "R", "F", "D", "L", "B", "M", "E", "S", "x", "y", "z", "Rw", "Uw'"])
             for _ in range(30)]
    assert K.check_facelets(P3.apply(P3.solved(), names)) is None


@given(st.integers(0, 10 ** 9), st.sampled_from(K.ROTATIONS))
def test_normalize_and_map_moves_solve_rotated_cube(seed, rot):
    moves = P3.random_moves(20, random.Random(seed))
    s = P3.apply(P3.solved(), moves)
    t = P3.apply(s, rot) if rot else s
    norm, rot2 = K.normalize(t)
    assert [int(norm[c]) for c in K.CENTERS] == list(range(6))
    solution = K.map_moves(C.inverse(moves), rot2)
    assert P3.is_solved(P3.apply(t, solution))


def test_24_rotations():
    assert len(K.ROTATIONS) == 24


def test_simplify_cancels_and_merges():
    assert C.simplify(P3.moves("R R'")) == []
    assert P3.names(C.simplify(P3.moves("R L R'"))) == ["L"]
    assert C.simplify(P3.moves("U2 U2")) == []
    assert P3.names(C.simplify(P3.moves("U U F F'"))) == ["U2"]


def test_solved_means_every_sticker_matches():
    assert not P3.is_solved(P3.apply(P3.solved(), "M2"))     # углы на месте, остальное нет
    assert not P4.is_solved(P4.apply(P4.solved(), "2R2"))
    assert P3.is_solved(P3.apply(P3.solved(), "x y"))        # повёрнутый целиком - собран
    assert P4.is_solved(P4.apply(P4.solved(), "z'"))


def test_corner_with_white_and_yellow_swapped_is_caught():
    s = P3.solved()
    a = K.CORNER_FACELET[0][0]       # белая наклейка URF
    b = K.CORNER_FACELET[4][0]       # жёлтая наклейка DFR
    s[a], s[b] = s[b], s[a]
    assert K.check_facelets(s) == "corner"
