"""Законы решателя Kociemba: таблицы ходов совпадают с умножением кубиков, таблицы
отсечения - точные расстояния (0 только в цели, соседи отличаются не больше чем на 1,
не больше числа ходов от цели), решения собирают кубик, средняя длина не больше 22.

Таблицы строятся заново один раз на прогон (так мутанты в построении ловятся), поиск
ограничен числом позиций, а не временем: итог не зависит от скорости машины."""
import random
import statistics
import threading
import time

import numpy as np
import pytest

from game.ai import kociemba as KO
from game.core import cube as C
from game.core import cubie as K

SAMPLE = 20
NODES = 60_000


@pytest.fixture(scope="session")
def tables(tmp_path_factory):
    path = tmp_path_factory.mktemp("koc") / "kociemba.npz"
    steps = []
    t = KO.Tables.load_or_build(str(path), progress=lambda f, s: steps.append(f))
    assert steps[0] == 0.0 and steps[-1] == 1.0 and steps == sorted(steps)
    assert path.exists()
    return t


@pytest.fixture(scope="session")
def solver(tables):
    return KO.Solver(tables)


def g1_cube(rng, length=25):
    cc = K.CubieCube()
    for _ in range(length):
        cc = cc * K.MOVE_CUBES[rng.choice(KO.P2_MOVES)]
    return cc


def test_coordinates_round_trip():
    co = KO.twist_decode(np.arange(KO.N_TWIST))
    assert (KO.twist_of(co) == np.arange(KO.N_TWIST)).all()
    assert (co.sum(1) % 3 == 0).all()
    eo = KO.flip_decode(np.arange(KO.N_FLIP))
    assert (KO.flip_of(eo) == np.arange(KO.N_FLIP)).all()
    assert (eo.sum(1) % 2 == 0).all()
    p = KO.all_perms(5)
    assert (KO.perm_rank(p) == np.arange(120)).all()
    assert len(KO.SLICE_MASKS) == KO.N_SLICE
    assert KO.slice_of(list(range(12))) == KO.SLICE_SOLVED


def test_move_tables_match_cubie_multiplication(tables, solver):
    rng = random.Random(3)
    for _ in range(60):
        cc = K.random_cube(rng)
        tw, fl, sl, cp = solver.coords(cc)
        for m in range(18):
            nx = cc * K.MOVE_CUBES[m]
            assert solver.coords(nx) == (tables.twist_move[tw, m], tables.flip_move[fl, m],
                                         tables.slice_move[sl, m], tables.cp_move[cp, m])
    for _ in range(60):
        cc = g1_cube(rng)
        ud = KO.perm_rank(cc.ep[:8])
        sp = KO.perm_rank([e - 8 for e in cc.ep[8:]])
        for k, m in enumerate(KO.P2_MOVES):
            nx = cc * K.MOVE_CUBES[m]
            assert KO.perm_rank(nx.ep[:8]) == tables.ud_move[ud, k]
            assert KO.perm_rank([e - 8 for e in nx.ep[8:]]) == tables.slicep_move[sp, k]


def test_pruning_tables_are_exact_distances(tables):
    """0 только в цели; у каждой позиции есть сосед на 1 ближе и нет соседа ближе на 2."""
    rng = np.random.default_rng(1)
    for prune, m1, m2, start, moves in (
            (tables.prune_twist_slice, tables.twist_move, tables.slice_move, KO.SLICE_SOLVED, list(range(18))),
            (tables.prune_flip_slice, tables.flip_move, tables.slice_move, KO.SLICE_SOLVED, list(range(18))),
            (tables.prune_cp_slicep, tables.cp_move[:, list(KO.P2_MOVES)], tables.slicep_move, 0, list(range(10))),
            (tables.prune_ud_slicep, tables.ud_move, tables.slicep_move, 0, list(range(10)))):
        n2 = m2.shape[0]
        assert prune[start] == 0 and (prune == 0).sum() == 1
        assert prune.max() < 20
        idx = rng.integers(0, prune.size, 3000)
        c1, c2 = idx // n2, idx % n2
        nb = prune[m1[c1][:, moves].astype(np.int64) * n2 + m2[c2][:, moves]]
        d = prune[idx].astype(int)[:, None]
        assert (np.abs(nb.astype(int) - d) <= 1).all()
        has_closer = (nb.astype(int) == d - 1).any(1)
        assert has_closer[prune[idx] > 0].all()


def test_pruning_never_overestimates(solver):
    rng = random.Random(7)
    for k in range(1, 9):
        for _ in range(20):
            cc = K.CubieCube()
            for _ in range(k):
                cc = cc * K.MOVE_CUBES[rng.randrange(18)]
            assert solver.phase1_estimate(cc) <= k


def test_trivial_cubes(solver):
    assert solver.solve(K.CubieCube()).moves == []
    r = solver.solve(K.MOVE_CUBES[3])                 # R
    assert r.names == ["R'"]
    cc = KO.apply_moves(K.CubieCube(), [0, 4, 7])     # U R2 F2: решение F2 R2 U'
    r = solver.solve(cc, node_limit=NODES)
    assert len(r) == 3 and KO.apply_moves(cc, r.moves).is_solved()


def test_random_cubes_are_solved_short(solver):
    rng = random.Random(2026)
    lengths = []
    for _ in range(SAMPLE):
        cc = K.random_cube(rng)
        r = solver.solve(cc, time_limit=1e9, node_limit=NODES)
        assert r is not None
        assert KO.apply_moves(cc, r.moves).is_solved()
        assert r.solutions == sorted(r.solutions, reverse=True) and r.solutions[-1] == len(r)
        for a, b in zip(r.moves, r.moves[1:]):
            assert a // 3 != b // 3                        # без двух ходов одной грани подряд
        lengths.append(len(r))
    assert statistics.mean(lengths) <= 22, lengths


def test_solution_works_on_facelets(solver):
    pz = C.Puzzle(3)
    moves = pz.random_moves(30, random.Random(9))
    s = pz.apply(pz.solved(), moves)
    r = solver.solve(K.from_facelets(s), node_limit=NODES)
    assert pz.is_solved(pz.apply(s, r.names))


def test_invalid_cube_is_refused(solver):
    cc = K.CubieCube()
    cc.co[0] = 1
    with pytest.raises(K.InvalidCube):
        solver.solve(cc)


def test_stop_event_cancels(solver):
    ev = threading.Event()
    ev.set()
    t0 = time.perf_counter()
    assert solver.solve(K.random_cube(random.Random(1)), stop=ev, time_limit=1e9, hard_limit=8) is None
    assert time.perf_counter() - t0 < 2.0


def test_broken_or_old_table_file_is_rebuilt(tmp_path, tables):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"not a table")
    assert KO.Tables.load(str(bad)) is None
    tables.save(str(tmp_path / "ok.npz"))
    again = KO.Tables.load(str(tmp_path / "ok.npz"))
    assert again is not None and (again.prune_flip_slice == tables.prune_flip_slice).all()
    data = dict(np.load(str(tmp_path / "ok.npz")))
    data["version"] = np.array(KO.TABLES_VERSION + 1)
    np.savez(str(tmp_path / "old.npz"), **data)
    assert KO.Tables.load(str(tmp_path / "old.npz")) is None
