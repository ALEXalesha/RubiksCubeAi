"""Законы 2×2: перебор находит все 3 674 160 позиций с известным распределением по
расстояниям, решения оптимальны и собирают кубик в любом повороте, невозможные ловятся."""
import random

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from game.ai import pocket as PK
from game.core import cube as C
from game.core import cubie as K

P2 = C.Puzzle(2)


@pytest.fixture(scope="session")
def table(pocket_table):
    return pocket_table


def test_distance_distribution_is_the_known_one(table):
    counts = np.bincount(table.dist)
    assert tuple(counts.tolist()) == PK.DISTRIBUTION
    assert counts.sum() == PK.N_STATES == 3_674_160


def test_move_tables_match_cubie_multiplication(table):
    rng = random.Random(4)
    for _ in range(200):
        cc = K.CubieCube()
        for _ in range(20):
            cc = cc * K.MOVE_CUBES[rng.choice(PK.MOVE_IDS)]
        code = PK.encode(cc.cp, cc.co)
        for k, mid in enumerate(PK.MOVE_IDS):
            nx = cc * K.MOVE_CUBES[mid]
            p, t = code // PK.N_TWIST, code % PK.N_TWIST
            assert PK.encode(nx.cp, nx.co) == table.pm[p, k] * PK.N_TWIST + table.tm[t, k]


@settings(max_examples=150)
@given(st.integers(0, 10 ** 9), st.sampled_from(K.ROTATIONS))
def test_optimal_solution_solves_any_orientation(table, seed, rot):
    rng = random.Random(seed)
    k = rng.randrange(0, 25)
    moves = P2.random_moves(k, rng)
    s = P2.apply(P2.solved(), moves)
    if rot:
        s = P2.apply(s, rot)
    sol = table.solve_state(s)
    assert P2.is_solved(P2.apply(s, sol))
    assert len(sol) == table.distance_of_state(s) <= min(11, k)


def test_distance_drops_by_at_most_one_per_move(table):
    rng = np.random.default_rng(5)
    idx = rng.integers(0, PK.N_STATES, 5000)
    p, t = idx // PK.N_TWIST, idx % PK.N_TWIST
    nb = table.dist[table.pm[p] * PK.N_TWIST + table.tm[t]].astype(int)
    d = table.dist[idx].astype(int)[:, None]
    assert (np.abs(nb - d) <= 1).all()
    assert (nb == d - 1).any(1)[d[:, 0] > 0].all()


def test_impossible_2x2_is_caught():
    s = P2.apply(P2.solved(), "R U F'")
    assert PK.check_state(s) is None
    t = s.copy()
    a, b, c = (int(i) for i in np.flatnonzero(PK.TO3 == K.CORNER_FACELET[0][0])), \
        (int(i) for i in np.flatnonzero(PK.TO3 == K.CORNER_FACELET[0][1])), \
        (int(i) for i in np.flatnonzero(PK.TO3 == K.CORNER_FACELET[0][2]))
    a, b, c = next(a), next(b), next(c)
    t[a], t[b], t[c] = t[b], t[c], t[a]               # повернуть один угол
    assert PK.check_state(t) == "twist"
    t = P2.solved().copy()
    t[0] = C.R
    assert PK.check_state(t) == "count"
    t = P2.solved().copy()
    a = next(int(i) for i in np.flatnonzero(PK.TO3 == K.CORNER_FACELET[0][0]))   # белая URF
    b = next(int(i) for i in np.flatnonzero(PK.TO3 == K.CORNER_FACELET[4][0]))   # жёлтая DFR
    t[a], t[b] = t[b], t[a]
    assert PK.check_state(t) == "corner"
