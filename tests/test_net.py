"""Законы сети и обучения: вход - одна единица на наклейку, цель итерации ценности
считается по формуле, малое обучение на 2×2 учится собирать короткие перемешивания,
снимок продолжает обучение ровно с того же места, A* с сетью собирает кубик."""
import numpy as np
import pytest

from game.ai import astar
from game.ai import pocket as PK
from game.ai.net import ValueNet, move_perms, train_moves
from game.core import cube as C
from game.core import cubie as K
from game.train import davi


def test_encoding_is_one_hot_without_centers():
    net = ValueNet(3, (16,))
    pz = C.Puzzle(3)
    x = net.encode(pz.apply(pz.solved(), "R U")[None])
    assert x.shape == (1, 48 * 6)
    assert (x.reshape(48, 6).sum(1) == 1).all()
    net2 = ValueNet(2, (16,))
    assert net2.encode(C.Puzzle(2).solved()[None]).shape == (1, 24 * 6)


def test_2x2_training_moves_keep_dbl_in_place():
    pz = C.Puzzle(2)
    for m in train_moves(2):
        s = pz.apply(pz.solved(), [m])
        assert [int(v) for v in np.asarray(PK._as3(s))[list(K.CORNER_FACELET[6])]] == [C.D, C.B, C.L]


def test_scrambled_batch_depths_and_validity():
    rng = np.random.default_rng(0)
    perms = move_perms(3)
    states, k = davi.scrambled_batch(3, perms, rng, 200, 12)
    assert k.min() >= 0 and k.max() <= 12
    assert davi.is_solved_batch(states[k == 0]).all()
    for s in states[:30]:
        assert K.check_facelets(s) is None


class Const:
    """Сеть-заглушка: оценка = номер грани первой наклейки (для проверки формулы цели)."""
    n = 3

    def predict(self, states):
        return np.asarray(states)[:, 0].astype(np.float32) + 5.0


def test_targets_follow_the_bellman_formula():
    pz = C.Puzzle(3)
    perms = move_perms(3)
    s = np.stack([pz.solved(), pz.apply(pz.solved(), "R"), pz.apply(pz.solved(), "R U F")])
    y = davi.targets(Const(), s, perms)
    assert y[0] == 0.0
    assert y[1] == 1.0                                 # сосед R' собран - оценка 0
    children = s[2][perms]
    expect = 1 + min(0.0 if pz.is_solved(c) else float(c[0]) + 5.0 for c in children)
    assert y[2] == expect


@pytest.fixture(scope="module")
def tiny():
    t = davi.Trainer(None, n=2, hidden=(64,), batch=256, kmax=3, lr=3e-3, sync=25, seed=1, log=False)
    t.train(400, eval_every=0, snapshot_every=0)
    return t


def test_tiny_training_learns_short_scrambles(tiny):
    ev = tiny.evaluate(per_k=100, ks=[1, 2, 3])
    assert ev[1]["greedy"] == 1.0
    assert ev[2]["greedy"] >= 0.9
    assert ev[1]["mean"] < ev[2]["mean"] < ev[3]["mean"]


def test_astar_with_net_solves_and_respects_node_limit(tiny):
    pz = C.Puzzle(2)
    s = pz.apply(pz.solved(), "R U' F2")
    r = astar.solve(tiny.net, s, max_nodes=5000)
    assert r.solved and pz.is_solved(pz.apply(s, r.moves))
    assert r.depth == len(r.moves)
    assert r.nodes <= 5000 + 64
    hard = pz.apply(pz.solved(), "R U F' R2 U' F R' U2 F' R U'")
    r = astar.solve(tiny.net, hard, max_nodes=64, batch=16)
    assert r.nodes < 64 + 16            # перебор кончается на пачке, где превышен предел
    if r.solved:
        assert pz.is_solved(pz.apply(hard, r.moves))


class Oracle:
    """Точное расстояние 2×2 из таблицы вместо сети: A* обязан найти оптимум."""
    n = 2

    def __init__(self, table):
        self.table = table

    def predict(self, states):
        return np.array([self.table.distance(PK.encode(*PK.to_corners(s))) for s in states], np.float32)


def test_astar_with_exact_heuristic_is_optimal(pocket_table):
    pz = C.Puzzle(2)
    rng = np.random.default_rng(3)
    perms = move_perms(2)
    states, _ = davi.scrambled_batch(2, perms, rng, 10, 12, kmin=8)
    for s in states:
        r = astar.solve(Oracle(pocket_table), s, weight=1.0, batch=1)
        assert r.solved and len(r.moves) == pocket_table.distance_of_state(s) == r.depth


def test_snapshot_resume_continues_exactly(tmp_path):
    a = davi.Trainer(str(tmp_path / "a"), n=2, hidden=(32,), batch=64, kmax=4, sync=5, seed=2, log=False)
    for _ in range(12):
        a.step()
    a.save_snapshot()
    b = davi.Trainer(str(tmp_path / "a"), n=2, hidden=(32,), batch=64, kmax=4, sync=5, seed=9, log=False)
    assert b.load_snapshot() and b.step_no == 12
    for _ in range(8):
        a.step()
        b.step()
    for (_, p, _), (_, q, _) in zip(a.net.params(), b.net.params()):
        assert np.array_equal(p, q)


def test_save_and_load_net(tmp_path):
    net = ValueNet(3, (8, 4), seed=4)
    path = str(tmp_path / "m.npz")
    net.save(path, {"x": 1})
    back = ValueNet.load(path, 3)
    pz = C.Puzzle(3)
    s = np.stack([pz.solved(), pz.apply(pz.solved(), "R U")])
    assert np.allclose(net.predict(s), back.predict(s))
    assert back.hidden == (8, 4)
