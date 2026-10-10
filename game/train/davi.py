"""Обучение сети итерацией ценности (DAVI, как в DeepCubeA), без решателя-учителя.

Шаг обучения:
1. пачка позиций: собранный кубик, перемешанный на k случайных ходов, k от 0 до kmax;
2. цель для каждой: 0 у собранной, иначе 1 + min по ходам оценки соседа целевой сетью
   (собранный сосед - 0);
3. шаг Adam по средней квадратичной ошибке.
Целевая сеть - копия обучаемой, обновляется раз в sync шагов.

Снимок (веса обеих сетей, Adam, номер шага, журнал оценки) пишется атомарно каждые
snapshot_every шагов и в конце; прерванное обучение продолжается с него.
"""
import os
import time

import numpy as np

from aikit.nn.losses import mse
from aikit.nn.optim import Adam
from aikit.trainlog import TrainLog, load_snapshot, save_snapshot
from game.ai.net import ValueNet, move_perms
from game.core import cube as C


def scrambled_batch(n, perms, rng, size, kmax, kmin=0):
    """(позиции, k): позиции перемешаны от собранной на k случайных ходов."""
    pz = C.Puzzle(n)
    states = np.tile(pz.solved(), (size, 1))
    k = rng.integers(kmin, kmax + 1, size)
    for step in range(kmax):
        live = np.flatnonzero(k > step)
        if not live.size:
            break
        m = rng.integers(0, len(perms), live.size)
        states[live] = np.take_along_axis(states[live], perms[m], 1)
    return states, k


def is_solved_batch(states):
    faces = states.reshape(len(states), 6, -1)
    return (faces == faces[:, :, :1]).all((1, 2))


def targets(target_net, states, perms):
    """1 + min по ходам оценки соседа; у собранной позиции - 0."""
    b = len(states)
    children = states[:, perms].reshape(b * len(perms), -1)
    h = target_net.predict(children)
    h[is_solved_batch(children)] = 0.0
    y = 1.0 + h.reshape(b, len(perms)).min(1)
    y[is_solved_batch(states)] = 0.0
    return y.astype(np.float32)


class Trainer:
    def __init__(self, run, n=3, hidden=(512, 512, 256), batch=1000, kmax=None, lr=1e-3, sync=500,
                 seed=0, log=True):
        self.run = run
        self.n = n
        self.batch = batch
        self.kmax = kmax or {2: 14, 3: 30}.get(n, 30)
        self.sync = sync
        self.net = ValueNet(n, hidden, seed)
        self.target = ValueNet(n, hidden, seed)
        self.target.copy_from(self.net)
        self.opt = Adam(self.net.params(), lr=lr)
        self.perms = move_perms(n)
        self.rng = np.random.default_rng(seed + 1)
        self.step_no = 0
        self.history = []
        self.train_seconds = 0.0
        self.log = TrainLog(os.path.join(run, "train.log"), echo=log) if run else (lambda text: None)

    @property
    def snapshot_path(self):
        return os.path.join(self.run, "snapshot.npz")

    def step(self):
        t0 = time.perf_counter()
        states, _ = scrambled_batch(self.n, self.perms, self.rng, self.batch, self.kmax)
        y = targets(self.target, states, self.perms)
        pred = self.net.model.forward(self.net.encode(states))[:, 0]
        loss, grad = mse(pred, y)
        for _, _, g in self.net.params():
            g[...] = 0
        self.net.model.backward(grad[:, None])
        self.opt.step()
        self.step_no += 1
        if self.step_no % self.sync == 0:
            self.target.copy_from(self.net)
        self.train_seconds += time.perf_counter() - t0
        return loss

    def evaluate(self, per_k=200, ks=None, seed=123):
        """Средняя оценка сети по глубине перемешивания и доля, которую жадный спуск по
        оценке собирает не длиннее k ходов (быстрая проверка, не паспорт)."""
        rng = np.random.default_rng(seed)
        ks = ks or sorted({1, 2, 3, 5, 7, 10, 15, 20, self.kmax})
        out = {}
        for k in ks:
            if k > self.kmax:
                continue
            states, _ = scrambled_batch(self.n, self.perms, rng, per_k, k, kmin=k)
            pred = self.net.predict(states)
            solved = greedy_solve_rate(self.net, states, self.perms, k + 4)
            out[k] = {"mean": round(float(pred.mean()), 2), "greedy": round(float(solved), 3)}
        return out

    def train(self, steps, eval_every=500, snapshot_every=200):
        losses = []
        started = time.perf_counter()
        for _ in range(steps):
            losses.append(self.step())
            if self.step_no % 50 == 0:
                rate = (time.perf_counter() - started) / len(losses)
                self.log(f"шаг {self.step_no}: потеря {np.mean(losses[-50:]):.3f}, {rate:.2f} с/шаг")
            if eval_every and self.step_no % eval_every == 0:
                ev = self.evaluate()
                self.history.append({"step": self.step_no, "eval": ev})
                self.log(f"оценка на шаге {self.step_no}: " +
                         ", ".join(f"k={k}: ~{v['mean']} жадно {v['greedy']:.0%}" for k, v in ev.items()))
            if snapshot_every and self.step_no % snapshot_every == 0:
                self.save_snapshot()
        self.save_snapshot()
        return losses

    def save_snapshot(self):
        if not self.run:
            return
        arrays = {f"net.{k}": v for k, v in self.net.arrays().items()}
        arrays.update({f"target.{k}": v for k, v in self.target.arrays().items()})
        for i, (m, v) in enumerate(zip(self.opt.m, self.opt.v)):
            arrays[f"adam.m.{i}"] = m
            arrays[f"adam.v.{i}"] = v
        meta = {"n": self.n, "hidden": list(self.net.hidden), "step": self.step_no, "adam_t": self.opt.t,
                "batch": self.batch, "kmax": self.kmax, "sync": self.sync, "history": self.history,
                "train_seconds": round(self.train_seconds, 1),
                "rng": self.rng.bit_generator.state}
        save_snapshot(self.snapshot_path, arrays, meta)

    def load_snapshot(self):
        got = load_snapshot(self.snapshot_path)
        if got is None:
            return False
        arrays, meta = got
        self.net.load_arrays({k[4:]: v for k, v in arrays.items() if k.startswith("net.")})
        self.target.load_arrays({k[7:]: v for k, v in arrays.items() if k.startswith("target.")})
        for i in range(len(self.opt.m)):
            self.opt.m[i][...] = arrays[f"adam.m.{i}"]
            self.opt.v[i][...] = arrays[f"adam.v.{i}"]
        self.opt.t = meta["adam_t"]
        self.step_no = meta["step"]
        self.history = meta.get("history", [])
        self.train_seconds = meta.get("train_seconds", 0.0)
        if "rng" in meta:
            self.rng.bit_generator.state = meta["rng"]
        return True


def greedy_solve_rate(net, states, perms, limit):
    """Доля позиций, которые собирает жадный спуск (ход к соседу с наименьшей оценкой)."""
    cur = states.copy()
    done = is_solved_batch(cur)
    for _ in range(limit):
        live = np.flatnonzero(~done)
        if not live.size:
            break
        children = cur[live][:, perms]
        b, m, s = children.shape
        flat = children.reshape(b * m, s)
        h = net.predict(flat)
        h[is_solved_batch(flat)] = -1.0
        best = h.reshape(b, m).argmin(1)
        cur[live] = children[np.arange(b), best]
        done[live] = is_solved_batch(cur[live])
    return done.mean()
