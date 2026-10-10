"""Сеть оценки на numpy по идее DeepCubeA: сколько ходов осталось до сборки.

Вход - цвет каждой наклейки, кроме центров (они на месте), в одну единицу из шести:
3×3 - 48·6 = 288 чисел, 2×2 - 24·6 = 144 (2×2 сначала поворачивается так, чтобы угол
DBL стоял на месте, - как у точного решателя). Выход - одно число, «ходов осталось».

Сеть учится сама, без решателя: итерация ценности (DAVI). Берутся позиции, перемешанные
от собранной на k случайных ходов, и для каждой цель = 1 + min по ходам оценки соседа
(у собранного - 0); соседей оценивает «целевая» копия сети, которая обновляется реже.
Так оценка расходится от собранного кубика всё дальше, как волна.
"""
import json
import os

import numpy as np

from aikit.nn import io
from aikit.nn.layers import Dense, ReLU, Sequential
from game.core import cube as C

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def model_path(n):
    return os.path.join(MODEL_DIR, f"cube{n}-value.npz")


def passport_path(n):
    return os.path.join(MODEL_DIR, f"cube{n}-value.json")


def train_moves(n):
    """Ходы, которыми сеть видит мир: 3×3 - 18 ходов граней, 2×2 - U R F (DBL неподвижен)."""
    pz = C.Puzzle(n)
    if n == 2:
        return [pz.face_move(f, 1, t) for f in (C.U, C.R, C.F) for t in (1, 2, 3)]
    return pz.face_moves()


def move_perms(n):
    pz = C.Puzzle(n)
    return np.stack([pz.perm(m) for m in train_moves(n)]).astype(np.int64)


def input_stickers(n):
    """Номера наклеек, которые идут на вход (у 3×3 без центров)."""
    pz = C.Puzzle(n)
    if n % 2 == 1:
        centers = {f * n * n + (n * n) // 2 for f in range(6)}
        return np.array([i for i in range(pz.size) if i not in centers])
    return np.arange(pz.size)


class ValueNet:
    def __init__(self, n, hidden=(512, 512, 256), seed=0):
        self.n = n
        self.hidden = tuple(hidden)
        self.stickers = input_stickers(n)
        self.features = len(self.stickers) * 6
        rng = np.random.default_rng(seed)
        layers = []
        prev = self.features
        for h in self.hidden:
            layers += [Dense(prev, h, rng), ReLU()]
            prev = h
        layers.append(Dense(prev, 1, rng))
        self.model = Sequential(layers)

    def encode(self, states):
        """(B, 6N²) цвета -> (B, признаки) float32."""
        s = np.asarray(states)[:, self.stickers].astype(np.int64)
        x = np.zeros((s.shape[0], s.shape[1], 6), np.float32)
        np.put_along_axis(x, s[..., None], 1.0, axis=2)
        return x.reshape(s.shape[0], -1)

    def predict(self, states, chunk=4096):
        """Оценка «ходов осталось» для пачки состояний (не меньше 0)."""
        states = np.asarray(states)
        out = np.empty(len(states), np.float32)
        for lo in range(0, len(states), chunk):
            out[lo:lo + chunk] = self.model.forward(self.encode(states[lo:lo + chunk]))[:, 0]
        return np.maximum(out, 0.0)

    def params(self):
        return self.model.params()

    def arrays(self):
        return {name: p.copy() for name, p, _ in self.params()}

    def load_arrays(self, arrays):
        for name, p, _ in self.params():
            if arrays[name].shape != p.shape:
                raise ValueError(f"{name}: в файле {arrays[name].shape}, в сети {p.shape}")
            p[...] = arrays[name]

    def copy_from(self, other):
        for (_, p, _), (_, q, _) in zip(self.params(), other.params()):
            p[...] = q

    def save(self, path, meta=None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        arrays = self.arrays()
        arrays["__hidden__"] = np.array(self.hidden)
        io.save(path, arrays)
        if meta is not None:
            with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8", newline="\n") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path, n):
        arrays = io.load(path)
        net = cls(n, tuple(int(h) for h in arrays.pop("__hidden__")))
        net.load_arrays(arrays)
        return net


_DEFAULT = {}


def default_net(n):
    """Готовая сеть из game/models или None (ещё не обучена)."""
    if n not in _DEFAULT:
        path = model_path(n)
        try:
            _DEFAULT[n] = ValueNet.load(path, n) if os.path.exists(path) else None
        except (OSError, ValueError, KeyError):
            _DEFAULT[n] = None
    return _DEFAULT[n]
