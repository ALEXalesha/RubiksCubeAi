"""Взвешенный A* с оценкой сети, пачками (как BWAS в DeepCubeA).

f = weight·g + h: g - сколько ходов сделано, h - оценка сети «сколько осталось». Каждый
шаг из очереди берутся batch лучших позиций, все их соседи оцениваются сетью одним
вызовом. Поиск кончается, когда среди новых позиций появился собранный кубик, или когда
просмотрено max_nodes позиций (тогда решения нет - честно «не нашла»).
"""
import heapq
import time

import numpy as np

from game.ai.net import move_perms, train_moves
from game.core import cube as C


class SearchResult:
    def __init__(self, moves, nodes, seconds, estimate, depth=None):
        self.moves = moves              # Move или None, если не нашла
        self.depth = depth              # g собранной позиции - должно совпасть с длиной пути
        self.nodes = nodes
        self.seconds = seconds
        self.estimate = estimate        # оценка сети для начальной позиции

    @property
    def solved(self):
        return self.moves is not None


def solve(net, state, weight=0.6, batch=64, max_nodes=50_000, time_limit=None, stop=None):
    """Решение для развёртки state (3×3 - центры на месте, 2×2 - DBL на месте)."""
    n = net.n
    pz = C.Puzzle(n)
    moves = train_moves(n)
    perms = move_perms(n)
    t0 = time.perf_counter()
    start = np.asarray(state, np.uint8)
    h0 = float(net.predict(start[None])[0])
    if pz.is_solved(start):
        return SearchResult([], 0, 0.0, h0)
    key0 = start.tobytes()
    parent = {key0: (None, None)}
    g_of = {key0: 0}
    heap = [(h0, 0, key0)]
    counter = 1
    nodes = 0
    solved_key = None
    while heap and solved_key is None:
        if nodes >= max_nodes or (time_limit and time.perf_counter() - t0 > time_limit):
            break
        if stop is not None and stop.is_set():
            break
        picked = []
        while heap and len(picked) < batch:
            f, _, key = heapq.heappop(heap)
            picked.append(key)
        states = np.stack([np.frombuffer(k, np.uint8) for k in picked])
        nodes += len(picked)
        children = states[:, perms].reshape(-1, start.size)     # [позиция, ход] подряд
        keys = [c.tobytes() for c in children]
        fresh = []
        for j, k in enumerate(keys):
            src = picked[j // len(moves)]
            g = g_of[src] + 1
            if k in g_of and g_of[k] <= g:
                continue
            g_of[k] = g
            parent[k] = (src, j % len(moves))
            fresh.append(j)
        if not fresh:
            continue
        fresh_states = children[fresh]
        faces = fresh_states.reshape(len(fresh), 6, -1)
        solved_mask = (faces == faces[:, :, :1]).all((1, 2))
        if solved_mask.any():
            solved_key = keys[fresh[int(np.flatnonzero(solved_mask)[0])]]
            break
        h = net.predict(fresh_states)
        for j, hv in zip(fresh, h):
            k = keys[j]
            heapq.heappush(heap, (weight * g_of[k] + float(hv), counter, k))
            counter += 1
    seconds = time.perf_counter() - t0
    if solved_key is None:
        return SearchResult(None, nodes, seconds, h0)
    path = []
    k = solved_key
    while parent[k][0] is not None:
        k, mi = parent[k]
        path.append(moves[mi])
    path.reverse()
    return SearchResult(path, nodes, seconds, h0, g_of[solved_key])
