"""Решатели для окна: одна точка входа на все размеры кубика, без Qt.

Окно отдаёт развёртку как она лежит (центры 3×3 могут быть повёрнуты после M или x,
2×2 повёрнут как угодно) и получает запись ходов именно для этой развёртки.

- 3×3: Kociemba (таблицы в папке данных, при первом запуске строятся);
- 2×2: таблица расстояний всех позиций - оптимальное решение;
- 4×4: решателя нет (в spec он не требуется), окно об этом говорит.
"""
import os
import threading

from game.core import cube as C
from game.core import cubie as K


class Cancelled(Exception):
    pass


class Solution:
    """Запись ходов и «мысли»: оценка, сколько осталось, просмотренные позиции, время."""

    def __init__(self, moves, method, nodes=0, seconds=0.0, estimate=None, optimal=False, found=()):
        self.moves = list(moves)
        self.method = method          # "kociemba", "bfs"
        self.nodes = nodes
        self.seconds = seconds
        self.estimate = estimate      # нижняя оценка (3×3) или точное расстояние (2×2)
        self.optimal = optimal
        self.found = list(found)      # длины решений по ходу поиска

    def __len__(self):
        return len(self.moves)


class Brain:
    """Держит таблицы (загружает или строит один раз на процесс) и решает кубики."""

    _lock = threading.Lock()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.kociemba = None
        self.pocket = None

    def tables_path(self, name):
        return os.path.join(self.data_dir, "tables", name)

    def ready(self, n):
        return (n == 3 and self.kociemba is not None) or (n == 2 and self.pocket is not None)

    def supports(self, n):
        return n in (2, 3)

    def prepare(self, n, progress=None, stop=None):
        """Загрузить или построить таблицы для размера n. progress(доля, подпись)."""
        def say(frac, text):
            if stop is not None and stop.is_set():
                raise Cancelled
            if progress is not None:
                progress(frac, text)

        with self._lock:
            if n == 3 and self.kociemba is None:
                from game.ai import kociemba as KO
                tables = KO.Tables.load_or_build(self.tables_path(f"kociemba-v{KO.TABLES_VERSION}.npz"), say)
                self.kociemba = KO.Solver(tables)
            elif n == 2 and self.pocket is None:
                from game.ai import pocket as PK
                self.pocket = PK.Table.load_or_build(self.tables_path(f"pocket-v{PK.TABLE_VERSION}.npz"), say)

    @staticmethod
    def check(n, state):
        """Причина, по которой кубик нельзя собрать, или None."""
        if n == 3:
            return K.check_facelets(state)
        if n == 2:
            from game.ai import pocket as PK
            return PK.check_state(state)
        return None

    def solve(self, n, state, time_limit=1.0, stop=None, node_limit=None):
        """Solution для развёртки state или None (отменено). InvalidCube - кубик невозможен."""
        pz = C.Puzzle(n)
        if pz.is_solved(state):
            return Solution([], "none", optimal=True, estimate=0)
        self.prepare(n, stop=stop)
        if n == 2:
            from game.ai import pocket as PK
            moves = self.pocket.solve_state(state)
            return Solution(moves, "bfs", nodes=len(moves) * 9, estimate=len(moves), optimal=True,
                            found=[len(moves)])
        if n != 3:
            raise ValueError("для этого размера решателя нет")
        norm, rot = K.normalize(state)
        cc = K.from_facelets(norm)
        cc.check()
        estimate = int(self.kociemba.phase1_estimate(cc))
        r = self.kociemba.solve(cc, time_limit=time_limit, stop=stop, node_limit=node_limit)
        if r is None:
            if stop is not None and stop.is_set():
                return None
            raise RuntimeError("решение не найдено")
        moves = [pz.name(m) for m in K.map_moves(r.names, rot)]
        return Solution(moves, "kociemba", nodes=r.nodes, seconds=r.seconds, estimate=estimate,
                        found=r.solutions)
