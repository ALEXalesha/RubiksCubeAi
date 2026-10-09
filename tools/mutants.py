# Мутанты для tools/mutate.py: (имя, файл, было, стало, проверки).
M = "game/core/cube.py"
K = "game/core/cubie.py"
TC = "tests/test_cube.py"
MUTANTS = [
    ("cube: поворот вокруг x в другую сторону", M, "x, y, z = x, -z, y", "x, y, z = x, z, -y", TC),
    ("cube: грань крутится не туда", M, "q = (-turns) % 4 if sign > 0 else turns % 4", "q = turns % 4", TC),
    ("cube: слой не тот", M, "coord = 2 * layer - (self.n - 1)", "coord = 2 * layer - self.n + 3", TC),
    ("cube: штрих как простой ход", M, "turns, t = 3, t[:-1]", "turns, t = 1, t[:-1]", TC),
    ("cube: двойной слой - один", M, "depths = range(depth) if wide else (depth - 1,)",
     "depths = (depth - 1,)", TC),
    ("cube: склейка не сокращает", M, "                if q:\n", "                if True:\n", TC),
    ("cube: собран, если совпали углы", M, "return bool((faces == faces[:, :1]).all())",
     "return bool((faces[:, -1:] == faces[:, :1]).all())", TC),
    ("cubie: чётность не проверяется", K, "if parity(self.cp) != parity(self.ep):", "if False:", TC),
    ("cubie: повёрнутый угол не ловится", K, "if sum(self.co) % 3:", "if False:", TC),
    ("cubie: перевёрнутое ребро не ловится", K, "if sum(self.eo) % 2:", "if False:", TC),
    ("cubie: ориентация ребра теряется", K, "cc.ep[i], cc.eo[i] = j, 1", "cc.ep[i], cc.eo[i] = j, 0", TC),
    ("cubie: умножение углов без поворота хода", K,
     "co = [(self.co[other.cp[i]] + other.co[i]) % 3 for i in range(8)]",
     "co = [self.co[other.cp[i]] for i in range(8)]", TC),
    ("cubie: случайный без чётности", K, "ep[0], ep[1] = ep[1], ep[0]", "pass", TC),
    ("cubie: случайный без суммы поворотов", K, "co.append((-sum(co)) % 3)", "co.append(0)", TC),
    ("cubie: поворот обратно не туда", K, "normal = C.rotate(normal, r.axis, -r.q)",
     "normal = C.rotate(normal, r.axis, r.q)", TC),
    ("cubie: два цвета U/D в углу", K, "and CORNER_COLOR[j][0] == cols[ori]), None)", "), None)", TC),
]
KO = "game/ai/kociemba.py"
TK = "tests/test_kociemba.py"
MUTANTS += [
    ("kociemba: ходы фазы 2 с R вместо R2", KO, "P2_MOVES = (0, 1, 2, 4, 7, 9, 10, 11, 13, 16)",
     "P2_MOVES = (0, 1, 2, 3, 7, 9, 10, 11, 13, 16)", TK),
    ("kociemba: цель среднего слоя не та", KO, "SLICE_INDEX[0xF00]", "SLICE_INDEX[0x0F0]", TK),
    ("kociemba: расстояния через одно", KO, "dist[cand[dist[cand] == 255]] = depth + 1",
     "dist[cand[dist[cand] == 255]] = depth + 2", TK),
    ("kociemba: поворот восьмого угла", KO, "co[..., 7] = (-co[..., :7].sum(-1)) % 3", "co[..., 7] = 0", TK),
    ("kociemba: рёбра после фазы 1 не так", KO, "ep = [ep[mep[i]] for i in range(12)]",
     "ep = [mep[ep[i]] for i in range(12)]", TK),
    # «фаза 2 забывает средний слой» (return cp == 0 and ud == 0) - равносильный мутант:
    # отсечение точное, на последнем ходе пропускает только позицию с ud = sp = 0.
    ("kociemba: углы фазы 2 не тем ходом", KO, "cp2 = CM[cp * 18 + m]", "cp2 = CM[cp * 18 + k]", TK),
    ("kociemba: отсечение строже, чем можно", KO, "                if h >= togo:\n",
     "                if h >= togo - 1:\n", TK),
    ("kociemba: старая версия таблиц читается", KO, 'if int(data["version"]) != TABLES_VERSION:',
     "if False:", TK),
    ("kociemba: отмена не слушается", KO,
     "            if stop is not None and stop.is_set():\n                raise _Stop",
     "            if False:\n                raise _Stop", TK),
    ("kociemba: улучшение не короче", KO,
     "limit = min((len(best[0]) - 1 if best[0] is not None", "limit = min((len(best[0]) + 1 if best[0] is not None", TK),
]
PK = "game/ai/pocket.py"
TP = "tests/test_pocket.py"
MUTANTS += [
    ("pocket: расстояния через одно", PK, "dist[cand[dist[cand] == 255]] = depth + 1",
     "dist[cand[dist[cand] == 255]] = depth + 2", TP),
    ("pocket: повороты углов не в номере", PK, "twist = twist * 3 + co[i]", "twist = twist * 3", TP),
    ("pocket: неподвижный угол зеркальный", PK, "== [C.D, C.B, C.L]", "== [C.D, C.L, C.B]", TP),
    ("pocket: поворот обратно не туда", PK, "normal = C.rotate(normal, r.axis, -r.q)",
     "normal = C.rotate(normal, r.axis, r.q)", TP),
    ("pocket: решение не по убыванию", PK, "if self.dist[nxt] == d - 1:", "if self.dist[nxt] <= d:", TP),
    ("pocket: повёрнутый угол не ловится", PK, "    if sum(cc.co) % 3:\n        raise K.InvalidCube(\"twist\")",
     "    if False:\n        raise K.InvalidCube(\"twist\")", TP),
]
V = "game/ui/cubeview.py"
W = "game/ui/widget.py"
TU = "tests/test_ui.py"
MUTANTS += [
    ("view: слой крутится против мыши", V, "q = 1 if r[rot_axis] > 0 else 3", "q = 3 if r[rot_axis] > 0 else 1", TU),
    ("view: видны задние грани", V, "return area < 0", "return area > 0", TU),
    ("view: тянуть за наклейку - вращать вид", V, '                    self.drag_mode = "turn"\n                    self.userTurn.emit(move)',
     '                    pass', TU),
    ("view: Shift не обращает ход", V, "            name += \"'\"", "            pass", TU),
    ("widget: отмена повторяет ход", W, 'self.view.turn(move.inverse(), "undo")', 'self.view.turn(move, "undo")', TU),
    ("widget: ход человека не сбрасывает решение ИИ", W,
     "        if self.solution is not None:\n            self.drop_solution()\n        self.view.turn(move, \"user\")",
     "        self.view.turn(move, \"user\")", TU),
    ("widget: рекорд не пишется", W, 'save_json(self.ctx.file("records.json"), self.records)', "pass", TU),
    ("widget: назад - вперёд", W, 'self.view.turn(move.inverse(), "back")', 'self.view.turn(move, "back")', TU),
    ("widget: невозможный кубик отдаётся ИИ", W, "        if why:\n            self.set_status", "        if False:\n            self.set_status", TU),
    ("widget: закрытие не ждёт потоки", W, "return self.ai.shutdown(2000)", "return True", TU),
]
