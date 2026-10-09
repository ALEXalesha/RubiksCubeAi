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
