# Rubik's Cube AI

**A 3D Rubik's cube in Qt that you can solve yourself or hand over to the AI.**

[Русская версия](README.ru.md)

![The AI solving a 3×3 cube](docs/preview.png)

[![Status](https://img.shields.io/badge/status-in_development-orange)](#) [![CI](https://github.com/ALEXalesha/RubiksCubeAi/actions/workflows/ci.yml/badge.svg)](https://github.com/ALEXalesha/RubiksCubeAi/actions/workflows/ci.yml) [![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![PySide6](https://img.shields.io/badge/PySide6-41CD52?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython-6/) [![Built with Claude Code](https://img.shields.io/badge/Built_with-Claude_Code-D97757?logo=claude&logoColor=white)](https://claude.com/claude-code)

Code, tests and docs are written by Claude (Anthropic) in Claude Code. Ideas, testing and decisions - Alexey.

Work in progress (0.1.0). Python, PySide6 and numpy; no pygame.

## What works

- **2×2, 3×3 and 4×4 cubes** drawn in 3D with plain QPainter: the turning layer is animated, cubies are sorted by depth.
- **Mouse:** drag a sticker to turn its layer, drag the background to rotate the whole cube. The cursor is never captured.
- **Keys:** U D L R F B (Shift for the reverse turn), M E S, X Y Z; Ctrl+Z undoes a move. You can also type a sequence like `R U R' U'`.
- **"I solve" mode:** scramble, a clock that starts with your first move, undo, a hint that highlights the layer to turn, records per cube size.
- **"AI solves" mode:** the AI finds a solution in a background thread and plays it move by move: to start, back, pause, forward, turn speed.
- **"Learn" mode:** the beginner layer-by-layer method in 7 steps; the AI turns the cube white side down, explains each step in words and highlights the piece being placed.
- **"My own cube":** paint the stickers on an unfolded net; the program checks that such a cube can exist and says what is wrong if not.
- **AI thoughts:** a lower and an upper bound for the moves left, how many positions were searched, how the solution got shorter during the search.
- **Impossible cubes** are explained: a twisted corner, a flipped edge, two swapped pieces, wrong colors.
- Dark and light themes, Russian and English, the window remembers its place.

## The AI

| Cube | Method | Result |
|---|---|---|
| 3×3 | Kociemba two-phase algorithm: move and pruning tables in numpy (built once, about 20 s, then read from a file), IDA* search that keeps improving the solution | 1000 random cubes: all solved, 20.81 moves on average (at most 23), 1.37 s per cube |
| 2×2 | Breadth-first search over all 3 674 160 positions | Optimal solutions, 8.76 moves on average, at most 11 |
| 2×2 | Neural network that taught itself (DeepCubeA idea: value iteration, weighted A*), 0.46 h of training | 1000 random cubes: all solved, 8.95 moves on average, optimal in 82.1% of cases, 0.04 s |
| 3×3 | The same network, 1 hour of training (6000 steps) | Scrambled by 3-7 moves: 100%; by 10: 93%; by 13: 30%; by 16 and more: 0%. Random cubes: 0 of 1000 within 5000 positions. Much weaker than Kociemba so far; training continues |
| 3×3 | Layer-by-layer beginner method written as rules ("Learn" mode) | Any cube, about 160 moves, every step explained with the piece highlighted |
| 4×4 | No solver yet | You solve it yourself |

All numbers are in [game/models/passport.json](game/models/passport.json); tests recompute the first cubes with the same seeds.

| | |
|---|---|
| ![Light theme](docs/preview_light.png) | ![2×2 hint](docs/preview_2x2.png) |
| ![Learn mode](docs/preview_learn.png) | ![4×4](docs/preview_4x4.png) |

## Run

```
pip install -r requirements.txt
python main.py
```

Tests: `pip install -r requirements-dev.txt`, then `python -m pytest -q`. Mutation check: `python tools/mutate.py`.

## License

MIT, see [LICENSE](LICENSE).
