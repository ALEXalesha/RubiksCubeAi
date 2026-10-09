"""Самопроверка без экрана: python main.py --selftest итог.txt.

Нужна прежде всего собранному exe: он без консоли, поэтому итог пишется в файл. Окно
создаётся offscreen, данные - во временной папке (таблицы строятся там же заново),
рекорды игрока не трогаются. Проверяется: законы модели, таблицы, окно - ИИ собирает
перемешанные 3×3 и 2×2 через окно до конца.
"""
import os
import random
import shutil
import tempfile
import time
import traceback


def selftest(out_path):
    lines = []
    code = 0
    home = None
    try:
        from aikit import offscreen
        offscreen.setup(force=True)
        home = tempfile.mkdtemp(prefix="rubikscubeai-selftest-")
        os.environ["RUBIKSCUBEAI_HOME"] = home
        from PySide6.QtWidgets import QApplication

        import aikit
        from aikit.host import make_window
        from game import APP_NAME, GAME_INFO, create_widget
        from game.core import cube as C
        from game.version import VERSION

        lines.append(f"RubiksCubeAi {VERSION} самопроверка (aikit {aikit.__version__})")
        lines.append(f"собранная версия: {'да' if aikit.frozen() else 'нет'}")
        t0 = time.time()
        rng = random.Random(1)
        for n in (2, 3, 4):
            pz = C.Puzzle(n)
            for _ in range(20):
                moves = pz.random_moves(30, rng)
                s = pz.apply(pz.solved(), moves)
                if not pz.is_solved(pz.apply(s, C.inverse(moves))):
                    raise RuntimeError(f"{n}x{n}: ходы и обратные не вернули кубик")
        lines.append(f"модель: 60 перемешиваний 2×2, 3×3, 4×4 и обратно, {time.time() - t0:.2f} с")

        app = QApplication.instance() or QApplication([])
        window, ctx = make_window(GAME_INFO, create_widget, APP_NAME, root=home)
        window.show()
        g = window.game
        g.change("turn_ms", 0)
        for n in (3, 2):
            g.set_size(n)
            g.scramble(seed=7)
            t0 = time.time()
            g.ai_solve()
            end = time.time() + 240
            while g.solution is None and time.time() < end:
                app.processEvents()
                time.sleep(0.005)
            if g.solution is None:
                raise RuntimeError(f"{n}x{n}: ИИ не нашёл решения: {g.status.text()}")
            built = time.time() - t0
            total = len(g.solution["moves"])
            end = time.time() + 60
            while (g.solution["index"] < total or g.view.animating()) and time.time() < end:
                app.processEvents()
                time.sleep(0.002)
            if not g.view.pz.is_solved(g.view.state):
                raise RuntimeError(f"{n}x{n}: решение ИИ не собрало кубик")
            image = window.grab().toImage()
            if image.isNull() or image.width() < 300:
                raise RuntimeError("окно не нарисовалось")
            lines.append(f"{n}×{n}: ИИ собрал за {total} ходов, таблицы и поиск {built:.1f} с, "
                         f"кадр {image.width()}x{image.height()}")
        t0 = time.time()
        window.close()
        app.processEvents()
        lines.append(f"окно закрыто за {time.time() - t0:.2f} с")
        lines.append("ИТОГ: работает")
    except Exception:  # noqa: BLE001 - любая ошибка должна попасть в файл
        lines.append("ИТОГ: ошибка")
        lines.append(traceback.format_exc())
        code = 1
    if home:
        shutil.rmtree(home, ignore_errors=True)        # своя временная папка с таблицами
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return code
