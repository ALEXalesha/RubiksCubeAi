"""Кадры окна для README и списка игр (без экрана):

    python tools/make_previews.py [папка с таблицами]

docs/preview.png - 16:9 для списка игр; остальные - для README, в тёмной и светлой теме.
Таблицы ИИ берутся из папки (по умолчанию .cache) или строятся.
"""
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main(argv):
    from aikit import offscreen
    offscreen.setup(force=True)
    from PySide6.QtWidgets import QApplication

    from aikit.host import make_window
    from game import APP_NAME, GAME_INFO, create_widget
    from game.ai import kociemba as KO
    from game.ai import pocket as PK

    cache = argv[1] if len(argv) > 1 else os.path.join(ROOT, ".cache")
    app = QApplication.instance() or QApplication([])
    home = tempfile.mkdtemp(prefix="rubikscubeai-previews-")
    window, ctx = make_window(GAME_INFO, create_widget, APP_NAME, root=home)
    window.resize(1280, 720)
    window.show()
    g = window.game
    g.brain.kociemba = KO.Solver(KO.Tables.load_or_build(os.path.join(cache, "kociemba.npz")))
    g.brain.pocket = PK.Table.load_or_build(os.path.join(cache, "pocket.npz"))
    g.change("turn_ms", 0)

    def settle(sec=0.4):
        end = time.time() + sec
        while time.time() < end:
            app.processEvents()
            time.sleep(0.01)

    def shot(name):
        settle()
        window.grab().save(os.path.join(ROOT, "docs", name))
        print("docs/" + name)

    def solving(theme_name, name, size=3):
        ctx.set_theme(theme_name)
        g.set_size(size)
        g.change("show_thoughts", True)
        g.scramble(seed=2026)
        g.ai_solve()
        end = time.time() + 60
        while g.solution is None and time.time() < end:
            app.processEvents()
            time.sleep(0.01)
        g.pause()
        for _ in range(len(g.solution["moves"]) // 2):
            g.step_forward()
        g.change("turn_ms", 400)
        g.step_forward()
        settle(0.15)                         # кадр посреди поворота слоя
        g.view.timer.stop()
        shot(name)
        g.view.finish_all()
        g.change("turn_ms", 0)

    solving("dark", "preview.png")
    solving("light", "preview_light.png")
    ctx.set_theme("dark")
    g.set_mode("self")
    g.set_size(2)
    g.scramble(seed=4)
    g.hint()
    end = time.time() + 30
    while g.hint_move is None and time.time() < end:
        app.processEvents()
        time.sleep(0.01)
    shot("preview_2x2.png")
    g.set_size(4)
    g.scramble(seed=4)
    g.view.yaw, g.view.pitch = -28, 32
    shot("preview_4x4.png")
    g.shutdown()
    window.close()


if __name__ == "__main__":
    main(sys.argv)
