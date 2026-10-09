"""Кубик Рубика, который собирает ИИ.

    python main.py                         # игра
    python main.py --offscreen             # без экрана (проверки)
    python main.py --selftest итог.txt     # самопроверка без экрана, итог в файл
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Кубик Рубика и ИИ")
    ap.add_argument("--selftest", metavar="ФАЙЛ", nargs="?", const="selftest.txt",
                    help="без окна собрать кубики через окно и записать итог в файл")
    ap.add_argument("--offscreen", action="store_true", help="без экрана")
    args = ap.parse_args(argv)
    if args.selftest:
        from game.selftest import selftest
        return selftest(args.selftest)
    if args.offscreen:
        from aikit import offscreen
        offscreen.setup(force=True)
    import aikit
    from game import APP_NAME, GAME_INFO, create_widget
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    icon = None
    try:
        from game.ui.icon import app_icon
        icon = app_icon()
    except ImportError:
        pass
    return aikit.run_standalone(GAME_INFO, create_widget, APP_NAME, icon=icon)


if __name__ == "__main__":
    sys.exit(main())
