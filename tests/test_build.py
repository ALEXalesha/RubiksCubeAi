"""Сборка: версия в двух местах совпадает, в сборку не попадает лишнее, данные - внутри."""
import os

import build
from game.version import VERSION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_version_same_in_installer_and_game():
    assert build.version() == VERSION


def test_no_pygame_no_torch_in_build_and_requirements():
    assert "pygame" in build.EXCLUDE and "torch" in build.EXCLUDE
    req = open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8").read().lower()
    assert "pygame" not in req and "torch" not in req


def test_models_and_aikit_version_go_into_build():
    src = open(os.path.join(ROOT, "build.py"), encoding="utf-8").read()
    assert "'game', 'models'" in src and "'aikit', 'VERSION'" in src


def test_trim_moves_files_instead_of_deleting(tmp_path):
    qt = tmp_path / "app" / "_internal" / "PySide6"
    (qt / "translations").mkdir(parents=True)
    (qt / "opengl32sw.dll").write_bytes(b"x" * 10)
    (qt / "translations" / "qt_de.qm").write_bytes(b"y" * 5)
    (qt / "translations" / "qt_ru.qm").write_bytes(b"z")
    keep = tmp_path / "trimmed"
    freed = build.trim(str(tmp_path / "app"), str(keep))
    assert freed == 15
    assert (keep / "_internal" / "PySide6" / "opengl32sw.dll").exists()
    assert (qt / "translations" / "qt_ru.qm").exists()


def test_installer_is_per_user_and_russian():
    nsi = open(os.path.join(ROOT, "installer.nsi"), encoding="utf-8").read()
    assert "RequestExecutionLevel user" in nsi and 'MUI_LANGUAGE "Russian"' in nsi
    assert '!define APP "RubiksCubeAi"' in nsi
