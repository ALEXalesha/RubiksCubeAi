import os
import sys

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Без экрана: Qt рисует в память. Ставится до любого импорта Qt.
os.environ["QT_QPA_PLATFORM"] = "offscreen"
_fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
if os.path.isdir(_fonts):
    os.environ.setdefault("QT_QPA_FONTDIR", _fonts)

from hypothesis import settings  # noqa: E402

settings.register_profile("default", max_examples=400, deadline=None)
settings.register_profile("thorough", max_examples=4000, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def koc_tables(tmp_path_factory):
    """Таблицы Kociemba - один раз на прогон, заново (мутанты построения ловятся)."""
    from game.ai import kociemba as KO
    path = tmp_path_factory.mktemp("koc") / "kociemba.npz"
    steps = []
    t = KO.Tables.load_or_build(str(path), progress=lambda f, s: steps.append(f))
    assert steps[0] == 0.0 and steps[-1] == 1.0 and steps == sorted(steps)
    assert path.exists()
    return t


@pytest.fixture(scope="session")
def pocket_table(tmp_path_factory):
    from game.ai import pocket as PK
    path = tmp_path_factory.mktemp("pocket") / "pocket.npz"
    t = PK.Table.load_or_build(str(path))
    assert path.exists() and PK.Table.load(str(path)) is not None
    return t
