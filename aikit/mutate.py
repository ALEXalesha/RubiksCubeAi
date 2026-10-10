"""Мутационная проверка для всех игр: нарочно ломает код и смотрит, что тесты это ловят.

Мутант - (имя, файл, было, стало, тесты): строка «было» должна встречаться в файле ровно
один раз. Файл правится на месте, гоняются тесты мутанта, затем файл восстанавливается
(в finally - и при Ctrl+C).

Два правила (BlockDropAi, 10.10):
- сначала те же тесты гоняются БЕЗ мутанта. Красные - прогон останавливается: «пойман» на
  уже красных тестах ничего не значит;
- тесты зависли дольше timeout - мутант пойман (зависание - тоже провал).

Пример tools/mutate.py игры:

    from aikit.mutate import main
    MUTANTS = [("правила: бьёт равной", "game/core/cards.py", "a > b", "a >= b", "tests/test_cards.py")]
    sys.exit(main(ROOT, MUTANTS, sys.argv[1:]))
"""
import os
import shlex
import subprocess
import sys

CAUGHT = "пойман"
ALIVE = "ВЫЖИЛ"


class BaselineRed(Exception):
    """Тесты красные без мутанта."""


def pytest_ok(root, tests, python=None, timeout=900):
    """Зелёные ли тесты. Зависли дольше timeout - не зелёные."""
    cmd = [python or sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *shlex.split(tests)]
    try:
        r = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False
    return r.returncode == 0


def run_mutant(root, mutant, python=None, timeout=900, baseline=None, check=pytest_ok):
    """Вердикт одного мутанта: CAUGHT, ALIVE или текст о ненайденной строке.
    baseline - словарь {тесты: зелёные ли без мутантов}, общий на прогон."""
    name, rel, old, new, tests = mutant
    baseline = {} if baseline is None else baseline
    if tests not in baseline:
        baseline[tests] = check(root, tests, python, timeout)
    if not baseline[tests]:
        raise BaselineRed(f"{name}: тесты «{tests}» красные без мутанта - сначала починить их")
    path = os.path.join(root, rel)
    with open(path, encoding="utf-8", newline="") as f:
        src = f.read()
    if src.count(old) != 1:
        return f"строка найдена {src.count(old)} раз"
    try:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(src.replace(old, new))
        return ALIVE if check(root, tests, python, timeout) else CAUGHT
    finally:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(src)


def main(root, mutants, argv=(), python=None, timeout=900, out=print, check=pytest_ok):
    """Прогнать мутантов (argv[0] - отбор по слову в имени). Код возврата: 0 - все пойманы,
    1 - кто-то выжил или строка не нашлась, 2 - тесты красные без мутанта (прогон остановлен)."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    flt = argv[0] if argv else ""
    chosen = [m for m in mutants if flt in m[0]]
    baseline, alive = {}, 0
    for m in chosen:
        try:
            verdict = run_mutant(root, m, python, timeout, baseline, check)
        except BaselineRed as e:
            out(f"СТОП     {e}")
            return 2
        out(f"{verdict:8} {m[0]}")
        alive += verdict != CAUGHT
    out(f"итого {len(chosen)}, не пойманы {alive}")
    return 1 if alive else 0
