"""aikit.mutate: красные без мутанта - стоп, зависание - пойман, файл всегда восстанавливается."""
import os

from aikit import mutate


def project(tmp_path, test_body="assert mod.f() == 1"):
    (tmp_path / "mod.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "test_mod.py").write_text(f"import mod\n\n\ndef test_f():\n    {test_body}\n", encoding="utf-8")
    return str(tmp_path)


def fake_check(want):
    """Вместо pytest в подпроцессе (медленно на занятом ПК): тест «mod.f() == want»."""
    def check(root, tests, python, timeout):
        ns = {}
        exec(open(os.path.join(root, "mod.py"), encoding="utf-8").read(), ns)
        return ns["f"]() == want
    return check


def test_caught_alive_and_restored(tmp_path):
    root = project(tmp_path)
    lines = []
    mutants = [("ломает", "mod.py", "return 1", "return 2", "test_mod.py"),
               ("ничего не меняет", "mod.py", "return 1", "return 1  # тот же", "test_mod.py"),
               ("нет строки", "mod.py", "return 3", "return 4", "test_mod.py")]
    code = mutate.main(root, mutants, out=lines.append, timeout=60, check=fake_check(1))
    assert lines[0].startswith(mutate.CAUGHT) and lines[1].startswith(mutate.ALIVE)
    assert "строка найдена 0 раз" in lines[2] and lines[-1] == "итого 3, не пойманы 2"
    assert code == 1
    assert (tmp_path / "mod.py").read_text(encoding="utf-8") == "def f():\n    return 1\n"


def test_red_tests_without_mutant_stop_the_run(tmp_path):
    root = project(tmp_path, test_body="assert mod.f() == 5")
    lines = []
    code = mutate.main(root, [("любой", "mod.py", "return 1", "return 2", "test_mod.py")],
                       out=lines.append, timeout=60, check=fake_check(5))
    assert code == 2 and lines[0].startswith("СТОП") and "красные без мутанта" in lines[0]
    assert (tmp_path / "mod.py").read_text(encoding="utf-8") == "def f():\n    return 1\n"


def test_hang_counts_as_caught(tmp_path):
    """Настоящий pytest в подпроцессе: зависший прогон обрывается по тайм-ауту."""
    root = project(tmp_path)
    calls = []

    def check(root_, tests, python, timeout):
        calls.append(timeout)
        return mutate.pytest_ok(root_, tests, python, timeout if len(calls) == 1 else 3)
    verdict = mutate.run_mutant(root, ("вечный цикл", "mod.py", "return 1", "while True:\n        pass", "test_mod.py"),
                                timeout=60, check=check)
    assert verdict == mutate.CAUGHT
    assert os.path.exists(os.path.join(root, "mod.py"))
    assert "return 1" in (tmp_path / "mod.py").read_text(encoding="utf-8")
