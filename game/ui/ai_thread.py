"""Работа ИИ в отдельном потоке: таблицы, решение, подсказка, «мысли». Окно не подвисает.

Каждый запрос получает номер (token); ответ со старым номером (кубик уже другой)
окно выбрасывает. Остановка - флагом threading.Event: поиск Kociemba проверяет его каждые
256 позиций, построение таблиц - на каждом куске. При закрытии окна потоки ждут не
дольше 2 секунд (контракт aikit).
"""
import threading
import time

from PySide6.QtCore import QObject, QThread, Signal

# Потоки, не успевшие остановиться за shutdown() (машина перегружена): окно можно разрушать,
# поток доработает до флага остановки сам и только потом удалится. Урок DurakAi 1.0.0:
# разрушение окна с работающим дочерним QThread - qFatal и падение всей программы.
_ORPHANS = []
_UNDERTAKER = None


class _Undertaker(QObject):
    """Живёт в потоке окна: сигнал finished сироты приходит сюда очередью."""

    def bury(self):
        for j in list(_ORPHANS):
            if j.isFinished():
                j.wait()
                _ORPHANS.remove(j)


def _undertaker():
    global _UNDERTAKER
    if _UNDERTAKER is None:
        _UNDERTAKER = _Undertaker()
    return _UNDERTAKER


class _Job(QThread):
    done = Signal(int, str, object)
    progress = Signal(int, float, str)

    def __init__(self, token, kind, fn):
        super().__init__()
        self.token = token
        self.kind = kind
        self.fn = fn
        self.stop_event = threading.Event()

    def run(self):
        try:
            payload = self.fn(self.stop_event, lambda f, s: self.progress.emit(self.token, f, s))
        except Exception as e:  # noqa: BLE001 - ошибку покажет окно, поток не должен падать молча
            payload = e
        if not self.stop_event.is_set():
            self.done.emit(self.token, self.kind, payload)


class AiController(QObject):
    """start(token, kind, fn): fn(stop, progress) в потоке. Ответ - ready(token, kind,
    результат или исключение), ход построения таблиц - progress(token, доля, подпись)."""

    ready = Signal(int, str, object)
    progress = Signal(int, float, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.jobs = []

    def start(self, token, kind, fn):
        self._reap()
        # без родителя: поток живёт, пока на него есть ссылка в self.jobs (или в _ORPHANS)
        job = _Job(token, kind, fn)
        job.done.connect(self.ready)
        job.progress.connect(self.progress)
        job.finished.connect(self._reap)
        self.jobs.append(job)
        job.start()
        return job

    def _reap(self):
        """Кончившиеся потоки - из списка; последняя ссылка уходит, объект удаляется."""
        alive = []
        for j in self.jobs:
            if j.isFinished():
                j.wait()
            elif j.isRunning():
                alive.append(j)
        self.jobs = alive

    def cancel(self, kind=None):
        for j in self.jobs:
            if kind is None or j.kind == kind:
                j.stop_event.set()

    def busy(self, kind=None):
        self._reap()
        return any(kind is None or j.kind == kind for j in self.jobs)

    def shutdown(self, timeout_ms=2000):
        """Остановить всё и дождаться потоков (не дольше timeout_ms на всё)."""
        self.cancel()
        deadline = time.monotonic() + timeout_ms / 1000
        for j in self.jobs:
            left = max(0, int((deadline - time.monotonic()) * 1000))
            j.wait(left)
        self._reap()
        for j in self.jobs:
            j.done.disconnect(self.ready)
            j.progress.disconnect(self.progress)
            _ORPHANS.append(j)
            j.finished.connect(_undertaker().bury)
        self.jobs = []
        _undertaker().bury()
        return not _ORPHANS
