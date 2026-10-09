"""Работа ИИ в отдельном потоке: таблицы, решение, подсказка, «мысли». Окно не подвисает.

Каждый запрос получает номер (token); ответ со старым номером (кубик уже другой)
окно выбрасывает. Остановка - флагом threading.Event: поиск Kociemba проверяет его каждые
256 позиций, построение таблиц - на каждом куске. При закрытии окна потоки ждут не
дольше 2 секунд (контракт aikit).
"""
import threading
import time

from PySide6.QtCore import QObject, QThread, Signal


class _Job(QThread):
    done = Signal(int, str, object)
    progress = Signal(int, float, str)

    def __init__(self, token, kind, fn, parent=None):
        super().__init__(parent)
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
        job = _Job(token, kind, fn, self)
        job.done.connect(self.ready)
        job.progress.connect(self.progress)
        self.jobs.append(job)
        job.start()
        return job

    def _reap(self):
        self.jobs = [j for j in self.jobs if j.isRunning()]

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
        return not self.jobs
