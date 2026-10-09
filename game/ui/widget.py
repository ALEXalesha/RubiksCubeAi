"""Виджет игры (create_widget по контракту): кубик и боковая панель.

Режимы:
- «Собираю сам»: перемешать, крутить мышью и клавишами, время с первого хода, отмена,
  подсказка, рекорды по размерам;
- «ИИ собирает»: ИИ ищет решение в потоке и показывает его по ходам: пауза, назад,
  вперёд, к началу, скорость.

Окно ему даёт оболочка (aikit.run_standalone или ИИ-Игротека). Тема и язык - из ctx, на
лету по ctx.changed. Файлы - только в ctx.data_dir.
"""
import gc
import html
import random
import time

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLineEdit, QScrollArea, QSlider,
                               QVBoxLayout, QWidget)

from aikit import settings as st
from aikit import theme
from aikit.storage import load_json, save_json
from aikit.widgets import button, card, label, restyle, segmented
from game.ai.solvers import Brain, Cancelled
from game.core import cube as C
from game.core.cubie import InvalidCube
from game.ui.ai_thread import AiController
from game.ui.cubeview import CubeView
from game.ui.i18n import translator
from game.version import VERSION

SIZES = (2, 3, 4)
SCRAMBLE_LEN = {2: 11, 3: 25, 4: 40}
SCHEMA = {
    "size": (3, st.choice(*SIZES)),
    "mode": ("self", st.choice("self", "ai")),
    "turn_ms": (180, st.int_range(0, 800)),
    "show_thoughts": (False, st.boolean()),
}
AI_TIME = 1.0               # секунд на поиск решения Kociemba (улучшение после первого)
THOUGHT_TIME = 0.3
AI_PAUSE_MS = 120           # пауза между ходами решения сверх анимации


def fmt_clock(seconds):
    m, s = divmod(max(0.0, seconds), 60)
    return f"{int(m)}:{s:04.1f}"


class GameWidget(QWidget):
    def __init__(self, ctx):
        super().__init__()
        gc.freeze()
        self.ctx = ctx
        self.t = translator(ctx.lang)
        self.settings = st.Settings(SCHEMA, ctx.file("settings.json"))
        self.brain = Brain(ctx.data_dir)
        self.ai = AiController(self)
        self.ai.ready.connect(self.on_ai)
        self.ai.progress.connect(self.on_progress)
        self.token = 0
        self.thought_token = 0
        self.rng = random.Random()
        self.history = []                # ходы человека после перемешивания (для отмены)
        self.timing = None               # {"start", "moves"} - идёт время сборки
        self.solution = None             # {"moves", "index", "playing", "start", "info"}
        self.status_key = ("app_sub", {})
        self.thought_text = ""
        self.hint_move = None
        self.records = load_json(ctx.file("records.json"))
        if not isinstance(self.records, dict):
            self.records = {}

        self.view = CubeView(self.settings["size"])
        self.view.turn_ms = self.settings["turn_ms"]
        self.view.userTurn.connect(self.on_user_turn)
        self.view.moveFinished.connect(self.on_move_finished)

        self.clock = QTimer(self)
        self.clock.setInterval(100)
        self.clock.timeout.connect(self.refresh_clock)
        self.step_timer = QTimer(self)
        self.step_timer.setSingleShot(True)
        self.step_timer.timeout.connect(self.play_step)
        self.thought_timer = QTimer(self)
        self.thought_timer.setSingleShot(True)
        self.thought_timer.setInterval(200)
        self.thought_timer.timeout.connect(self.request_thoughts)

        root = QHBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(14)
        self.setObjectName("page")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        root.addWidget(self.view, 1)
        root.addWidget(self._side())

        for keys, slot in (("Ctrl+Z", self.undo), ("Space", self.toggle_play), ("H", self.hint),
                           ("N", self.scramble)):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(slot)
        ctx.changed.connect(self.on_ctx_changed)
        self.apply_theme()
        self.retranslate()
        self.sync_controls()

    def minimumSizeHint(self):
        return QSize(980, 660)

    # --- панель ---

    def _side(self):
        side = card()
        side.setFixedWidth(360)
        outer = QVBoxLayout(side)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        col = QVBoxLayout(body)
        col.setContentsMargins(16, 14, 16, 14)
        col.setSpacing(8)
        self.title = label("", "heading", wrap=True)
        col.addWidget(self.title)

        def row(widget, caption_key):
            cap = label("", "muted")
            self.captions[caption_key] = cap
            line = QHBoxLayout()
            line.addWidget(cap)
            line.addStretch(1)
            line.addWidget(widget)
            col.addLayout(line)

        self.captions = {}
        w, self.size_buttons = segmented({n: f"{n}×{n}" for n in SIZES}, self.set_size)
        row(w, "size")
        w, self.mode_buttons = segmented({"self": "", "ai": ""}, self.set_mode)
        col.addWidget(w)

        self.status = label("", "section", wrap=True)
        self.status.setMinimumHeight(44)
        self.status.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        col.addWidget(self.status)
        clock_row = QHBoxLayout()
        self.clock_label = label("0:00.0", "bigNumber")
        self.moves_label = label("", "muted")
        clock_row.addWidget(self.clock_label)
        clock_row.addStretch(1)
        clock_row.addWidget(self.moves_label)
        col.addLayout(clock_row)

        r1 = QHBoxLayout()
        self.scramble_button = button("", "primary", self.scramble)
        self.reset_button = button("", None, self.reset)
        r1.addWidget(self.scramble_button)
        r1.addWidget(self.reset_button)
        col.addLayout(r1)
        r2 = QHBoxLayout()
        self.undo_button = button("", None, self.undo)
        self.hint_button = button("", None, self.hint)
        r2.addWidget(self.undo_button)
        r2.addWidget(self.hint_button)
        col.addLayout(r2)
        self.solve_button = button("", "primary", self.ai_solve)
        col.addWidget(self.solve_button)

        self.player = QWidget()
        pl = QVBoxLayout(self.player)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(6)
        prow = QHBoxLayout()
        self.first_button = button("", None, self.to_start)
        self.back_button = button("", None, self.step_back)
        self.play_button = button("", "primary", self.toggle_play)
        self.forward_button = button("", None, self.step_forward)
        for b in (self.first_button, self.back_button, self.play_button, self.forward_button):
            prow.addWidget(b)
        pl.addLayout(prow)
        self.solution_label = label("", None, wrap=True)
        self.solution_label.setTextFormat(Qt.TextFormat.RichText)
        pl.addWidget(self.solution_label)
        col.addWidget(self.player)

        self.speed_caption = label("", "muted")
        col.addWidget(self.speed_caption)
        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(0, 700)
        self.speed.setInvertedAppearance(True)
        self.speed.setValue(self.settings["turn_ms"])
        self.speed.valueChanged.connect(lambda v: self.change("turn_ms", int(v)))
        col.addWidget(self.speed)

        self.thoughts_box = QCheckBox()
        self.thoughts_box.setChecked(self.settings["show_thoughts"])
        self.thoughts_box.toggled.connect(lambda v: self.change("show_thoughts", bool(v)))
        col.addWidget(self.thoughts_box)
        self.thoughts_label = label("", "muted", wrap=True)
        col.addWidget(self.thoughts_label)

        self.notation_caption = label("", "muted")
        col.addWidget(self.notation_caption)
        nrow = QHBoxLayout()
        self.notation = QLineEdit()
        self.notation.returnPressed.connect(self.apply_notation)
        self.apply_button = button("", None, self.apply_notation)
        nrow.addWidget(self.notation, 1)
        nrow.addWidget(self.apply_button)
        col.addLayout(nrow)

        self.records_label = label("", "hint", wrap=True)
        col.addWidget(self.records_label)
        w, self.theme_buttons = segmented({"dark": "", "light": ""}, self.ctx.set_theme)
        row(w, "theme")
        w, self.lang_buttons = segmented({"ru": "RU", "en": "EN"}, self.ctx.set_lang)
        row(w, "lang")
        self.keys_label = label("", "hint", wrap=True)
        col.addWidget(self.keys_label)
        col.addStretch(1)
        self.foot = label("", "hint")
        col.addWidget(self.foot)
        return side

    # --- общее ---

    def change(self, key, value):
        if self.settings[key] == value:
            return
        self.settings.set(key, value)
        if key == "turn_ms":
            self.view.turn_ms = value
        elif key == "show_thoughts":
            self.thought_text = ""
            self.schedule_thoughts()
            self.refresh_texts()

    def on_ctx_changed(self, what):
        if what == "theme":
            self.apply_theme()
        elif what == "lang":
            self.t = translator(self.ctx.lang)
            self.retranslate()
        self.sync_controls()

    def apply_theme(self):
        c = theme.palette(self.ctx.theme)
        self.view.set_theme(c["bg"], c["select"])
        # поля ввода в общей теме aikit нет - стиль здесь, в цвет темы
        self.notation.setStyleSheet(
            f"QLineEdit {{ background: {c['surface2']}; border: 1px solid {c['border']}; border-radius: 8px;"
            f" padding: 6px 10px; color: {c['text']}; }} QLineEdit:focus {{ border-color: {c['accent']}; }}")

    def retranslate(self):
        t = self.t
        self.title.setText(t("app_title"))
        for key, cap in self.captions.items():
            cap.setText(t(key))
        self.mode_buttons["self"].setText(t("mode_self"))
        self.mode_buttons["ai"].setText(t("mode_ai"))
        self.scramble_button.setText(t("scramble"))
        self.reset_button.setText(t("reset"))
        self.undo_button.setText(t("undo"))
        self.hint_button.setText(t("hint"))
        self.solve_button.setText(t("ai_solve"))
        self.first_button.setText(t("first"))
        self.back_button.setText(t("back"))
        self.forward_button.setText(t("forward"))
        self.speed_caption.setText(t("speed"))
        self.thoughts_box.setText(t("thoughts"))
        self.notation_caption.setText(t("notation"))
        self.notation.setPlaceholderText(t("notation_ph"))
        self.apply_button.setText(t("apply"))
        self.theme_buttons["dark"].setText(t("dark"))
        self.theme_buttons["light"].setText(t("light"))
        self.keys_label.setText(t("keys"))
        self.foot.setText(t("footer", v=VERSION))
        self.refresh_texts()

    def sync_controls(self):
        self.size_buttons[self.n].setChecked(True)
        self.mode_buttons[self.settings["mode"]].setChecked(True)
        self.theme_buttons.get(self.ctx.theme, self.theme_buttons["dark"]).setChecked(True)
        self.lang_buttons[self.ctx.lang].setChecked(True)
        self.thoughts_box.blockSignals(True)
        self.thoughts_box.setChecked(self.settings["show_thoughts"])
        self.thoughts_box.blockSignals(False)
        ai = self.settings["mode"] == "ai"
        self.player.setVisible(ai)
        self.undo_button.setEnabled(bool(self.history) and not ai)
        self.hint_button.setEnabled(self.brain.supports(self.n))
        self.solve_button.setEnabled(self.brain.supports(self.n))
        sol = self.solution
        self.first_button.setEnabled(bool(sol) and sol["index"] > 0)
        self.back_button.setEnabled(bool(sol) and sol["index"] > 0)
        self.forward_button.setEnabled(bool(sol) and sol["index"] < len(sol["moves"]))
        self.play_button.setEnabled(bool(sol) and sol["index"] < len(sol["moves"]))
        self.play_button.setText(self.t("pause") if sol and sol["playing"] else self.t("play"))

    @property
    def n(self):
        return self.view.n

    def set_status(self, key, **kw):
        self.status_key = (key, kw)
        self.refresh_texts()

    def refresh_texts(self):
        key, kw = self.status_key
        self.status.setText(self.t(key, **kw))
        restyle(self.status, "statusWin" if key in ("solved", "solved_time", "solved_record", "solution_done") else "section")
        self.refresh_clock()
        sol = self.solution
        if sol:
            parts = []
            for i, m in enumerate(sol["moves"]):
                m = html.escape(m)
                if i < sol["index"]:
                    parts.append(f"<span style='color:{theme.palette(self.ctx.theme)['muted']}'>{m}</span>")
                elif i == sol["index"]:
                    parts.append(f"<b><u>{m}</u></b>")
                else:
                    parts.append(m)
            self.solution_label.setText(
                html.escape(self.t("step", i=sol["index"], n=len(sol["moves"]))) + "<br>" + " ".join(parts))
        else:
            self.solution_label.setText("")
        self.thoughts_label.setVisible(self.settings["show_thoughts"])
        self.thoughts_label.setText(self.thought_text)
        items = []
        for n in SIZES:
            r = self.records.get(str(n))
            if isinstance(r, dict) and isinstance(r.get("time"), (int, float)):
                items.append(self.t("record_item", size=f"{n}×{n}", time=fmt_clock(r["time"]),
                                    moves=r.get("moves", "?")))
        self.records_label.setText(self.t("records", text="; ".join(items)) if items else self.t("no_records"))
        self.sync_controls()

    def refresh_clock(self):
        if self.timing and self.timing.get("start") is not None:
            end = self.timing.get("end") or time.monotonic()
            self.clock_label.setText(fmt_clock(end - self.timing["start"]))
        elif not self.timing:
            self.clock_label.setText(fmt_clock(0))
        moves = self.timing["moves"] if self.timing else len(self.history)
        self.moves_label.setText(self.t("turning", moves=moves))

    def shutdown(self):
        """Окно закрывается: таймеры стоп, поиски остановить и дождаться (не дольше 2 с)."""
        self._invalidate()
        self.thought_token += 1
        self.clock.stop()
        self.thought_timer.stop()
        self.view.stop_animation()
        return self.ai.shutdown(2000)

    def _invalidate(self):
        """Кубик стал другим: ответы ИИ на старый запрос не нужны."""
        self.token += 1
        self.step_timer.stop()
        self.ai.cancel("solve")
        self.ai.cancel("hint")
        self.hint_move = None
        self.view.highlight = set()

    # --- размер, режим ---

    def set_size(self, n):
        if n == self.n:
            return
        self.settings.set("size", n)
        self._invalidate()
        self.solution = None
        self.history = []
        self.timing = None
        self.clock.stop()
        self.view.set_puzzle(n)
        self.set_status("app_sub")
        self.schedule_thoughts()

    def set_mode(self, mode):
        if self.settings["mode"] == mode:
            return
        self.settings.set("mode", mode)
        if mode == "self":
            self.pause()
        self.sync_controls()

    # --- ходы человека ---

    def on_user_turn(self, move):
        if self.solution is not None:
            self.drop_solution()
        self.view.turn(move, "user")

    def on_move_finished(self, move, source):
        if source == "user":
            self.history.append(move)
            if self.timing is not None and self.timing.get("end") is None:
                if self.timing["start"] is None:
                    self.timing["start"] = time.monotonic()
                    self.clock.start()
                self.timing["moves"] += 1
        if source in ("user", "undo", "custom"):
            self._invalidate()
        if self.view.pz.is_solved(self.view.state) and not self.view.animating():
            self.on_solved(source)
        elif source == "user" and self.status_key[0] in ("solved", "solved_time", "solved_record", "already", "solution_done"):
            self.set_status("app_sub")
        if source == "ai" and self.solution is not None and self.solution["playing"]:
            if self.solution["index"] < len(self.solution["moves"]):
                if not self.view.animating():
                    self.step_timer.start(AI_PAUSE_MS)
            else:
                self.solution["playing"] = False
        self.refresh_texts()
        self.schedule_thoughts()

    def on_solved(self, source):
        if self.timing is not None and self.timing.get("start") is not None and self.timing.get("end") is None:
            self.timing["end"] = time.monotonic()
            self.clock.stop()
            spent = self.timing["end"] - self.timing["start"]
            moves = self.timing["moves"]
            self.ctx.stats.record(f"{self.n}x{self.n}", "win")
            self.ctx.sound.play("win")
            old = self.records.get(str(self.n))
            if not isinstance(old, dict) or not isinstance(old.get("time"), (int, float)) or spent < old["time"]:
                self.records[str(self.n)] = {"time": round(spent, 2), "moves": moves}
                save_json(self.ctx.file("records.json"), self.records)
                self.set_status("solved_record", time=fmt_clock(spent), moves=moves)
            else:
                self.set_status("solved_time", time=fmt_clock(spent), moves=moves)
        elif source == "ai" and self.solution is not None:
            self.set_status("solution_done", n=len(self.solution["moves"]))
        elif source in ("user", "undo"):
            self.set_status("solved")

    def scramble(self, seed=None):
        rng = random.Random(seed) if seed is not None else self.rng
        moves = self.view.pz.random_moves(SCRAMBLE_LEN[self.n], rng)
        self._new_position(self.view.pz.apply(self.view.pz.solved(), moves))
        self.timing = {"start": None, "moves": 0, "end": None}
        self.set_status("scrambled", n=len(moves))
        self.refresh_texts()
        return moves

    def reset(self):
        self._new_position(self.view.pz.solved())
        self.set_status("app_sub")

    def _new_position(self, state):
        self._invalidate()
        self.solution = None
        self.history = []
        self.timing = None
        self.clock.stop()
        self.view.set_state(state)
        self.thought_text = ""
        self.schedule_thoughts()
        self.refresh_texts()

    def undo(self):
        if not self.history or self.settings["mode"] == "ai":
            return
        move = self.history.pop()
        self.view.turn(move.inverse(), "undo")

    def apply_notation(self):
        text = self.notation.text().strip()
        if not text:
            return
        try:
            moves = self.view.pz.moves(text)
        except C.NotationError:
            self.set_status("bad_notation", text=text)
            return
        self.drop_solution()
        self.timing = None
        self.history = []
        for m in moves:
            self.view.turn(m, "custom")
        self.set_status("custom")

    # --- ИИ ---

    def _check_cube(self):
        if not self.brain.supports(self.n):
            self.set_status("no_solver", size=f"{self.n}×{self.n}")
            return False
        why = self.brain.check(self.n, self.view.state)
        if why:
            self.set_status("invalid", why=self.t("why_" + why))
            return False
        return True

    def _job(self, kind, time_limit):
        n, state = self.n, self.view.state.copy()
        brain = self.brain

        def fn(stop, progress):
            if not brain.ready(n):
                brain.prepare(n, progress, stop)
            return brain.solve(n, state, time_limit=time_limit, stop=stop)
        return fn

    def ai_solve(self):
        self.view.finish_all()
        if self.view.pz.is_solved(self.view.state):
            self.set_status("already")
            return
        if not self._check_cube():
            return
        if self.settings["mode"] != "ai":
            self.set_mode("ai")
        self._invalidate()
        self.drop_solution()
        self.set_status("thinking")
        self.ai.start(self.token, "solve", self._job("solve", AI_TIME))

    def hint(self):
        self.view.finish_all()
        if self.view.pz.is_solved(self.view.state):
            self.set_status("already")
            return
        if not self._check_cube():
            return
        self._invalidate()
        self.set_status("thinking")
        self.ai.start(self.token, "hint", self._job("hint", 0.3))

    def on_progress(self, token, frac, _text):
        if token in (self.token, self.thought_token) and self.status_key[0] in ("thinking", "preparing"):
            self.set_status("preparing", pct=int(frac * 100))

    def on_ai(self, token, kind, payload):
        if kind == "thoughts":
            if token == self.thought_token:
                self.show_thoughts(payload)
            return
        if token != self.token:
            return
        if isinstance(payload, InvalidCube):
            self.set_status("invalid", why=self.t("why_" + payload.reason))
            return
        if isinstance(payload, Cancelled) or payload is None:
            return
        if isinstance(payload, Exception):
            self.set_status("error", e=payload)
            return
        if kind == "hint":
            if not payload.moves:
                self.set_status("already")
                return
            move = self.view.pz.parse(payload.moves[0])
            self.hint_move = move
            self.view.highlight = self.layer_stickers(move)
            self.view.update()
            self.set_status("hint_is", move=payload.moves[0])
        elif kind == "solve":
            self.solution = {"moves": payload.moves, "index": 0, "playing": True,
                             "start": self.view.state.copy(), "info": payload}
            self.set_status("solution", n=len(payload.moves))
            self.show_thoughts(payload)
            self.step_timer.start(AI_PAUSE_MS)

    def layer_stickers(self, move):
        """Наклейки слоя, который поворачивает ход (для подсветки подсказки)."""
        pz = self.view.pz
        out = set()
        for i in range(pz.size):
            if pz.layer_of(i, move.axis) in move.layers:
                out.add(i)
        return out

    # --- проигрыватель решения ---

    def drop_solution(self):
        self.step_timer.stop()
        self.solution = None
        self.refresh_texts()

    def play_step(self):
        sol = self.solution
        if not sol or not sol["playing"] or self.view.animating():
            return
        self.step_forward()

    def step_forward(self):
        sol = self.solution
        if not sol or sol["index"] >= len(sol["moves"]) or self.view.animating():
            return
        move = self.view.pz.parse(sol["moves"][sol["index"]])
        sol["index"] += 1
        self.view.turn(move, "ai")
        self.refresh_texts()

    def step_back(self):
        sol = self.solution
        if not sol or sol["index"] == 0:
            return
        self.pause()
        self.view.finish_all()
        sol["index"] -= 1
        move = self.view.pz.parse(sol["moves"][sol["index"]])
        self.view.turn(move.inverse(), "back")
        if self.status_key[0] == "solution_done":
            self.set_status("solution", n=len(sol["moves"]))
        self.refresh_texts()

    def to_start(self):
        sol = self.solution
        if not sol:
            return
        self.pause()
        self.view.set_state(sol["start"])
        sol["index"] = 0
        self.set_status("solution", n=len(sol["moves"]))
        self.refresh_texts()

    def toggle_play(self):
        sol = self.solution
        if not sol:
            return
        if sol["playing"]:
            self.pause()
            return
        if sol["index"] >= len(sol["moves"]):
            return
        sol["playing"] = True
        self.refresh_texts()
        self.play_step()

    def pause(self):
        if self.solution:
            self.solution["playing"] = False
        self.step_timer.stop()
        self.refresh_texts()

    # --- мысли ИИ ---

    def schedule_thoughts(self):
        self.thought_token += 1
        self.ai.cancel("thoughts")
        if self.settings["show_thoughts"]:
            self.thought_timer.start()

    def request_thoughts(self):
        if not self.settings["show_thoughts"] or self.view.animating():
            if self.settings["show_thoughts"]:
                self.thought_timer.start()
            return
        pz = self.view.pz
        if pz.is_solved(self.view.state):
            self.thought_text = self.t("th_solved")
            self.refresh_texts()
            return
        if not self.brain.supports(self.n):
            self.thought_text = self.t("th_none")
            self.refresh_texts()
            return
        if self.brain.check(self.n, self.view.state):
            self.thought_text = ""
            self.refresh_texts()
            return
        self.thought_text = self.t("th_wait")
        self.refresh_texts()
        self.ai.start(self.thought_token, "thoughts", self._job("thoughts", THOUGHT_TIME))

    def show_thoughts(self, sol):
        if isinstance(sol, Exception) or sol is None:
            return
        if sol.optimal:
            text = self.t("th_exact", n=len(sol))
        else:
            text = self.t("th_bound", low=sol.estimate, high=len(sol))
            found = " → ".join(str(x) for x in sol.found)
            text += "\n" + self.t("th_search", nodes=f"{sol.nodes:,}".replace(",", " "),
                                  sec=f"{sol.seconds:.2f}", found=found)
        self.thought_text = text
        self.refresh_texts()
