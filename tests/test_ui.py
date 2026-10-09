"""Окно без экрана (Qt offscreen): кубик рисуется, слой поворачивается мышью и клавишами,
фон вращает вид без захвата мыши, ИИ собирает и показывает решение по ходам, подсказка,
отмена, рекорды, темы и языки на лету, остановка за 2 секунды. Ожидания - с тайм-аутом;
мышь - собранными событиями, настоящий курсор не трогается."""
import json
import os
import time

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QWidget

from game import APP_NAME, GAME_INFO, create_widget
from game.ai import kociemba as KO
from game.core import cube as C
from game.ui.i18n import EN, RU

P3 = C.Puzzle(3)


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(app, tmp_path, koc_tables, pocket_table):
    from aikit.host import make_window
    w, ctx = make_window(GAME_INFO, create_widget, APP_NAME, root=str(tmp_path / "home"))
    w.resize(1280, 820)
    w.show()
    g = w.game
    g.brain.kociemba = KO.Solver(koc_tables)
    g.brain.pocket = pocket_table
    g.change("turn_ms", 0)
    app.processEvents()
    yield w
    g.shutdown()
    w.close()


def wait(app, cond, limit=20.0):
    end = time.time() + limit
    while not cond() and time.time() < end:
        app.processEvents()
        time.sleep(0.003)
    return cond()


def mouse(widget, kind, pos, buttons=Qt.MouseButton.LeftButton):
    types = {"press": QEvent.Type.MouseButtonPress, "move": QEvent.Type.MouseMove,
             "release": QEvent.Type.MouseButtonRelease}
    button = Qt.MouseButton.NoButton if kind == "move" else Qt.MouseButton.LeftButton
    ev = QMouseEvent(types[kind], QPointF(pos), widget.mapToGlobal(QPointF(pos)), button,
                     buttons if kind != "release" else Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(widget, ev)


def drag(app, widget, start, end, steps=6):
    mouse(widget, "press", start)
    for k in range(1, steps + 1):
        t = k / steps
        mouse(widget, "move", QPointF(start.x() + (end.x() - start.x()) * t, start.y() + (end.y() - start.y()) * t))
    mouse(widget, "release", end)
    app.processEvents()


def key(widget, k, shift=False):
    mods = Qt.KeyboardModifier.ShiftModifier if shift else Qt.KeyboardModifier.NoModifier
    QApplication.sendEvent(widget, QKeyEvent(QEvent.Type.KeyPress, k, mods))


def test_translations_complete():
    assert set(RU) == set(EN)


def test_cube_is_drawn_with_all_visible_stickers(app, win):
    v = win.game.view
    img = v.grab().toImage()
    assert not img.isNull()
    seen = {i for _, i in v.hits}
    # видно три грани по 9 наклеек
    assert len(seen) == 27
    assert len({tuple(P3.normal[i]) for i in seen}) == 3
    c = img.pixelColor(v.width() // 2, v.height() // 2)
    assert c != v.background


def test_dragging_a_sticker_turns_its_layer(app, win):
    v = win.game.view
    v.grab()
    idx = 20                         # F3: верхний правый угол передней грани
    a = v.sticker_center(idx)
    b = v.sticker_center(26)         # F9: под ним, нижний правый
    drag(app, v, a, QPointF(a.x() + (b.x() - a.x()), a.y() + (b.y() - a.y())))
    assert wait(app, lambda: not v.animating())
    assert (v.state == P3.apply(P3.solved(), "R'")).all()
    assert QWidget.mouseGrabber() is None
    assert win.game.history == [P3.parse("R'")]


def test_dragging_the_middle_turns_a_slice(app, win):
    v = win.game.view
    v.grab()
    a = v.sticker_center(22)         # центр F
    b = v.sticker_center(23)         # справа от центра
    drag(app, v, a, QPointF(b.x() + (b.x() - a.x()), b.y() + (b.y() - a.y())))
    assert wait(app, lambda: not v.animating())
    assert (v.state == P3.apply(P3.solved(), "E")).all()      # средний ряд вправо - как D


def test_dragging_the_background_rotates_only_the_view(app, win):
    v = win.game.view
    v.grab()
    yaw, pitch = v.yaw, v.pitch
    drag(app, v, QPointF(20, 20), QPointF(140, 80))
    assert v.yaw != yaw and v.pitch != pitch
    assert P3.is_solved(v.state) and (v.state == P3.solved()).all()
    assert QWidget.mouseGrabber() is None
    assert not win.game.history


def test_keys_turn_faces(app, win):
    v = win.game.view
    key(v, Qt.Key.Key_R)
    key(v, Qt.Key.Key_U, shift=True)
    assert wait(app, lambda: not v.animating())
    assert (v.state == P3.apply(P3.solved(), "R U'")).all()
    win.game.undo()
    assert wait(app, lambda: not v.animating())
    assert (v.state == P3.apply(P3.solved(), "R")).all()


def test_turns_are_animated(app, win):
    g = win.game
    g.change("turn_ms", 120)
    g.view.turn("R", "user")
    assert g.view.animating()
    assert (g.view.state == P3.solved()).all()        # пока крутится, состояние прежнее
    g.view.grab()
    assert wait(app, lambda: not g.view.animating(), 5)
    assert (g.view.state == P3.apply(P3.solved(), "R")).all()


def test_ai_solves_and_plays_the_solution(app, win):
    g = win.game
    g.scramble(seed=11)
    start = g.view.state.copy()
    g.ai_solve()
    assert g.settings["mode"] == "ai" and g.player.isVisible()
    assert wait(app, lambda: g.solution is not None, 30), g.status.text()
    n = len(g.solution["moves"])
    assert 0 < n <= 26
    assert wait(app, lambda: g.solution["index"] == n and not g.view.animating(), 30)
    assert P3.is_solved(g.view.state)
    assert str(n) in g.status.text()
    g.step_back()
    assert wait(app, lambda: not g.view.animating())
    last = P3.parse(g.solution["moves"][-1])
    assert (g.view.state == P3.apply(P3.solved(), [last.inverse()])).all()
    assert g.solution["index"] == n - 1
    g.step_forward()
    assert wait(app, lambda: not g.view.animating())
    assert P3.is_solved(g.view.state)
    g.to_start()
    assert (g.view.state == start).all() and g.solution["index"] == 0
    g.toggle_play()
    assert g.solution["playing"]
    g.toggle_play()
    assert not g.solution["playing"]


def test_user_turn_drops_the_ai_solution(app, win):
    g = win.game
    g.scramble(seed=12)
    g.ai_solve()
    assert wait(app, lambda: g.solution is not None, 30)
    g.pause()
    g.on_user_turn(P3.parse("U"))
    assert g.solution is None


def test_hint_on_2x2_is_an_optimal_move(app, win, pocket_table):
    g = win.game
    g.set_size(2)
    g.scramble(seed=5)
    d = pocket_table.distance_of_state(g.view.state)
    g.hint()
    assert wait(app, lambda: g.hint_move is not None, 30), g.status.text()
    after = g.view.pz.apply(g.view.state, [g.hint_move])
    assert pocket_table.distance_of_state(after) == d - 1
    assert g.view.highlight


def test_self_solve_sets_a_record(app, win):
    g = win.game
    moves = g.scramble(seed=3)
    assert g.timing is not None and g.timing["start"] is None
    for m in C.inverse(moves):
        g.on_user_turn(m)
    assert wait(app, lambda: not g.view.animating())
    assert P3.is_solved(g.view.state)
    rec = json.load(open(g.ctx.file("records.json"), encoding="utf-8"))
    assert rec["3"]["moves"] == len(moves)
    assert g.ctx.stats.entry("3x3")["wins"] == 1
    assert "0:" in g.status.text()


def test_impossible_cube_is_explained(app, win):
    g = win.game
    s = P3.solved().copy()
    s[8], s[9], s[20] = s[9], s[20], s[8]         # повернуть угол URF на месте
    g.view.set_state(s)
    g.ai_solve()
    assert g.solution is None
    assert RU["why_twist"] in g.status.text()


def test_4x4_has_no_solver_but_turns(app, win):
    g = win.game
    g.set_size(4)
    assert not g.solve_button.isEnabled() and not g.hint_button.isEnabled()
    g.scramble(seed=1)
    assert not g.view.pz.is_solved(g.view.state)
    g.view.grab()
    assert len({i for _, i in g.view.hits}) == 48


def test_notation_input(app, win):
    g = win.game
    g.notation.setText("R U R' U'")
    g.apply_notation()
    assert wait(app, lambda: not g.view.animating())
    assert (g.view.state == P3.apply(P3.solved(), "R U R' U'")).all()
    g.notation.setText("Q")
    g.apply_notation()
    assert "Q" in g.status.text()


def test_thoughts_show_bounds(app, win):
    g = win.game
    g.change("show_thoughts", True)
    g.scramble(seed=8)
    assert wait(app, lambda: "Kociemba" in g.thoughts_label.text(), 30), g.thoughts_label.text()
    assert g.thoughts_label.isVisible()


def test_theme_and_language_switch_live(app, win):
    g = win.game
    g.ctx.set_lang("en")
    assert g.title.text() == EN["app_title"]
    assert g.scramble_button.text() == EN["scramble"]
    bg = g.view.background.name()
    g.ctx.set_theme("light" if g.ctx.theme == "dark" else "dark")
    assert g.view.background.name() != bg
    g.ctx.set_lang("ru")
    assert g.title.text() == RU["app_title"]


def test_close_stops_table_building_within_two_seconds(app, tmp_path):
    from aikit.host import make_window
    w, _ = make_window(GAME_INFO, create_widget, APP_NAME, root=str(tmp_path / "fresh"))
    w.show()
    g = w.game
    g.scramble(seed=1)
    g.ai_solve()                       # таблиц нет - строятся в потоке
    assert wait(app, lambda: g.ai.busy(), 5)
    time.sleep(0.3)
    t0 = time.time()
    assert g.shutdown()
    assert time.time() - t0 < 2.0
    assert not g.ai.busy()                 # потоки действительно остановлены
    w.close()
    assert os.path.exists(os.path.join(str(tmp_path / "fresh"), "window.json"))


def test_step_back_undoes_exactly_the_last_move(app, win):
    g = win.game
    g.view.set_state(P3.apply(P3.solved(), "R U"))
    g.ai_solve()
    assert wait(app, lambda: g.solution is not None, 30)
    assert g.solution["moves"] == ["U'", "R'"]
    assert wait(app, lambda: g.solution["index"] == 2 and not g.view.animating(), 10)
    g.step_back()
    assert wait(app, lambda: not g.view.animating())
    assert (g.view.state == P3.apply(P3.solved(), "R")).all()
    g.step_back()
    assert wait(app, lambda: not g.view.animating())
    assert (g.view.state == P3.apply(P3.solved(), "R U")).all()
