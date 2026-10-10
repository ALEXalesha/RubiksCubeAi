"""Переключатели aikit: ширина под жирный текст, сетка, перекладка в сетку."""
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QGridLayout, QHBoxLayout


def bold_width(b):
    f = QFont(b.font())
    f.setPixelSize(14)
    f.setWeight(QFont.Weight.Bold)
    return QFontMetrics(f).horizontalAdvance(b.text())


def test_fit_segments_leaves_room_for_bold_text(qapp):
    from aikit.widgets import fit_segments, segmented
    _, buttons = segmented({"a": "Соревнование", "b": "ИИ"}, lambda v: None)
    for b in buttons.values():
        b.setMinimumWidth(0)
    fit_segments(buttons)
    for b in buttons.values():
        assert b.minimumWidth() >= bold_width(b) + 20


def test_segmented_fits_bold_text_from_the_start(qapp):
    from aikit.widgets import segmented
    _, buttons = segmented({"x": "Переводной дурак"}, lambda v: None)
    assert buttons["x"].minimumWidth() >= bold_width(buttons["x"]) + 20


def test_fit_bold_takes_a_list_too(qapp):
    from aikit.widgets import button, fit_bold
    b = button("Ускоренная игра", "seg")
    fit_bold([b])
    assert b.minimumWidth() >= bold_width(b) + 20


def test_segmented_in_columns_and_seg_grid(qapp):
    from aikit.widgets import seg_grid, segmented
    chosen = []
    box, buttons = segmented({k: k for k in "abcde"}, chosen.append, columns=2)
    lay = box.layout()
    assert isinstance(lay, QGridLayout)
    assert lay.getItemPosition(lay.indexOf(buttons["c"]))[:2] == (1, 0)
    assert lay.getItemPosition(lay.indexOf(buttons["e"]))[:2] == (2, 0)
    buttons["b"].click()
    buttons["e"].click()
    assert chosen == ["b", "e"] and buttons["e"].isChecked() and not buttons["b"].isChecked()
    box2, buttons2 = seg_grid({k: k for k in "abc"}, chosen.append, cols=3)
    assert box2.layout().getItemPosition(box2.layout().indexOf(buttons2["c"]))[:2] == (0, 2)
    row, _ = segmented({"a": "a"}, chosen.append)
    assert isinstance(row.layout(), QHBoxLayout)


def test_regrid_keeps_buttons_alive_and_exclusive(qapp):
    from aikit.widgets import regrid, segmented
    chosen = []
    box, buttons = segmented({k: k for k in "abcd"}, chosen.append)
    regrid(box, buttons, columns=2)
    assert isinstance(box.layout(), QGridLayout)
    assert box.layout().getItemPosition(box.layout().indexOf(buttons["c"]))[:2] == (1, 0)
    buttons["a"].click()
    buttons["d"].click()
    assert chosen == ["a", "d"] and buttons["d"].isChecked() and not buttons["a"].isChecked()
