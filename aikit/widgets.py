"""Общие виджеты: выпадающий список со своей стрелкой, подпись с переносом, кнопки,
переключатели в ряд и сеткой.

Две ловушки Qt, на которые уже наступали:

- Подпись с переносом строк нельзя ограничивать по высоте (setFixedHeight/Maximum): на
  настоящем экране текст шире, чем в offscreen, переносится на лишнюю строку и обрезается
  (TicTacToeAi 1.0.0). Только setMinimumHeight. label(..., wrap=True) так и делает.
- У скруглённого QComboBox кнопка ::drop-down без стиля рисуется квадратом Fusion поверх
  скругления (AiCar 2.0.0). В теме ::drop-down прозрачный, ::down-arrow пустой, а стрелку
  рисует Combo.paintEvent: треугольник из рамок QSS Qt не умеет, выходит прямоугольник.
"""
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPolygonF
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
                               QPushButton, QWidget)


class Combo(QComboBox):
    """QComboBox со стрелкой, нарисованной кодом в цвет текста темы."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.arrow_color = QColor("#e6e8ec")

    def set_arrow_color(self, color):
        self.arrow_color = QColor(color)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(self.arrow_color)
        cx = self.width() - 16
        cy = self.height() / 2 + 1
        p.drawPolygon(QPolygonF([QPointF(cx - 5, cy - 3), QPointF(cx + 5, cy - 3), QPointF(cx, cy + 3)]))
        p.end()


def label(text="", name=None, wrap=False):
    """Подпись. С переносом - без ограничения высоты сверху (см. начало файла)."""
    w = QLabel(text)
    if name:
        w.setObjectName(name)
    w.setWordWrap(wrap)
    return w


def card():
    frame = QFrame()
    frame.setObjectName("card")
    return frame


def button(text, name=None, slot=None):
    b = QPushButton(text)
    if name:
        b.setObjectName(name)
    b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    if slot:
        b.clicked.connect(slot)
    return b


def segmented(options, slot, columns=None):
    """Кнопки-переключатели: {значение: подпись}. Возвращает (виджет, {значение: кнопка}).
    columns=None - в один ряд; число - сеткой в столько столбцов (в ряд не влезают).
    Ширина кнопок - сразу под жирный текст (fit_segments); после смены надписей - снова."""
    box = QWidget()
    if columns:
        lay = QGridLayout(box)
    else:
        lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    group = QButtonGroup(box)
    group.setExclusive(True)
    buttons = {}
    for i, (value, text) in enumerate(options.items()):
        b = button(text, "seg")
        b.setCheckable(True)
        b.clicked.connect(lambda _=False, v=value: slot(v))
        group.addButton(b)
        if columns:
            lay.addWidget(b, i // columns, i % columns)
        else:
            lay.addWidget(b)
        buttons[value] = b
    fit_segments(buttons)
    return box, buttons


def seg_grid(options, slot, cols=2):
    """Переключатели сеткой (BlockDropAi) - то же, что segmented(..., columns=cols)."""
    return segmented(options, slot, columns=cols)


def fit_segments(*groups):
    """Кнопки-переключатели шире жирного текста: выбранная кнопка пишется жирным (тема), а
    ширину Qt считает по обычному шрифту - надпись обрезалась (2048Ai, BlockDropAi).
    Звать после каждой смены надписей (язык). groups - словари {значение: кнопка} или
    списки кнопок."""
    for group in groups:
        for b in (group.values() if isinstance(group, dict) else group):
            # Шрифт - как в теме (14px), жирный с запасом: у кнопки при постройке ещё
            # шрифт по умолчанию, тема применяется позже.
            f = QFont(b.font())
            f.setPixelSize(max(14, f.pixelSize()))
            f.setWeight(QFont.Weight.Bold)
            b.setMinimumWidth(QFontMetrics(f).horizontalAdvance(b.text()) + 30)


fit_bold = fit_segments          # имя из BlockDropAi


def regrid(box, buttons, columns):
    """Переложить кнопки готового переключателя в сетку (2048Ai: боковая панель узкая).
    Старая раскладка уходит во временный виджет и удаляется вместе с ним."""
    old = box.layout()
    for b in buttons.values():
        old.removeWidget(b)                # кнопки остаются детьми box
    QWidget().setLayout(old)
    grid = QGridLayout(box)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setSpacing(6)
    for i, b in enumerate(buttons.values()):
        grid.addWidget(b, i // columns, i % columns)


def restyle(widget, object_name):
    """Сменить objectName и перечитать стиль - так метка меняет цвет по QSS."""
    widget.setObjectName(object_name)
    widget.style().unpolish(widget)
    widget.style().polish(widget)
