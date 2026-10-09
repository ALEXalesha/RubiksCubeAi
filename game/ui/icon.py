"""Иконка программы: тот же объёмный кубик, что в окне, на прозрачном фоне."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QImage, QPainter, QPixmap


def render(size=256):
    from game.ui.cubeview import CubeView
    view = CubeView(3)
    view.setMinimumSize(16, 16)
    view.resize(size, size)
    view.background = Qt.GlobalColor.transparent
    view.state = view.pz.apply(view.pz.solved(), "R U R' F")
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    view.render(p)
    p.end()
    view.deleteLater()
    return img


def app_icon():
    icon = QIcon()
    for s in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(QPixmap.fromImage(render(s)))
    return icon
