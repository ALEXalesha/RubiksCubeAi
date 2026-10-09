"""RubiksCubeAi: кубик Рубика, который собирает ИИ.

Вход игры по контракту PythonAiGames: GAME_INFO и create_widget(ctx). Qt загружается
только в create_widget - модель куба, решатели и обучение работают без него.
"""
from game.version import VERSION

GAME_INFO = {
    "id": "rubikscube",
    "title": {"ru": "Кубик Рубика", "en": "Rubik's Cube"},
    "summary": {"ru": "Кубик 2×2, 3×3 и 4×4: собирай сам или смотри, как собирает ИИ",
                "en": "2×2, 3×3 and 4×4 cube: solve it yourself or watch the AI solve it"},
    "preview": "docs/preview.png",
    "version": VERSION,
}

APP_NAME = "RubiksCubeAi"


def create_widget(ctx):
    """ctx: aikit.Context. Виджет с кубом, меню и панелями."""
    from game.ui.widget import GameWidget
    return GameWidget(ctx)
