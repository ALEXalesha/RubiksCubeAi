"""Законы послойного метода: собирает любой возможный кубик (и с повёрнутыми центрами),
шаги идут по порядку, поставленный кубик стоит на месте до конца своего слоя, ходов
в разумных пределах (начинающий собирает за 100-200)."""
import random

import numpy as np
from hypothesis import given, settings, strategies as st

from game.ai import beginner as BG
from game.core import cube as C
from game.core import cubie as K

P3 = C.Puzzle(3)
CENTER_OF = {i: K.CENTERS[[f for f in range(6) if tuple(P3.normal[K.CENTERS[f]]) == tuple(P3.normal[i])][0]]
             for i in range(54)}


def piece_solved(state, colors):
    idx = BG.piece_stickers(state, colors)
    assert len(idx) == len(colors)
    return all(state[i] == state[CENTER_OF[i]] for i in idx)


@settings(max_examples=150)
@given(st.integers(0, 10 ** 9), st.sampled_from(["", "x", "y M", "z2 E", "S x'"]))
def test_beginner_method_solves_any_cube(seed, turn):
    rng = random.Random(seed)
    s = K.random_cube(rng).to_facelets()
    if turn:
        s = P3.apply(s, turn)
    plan = BG.plan(s)
    order = [BG.STEPS.index(step) for step, _, _ in plan.steps]
    assert order == sorted(order)
    placed = []
    cur = s
    for step, colors, moves in plan.steps:
        cur = P3.apply(cur, moves)
        if colors is not None:
            placed.append(colors)
            for c in placed:                      # всё поставленное раньше стоит на месте
                assert piece_solved(cur, c), (step, c)
        elif step == "ycross":
            placed = []                           # дальше первые два слоя проверяет сама сборка
    assert P3.is_solved(cur)
    assert 0 < len(plan.moves) <= 260


def test_known_counts_of_pieces_per_step():
    plan = BG.plan(K.random_cube(random.Random(5)).to_facelets())
    kinds = {}
    for step, colors, _ in plan.steps:
        kinds.setdefault(step, []).append(colors)
    assert len(kinds.get("cross", [])) <= 4 and len(kinds.get("corners", [])) <= 4
    assert all(0 in c for c in kinds.get("cross", []) + kinds.get("corners", []))   # белые
    assert all(0 not in c and 3 not in c for c in kinds.get("middle", []))          # без белого и жёлтого


def test_solved_cube_needs_no_moves():
    assert BG.plan(P3.solved()).moves == []


def test_piece_stickers_by_colors():
    s = P3.apply(P3.solved(), "R U")
    assert len(BG.piece_stickers(s, (0, 1, 2))) == 3
    assert len(BG.piece_stickers(s, (0, 2))) == 2
    assert BG.piece_stickers(s, (0, 3)) == set()          # бело-жёлтого ребра нет


@settings(max_examples=60)
@given(st.integers(0, 10 ** 9))
def test_plan_for_a_cube_already_turned_white_down(seed):
    s = K.random_cube(random.Random(seed)).to_facelets()
    turned = P3.apply(s, BG.white_down(s))
    assert turned[K.CENTERS[C.D]] == C.U                  # белый центр внизу
    plan = BG.plan(turned, as_is=True)
    assert P3.is_solved(P3.apply(turned, plan.moves))
    corners = [m for step, _, ms in plan.steps if step == "corners" for m in ms]
    assert not any(m.startswith(("D", "x", "y", "z")) for m in corners)   # углы - сверху, белое не трогаем
