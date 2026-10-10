"""Паспорт ИИ: цифры на месте и честные - тесты пересчитывают первые кубики с теми же
зёрнами и пределами и сверяют длины решений."""
import json
import os

import pytest

from game.ai.net import passport_path
from game.train import passport as PP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    with open(PP.PASSPORT, encoding="utf-8") as f:
        return json.load(f)


def test_kociemba_passport_meets_the_spec():
    k = load()["kociemba3"]
    assert k["count"] == 1000 and k["solved"] == 1000
    assert k["mean_length"] <= 22
    assert k["node_limit"] == PP.NODES and k["seed_base"] == PP.SEED_BASE


@pytest.mark.slow
def test_kociemba_passport_recomputes(koc_tables, monkeypatch):
    from game.ai import kociemba as KO
    monkeypatch.setitem(PP._W, "koc", KO.Solver(koc_tables))
    k = load()["kociemba3"]
    for i in range(3):
        row = PP.kociemba_case(i)
        assert row["solved"] and row["length"] == k["lengths"][i]


def test_net2_passport_is_honest_and_recomputes(pocket_table, monkeypatch):
    p = load()["net2"]
    assert p["count"] == 1000
    assert p["mean_length"] >= p["mean_optimal"]          # сеть не бывает короче оптимума
    assert 0 <= p["optimal_share"] <= p["solved_share"] <= 1
    monkeypatch.setitem(PP._W, "pocket", pocket_table)
    for i in range(5):
        row = PP.net_case((2, i, p["max_nodes"]))
        assert row["length"] == p["lengths"][i]
        assert row["length"] is None or row["length"] >= row["optimal"]


def test_model_metadata_has_training_time():
    with open(passport_path(2), encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["train_seconds"] > 0 and meta["steps"] > 0


def test_net3_passport_is_honest():
    d = load()
    p = d["net3"]
    assert p["count"] == 1000 and 0 <= p["solved"] <= 1000
    table = d["net3_by_depth"]["table"]
    shares = [table[k]["solved_share"] for k in sorted(table, key=int)]
    assert shares == sorted(shares, reverse=True)          # чем глубже перемешано, тем реже
    assert shares[0] == 1.0


@pytest.mark.slow
def test_net3_passport_recomputes():
    p = load()["net3"]
    for i in range(2):
        row = PP.net_case((3, i, p["max_nodes"]))
        assert row["length"] == p["lengths"][i]
