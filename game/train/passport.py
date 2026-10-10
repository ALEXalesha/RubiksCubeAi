"""Паспорт ИИ: цифры решателей на случайных кубиках с постоянными зёрнами.

    python -m game.train.passport kociemba --count 1000
    python -m game.train.passport net3 --count 1000 --max-nodes 5000
    python -m game.train.passport net2 --count 1000
    python -m game.train.passport trial              # 10 кубиков каждого вида и прогноз времени

Kociemba ограничен числом позиций (NODES), а не временем: длины решений повторяются на
любой машине, и тесты пересчитывают первые кубики с теми же зёрнами. Время - замер
этой машины, пишется рядом. Работников не больше 2 (ПК делят и другие задачи), в каждом
один поток BLAS. Итог дописывается в game/models/passport.json своим разделом.
"""
import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

PASSPORT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "passport.json")
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".cache")
NODES = 100_000
SEED_BASE = 20261010
WEIGHT, BATCH = 0.6, 64

_W = {}


def _kociemba():
    if "koc" not in _W:
        from game.ai import kociemba as KO
        _W["koc"] = KO.Solver(KO.Tables.load_or_build(os.path.join(CACHE, "kociemba.npz")))
    return _W["koc"]


def kociemba_case(i):
    """Кубик i: равномерно случайный, решение с пределом NODES позиций."""
    import random
    from game.ai import kociemba as KO
    from game.core import cubie as K
    cc = K.random_cube(random.Random(SEED_BASE + i))
    t0 = time.perf_counter()
    r = _kociemba().solve(cc, time_limit=1e9, node_limit=NODES)
    sec = time.perf_counter() - t0
    ok = r is not None and KO.apply_moves(cc, r.moves).is_solved()
    return {"i": i, "solved": ok, "length": len(r) if r else None, "first": r.solutions[0] if r else None,
            "seconds": round(sec, 3), "nodes": r.nodes if r else None}


def net_case(args):
    """Сеть n×n и взвешенный A* на кубике i (3×3 - случайный, 2×2 - случайный через
    30 ходов и поворот к DBL)."""
    n, i, max_nodes = args
    import random
    from game.ai import astar
    from game.ai.net import default_net
    from game.core import cube as C
    from game.core import cubie as K
    net = default_net(n)
    pz = C.Puzzle(n)
    if n == 3:
        state = K.random_cube(random.Random(SEED_BASE + i)).to_facelets()
    else:
        from game.ai import pocket as PK
        rng = random.Random(SEED_BASE + i)
        state, _ = PK.normalize2(pz.apply(pz.solved(), pz.random_moves(30, rng)))
    r = astar.solve(net, state, weight=WEIGHT, batch=BATCH, max_nodes=max_nodes)
    ok = r.solved and pz.is_solved(pz.apply(state, r.moves))
    out = {"i": i, "solved": ok, "length": len(r.moves) if ok else None, "seconds": round(r.seconds, 3),
           "nodes": r.nodes, "estimate": round(r.estimate, 2)}
    if n == 2:
        if "pocket" not in _W:
            from game.ai import pocket as PK
            _W["pocket"] = PK.Table.load_or_build(os.path.join(CACHE, "pocket.npz"))
        out["optimal"] = _W["pocket"].distance_of_state(state)
    return out


def depth_case(args):
    """Сеть 3×3 на кубике, перемешанном ровно k случайными ходами."""
    k, i, max_nodes = args
    import random
    from game.ai import astar
    from game.ai.net import default_net
    from game.core import cube as C
    pz = C.Puzzle(3)
    rng = random.Random(SEED_BASE * 7 + k * 1000 + i)
    moves = [m for m in pz.random_moves(k, rng)]
    state = pz.apply(pz.solved(), moves)
    r = astar.solve(default_net(3), state, weight=WEIGHT, batch=BATCH, max_nodes=max_nodes)
    ok = r.solved and pz.is_solved(pz.apply(state, r.moves))
    return {"k": k, "solved": ok, "length": len(r.moves) if ok else None, "seconds": round(r.seconds, 3)}


def summary(rows, extra=None):
    done = [r for r in rows if r["solved"]]
    lens = [r["length"] for r in done]
    out = {"count": len(rows), "solved": len(done), "solved_share": round(len(done) / max(1, len(rows)), 4),
           "mean_length": round(statistics.mean(lens), 2) if lens else None,
           "median_length": statistics.median(lens) if lens else None,
           "max_length": max(lens) if lens else None,
           "mean_seconds": round(statistics.mean(r["seconds"] for r in rows), 3),
           "lengths": [r["length"] for r in rows[:20]]}
    if extra:
        out.update(extra)
    return out


def write(section, data):
    try:
        with open(PASSPORT, encoding="utf-8") as f:
            passport = json.load(f)
    except (OSError, ValueError):
        passport = {}
    passport[section] = data
    os.makedirs(os.path.dirname(PASSPORT), exist_ok=True)
    with open(PASSPORT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(passport, f, ensure_ascii=False, indent=1)


def run(fn, items, workers):
    t0 = time.perf_counter()
    if workers > 1:
        with mp.Pool(workers) as pool:
            rows = pool.map(fn, items, chunksize=1)
    else:
        rows = [fn(x) for x in items]
    return rows, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m game.train.passport")
    ap.add_argument("what", choices=("kociemba", "net3", "net2", "depth3", "trial"))
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--max-nodes", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    w = min(2, args.workers)
    machine = {"workers": w, "date": time.strftime("%Y-%m-%d")}

    if args.what == "trial":
        out = {}
        rows, sec = run(kociemba_case, range(10), 1)
        out["kociemba_sec_per_cube"] = round(sec / 10, 2)
        out["kociemba_1000_minutes"] = round(sec / 10 * 1000 / w / 60, 1)
        from game.ai.net import default_net
        for n in (2, 3):
            if default_net(n) is not None:
                rows, sec = run(net_case, [(n, i, args.max_nodes) for i in range(10)], 1)
                out[f"net{n}_sec_per_cube"] = round(sec / 10, 2)
                out[f"net{n}_1000_minutes"] = round(sec / 10 * 1000 / w / 60, 1)
        print(json.dumps(out, ensure_ascii=False))
        return 0
    if args.what == "kociemba":
        rows, sec = run(kociemba_case, range(args.count), w)
        firsts = [r["first"] for r in rows if r["first"]]
        data = summary(rows, {"node_limit": NODES, "seed_base": SEED_BASE, "wall_minutes": round(sec / 60, 1),
                              "mean_first_solution": round(statistics.mean(firsts), 2), **machine})
        write("kociemba3", data)
    elif args.what in ("net2", "net3"):
        n = int(args.what[-1])
        from game.ai.net import passport_path
        rows, sec = run(net_case, [(n, i, args.max_nodes) for i in range(args.count)], w)
        extra = {"max_nodes": args.max_nodes, "weight": WEIGHT, "batch": BATCH, "seed_base": SEED_BASE,
                 "wall_minutes": round(sec / 60, 1), **machine}
        try:
            with open(passport_path(n), encoding="utf-8") as f:
                meta = json.load(f)
            extra["train_hours"] = round(meta.get("train_seconds", 0) / 3600, 2)
            extra["train_steps"] = meta.get("steps")
        except (OSError, ValueError):
            pass
        if n == 2:
            done = [r for r in rows if r["solved"]]
            extra["mean_optimal"] = round(statistics.mean(r["optimal"] for r in rows), 2)
            extra["optimal_share"] = round(sum(r["length"] == r["optimal"] for r in done) / max(1, len(rows)), 4)
        write(f"net{n}", summary(rows, extra))
    elif args.what == "depth3":
        items = [(k, i, args.max_nodes) for k in (3, 5, 7, 10, 13, 16, 20) for i in range(args.count)]
        rows, sec = run(depth_case, items, w)
        table = {}
        for k in sorted({r["k"] for r in rows}):
            rk = [r for r in rows if r["k"] == k]
            table[str(k)] = {"solved_share": round(sum(r["solved"] for r in rk) / len(rk), 3),
                             "mean_length": round(statistics.mean(r["length"] for r in rk if r["solved"]), 2)
                             if any(r["solved"] for r in rk) else None}
        write("net3_by_depth", {"per_depth": args.count, "max_nodes": args.max_nodes, "table": table,
                                "wall_minutes": round(sec / 60, 1), **machine})
    print(json.dumps(json.load(open(PASSPORT, encoding="utf-8")), ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
