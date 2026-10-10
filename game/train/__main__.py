"""Обучение сети по команде (итерация ценности, game/train/davi.py):

    python -m game.train --size 3 --trial                 # пробный прогон и прогноз времени
    python -m game.train --size 3 --steps 20000           # с нуля
    python -m game.train --size 3 --steps 20000 --resume  # продолжить со снимка
    python -m game.train --size 3 --publish               # сеть из снимка -> game/models

Потоки BLAS ограничены (--threads, по умолчанию 2): ПК делят и другие задачи. Журнал и
снимок - в --run (runs/cube3).
"""
import argparse
import json
import os
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m game.train")
    ap.add_argument("--size", type=int, choices=(2, 3), default=3)
    ap.add_argument("--run")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--kmax", type=int)
    ap.add_argument("--hidden", default="512,512,256")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sync", type=int, default=500, help="обновить цель не реже, чем раз в столько шагов")
    ap.add_argument("--sync-loss", type=float, default=0.05, help="обновить цель, когда потеря ниже")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--trial", type=int, nargs="?", const=60, metavar="ШАГОВ",
                    help="пробный прогон: столько шагов во временной папке и прогноз на --steps")
    ap.add_argument("--publish", action="store_true", help="сеть из снимка в game/models")
    args = ap.parse_args(argv)

    from aikit.trainlog import limit_threads
    limit_threads(args.threads, force=True)          # до импорта numpy

    from game.ai.net import model_path
    from game.train.davi import Trainer
    hidden = tuple(int(x) for x in args.hidden.split(","))
    run = args.run or os.path.join("runs", f"cube{args.size}")

    if args.trial:
        import tempfile
        tmp = tempfile.mkdtemp(prefix=f"cube{args.size}-trial-")
        t = Trainer(tmp, args.size, hidden, args.batch, args.kmax, args.lr, args.sync, log=False,
                    sync_loss=args.sync_loss)
        t.step()                                        # первый шаг - разогрев
        t0 = time.perf_counter()
        for _ in range(args.trial):
            loss = t.step()
        per_step = (time.perf_counter() - t0) / args.trial
        e0 = time.perf_counter()
        t.evaluate(per_k=100)
        eval_s = time.perf_counter() - e0
        evals = args.steps // args.eval_every if args.eval_every else 0
        total = per_step * args.steps + eval_s * 2 * evals
        print(json.dumps({"size": args.size, "hidden": hidden, "batch": args.batch, "threads": args.threads,
                          "sec_per_step": round(per_step, 3), "last_loss": round(float(loss), 3),
                          "eval_sec": round(eval_s * 2, 1), "steps": args.steps,
                          "forecast_hours": round(total / 3600, 2), "trial_dir": tmp}, ensure_ascii=False))
        return 0

    t = Trainer(run, args.size, hidden, args.batch, args.kmax, args.lr, args.sync, sync_loss=args.sync_loss)
    if args.publish:
        if not t.load_snapshot():
            print("нет снимка", file=sys.stderr)
            return 1
        meta = {"size": args.size, "hidden": list(hidden), "steps": t.step_no, "batch": t.batch,
                "syncs": t.syncs,
                "kmax": t.kmax, "train_seconds": round(t.train_seconds), "history": t.history}
        t.net.save(model_path(args.size), meta)
        print(f"сеть шага {t.step_no} -> {model_path(args.size)}")
        return 0
    if args.resume and t.load_snapshot():
        t.log(f"продолжаю с шага {t.step_no}")
    else:
        t.log(f"с нуля: {args.size}x{args.size}, слои {hidden}, пачка {args.batch}, kmax {t.kmax}, "
              f"потоков {args.threads}")
    left = max(0, args.steps - t.step_no)
    t.train(left, eval_every=args.eval_every)
    t.log(f"готово: шаг {t.step_no}, {t.train_seconds / 3600:.2f} ч счёта")
    return 0


if __name__ == "__main__":
    sys.exit(main())
