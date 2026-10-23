"""Train the learned detector (adversary level A3) on institutional labels from background runs.

    python -m scripts.train_detector                         # 40 train + 8 validation seeds, 30 epochs
    python -m scripts.train_detector --quick                 # smoke test, seconds
    python -m scripts.train_detector --epochs 60 --lr 0.01 --preset A1-strict

The classifier never sees an executor: it learns what institutional metaorders look like in public
data. One line per epoch: training loss, validation ROC AUC for the buy class, the sell class and
the unsigned signal. Weights go to outputs/detectors/learned.npz (loaded by the E2/E3 scripts).
"""

import argparse
import time

import numpy as np

from order_book.detectors.learned import collect_dataset, train
from order_book.experiments import ADVERSARIES
from scripts import OUTPUTS

WEIGHTS_PATH = OUTPUTS / "detectors" / "learned.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", type=int, default=40, help="training seeds 100..100+n-1")
    parser.add_argument("--val-seeds", type=int, default=8, help="validation seeds 500..500+n-1")
    parser.add_argument("--steps", type=int, default=20_000, help="steps recorded per seed after burn-in")
    parser.add_argument("--burn-in", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--preset", choices=sorted(ADVERSARIES), default="A1-medium",
                        help="market the data is collected in (the makers' reaction is part of what is learnt)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=str(WEIGHTS_PATH))
    args = parser.parse_args()
    if args.quick:
        args.train_seeds, args.val_seeds, args.steps, args.burn_in, args.epochs = 2, 1, 3_000, 1_000, 3
    cfg = ADVERSARIES[args.preset].config()
    t0 = time.perf_counter()
    print(f"[train_detector] collecting {args.train_seeds} training seeds x {args.steps} steps ({args.preset})")
    x, y = collect_dataset(range(100, 100 + args.train_seeds), args.steps, cfg, args.burn_in)
    print(f"[train_detector] collecting {args.val_seeds} validation seeds")
    xv, yv = collect_dataset(range(500, 500 + args.val_seeds), args.steps, cfg, args.burn_in)
    counts = np.bincount(y, minlength=3)
    print(f"[train_detector] {len(y)} samples; class shares none/buy/sell = "
          f"{counts[0] / len(y):.2f}/{counts[1] / len(y):.2f}/{counts[2] / len(y):.2f}; "
          f"data in {time.perf_counter() - t0:.0f}s")
    params = train(x, y, xv, yv, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, l2=args.l2)
    WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    params.save(args.out)
    print(f"[train_detector] saved {args.out}; feature weights (buy - sell column):")
    for name, w in zip(params.feature_names, params.weights[:, 1] - params.weights[:, 2]):
        print(f"    {name:<24s} {w:+.3f}")


if __name__ == "__main__":
    main()
