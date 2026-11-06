"""Run :mod:`filseg.train` for several folds one after another (e.g. on one GPU).

Each fold is trained in its own subprocess (clean GPU memory) with output in
``<out>/fold<k>`` (``fold-1`` = all images, no validation).

Usage:
    python -m filseg.cv --folds 1 3 --out /kaggle/working/r34_1024 -- --res 1024 --encoder resnet34 ...
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-hours", type=float, default=11.0, help="total budget, split over the remaining folds")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("train_args", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    train_args = args.train_args[1:] if args.train_args[:1] == ["--"] else args.train_args
    if "--max-hours" in train_args:
        raise SystemExit("pass --max-hours to filseg.cv, not to the training arguments")

    t0 = time.time()
    codes = []
    for i, fold in enumerate(args.folds):
        left = args.max_hours - (time.time() - t0) / 3600
        per_fold = max(left / (len(args.folds) - i), 0.05)
        cmd = [sys.executable, "-m", "filseg.train", "--fold", str(fold), "--out", str(args.out / f"fold{fold}"),
               "--max-hours", f"{per_fold:.3f}", *train_args]
        print(f"[cv] fold {fold}: budget {per_fold:.2f} h -> {' '.join(cmd[1:])}", flush=True)
        if args.dry_run:
            codes.append(0)
            continue
        codes.append(subprocess.run(cmd).returncode)
        print(f"[cv] fold {fold} finished with code {codes[-1]} after {(time.time() - t0) / 3600:.2f} h", flush=True)
    if not args.dry_run:
        sys.exit(max(codes))


if __name__ == "__main__":
    main()
