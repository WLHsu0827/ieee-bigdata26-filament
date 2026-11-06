"""Validate a submission CSV before uploading (see :func:`filseg.submission.check`).

Usage:
    python scripts/check_submission.py submission.csv [--test-dir <MAGFiLO>/test/test_images]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.submission import check  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--test-dir", type=Path, default=None)
    args = ap.parse_args()
    test_ids = {p.stem for p in args.test_dir.glob("*.jp*g")} if args.test_dir else None
    rep = check(pd.read_csv(args.csv, dtype=str, keep_default_na=False), test_ids)
    for k, v in rep.items():
        print(f"{k}: {v}")
    sys.exit(1 if rep["problems"] else 0)


if __name__ == "__main__":
    main()
