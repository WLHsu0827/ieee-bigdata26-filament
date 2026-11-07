"""Run several Python command lines one after another, stopping at the first failure.

Lets one Kaggle worker process (one GPU) run a short pipeline, e.g. pseudo-labelling and then
training:
    python scripts/chain.py scripts/pseudo_label.py ... ++ -m filseg.train ...
"""
from __future__ import annotations

import subprocess
import sys


def split(argv: list[str]) -> list[list[str]]:
    cmds, cur = [], []
    for a in argv:
        if a == "++":
            cmds.append(cur)
            cur = []
        else:
            cur.append(a)
    cmds.append(cur)
    return [c for c in cmds if c]


def main() -> None:
    for cmd in split(sys.argv[1:]):
        print("chain:", " ".join(cmd), flush=True)
        rc = subprocess.run([sys.executable] + cmd).returncode
        if rc:
            sys.exit(rc)


if __name__ == "__main__":
    main()
