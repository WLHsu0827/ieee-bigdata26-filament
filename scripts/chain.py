"""Run several Python command lines one after another, stopping at the first failure.

Lets one Kaggle worker process (one GPU) run a short pipeline, e.g. pseudo-labelling and then
training:
    python scripts/chain.py scripts/pseudo_label.py ... ++ -m filseg.train ...

With ``--keep-going`` as the first argument every command runs even if an earlier one failed
(independent experiments sharing a GPU); the exit code is then the largest one.
"""
from __future__ import annotations

import subprocess
import sys


def split(argv: list[str]) -> list[list[str]]:
    """Split the arguments at ``++`` into command lines (empty ones dropped)."""
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
    argv = sys.argv[1:]
    keep_going = bool(argv) and argv[0] == "--keep-going"
    worst = 0
    for cmd in split(argv[1:] if keep_going else argv):
        print("chain:", " ".join(cmd), flush=True)
        rc = subprocess.run([sys.executable] + cmd).returncode
        if rc and not keep_going:
            sys.exit(rc)
        if rc:
            print(f"chain: exit code {rc}, continuing (--keep-going)", flush=True)
        worst = max(worst, rc)
    sys.exit(worst)


if __name__ == "__main__":
    main()
