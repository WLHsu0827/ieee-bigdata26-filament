"""Small helpers shared by the command-line scripts."""
from __future__ import annotations

import glob
from pathlib import Path


def expand(paths) -> list[Path]:
    """Expand glob patterns (``**`` allowed) without relying on the shell; keep order, drop duplicates."""
    out: list[Path] = []
    for p in paths:
        p = str(p)
        hits = sorted(glob.glob(p, recursive=True)) if any(ch in p for ch in "*?[") else [p]
        if not hits:
            raise FileNotFoundError(f"no files match {p}")
        out += [Path(h) for h in hits if Path(h) not in out]
    return out


def find_test_dir(hint: str | None = None) -> Path:
    """The competition's test image folder (``hint`` or found under /kaggle/input)."""
    if hint:
        return Path(hint)
    for p in Path("/kaggle/input").rglob("test_images"):
        if p.is_dir():
            return p
    raise FileNotFoundError("test_images folder not found; pass --test-dir")
