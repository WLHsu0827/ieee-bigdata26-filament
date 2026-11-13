"""Check the committed weights without a GPU, competition data or network access.

Verifies the SHA-256 of every weight file listed in ``weights/manifest.json``, loads all networks
on the CPU with network access blocked (so inference downloads nothing, e.g. no pretrained encoder
weights), runs a small forward pass through each, and applies every post-processing configuration
of the manifest (``full`` with the learned filter, ``fast`` and the ``rule`` variant) to a synthetic
probability map. Takes about a minute.

Usage:
    python scripts/check_weights.py
"""
from __future__ import annotations

import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def block_network() -> None:
    """Make every socket connection of this process raise, so nothing can be downloaded during the check."""
    def blocked(*args, **kwargs):
        raise RuntimeError("network access attempted")

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked


def synthetic_map(size: int = 2048) -> np.ndarray:
    """Probability map with four confident, elongated filament-like ridges (values in [0, 1]); every
    post-processing configuration keeps the three larger ones."""
    import cv2

    canvas = np.zeros((size, size), np.float32)
    for (x0, y0, x1, y1, width) in [(600, 500, 900, 620, 18), (1200, 1300, 1500, 1250, 12),
                                    (800, 1500, 860, 1700, 9), (1400, 600, 1440, 610, 6)]:
        cv2.line(canvas, (x0, y0), (x1, y1), 1.0, width)
    return np.clip(cv2.GaussianBlur(canvas, (0, 0), 1.0), 0.0, 1.0)


def main() -> None:
    block_network()
    import torch

    from filseg.infer import load_models
    from filseg.postprocess import PostConfig, instances_from_prob

    t0 = time.time()
    manifest = json.loads((ROOT / "weights" / "manifest.json").read_text())
    paths = []
    for entry in manifest["models"]:
        path = ROOT / "weights" / entry["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise SystemExit(f"SHA-256 mismatch for {path.name}: the file differs from the manifest")
        paths.append(path)
    print(f"{len(paths)} weight files match the SHA-256 in weights/manifest.json")

    models = load_models(paths, torch.device("cpu"))
    x = torch.zeros(1, 1, 256, 256)
    with torch.no_grad():
        for (net, _, _), path in zip(models, paths, strict=True):
            out = net(x)
            if out.shape[-2:] != x.shape[-2:] or not torch.isfinite(out).all():
                raise SystemExit(f"unexpected output of {path.name}: {tuple(out.shape)}")
    print(f"{len(models)} networks loaded and run on the CPU without network access "
          f"(input resolution / normalisation: {sorted({(r, n) for _, r, n in models})})")

    prob = synthetic_map()
    configs = {"full": manifest["post"], "fast": manifest["fast"]["post"], **manifest.get("variants", {})}
    for name, spec in configs.items():
        lab, stats = instances_from_prob(prob, PostConfig(**spec))
        labels = sorted(set(np.unique(lab).tolist()) - {0})
        if not stats or labels != list(range(1, len(stats) + 1)):
            raise SystemExit(f"post-processing {name!r}: no instance kept or inconsistent instance labels")
        kind = "learned filter" if spec.get("scorer") else "rule"
        print(f"post-processing {name!r} ({kind}): {len(stats)} of 4 synthetic ridges kept")
    print(f"all checks passed in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
