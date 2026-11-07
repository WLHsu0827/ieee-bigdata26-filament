"""Verify the external (unlabelled) GONG images used for semi-supervised training.

1. Date gap: every external image must be taken at least ``--min-gap-days`` days away from every
   competition image (train and test; dates from the ``YYYYMMDDhhmmss<site>h`` file names).
2. Hashes (optional): SHA-256 of the original archive files (downloaded again from the manifest
   URLs, ``--originals``) and of the processed files actually used (``--images`` folder, e.g. the
   unzipped private dataset). With both, every original is re-processed exactly as in
   ``fetch_gong.py`` and must reproduce the processed file byte for byte.

``--write`` stores the hashes in the manifest. Exit code 1 if any check fails.

Usage:
    python scripts/check_external.py --manifest docs/external_gong_manifest.csv --data MAGFiLO_1.0_Kaggle_2026 \
        [--images gong/images --originals --write]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_gong import competition_days, get, mask_outside  # noqa: E402

from filseg.train import find_data_dir  # noqa: E402


def min_gaps(dates: list[dt.date], used: set[dt.date]) -> list[int]:
    """Smallest distance in days from each date to any date of ``used``."""
    ref = np.array(sorted(d.toordinal() for d in used))
    out = []
    for d in dates:
        k = np.searchsorted(ref, d.toordinal())
        cand = [abs(int(ref[j]) - d.toordinal()) for j in (k - 1, k) if 0 <= j < len(ref)]
        out.append(min(cand))
    return out


def process(raw: bytes) -> bytes:
    """The exact transformation of fetch_gong.py: grey, outside of the inscribed circle zeroed, JPEG q95."""
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    ok, buf = cv2.imencode(".jpeg", mask_outside(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=Path("docs/external_gong_manifest.csv"))
    ap.add_argument("--data", default=None, help="competition folder (default: $FILSEG_DATA or /kaggle/input)")
    ap.add_argument("--min-gap-days", type=int, default=2)
    ap.add_argument("--images", type=Path, default=None, help="folder with the processed <name>.jpeg files to hash")
    ap.add_argument("--originals", action="store_true", help="download the original archive files and hash them")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--write", action="store_true", help="store the hashes in the manifest")
    args = ap.parse_args()

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    used = competition_days(find_data_dir(args.data))
    dates = [dt.datetime.strptime(r["name"][:8], "%Y%m%d").date() for r in rows]
    gaps = min_gaps(dates, used)
    bad = [(r["name"], g) for r, g in zip(rows, gaps) if g < args.min_gap_days]
    dup = len({r["name"] for r in rows}) != len(rows)
    print(f"{len(rows)} external images, {len(used)} competition days; smallest gap {min(gaps)} days "
          f"(required >= {args.min_gap_days}); violations: {len(bad)}; duplicate names: {dup}")
    failed = bool(bad) or dup

    if args.images or args.originals:
        def work(r):
            out = {}
            raw = None
            if args.originals:
                raw = get(r["url"])
                out["sha256_original"] = hashlib.sha256(raw).hexdigest()
            if args.images:
                data = (args.images / f"{r['name']}.jpeg").read_bytes()
                out["sha256_processed"] = hashlib.sha256(data).hexdigest()
                if raw is not None:
                    out["reproduced"] = hashlib.sha256(process(raw)).hexdigest() == out["sha256_processed"]
            return out

        with ThreadPoolExecutor(args.workers) as ex:
            results = list(ex.map(work, rows))
        for r, res in zip(rows, results):
            r.update({k: v for k, v in res.items() if k != "reproduced"})
        if args.images and args.originals:
            n_ok = sum(res["reproduced"] for res in results)
            print(f"re-processing the originals reproduces {n_ok} / {len(rows)} processed files byte for byte")
            failed |= n_ok != len(rows)
        if args.write:
            fields = list(rows[0])
            with open(args.manifest, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
            print(f"hashes written to {args.manifest}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
