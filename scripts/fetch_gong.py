"""Fetch unlabelled GONG H-alpha images (external data for semi-supervised training).

The competition images are GONG H-alpha full-disk JPEGs (``YYYYMMDDhhmmss<site>h``); the
public NSO archive (https://gong2.nso.edu/HA/hag/) holds the same 2048 x 2048 images with
text overlays in the corners. This script samples days from the competition period that are at
least ``--margin-days`` + 1 days away from every competition (train *and* test) image, picks one
random image per day, removes the overlays (everything outside the inscribed circle is set to 0,
as in the competition images), keeps it only if its disk statistics lie within the range of the
training images, and stores it as JPEG.

No labels are downloaded; only the images. GONG data: NSO Integrated Synoptic Program (NISP).

Usage:
    python scripts/fetch_gong.py --out /kaggle/working/gong --n 1200
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.train import find_data_dir  # noqa: E402

BASE = "https://gong2.nso.edu/HA/hag"
NAME_RE = re.compile(r'href="(\d{14}[A-Z]h)\.jpg"')
# server-side filter of the day listing (Apache mod_autoindex): only images taken at minute 00, i.e. about one per
# hour and site -- a few KB instead of ~1 MB per day, which keeps the load on the public archive low
HOURLY = "?P=" + "%3F" * 10 + "00" + "%3F" * 3 + "h.jpg"
SITES = set("BCLMTU")  # the six GONG sites of the competition images
UA = "filseg-research/1.0 (solar filament segmentation; unlabelled images for semi-supervised training)"
SIZE = 2048


def competition_days(data_dir: Path) -> set[dt.date]:
    names = [p.stem for p in data_dir.rglob("*.jp*g")]
    return {dt.datetime.strptime(n[:8], "%Y%m%d").date() for n in names}


def candidate_days(used: set[dt.date], start: dt.date, end: dt.date, margin: int) -> list[dt.date]:
    """Days in [start, end] farther than ``margin`` days from every competition day."""
    blocked = {d + dt.timedelta(days=k) for d in used for k in range(-margin, margin + 1)}
    days, d = [], start
    while d <= end:
        if d not in blocked:
            days.append(d)
        d += dt.timedelta(days=1)
    return days


def disk_stats(img: np.ndarray) -> tuple[float, float, float]:
    """Mean, standard deviation and sharpness (variance of the Laplacian) inside the disk."""
    h = img.shape[0]
    yy, xx = np.ogrid[:h, :h]
    disk = (yy - (h - 1) / 2) ** 2 + (xx - (h - 1) / 2) ** 2 < (0.43 * h) ** 2
    f = img.astype(np.float32)
    lap = cv2.Laplacian(f, cv2.CV_32F)
    return float(f[disk].mean()), float(f[disk].std()), float(lap[disk].var())


def mask_outside(img: np.ndarray) -> np.ndarray:
    """Zero everything outside the inscribed circle (removes the archive's text overlays)."""
    h = img.shape[0]
    yy, xx = np.ogrid[:h, :h]
    out = img.copy()
    out[(yy - (h - 1) / 2) ** 2 + (xx - (h - 1) / 2) ** 2 > (h / 2) ** 2] = 0
    return out


def get(url: str, retries: int = 3) -> bytes:
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError("unreachable")


def fetch_day(day: dt.date, ranges: dict, out_dir: Path, seed: int, tries: int = 3) -> dict | None:
    rng = random.Random(f"{seed}-{day.isoformat()}")
    folder = f"{BASE}/{day:%Y%m}/{day:%Y%m%d}/"
    try:
        names = sorted(n for n in set(NAME_RE.findall(get(folder + HOURLY).decode("utf-8", "replace")))
                       if n[14] in SITES)
    except Exception:
        return None
    rng.shuffle(names)
    for name in names[:tries]:
        try:
            raw = np.frombuffer(get(folder + name + ".jpg"), np.uint8)
        except Exception:
            continue
        img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape != (SIZE, SIZE):
            continue
        img = mask_outside(img)
        m, s, sharp = disk_stats(img)
        ok = all(lo <= v <= hi for v, (lo, hi) in zip((m, s, sharp), (ranges["mean"], ranges["std"], ranges["sharp"])))
        if ok:
            cv2.imwrite(str(out_dir / f"{name}.jpeg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return dict(name=name, url=folder + name + ".jpg", date=day.isoformat(), site=name[14],
                        disk_mean=round(m, 2), disk_std=round(s, 2), sharpness=round(sharp, 2))
    return None


def reference_ranges(data_dir: Path, n: int, lo: float, hi: float, seed: int) -> dict:
    files = sorted((data_dir / "train").rglob("*.jp*g"))
    files = random.Random(seed).sample(files, min(n, len(files)))
    st = np.array([disk_stats(cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)) for f in files])
    q = np.percentile(st, [lo, hi], axis=0)
    return {k: (float(q[0, i]), float(q[1, i])) for i, k in enumerate(["mean", "std", "sharp"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="competition folder (train + test images; default: auto)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=1200, help="number of images to keep")
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default="2022-12-31")
    ap.add_argument("--margin-days", type=int, default=1, help="exclude days within this distance of competition days")
    ap.add_argument("--ref-images", type=int, default=200, help="training images used for the quality ranges")
    ap.add_argument("--ref-percentiles", type=float, nargs=2, default=[1.0, 99.0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--dry-run", action="store_true", help="only report the candidate days")
    args = ap.parse_args()

    data_dir = find_data_dir(args.data)
    used = competition_days(data_dir)
    days = candidate_days(used, dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end), args.margin_days)
    random.Random(args.seed).shuffle(days)
    print(f"{len(used)} competition days; {len(days)} candidate days at least {args.margin_days + 1} days away",
          flush=True)
    if args.dry_run:
        return
    ranges = reference_ranges(data_dir, args.ref_images, *args.ref_percentiles, args.seed)
    print("quality ranges (training images):", {k: tuple(round(x, 1) for x in v) for k, v in ranges.items()},
          flush=True)
    img_dir = args.out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    # checkpoint: accepted rows are appended to manifest.partial.csv after every batch; a restart keeps them
    partial = args.out / "manifest.partial.csv"
    fields = ["name", "url", "date", "site", "disk_mean", "disk_std", "sharpness"]
    rows = []
    if partial.exists():
        with open(partial, newline="") as f:
            rows = [r for r in csv.DictReader(f) if (img_dir / f"{r['name']}.jpeg").exists()]
        done_days = {r["date"] for r in rows}
        days = [d for d in days if d.isoformat() not in done_days]
        print(f"resuming with {len(rows)} images already fetched", flush=True)
    with open(partial, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    t0, pos, shown = time.time(), 0, len(rows)
    with ThreadPoolExecutor(args.workers) as ex:
        while len(rows) < args.n and pos < len(days):
            batch = days[pos:pos + 4 * args.workers]
            pos += len(batch)
            new = [r for r in ex.map(lambda d: fetch_day(d, ranges, img_dir, args.seed), batch) if r]
            rows += new
            with open(partial, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writerows(new)
            if len(rows) >= shown + 100 or len(rows) >= args.n or pos >= len(days):
                shown = len(rows)
                print(f"{len(rows)} images from {pos} days [{time.time() - t0:.0f}s]", flush=True)
    for extra in rows[args.n:]:
        (img_dir / f"{extra['name']}.jpeg").unlink(missing_ok=True)
    rows = rows[:args.n]
    if not rows:
        raise SystemExit("no image could be fetched")
    with open(args.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    partial.unlink()
    print(f"kept {len(rows)} images from {pos} candidate days in {time.time() - t0:.0f}s -> {img_dir}")


if __name__ == "__main__":
    main()
