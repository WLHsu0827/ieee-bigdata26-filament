"""Train a U-Net on MAGFiLO soft targets, tune post-processing on a hold-out
fold, and (optionally) write a submission for the test images.

Example (hold-out fold 0, 512 px):
    python -m filseg.train --data /path/MAGFiLO_1.0_Kaggle_2026 --out runs/r34_512 \
        --res 512 --encoder resnet34 --epochs 25 --bs 8 --fold 0 --predict-test

Training on all images (no validation) with a fixed post-processing config:
    python -m filseg.train ... --fold -1 --post runs/r34_512/post.json --predict-test
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import augment, data, submission
from .metrics import Crop
from .model import MEAN, STD, build_model, predict_images, save_checkpoint, to_tensor, upsample
from .postprocess import PostConfig, instances_from_prob
from .tune import grid_search, score


# ----------------------------------------------------------------------------- data
SMALL_REF_AREA = 1200.0   # median filament area (px at 2048)
SMALL_MAX_WEIGHT = 4.0


def small_instance_weights(entries: list[data.Entry], gamma: float, margin: int = 6) -> np.ndarray:
    """Pixel weights in [1, SMALL_MAX_WEIGHT] emphasising small filaments (full resolution).

    Every annotated filament of area ``A`` gets weight ``clip((A_ref / A) ** gamma, 1, max)`` on its
    pixels and a ``margin``-px band around them; overlapping weights take the maximum.
    """
    from scipy import ndimage as ndi

    wmap = np.ones((data.IMAGE_SIZE, data.IMAGE_SIZE), np.float32)
    k = np.ones((2 * margin + 1, 2 * margin + 1), bool)
    for e in entries:
        for m in e.masks():
            area = int(m.sum())
            w = float(np.clip((SMALL_REF_AREA / max(area, 1)) ** gamma, 1.0, SMALL_MAX_WEIGHT))
            if w <= 1.0:
                continue
            ys, xs = np.nonzero(m)
            y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, data.IMAGE_SIZE)
            x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, data.IMAGE_SIZE)
            region = ndi.binary_dilation(m[y0:y1, x0:x1], structure=k)
            sub = wmap[y0:y1, x0:x1]
            sub[region] = np.maximum(sub[region], w)
    return wmap


def _prepare_one(args):
    path, entries, res, gamma, norm = args
    from .model import image_stats, resize_image
    img = data.read_image(path)
    tgt = np.round(data.soft_target(entries, size=res) * 255).astype(np.uint8)
    wmap = None
    if gamma > 0:
        w = resize_image(small_instance_weights(entries, gamma), res)
        wmap = np.round((w - 1.0) / (SMALL_MAX_WEIGHT - 1.0) * 255).astype(np.uint8)
    return resize_image(img, res), tgt, wmap, image_stats(img, norm)


def prepare_arrays(data_dir: Path, groups: dict[str, list[data.Entry]], res: int, workers: int,
                   small_weight: float = 0.0, norm: str = "global"):
    """Images and soft targets at ``res`` (uint8), optional weight maps, per-image normalisation stats."""
    jobs = [(data_dir / "train" / "train_images" / es[0].file_name, es, res, small_weight, norm)
            for es in groups.values()]
    n = len(jobs)
    # preallocate and fill in place: peak memory stays ~1x the arrays even at 2048 px
    imgs = np.empty((n, res, res), np.uint8)
    tgts = np.empty((n, res, res), np.uint8)
    wmaps = np.empty((n, res, res), np.uint8) if small_weight > 0 else None
    stats = np.empty((n, 2), np.float32)
    pool = Pool(workers) if workers > 0 else None
    try:
        it = pool.imap(_prepare_one, jobs, chunksize=2) if pool else map(_prepare_one, jobs)
        for k, (img, tgt, wmap, st) in enumerate(it):
            imgs[k], tgts[k], stats[k] = img, tgt, st
            if wmaps is not None:
                wmaps[k] = wmap
    finally:
        if pool:
            pool.close()
            pool.join()
    return imgs, tgts, wmaps, stats


CACHE_VERSION = 1


def cached_arrays(cache_dir: Path | None, data_dir: Path, groups: dict[str, list[data.Entry]], res: int,
                  workers: int, small_weight: float = 0.0, norm: str = "global", log=print):
    """:func:`prepare_arrays` with an optional on-disk cache of uint8 ``.npy`` files.

    Several folds (and concurrent processes on one machine) need identical arrays. The
    first run writes them (each file atomically); later runs memory-map them read-only,
    which skips the CPU preparation and lets concurrent processes share one page-cache copy.
    """
    if cache_dir is None:
        return prepare_arrays(data_dir, groups, res, workers, small_weight, norm)
    import hashlib

    key = hashlib.sha1(json.dumps([CACHE_VERSION, res, norm, small_weight, list(groups)]).encode()).hexdigest()[:16]
    d = Path(cache_dir) / f"arrays_{res}_{norm}_{key}"
    names = ["imgs", "tgts", "stats"] + (["wmaps"] if small_weight > 0 else [])
    if all((d / f"{n}.npy").exists() for n in names):
        log(f"loading prepared arrays from {d}")
        arr = {n: np.load(d / f"{n}.npy", mmap_mode="r") for n in names}
        return arr["imgs"], arr["tgts"], arr.get("wmaps"), np.array(arr["stats"])
    imgs, tgts, wmaps, stats = prepare_arrays(data_dir, groups, res, workers, small_weight, norm)
    tmp = None
    try:
        d.mkdir(parents=True, exist_ok=True)
        for n, a in (("imgs", imgs), ("tgts", tgts), ("wmaps", wmaps), ("stats", stats)):
            if a is not None:
                tmp = d / f"{n}.{os.getpid()}.tmp.npy"
                np.save(tmp, a)
                os.replace(tmp, d / f"{n}.npy")
        log(f"prepared arrays cached in {d}")
    except OSError as e:  # e.g. disk full: training continues without the cache
        log(f"could not write the array cache ({e})")
        if tmp is not None and tmp.exists():
            tmp.unlink()
        return imgs, tgts, wmaps, stats
    # continue with the memory-mapped copies: frees this process's private copy, so concurrent
    # processes on one machine share a single page-cache copy of the arrays
    del imgs, tgts, wmaps
    arr = {n: np.load(d / f"{n}.npy", mmap_mode="r") for n in names}
    return arr["imgs"], arr["tgts"], arr.get("wmaps"), np.array(arr["stats"])


# ----------------------------------------------------------------------------- unlabelled images
def extra_pairs(patterns, target_dir: Path) -> list[tuple[Path, Path]]:
    """Unlabelled images (files / glob patterns) that have a teacher map ``<target_dir>/<stem>.png``."""
    import glob

    files: list[str] = []
    for p in patterns:
        p = str(p)
        files += sorted(glob.glob(p, recursive=True)) if any(c in p for c in "*?[") else [p]
    pairs = [(Path(f), Path(target_dir) / f"{Path(f).stem}.png") for f in dict.fromkeys(files)]
    return [(f, t) for f, t in pairs if t.exists()]


def _read_extra_image(args):
    from .model import image_stats, resize_image
    path, res, norm = args
    img = data.read_image(path)
    return resize_image(img, res), image_stats(img, norm)


def _read_extra_target(args):
    import cv2

    from .model import resize_image
    path, res = args
    tgt = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if tgt is None:
        raise FileNotFoundError(path)
    return resize_image(tgt, res)


def _locked_memmaps(d: Path, specs: dict, fill, owner: bool | None = None, log=print,
                    wait_hours: float = 3.0) -> dict[str, np.ndarray]:
    """Create the ``.npy`` arrays ``specs`` (name -> (dtype, shape)) in ``d`` once and memory-map them.

    The first process takes ``d/lock`` and writes the arrays through ``fill(arrays)`` straight into
    memory-mapped files (no in-memory copy); concurrent processes wait for ``d/done``.
    """
    d.mkdir(parents=True, exist_ok=True)
    done = d / "done"
    if not done.exists():
        if owner is None:
            try:
                os.close(os.open(d / "lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                owner = True
            except FileExistsError:
                owner = False
        if owner:
            arrs = {n: np.lib.format.open_memmap(d / f"{n}.npy", "w+", dt, shp) for n, (dt, shp) in specs.items()}
            fill(arrs)
            for a in arrs.values():
                a.flush()
            del arrs
            done.write_text("ok")
        else:
            log(f"waiting for another process to prepare {d}")
            deadline = time.time() + wait_hours * 3600
            while not done.exists():
                if time.time() > deadline:
                    raise TimeoutError(f"{d} was not completed by the process holding its lock")
                time.sleep(2)
    return {n: np.load(d / f"{n}.npy", mmap_mode="r") for n in specs}


def cached_extra(cache_dir: Path | None, pairs: list[tuple[Path, Path]], res: int, workers: int,
                 norm: str = "global", log=print, wait_hours: float = 3.0):
    """Images, teacher targets (uint8) and normalisation statistics of unlabelled images.

    Returns ``(imgs, tgts, stats, target_dir)``. The arrays live in memory-mapped ``.npy`` files, so
    thousands of 2048 px images fit. With ``cache_dir`` set, the images (and their statistics) are
    shared by every run on the same image list, the targets by runs with the same teacher maps, and
    one process prepares each set under a lock while concurrent processes (other GPUs) wait.
    ``target_dir`` can be deleted after training (``--extra-cleanup``) to free disk space.
    """
    import hashlib
    import shutil
    import tempfile

    def run(fn, jobs, put):
        pool = Pool(workers) if workers > 0 else None
        try:
            for k, out in enumerate(pool.imap(fn, jobs, chunksize=2) if pool else map(fn, jobs)):
                put(k, out)
        finally:
            if pool:
                pool.close()
                pool.join()

    n = len(pairs)
    if cache_dir is None:
        root, owner = Path(tempfile.mkdtemp(prefix="filseg_extra_")), True
    else:
        root, owner = Path(cache_dir), None
    key_img = hashlib.sha1(json.dumps([CACHE_VERSION, res, norm, [f.name for f, _ in pairs]]).encode()).hexdigest()[:16]
    key_tgt = hashlib.sha1(json.dumps([CACHE_VERSION, res, [str(t) for _, t in pairs]]).encode()).hexdigest()[:16]
    d_img, d_tgt = root / f"extra_img_{res}_{norm}_{key_img}", root / f"extra_tgt_{res}_{key_tgt}"
    root.mkdir(parents=True, exist_ok=True)
    log(f"unlabelled arrays: {n} images in {d_img}, targets in {d_tgt} (free disk {shutil.disk_usage(root).free / 1e9:.0f} GB)")

    def fill_img(a):
        def put(k, out):
            a["imgs"][k], a["stats"][k] = out
        run(_read_extra_image, [(f, res, norm) for f, _ in pairs], put)

    def fill_tgt(a):
        def put(k, out):
            a["tgts"][k] = out
        run(_read_extra_target, [(t, res) for _, t in pairs], put)

    ia = _locked_memmaps(d_img, {"imgs": (np.uint8, (n, res, res)), "stats": (np.float32, (n, 2))}, fill_img,
                         owner, log, wait_hours)
    ta = _locked_memmaps(d_tgt, {"tgts": (np.uint8, (n, res, res))}, fill_tgt, owner, log, wait_hours)
    return ia["imgs"], ta["tgts"], np.array(ia["stats"]), d_tgt

def _worker_init(_):
    import cv2

    cv2.setNumThreads(1)  # one OpenCV thread per data-loader worker (avoids oversubscription)


class TrainSet(Dataset):
    """Training samples of the images ``idx`` (indices into the full arrays, so nothing is copied)."""

    def __init__(self, imgs: np.ndarray, tgts: np.ndarray, crop: int, epoch_mult: int = 1,
                 wmaps: np.ndarray | None = None, stats: np.ndarray | None = None,
                 idx: np.ndarray | None = None):
        self.imgs, self.tgts, self.crop, self.mult, self.wmaps = imgs, tgts, crop, epoch_mult, wmaps
        self.stats = stats
        self.idx = np.arange(len(imgs)) if idx is None else np.asarray(idx)

    def __len__(self):
        return len(self.idx) * self.mult

    def __getitem__(self, i):
        rng = np.random.default_rng(torch.randint(0, 2**31, (1,)).item())
        i = int(self.idx[i % len(self.idx)])
        arrays = [self.imgs[i], self.tgts[i]] + ([self.wmaps[i]] if self.wmaps is not None else [])
        if self.crop < arrays[0].shape[0]:
            arrays = augment.crop_geometric(arrays, self.crop, rng, focus=1)
        else:
            arrays = augment.geometric(arrays, rng)
        img = augment.photometric(arrays[0].astype(np.float32) / 255.0, rng)
        tgt = np.stack([a.astype(np.float32) / 255.0 for a in arrays[1:]])
        st = tuple(self.stats[i]) if self.stats is not None else (MEAN, STD)
        return to_tensor(img, st), torch.from_numpy(tgt)


def ema_with_warmup(decay: float):
    """``multi_avg_fn`` for ``AveragedModel``: EMA whose decay ramps up as ``(1 + n) / (10 + n)``.

    Without the ramp, an EMA with decay 0.999 keeps a weight of 0.999**n on the
    (randomly initialised decoder of the) model at the first step, which makes
    early validation meaningless and can still bias short runs.
    """
    @torch.no_grad()
    def fn(ema_params, cur_params, num_averaged):
        n = float(num_averaged)
        if not (torch.is_floating_point(ema_params[0]) or torch.is_complex(ema_params[0])):
            torch._foreach_copy_(ema_params, cur_params)  # integer buffers, e.g. BatchNorm num_batches_tracked
            return
        torch._foreach_lerp_(ema_params, cur_params, 1.0 - min(decay, (1.0 + n) / (10.0 + n)))

    return fn


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE + global soft Dice. ``target`` is (B, 1, H, W) or (B, 2, H, W) with a weight channel."""
    logits = logits.float()
    t = target[:, :1]
    if target.shape[1] > 1:
        w = 1.0 + (SMALL_MAX_WEIGHT - 1.0) * target[:, 1:2]
        bce = (F.binary_cross_entropy_with_logits(logits, t, reduction="none") * w).sum() / w.sum()
    else:
        bce = F.binary_cross_entropy_with_logits(logits, t)
    p = torch.sigmoid(logits)
    inter = (p * t).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)
    return bce + dice


# ----------------------------------------------------------------------------- io
def find_data_dir(hint: str | None) -> Path:
    """The MAGFiLO folder: ``hint``, else ``$FILSEG_DATA``, else found under /kaggle/input."""
    if hint:
        return Path(hint)
    if os.environ.get("FILSEG_DATA"):
        return Path(os.environ["FILSEG_DATA"])
    for p in Path("/kaggle/input").rglob(data.TRAIN_JSON):
        return p.parent.parent
    raise FileNotFoundError("MAGFiLO data directory not found; pass --data or set FILSEG_DATA")


def write_submission(models, data_dir: Path, res: int, cfg: PostConfig, device, out_csv: Path,
                     tta: int, batch: int, log=print, norm: str = "global", limit: int = 0) -> pd.DataFrame:
    """Predict the test images, post-process, write the CSV and log per-image inference time.

    ``limit`` > 0 predicts only the first ``limit`` test images (debugging).
    """
    test_dir = data_dir / "test" / "test_images"
    files = sorted(test_dir.glob("*.jp*g"))
    if limit:
        files = files[:limit]
    rows = []
    test_probs = {}  # uint8 probability maps at the network resolution, for re-post-processing later
    t_read = t_model = t_post = 0.0
    for s in range(0, len(files), 16):
        chunk = files[s:s + 16]
        t0 = time.time()
        imgs = [data.read_image(f) for f in chunk]
        t1 = time.time()
        probs = predict_images(models, imgs, res, device, tta, batch, norm=norm)
        t2 = time.time()
        for f, p in zip(chunk, probs):
            test_probs[f.stem] = np.round(p.astype(np.float32) * 255).astype(np.uint8)
            lab, _ = instances_from_prob(upsample(p), cfg)
            rows += submission.rows_from_label(f.stem, lab)
        t_read, t_model, t_post = t_read + t1 - t0, t_model + t2 - t1, t_post + time.time() - t2
    df, report = submission.write(rows, out_csv, {f.stem for f in files})
    n = max(len(files), 1)
    timing = {"device": str(device), "tta": tta, "res": res, "n_models": len(models), "images": len(files),
              "read_s_per_image": t_read / n, "model_s_per_image": t_model / n, "post_s_per_image": t_post / n}
    (out_csv.parent / "timing.json").write_text(json.dumps(timing, indent=1))
    np.savez_compressed(out_csv.parent / "test_probs.npz", **test_probs)
    log(f"submission: {len(files)} images, {len(df)} filaments -> {out_csv}; check: {report}")
    log(f"inference time per image: read {t_read / n:.3f}s + model {t_model / n:.3f}s + post/encode {t_post / n:.3f}s"
        f" ({device}, {len(models)} model(s), tta={tta}, res={res})")
    return df


# ----------------------------------------------------------------------------- main
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--res", type=int, default=512, help="training/inference resolution (full image)")
    ap.add_argument("--crop", type=int, default=0, help="random crop size at --res (0 = full image)")
    ap.add_argument("--epoch-mult", type=int, default=1, help="samples per image per epoch (useful with --crop)")
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--norm", default="global", choices=["global", "disk"],
                    help="input normalisation: fixed constants or per-image statistics inside the solar disk")
    ap.add_argument("--weights", default="imagenet", help="encoder pre-training (\"none\" = random init)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--infer-bs", type=int, default=0, help="inference batch size (0 = --bs)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--fold", type=int, default=0, help="hold-out fold, -1 = train on everything")
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--tune-workers", type=int, default=0, help="processes for post-processing scoring (0 = all CPUs)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--post", type=Path, default=None, help="post-processing JSON (skips tuning)")
    ap.add_argument("--predict-test", action="store_true")
    ap.add_argument("--max-hours", type=float, default=11.0, help="stop training early to fit the time limit")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0 = off)")
    ap.add_argument("--small-weight", type=float, default=0.0,
                    help="gamma of the small-filament BCE weighting (A_ref/A)^gamma (0 = off)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N images")
    ap.add_argument("--limit-test", type=int, default=0, help="debug: predict only the first N test images")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="cache prepared training arrays here (memory-mapped by later folds / processes)")
    ap.add_argument("--extra-images", nargs="*", default=[],
                    help="unlabelled images (files / glob patterns) trained on teacher maps (semi-supervised)")
    ap.add_argument("--extra-targets", type=Path, default=None,
                    help="folder of teacher maps <image stem>.png (scripts/pseudo_label.py)")
    ap.add_argument("--extra-mult", type=int, default=2, help="samples per unlabelled image per epoch")
    ap.add_argument("--extra-cleanup", action="store_true",
                    help="delete the cached teacher targets after training (when no other process uses them)")
    args = ap.parse_args(argv)

    t_start = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    log_f = open(args.out / "log.txt", "a", encoding="utf-8")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"args={vars(args)} device={device} torch={torch.__version__}")
    (args.out / "args.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=1))

    data_dir = find_data_dir(args.data)
    entries = data.load_entries(data_dir / "train" / data.TRAIN_JSON)
    groups = data.group_by_image(entries)
    if args.limit:
        groups = dict(list(groups.items())[:args.limit])
    folds = data.image_folds(list(groups), args.nfolds, args.seed)
    t0 = time.time()
    imgs, tgts, wmaps, stats = cached_arrays(args.cache_dir, data_dir, groups, args.res, args.workers,
                                             args.small_weight, args.norm, log)
    ids = list(groups)
    log(f"prepared {len(ids)} images at {args.res}px in {time.time() - t0:.0f}s")

    is_val = np.array([folds[i] == args.fold for i in ids])
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    log(f"train={len(tr_idx)} val={len(va_idx)}")

    gt_val = {}
    for iid in (ids[i] for i in va_idx):
        for e in groups[iid]:
            gt_val[e.entry_id] = [Crop.from_full(m) for m in e.masks()]

    model = build_model(args.arch, args.encoder, None if args.weights == "none" else args.weights).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    ds = TrainSet(imgs, tgts, args.crop or args.res, args.epoch_mult, wmaps, stats, idx=tr_idx)
    n_extra, extra_tgt_dir = 0, None
    if args.extra_images:
        if wmaps is not None:
            raise SystemExit("--extra-images cannot be combined with --small-weight")
        pairs = extra_pairs(args.extra_images, args.extra_targets)
        if not pairs:
            raise SystemExit(f"no unlabelled image has a teacher map in {args.extra_targets}")
        e_imgs, e_tgts, e_stats, extra_tgt_dir = cached_extra(args.cache_dir, pairs, args.res, args.workers,
                                                               args.norm, log)
        ds = torch.utils.data.ConcatDataset(
            [ds, TrainSet(e_imgs, e_tgts, args.crop or args.res, args.extra_mult, None, e_stats)])
        n_extra = len(pairs)
        del e_imgs, e_tgts
        log(f"unlabelled: {n_extra} images x {args.extra_mult} samples per epoch, teacher maps {args.extra_targets}")
    del imgs, tgts, wmaps  # only the dataset keeps the arrays; they are released after training

    def make_loader(bs: int) -> DataLoader:
        return DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=args.workers,
                          pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
                          worker_init_fn=_worker_init)

    dl = make_loader(args.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = [args.epochs * len(dl)]  # mutable: shrunk if the time budget cannot fit all epochs
    warm = min(len(dl), 200)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total[0]) / total[0])))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = None
    if args.ema > 0:
        from torch.optim.swa_utils import AveragedModel
        ema = AveragedModel(model, multi_avg_fn=ema_with_warmup(args.ema), use_buffers=True)

    ckpt = args.out / "last.pt"
    start_epoch = 0
    if ckpt.exists():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]), opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]), scaler.load_state_dict(st["scaler"])
        if ema is not None and "ema" in st:
            ema.load_state_dict(st["ema"])
        start_epoch = st["epoch"] + 1
        args.epochs = st.get("epochs", args.epochs)
        total[0] = args.epochs * len(dl)
        log(f"resumed from epoch {start_epoch} of {args.epochs}")

    val_imgs = [None] * len(va_idx)  # lazily read full-res val images for prediction
    def val_probs(models, tta):
        nonlocal val_imgs
        if val_imgs[0] is None:
            val_imgs = [data.read_image(data_dir / "train" / "train_images" / groups[ids[i]][0].file_name)
                        for i in va_idx]
        p = predict_images(models, val_imgs, args.res, device, tta=tta, batch=args.infer_bs or args.bs, norm=args.norm)
        return {ids[i]: p[k] for k, i in enumerate(va_idx)}

    default_cfg = PostConfig(threshold=0.5, min_area=150)
    monitor_cfg = default_cfg
    if args.post and args.post.exists():  # mid-training validation comparable with the final number
        monitor_cfg = PostConfig(**json.loads(args.post.read_text()))
    budget = args.max_hours * 3600
    epoch = start_epoch
    while epoch < args.epochs:
        model.train()
        te, tl, n = time.time(), 0.0, 0
        try:
            for xb, yb in dl:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                if device.type == "cuda":
                    xb = xb.to(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss = loss_fn(model(xb), yb)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                if ema is not None:
                    ema.update_parameters(model)
                tl += loss.item()
                n += 1
        except torch.cuda.OutOfMemoryError:
            # an unexpected memory limit should not waste a GPU session: halve the batch and redo the epoch
            if epoch != start_epoch or args.bs < 2:
                raise
            xb = yb = loss = None
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            args.bs //= 2
            log(f"CUDA out of memory in the first epoch: restarting it with batch size {args.bs}")
            del dl
            dl = make_loader(args.bs)
            total[0] = args.epochs * len(dl)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            continue
        t_epoch = time.time() - te
        if epoch == start_epoch:
            # fit the remaining epochs into the time budget (reserve ~12% for tuning + test inference)
            left = budget * 0.88 - (time.time() - t_start)
            fit = epoch + 1 + int(left // max(t_epoch * 1.08, 1))
            if fit < args.epochs:
                log(f"time budget: reducing epochs {args.epochs} -> {fit}")
                args.epochs = max(fit, epoch + 1)
                total[0] = args.epochs * len(dl)
        msg = f"epoch {epoch + 1}/{args.epochs} loss={tl / max(n, 1):.4f} lr={sched.get_last_lr()[0]:.2e} {t_epoch:.0f}s"
        last = epoch + 1 == args.epochs
        over = time.time() - t_start > budget
        if len(va_idx) and ((epoch + 1) % args.val_every == 0 or last or over):
            net = ema.module if ema is not None else model
            net.eval()
            res = score(val_probs([net], tta=1), gt_val, monitor_cfg, workers=args.tune_workers or None)
            msg += f" | val(tta1) {res}"
        log(msg)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "epochs": args.epochs,
                    "ema": ema.state_dict() if ema is not None else None}, ckpt.with_suffix(".tmp"))
        os.replace(ckpt.with_suffix(".tmp"), ckpt)  # atomic: a concurrent uploader never sees a partial file
        if over:
            log("time budget reached, stopping training")
            break
        epoch += 1

    if ema is not None:
        model = ema.module
    model.eval()
    del dl, ds  # release the training arrays (and loader workers) before the final evaluation
    gc.collect()
    if args.extra_cleanup and extra_tgt_dir is not None:
        import shutil

        shutil.rmtree(extra_tgt_dir, ignore_errors=True)
        log(f"removed cached teacher targets {extra_tgt_dir}")
    save_checkpoint(model, args.out / "model.pt", arch=args.arch, encoder=args.encoder, res=args.res,
                    norm=args.norm, fold=args.fold, epochs=args.epochs, ema=args.ema)

    cfg = default_cfg
    val_res = None
    if args.post and args.post.exists():
        cfg = PostConfig(**json.loads(args.post.read_text()))
    if len(va_idx):
        probs = val_probs([model], tta=args.tta)
        np.savez_compressed(args.out / "val_probs.npz",
                            **{k: np.round(v.astype(np.float32) * 255).astype(np.uint8) for k, v in probs.items()})
        if args.post and args.post.exists():
            val_res = score(probs, gt_val, cfg, workers=args.tune_workers or None)
            log(f"val (tta{args.tta}) with given post {asdict(cfg)} -> {val_res}")
        else:
            log("tuning post-processing on hold-out fold")
            cfg, val_res, _ = grid_search(probs, gt_val, default_cfg, log=log, workers=args.tune_workers or None)
            log(f"best post {asdict(cfg)} -> {val_res}")
    (args.out / "post.json").write_text(json.dumps(asdict(cfg), indent=1))

    if args.predict_test:
        write_submission([model], data_dir, args.res, cfg, device, args.out / "submission.csv",
                         args.tta, args.infer_bs or args.bs, log, norm=args.norm, limit=args.limit_test)
    hours = (time.time() - t_start) / 3600
    done = {"hours": round(hours, 3), "epochs": args.epochs, "fold": args.fold, "res": args.res,
            "arch": args.arch, "encoder": args.encoder, "post": asdict(cfg), "extra_images": n_extra}
    if val_res is not None:
        done["val"] = {"pq": val_res.pq, "sq": val_res.sq, "rq": val_res.rq,
                       "tp": val_res.tp, "fp": val_res.fp, "fn": val_res.fn}
    (args.out / "done.json").write_text(json.dumps(done, indent=1))  # completion marker
    log(f"done in {hours:.2f} h")


if __name__ == "__main__":
    main()
