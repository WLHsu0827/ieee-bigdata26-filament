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

from . import augment, data, rle
from .metrics import Crop
from .model import build_model, predict_images, to_tensor, upsample
from .postprocess import PostConfig, instances_from_prob, label_to_masks
from .tune import grid_search, score


# ----------------------------------------------------------------------------- data
def _prepare_one(args):
    path, entries, res = args
    img = data.read_image(path)
    tgt = data.soft_target(entries, size=res)
    from .model import resize_image
    return resize_image(img, res), np.round(tgt * 255).astype(np.uint8)


def prepare_arrays(data_dir: Path, groups: dict[str, list[data.Entry]], res: int, workers: int):
    jobs = [(data_dir / "train" / "train_images" / es[0].file_name, es, res) for es in groups.values()]
    if workers > 0:
        with Pool(workers) as pool:
            out = pool.map(_prepare_one, jobs, chunksize=4)
    else:
        out = [_prepare_one(j) for j in jobs]
    imgs = np.stack([o[0] for o in out])
    tgts = np.stack([o[1] for o in out])
    return imgs, tgts


class TrainSet(Dataset):
    def __init__(self, imgs: np.ndarray, tgts: np.ndarray, crop: int, epoch_mult: int = 1):
        self.imgs, self.tgts, self.crop, self.mult = imgs, tgts, crop, epoch_mult

    def __len__(self):
        return len(self.imgs) * self.mult

    def __getitem__(self, i):
        rng = np.random.default_rng(torch.randint(0, 2**31, (1,)).item())
        i = i % len(self.imgs)
        img = self.imgs[i].astype(np.float32) / 255.0
        tgt = self.tgts[i].astype(np.float32) / 255.0
        img, tgt = augment.geometric(img, tgt, rng)
        img, tgt = augment.random_crop(img, tgt, self.crop, rng)
        img = augment.photometric(np.ascontiguousarray(img), rng)
        return to_tensor(img), torch.from_numpy(np.ascontiguousarray(tgt))[None]


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + target.sum() + 1)
    return bce + dice


# ----------------------------------------------------------------------------- io
def find_data_dir(hint: str | None) -> Path:
    if hint:
        return Path(hint)
    for p in Path("/kaggle/input").rglob(data.TRAIN_JSON):
        return p.parent.parent
    raise FileNotFoundError("MAGFiLO data directory not found; pass --data")


def write_submission(models, data_dir: Path, res: int, cfg: PostConfig, device, out_csv: Path,
                     tta: int, batch: int, log=print) -> pd.DataFrame:
    """Predict the test images, post-process, write the CSV and log per-image inference time."""
    test_dir = data_dir / "test" / "test_images"
    files = sorted(test_dir.glob("*.jp*g"))
    rows = []
    t_read = t_model = t_post = 0.0
    for s in range(0, len(files), 16):
        chunk = files[s:s + 16]
        t0 = time.time()
        imgs = [data.read_image(f) for f in chunk]
        t1 = time.time()
        probs = predict_images(models, imgs, res, device, tta, batch)
        t2 = time.time()
        for f, p in zip(chunk, probs):
            lab, _ = instances_from_prob(upsample(p), cfg)
            for k, m in enumerate(label_to_masks(lab), start=1):
                rows.append((f"{f.stem}_{k}", rle.encode(m)))
        t_read, t_model, t_post = t_read + t1 - t0, t_model + t2 - t1, t_post + time.time() - t2
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(out_csv, index=False)
    n = max(len(files), 1)
    timing = {"device": str(device), "tta": tta, "res": res, "n_models": len(models), "images": len(files),
              "read_s_per_image": t_read / n, "model_s_per_image": t_model / n, "post_s_per_image": t_post / n}
    (out_csv.parent / "timing.json").write_text(json.dumps(timing, indent=1))
    log(f"submission: {len(files)} images, {len(df)} filaments -> {out_csv}")
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
    ap.add_argument("--arch", default="Unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--weights", default="imagenet", help="encoder pre-training (\"none\" = random init)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--fold", type=int, default=0, help="hold-out fold, -1 = train on everything")
    ap.add_argument("--nfolds", type=int, default=5)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--post", type=Path, default=None, help="post-processing JSON (skips tuning)")
    ap.add_argument("--predict-test", action="store_true")
    ap.add_argument("--max-hours", type=float, default=11.0, help="stop training early to fit the time limit")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0 = off)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N images")
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

    data_dir = find_data_dir(args.data)
    entries = data.load_entries(data_dir / "train" / data.TRAIN_JSON)
    groups = data.group_by_image(entries)
    if args.limit:
        groups = dict(list(groups.items())[:args.limit])
    folds = data.image_folds(list(groups), args.nfolds, args.seed)
    t0 = time.time()
    imgs, tgts = prepare_arrays(data_dir, groups, args.res, args.workers)
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
    ds = TrainSet(imgs[tr_idx], tgts[tr_idx], args.crop or args.res)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=args.workers,
                    pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = [args.epochs * len(dl)]  # mutable: shrunk if the time budget cannot fit all epochs
    warm = min(len(dl), 200)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total[0]) / total[0])))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = None
    if args.ema > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)

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
        p = predict_images(models, val_imgs, args.res, device, tta=tta, batch=args.bs)
        return {ids[i]: p[k] for k, i in enumerate(va_idx)}

    default_cfg = PostConfig(threshold=0.5, min_area=150)
    budget = args.max_hours * 3600
    epoch = start_epoch
    while epoch < args.epochs:
        model.train()
        te, tl, n = time.time(), 0.0, 0
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
            res = score(val_probs([net], tta=1), gt_val, default_cfg)
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
    torch.save({"model": model.state_dict(), "arch": args.arch, "encoder": args.encoder, "res": args.res},
               args.out / "model.pt")

    cfg = default_cfg
    if args.post and args.post.exists():
        cfg = PostConfig(**json.loads(args.post.read_text()))
    elif len(va_idx):
        probs = val_probs([model], tta=args.tta)
        np.savez_compressed(args.out / "val_probs.npz",
                            **{k: np.round(v.astype(np.float32) * 255).astype(np.uint8) for k, v in probs.items()})
        log("tuning post-processing on hold-out fold")
        cfg, best, _ = grid_search(probs, gt_val, default_cfg, log=log)
        log(f"best post {asdict(cfg)} -> {best}")
    (args.out / "post.json").write_text(json.dumps(asdict(cfg), indent=1))

    if args.predict_test:
        write_submission([model], data_dir, args.res, cfg, device, args.out / "submission.csv",
                         args.tta, args.bs, log)
    log(f"done in {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
