"""Collect trained runs into ``weights/`` (fp16 checkpoints + ``manifest.json``) for the pipeline notebook.

Usage:
    python scripts/make_weights.py --runs runs/r34_f0 runs/r34_f1 ... --post sel/post_selected.json \
        --tta 4 --out weights
Each run directory must contain ``model.pt``; training arguments are read from
``args.json`` (or the first line of ``log.txt`` for older runs).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.model import load_checkpoint, save_checkpoint  # noqa: E402

SKIP_ARGS = {"data", "out", "predict_test", "post", "workers", "threads", "tune_workers", "limit"}


def train_args(run: Path) -> dict:
    if (run / "args.json").exists():
        return json.loads((run / "args.json").read_text())
    first = (run / "log.txt").read_text(encoding="utf-8").splitlines()[0]
    m = re.match(r"args=(\{.*\}) device=", first)
    text = re.sub(r"(Windows|Posix)Path\('([^']*)'\)", r"'\2'", m.group(1)) if m else "{}"
    return ast.literal_eval(text)


def command_line(args: dict) -> str:
    parts = []
    for k, v in args.items():
        if k in SKIP_ARGS or v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-")
        parts.append(flag if v is True else f"{flag} {v}")
    return " ".join(parts)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--post", type=Path, required=True, help="post-processing JSON (PostConfig fields)")
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("weights"))
    ap.add_argument("--fast-post", type=Path, default=None,
                    help="post-processing JSON of the faster profile (subset of the models, --fast-encoder)")
    ap.add_argument("--fast-encoder", default="tu-tf_efficientnetv2_s")
    ap.add_argument("--variant", action="append", default=[],
                    help="NAME=POST_JSON: extra post-processing of the full profile (written as submission_NAME.csv)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    models = []
    for run in args.runs:
        net, meta = load_checkpoint(run / "model.pt")
        targs = train_args(run)
        meta.setdefault("norm", targs.get("norm", "global"))
        meta.setdefault("fold", targs.get("fold"))
        name = f"unet_{meta['encoder']}_{meta['res']}_fold{meta['fold']}.pt".replace("/", "-")
        save_checkpoint(net, args.out / name, **{k: v for k, v in meta.items() if k != "model"})
        models.append({"file": name, "arch": meta["arch"], "encoder": meta["encoder"], "res": int(meta["res"]),
                       "norm": meta["norm"], "fold": meta["fold"], "sha256": sha256(args.out / name),
                       "train_args": command_line(targs)})
        print(f"{run} -> {name} ({(args.out / name).stat().st_size / 1e6:.1f} MB)")
    manifest = {"post": json.loads(args.post.read_text()), "tta": args.tta, "nfolds": 5, "seed": 2026,
                "models": models}
    if args.fast_post is not None:
        files = [m["file"] for m in models if m["encoder"] == args.fast_encoder]
        if not files:
            raise SystemExit(f"no model with encoder {args.fast_encoder} for the fast profile")
        manifest["fast"] = {"models": files, "post": json.loads(args.fast_post.read_text())}
    if args.variant:
        manifest["variants"] = {name: json.loads(Path(p).read_text())
                                for name, p in (v.split("=", 1) for v in args.variant)}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {args.out / 'manifest.json'} with {len(models)} models")


if __name__ == "__main__":
    main()
