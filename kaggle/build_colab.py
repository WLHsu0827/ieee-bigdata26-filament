"""Generate ``notebooks/colab_train.ipynb``: a self-contained, resumable Colab
(free T4) training notebook.

The notebook embeds the ``filseg`` package, downloads the competition data with
the user's Kaggle API token (stored as a Colab secret, never in the notebook),
trains with ``filseg.train``, and periodically uploads ``last.pt`` plus logs to
a *private* Kaggle dataset so that a disconnected session can resume and the
results can be retrieved without manual downloads.

Usage:
    python kaggle/build_colab.py --run c1-r34-1024 -- --res 1024 --encoder resnet34 ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_kernel import payload

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Solar filament segmentation -- Colab training run `{run}`

One-time setup (about 2 minutes):

1. **Runtime -> Change runtime type -> T4 GPU.**
2. On Kaggle: *Settings -> API Tokens -> Generate New Token* and copy it.
   (Do **not** use "Create Legacy API Key": that would expire the key already in use.)
3. In Colab, open the key icon (Secrets) in the left sidebar and add two secrets with
   *Notebook access* switched on:
   * `KAGGLE_API_TOKEN` = the token from step 2
   * `KAGGLE_USERNAME` = your Kaggle user name
4. **Runtime -> Run all.**

If the session disconnects, reconnect and *Run all* again: training resumes from the
last checkpoint, which is stored in the private Kaggle dataset `fil-colab-{run}`.
Results (weights, logs, validation scores, submission) are uploaded to the same
private dataset automatically; nothing needs to be downloaded by hand.
"""

CONFIG = """RUN = {run!r}
TRAIN_ARGS = {args!r}
SYNC_MINUTES = 20  # upload the checkpoint at most this often
"""

SETUP = r"""import os, subprocess, sys, json, time, shutil, threading, base64, io, zipfile, glob
from pathlib import Path
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle", "segmentation-models-pytorch", "pycocotools"], check=True)
from google.colab import userdata
os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
USER = userdata.get("KAGGLE_USERNAME").strip()
SLUG = f"{USER}/fil-colab-{RUN}"
WORK = Path("/content/work"); OUT = WORK / "out" / RUN; OUT.mkdir(parents=True, exist_ok=True)
def kaggle(*a, check=True):
    r = subprocess.run(["kaggle", *a], capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    return r
print("kaggle user:", USER)
"""

DATA = r"""DATA = WORK / "data"
if not (DATA / "MAGFiLO_1.0_Kaggle_2026").exists():
    kaggle("competitions", "download", "filament-segmentation-2026", "-p", str(DATA), "-q")
    for z in DATA.glob("*.zip"):
        zipfile.ZipFile(z).extractall(DATA); z.unlink()
print(sorted(p.name for p in (DATA / "MAGFiLO_1.0_Kaggle_2026").iterdir()))
"""

CODE = """PAYLOAD = "{payload}"
SRC = WORK / "src"; shutil.rmtree(SRC, ignore_errors=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall(SRC)
print("code unpacked to", SRC)
"""

RESUME = r"""SYNC = WORK / "sync"
def dataset_exists():
    return kaggle("datasets", "status", SLUG, check=False).returncode == 0
if not (OUT / "last.pt").exists() and dataset_exists():
    tmp = WORK / "resume"; shutil.rmtree(tmp, ignore_errors=True)
    r = kaggle("datasets", "download", SLUG, "-p", str(tmp), "--unzip", check=False)
    for f in ["last.pt", "log.txt"]:
        if (tmp / f).exists():
            shutil.copy(tmp / f, OUT / f)
    print("resume files:", sorted(p.name for p in OUT.iterdir()))
"""

TRAIN = r"""_lock = threading.Lock()
def sync(message, files):
    with _lock:
        shutil.rmtree(SYNC, ignore_errors=True); SYNC.mkdir(parents=True)
        for f in files:
            if (OUT / f).exists():
                shutil.copy(OUT / f, SYNC / f)
        (SYNC / "dataset-metadata.json").write_text(json.dumps(
            {"title": f"fil-colab-{RUN}", "id": SLUG, "licenses": [{"name": "other"}]}))
        if dataset_exists():
            r = kaggle("datasets", "version", "-p", str(SYNC), "-m", message, "-q", check=False)
        else:
            r = kaggle("datasets", "create", "-p", str(SYNC), "-q", check=False)  # private by default
        print(f"[sync] {message}: rc={r.returncode} {r.stderr.strip()[-200:]}", flush=True)

stop = threading.Event()
def periodic():
    last = None
    while not stop.wait(SYNC_MINUTES * 60):
        p = OUT / "last.pt"
        if p.exists() and p.stat().st_mtime != last:
            last = p.stat().st_mtime
            sync(f"checkpoint {time.strftime('%H:%M')}", ["last.pt", "log.txt"])
threading.Thread(target=periodic, daemon=True).start()

env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONUNBUFFERED="1")
cmd = [sys.executable, "-m", "filseg.train", "--data", str(DATA / "MAGFiLO_1.0_Kaggle_2026"), "--out", str(OUT), *TRAIN_ARGS]
proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end="")
rc = proc.wait(); stop.set()
print("training exit code", rc)
sync(f"final rc={rc}", ["model.pt", "post.json", "log.txt", "val_probs.npz", "submission.csv"])
"""


def cell(kind: str, src: str) -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "notebooks" / "colab_train.ipynb")
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    train_args = a.args[1:] if a.args and a.args[0] == "--" else a.args
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": [cell("markdown", INTRO.format(run=a.run)),
                  cell("code", CONFIG.format(run=a.run, args=train_args)),
                  cell("code", SETUP), cell("code", DATA),
                  cell("code", CODE.format(payload=payload())),
                  cell("code", RESUME), cell("code", TRAIN)],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {a.out} ({a.out.stat().st_size // 1024} KB) for run {a.run}: {' '.join(train_args)}")


if __name__ == "__main__":
    main()
