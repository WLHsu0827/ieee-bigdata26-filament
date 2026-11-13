"""Generate ``notebooks/colab_<run>.ipynb``: a self-contained, resumable Colab
(free T4) training notebook that runs one or more training jobs in sequence.

The notebook embeds the ``filseg`` package, downloads the competition data with
the user's Kaggle API token (stored as a Colab secret, never in the notebook),
trains each job with ``filseg.train``, and periodically uploads the running
job's checkpoint plus every finished job's outputs to a *private* Kaggle dataset
(``<job>__<file>``), so that a disconnected session resumes where it stopped and
results can be retrieved without manual downloads.

Usage:
    python kaggle/build_colab.py --run c2 --kaggle-user <user> \
        --job r34-f1 "--fold 1 --res 1024 --encoder resnet34 ..." --job r34-f2 "--fold 2 ..."
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from build_kernel import payload

ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Solar filament segmentation -- Colab training run `{run}` (jobs listed in the first code cell)

One-time setup (about 2 minutes):

1. **Runtime -> Change runtime type -> T4 GPU.**
2. On Kaggle: *Settings -> API -> Generate New Token* and copy it.
   (Do **not** use "Create Legacy API Key": that would expire the key already in use.)
3. In Colab, open the key icon (Secrets) in the left sidebar and add **one** secret with
   *Notebook access* switched on: `KAGGLE_API_TOKEN` = the token from step 2.
   (If Secrets are unavailable, e.g. in the VS Code Colab extension, the notebook asks
   for the token in a hidden prompt instead. The token is never printed or saved.)
4. **Runtime -> Run all.**

If the session disconnects, reconnect and *Run all* again: training resumes from the
last checkpoint, which is stored in the private Kaggle dataset `fil-colab-{run}`.
Results (weights, logs, validation scores, submission) are uploaded to the same
private dataset automatically; nothing needs to be downloaded by hand.
"""

CONFIG = """RUN = {run!r}
JOBS = {jobs!r}  # (name, filseg.train arguments), run one after another
SYNC_MINUTES = 20  # upload the checkpoint at most this often
DEFAULT_KAGGLE_USER = {user!r}  # public Kaggle user name; override with the optional secret KAGGLE_USERNAME
PIP_PACKAGES = {pip!r}  # network packages pinned to requirements.txt, so the weights load in the final pipeline
"""

SETUP = r"""import os, subprocess, sys, json, time, shutil, threading, base64, io, zipfile, getpass
from pathlib import Path
print(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle", "pycocotools", *PIP_PACKAGES], check=True)

def _secret(name):
    try:
        from google.colab import userdata
        value = userdata.get(name)
        return value.strip() if value else None
    except Exception:  # secret missing, notebook access off, or userdata unsupported
        return None

TOKEN = _secret("KAGGLE_API_TOKEN") or getpass.getpass("KAGGLE_API_TOKEN (hidden input): ").strip()
if not TOKEN:
    raise RuntimeError("KAGGLE_API_TOKEN is empty: add the Colab secret KAGGLE_API_TOKEN or paste the token at the prompt.")
USER = _secret("KAGGLE_USERNAME") or DEFAULT_KAGGLE_USER
SLUG = f"{USER}/fil-colab-{RUN}"
WORK = Path("/content/work"); OUT = WORK / "out" / RUN; OUT.mkdir(parents=True, exist_ok=True)  # one sub-folder per job

def kaggle(*a, check=True):
    # the token is passed only to the kaggle CLI subprocess, never to os.environ or the training process
    env = dict(os.environ, KAGGLE_API_TOKEN=TOKEN, KAGGLE_USERNAME=USER)
    r = subprocess.run(["kaggle", *a], capture_output=True, text=True, env=env)
    r.stdout, r.stderr = r.stdout.replace(TOKEN, "***"), r.stderr.replace(TOKEN, "***")
    if check and r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    return r

_auth = kaggle("competitions", "files", "filament-segmentation-2026", "--page-size", "1", check=False)
if _auth.returncode != 0:
    raise RuntimeError("Kaggle authentication failed. Check the Colab secret KAGGLE_API_TOKEN "
                     "(Secrets panel, Notebook access ON) or paste a valid token at the prompt.\n"
                     + (_auth.stderr or _auth.stdout)[-300:])
print("Kaggle authentication OK for", USER)
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
FINAL_FILES = ["model.pt", "post.json", "log.txt", "val_probs.npz", "test_probs.npz", "submission.csv",
               "timing.json", "done.json"]

def dataset_exists():
    return kaggle("datasets", "status", SLUG, check=False).returncode == 0

# after a disconnect: restore finished jobs and the checkpoint of the unfinished one
if dataset_exists() and not any((OUT / name).exists() for name, _ in JOBS):
    tmp = WORK / "resume"; shutil.rmtree(tmp, ignore_errors=True)
    kaggle("datasets", "download", SLUG, "-p", str(tmp), "--unzip", check=False)
    for name, _ in JOBS:
        for f in tmp.glob(f"{name}__*"):
            (OUT / name).mkdir(parents=True, exist_ok=True)
            shutil.copy(f, OUT / name / f.name.split("__", 1)[1])
for name, _ in JOBS:
    d = OUT / name
    state = "done" if (d / "done.json").exists() else ("resume" if (d / "last.pt").exists() else "new")
    print(f"{name}: {state}")
"""

TRAIN = r"""_lock = threading.Lock()
def sync(message, with_checkpoint):
    with _lock:
        shutil.rmtree(SYNC, ignore_errors=True); SYNC.mkdir(parents=True)
        for name, _ in JOBS:
            d = OUT / name
            done = (d / "done.json").exists()
            for f in FINAL_FILES + (["last.pt"] if with_checkpoint and not done else []):
                if (d / f).exists():
                    shutil.copy(d / f, SYNC / f"{name}__{f}")
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
        stamps = tuple((OUT / n / "last.pt").stat().st_mtime for n, _ in JOBS if (OUT / n / "last.pt").exists())
        if stamps and stamps != last:
            last = stamps
            sync(f"checkpoint {time.strftime('%H:%M')}", with_checkpoint=True)
threading.Thread(target=periodic, daemon=True).start()

env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONUNBUFFERED="1")
codes = {}
for name, args in JOBS:
    d = OUT / name
    if (d / "done.json").exists():
        print(f"===== {name}: already finished, skipping"); continue
    print(f"===== {name}: {' '.join(args)}", flush=True)
    cmd = [sys.executable, "-m", "filseg.train", "--data", str(DATA / "MAGFiLO_1.0_Kaggle_2026"), "--out", str(d), *args]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    codes[name] = proc.wait()
    print(f"===== {name}: exit code {codes[name]}", flush=True)
    sync(f"{name} finished rc={codes[name]}", with_checkpoint=False)
stop.set()
print("all jobs finished:", codes)
"""

def cell(kind: str, src: str) -> dict:
    """One nbformat-4 cell of type ``kind`` ("markdown" or "code"); code cells start unexecuted."""
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def pinned_packages(names=("segmentation-models-pytorch", "timm")) -> list[str]:
    """``name==version`` of the network packages, from requirements.txt."""
    pins = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip().lower()] = f"{name.strip()}=={version.strip()}"
    missing = [n for n in names if n not in pins]
    if missing:
        raise SystemExit(f"requirements.txt pins no version for {missing}")
    return [pins[n] for n in names]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name; results go to the private dataset fil-colab-<run>")
    ap.add_argument("--kaggle-user", required=True, help="public Kaggle user name owning the private result dataset")
    ap.add_argument("--job", nargs=2, action="append", metavar=("NAME", "ARGS"), required=True,
                    help='one training job: name and the filseg.train arguments as one string (repeatable)')
    ap.add_argument("--out", type=Path, default=None, help="default: notebooks/colab_<run>.ipynb")
    a = ap.parse_args()
    jobs = [(name, shlex.split(args)) for name, args in a.job]
    out = a.out or ROOT / "notebooks" / f"colab_{a.run}.ipynb"
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"},
                     "language_info": {"name": "python"}},
        "cells": [cell("markdown", INTRO.format(run=a.run)),
                  cell("code", CONFIG.format(run=a.run, jobs=jobs, user=a.kaggle_user, pip=pinned_packages())),
                  cell("code", SETUP), cell("code", DATA),
                  cell("code", CODE.format(payload=payload())),
                  cell("code", RESUME), cell("code", TRAIN)],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB), run {a.run}:")
    for name, args in jobs:
        print(f"  {name}: {' '.join(args)}")

if __name__ == "__main__":
    main()
