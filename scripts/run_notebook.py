"""Execute ``notebooks/pipeline.ipynb`` in a fresh copy of the repository (reproducibility check).

Copies the package, the notebook and the documentation figures into ``--workdir``, points the
notebook at the trained weights (folder containing ``manifest.json``) and the competition data,
executes it with nbconvert (profile and options through environment variables) and collects the
executed notebook, the submission, the ensemble probability maps and the package versions in
``--out``. Used on a Kaggle GPU to verify that the committed weights reproduce the submission
without any re-training.

Usage:
    python scripts/run_notebook.py --profile full --weights "/kaggle/input/**/manifest.json" --out out_full
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cli_utils import expand  # noqa: E402

PACKAGES = ["numpy", "pandas", "opencv-python-headless", "opencv-python", "scipy", "scikit-image", "pycocotools",
            "torch", "segmentation-models-pytorch", "timm", "matplotlib", "scikit-learn", "nbconvert", "ipykernel"]


def versions() -> dict:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="full", choices=["full", "fast"])
    ap.add_argument("--weights", required=True, help="manifest.json of the weights (file or glob)")
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variant", action="append", default=[],
                    help="NAME=POST_JSON added to the manifest's variants (as scripts/make_weights.py --variant)")
    args = ap.parse_args()

    weights = expand([args.weights])[0].parent
    work = args.workdir or Path(f"/tmp/repo_{args.profile}")
    if work.exists():
        shutil.rmtree(work)
    if args.variant:  # weights folder with links to the checkpoints and an extended manifest
        local = work.parent / f"weights_{args.profile}"
        if local.exists():
            shutil.rmtree(local)
        local.mkdir(parents=True)
        for f in weights.glob("*.pt"):
            (local / f.name).symlink_to(f)
        manifest = json.loads((weights / "manifest.json").read_text())
        manifest.setdefault("variants", {}).update(
            {name: json.loads(Path(p).read_text()) for name, p in (v.split("=", 1) for v in args.variant)})
        (local / "manifest.json").write_text(json.dumps(manifest, indent=1))
        weights = local
    shutil.copytree(ROOT / "filseg", work / "filseg", ignore=shutil.ignore_patterns("__pycache__"))
    (work / "notebooks").mkdir(parents=True)
    if not (ROOT / "notebooks" / "pipeline.ipynb").exists():  # e.g. a Kaggle payload ships only the generator
        subprocess.run([sys.executable, str(ROOT / "notebooks" / "build_pipeline_notebook.py")], check=True)
    shutil.copy(ROOT / "notebooks" / "pipeline.ipynb", work / "notebooks" / "pipeline.ipynb")
    if (ROOT / "docs").exists():
        shutil.copytree(ROOT / "docs", work / "docs")
    args.out.mkdir(parents=True, exist_ok=True)

    from filseg.train import find_data_dir

    env = dict(os.environ, FILSEG_PROFILE=args.profile, FILSEG_LIMIT_TEST=str(args.limit),
               FILSEG_WEIGHTS=str(weights), FILSEG_SAVE_PROBS=str(args.out.resolve() / f"test_probs_{args.profile}.npz"),
               FILSEG_DATA=str(find_data_dir(None)))
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
           "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=python3",
           "--output", f"executed_{args.profile}.ipynb", "--output-dir", str(args.out.resolve()),
           str(work / "notebooks" / "pipeline.ipynb")]
    print("versions:", json.dumps(versions()), flush=True)
    print("running:", " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, env=env, cwd=work / "notebooks").returncode
    print(f"nbconvert finished with code {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
    for f in work.glob("submission*.csv"):
        shutil.copy(f, args.out / f.name)
    (args.out / "versions.json").write_text(json.dumps(versions(), indent=1))
    executed = args.out / f"executed_{args.profile}.ipynb"
    if executed.exists():  # print the notebook's own report lines (inference time, submission check)
        nb = json.loads(executed.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            for o in cell.get("outputs", []):
                text = "".join(o.get("text", []))
                if "per image" in text or "filaments ->" in text:
                    print(text.strip(), flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
