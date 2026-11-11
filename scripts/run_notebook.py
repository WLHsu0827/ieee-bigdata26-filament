"""Execute ``notebooks/pipeline.ipynb`` in a fresh copy of the repository (reproducibility check).

Copies the package, the notebook and the documentation figures into ``--workdir``, points the
notebook at the trained weights (folder containing ``manifest.json``) and the competition data,
executes it with nbconvert (profile and options through environment variables) and collects the
executed notebook, the submission, the ensemble probability maps and the package versions in
``--out``. Used on a Kaggle GPU to verify that the committed weights reproduce the submission
without any re-training.

``--venv DIR`` checks the README instructions as a user would follow them: it creates a new virtual
environment, runs the README's ``pip install -r requirements.txt`` and ``python -m ipykernel install
--sys-prefix`` in it, then the README's nbconvert command with that environment activated (the installed
versions go to ``pip_freeze.txt``). The check fails unless the notebook reports the torch build of the new
environment, i.e. unless its cells really ran with the installed packages.

Usage:
    python scripts/run_notebook.py --profile full --weights "/kaggle/input/**/manifest.json" --out out_full
    python scripts/run_notebook.py --weights "/kaggle/input/**/manifest.json" --venv /tmp/fresh --limit 2 --out out_fresh
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from cli_utils import expand  # noqa: E402

PACKAGES = ["numpy", "pandas", "opencv-python-headless", "opencv-python", "scipy", "scikit-image", "pycocotools",
            "torch", "segmentation-models-pytorch", "timm", "matplotlib", "scikit-learn", "nbconvert", "ipykernel"]
VERSIONS_CODE = f"""
import json, platform
from importlib import metadata
out = {{"python": platform.python_version()}}
for p in {PACKAGES!r}:
    try:
        out[p] = metadata.version(p)
    except metadata.PackageNotFoundError:
        pass
print(json.dumps(out))
"""


def versions(python: str = sys.executable) -> dict:
    """Python and package versions of an interpreter (default: this one)."""
    res = subprocess.run([python, "-c", VERSIONS_CODE], capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def make_venv(venv: Path, out: Path) -> tuple[str, dict]:
    """Follow the README in a new virtual environment: create it, ``pip install -r requirements.txt`` and
    ``python -m ipykernel install --sys-prefix``. Writes ``pip freeze`` to ``out/pip_freeze.txt`` and returns
    the environment's python and the environment variables of the activated environment."""
    if venv.exists():
        shutil.rmtree(venv)
    if subprocess.run([sys.executable, "-m", "venv", str(venv)]).returncode != 0:  # e.g. no ensurepip
        shutil.rmtree(venv, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "virtualenv"], check=True)
        subprocess.run([sys.executable, "-m", "virtualenv", str(venv)], check=True)
    python = str(venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    env = dict(os.environ, VIRTUAL_ENV=str(venv), PATH=str(Path(python).parent) + os.pathsep + os.environ.get("PATH", ""))
    env.pop("JUPYTER_PATH", None)  # a clean machine has none; it would take precedence over the environment's kernel
    t0 = time.time()
    subprocess.run([python, "-m", "pip", "install", "--progress-bar", "off", "-r", str(ROOT / "requirements.txt")],
                   env=env, check=True)
    subprocess.run([python, "-m", "ipykernel", "install", "--sys-prefix"], env=env, check=True)
    freeze = subprocess.run([python, "-m", "pip", "freeze"], env=env, capture_output=True, text=True, check=True).stdout
    (out / "pip_freeze.txt").write_text(freeze)
    specs = json.loads(subprocess.run([python, "-m", "jupyter", "kernelspec", "list", "--json"], env=env,
                                      capture_output=True, text=True, check=True).stdout)["kernelspecs"]
    print(f"fresh environment {venv}: requirements installed in {(time.time() - t0) / 60:.1f} min; kernel python3: "
          f"{specs['python3']['spec']['argv'][0]} ({specs['python3']['resource_dir']})", flush=True)
    return python, env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="full", choices=["full", "fast"])
    ap.add_argument("--weights", required=True, help="manifest.json of the weights (file or glob)")
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variant", action="append", default=[],
                    help="NAME=POST_JSON added to the manifest's variants (as scripts/make_weights.py --variant)")
    ap.add_argument("--venv", type=Path, default=None,
                    help="create this virtual environment, install requirements.txt and run the README command in it")
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
    python, base_env = make_venv(args.venv, args.out) if args.venv else (sys.executable, dict(os.environ))

    from filseg.train import find_data_dir

    env = dict(base_env, FILSEG_PROFILE=args.profile, FILSEG_LIMIT_TEST=str(args.limit),
               FILSEG_WEIGHTS=str(weights), FILSEG_SAVE_PROBS=str(args.out.resolve() / f"test_probs_{args.profile}.npz"),
               FILSEG_DATA=str(find_data_dir(None)))
    cmd = [python, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute"]
    if not args.venv:  # the fresh-environment check runs the README command as written
        cmd += ["--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=python3"]
    cmd += ["--output", f"executed_{args.profile}.ipynb", "--output-dir", str(args.out.resolve()),
            str(work / "notebooks" / "pipeline.ipynb")]
    print("versions:", json.dumps(versions(python)), flush=True)
    print("running:", " ".join(cmd), flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, env=env, cwd=work / "notebooks").returncode
    print(f"nbconvert finished with code {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
    for f in work.glob("submission*.csv"):
        shutil.copy(f, args.out / f.name)
    (args.out / "versions.json").write_text(json.dumps(versions(python), indent=1))
    executed = args.out / f"executed_{args.profile}.ipynb"
    texts = []
    if executed.exists():  # print the notebook's own report lines (inference time, submission check)
        nb = json.loads(executed.read_text(encoding="utf-8"))
        texts = ["".join(o.get("text", [])) for cell in nb["cells"] for o in cell.get("outputs", [])]
        for text in texts:
            if "per image" in text or "filaments ->" in text:
                print(text.strip(), flush=True)
    if args.venv:  # the cells must have run with the packages of the fresh environment, not another installation
        torch_version = subprocess.run([python, "-c", "import torch; print(torch.__version__)"], env=env,
                                       capture_output=True, text=True, check=True).stdout.strip()
        ok = any(f"torch {torch_version} | device:" in t for t in texts)
        print(f"notebook cells ran with the fresh environment (torch {torch_version}): {ok}", flush=True)
        rc = rc or int(not ok)
    sys.exit(rc)


if __name__ == "__main__":
    main()
