"""Build the report PDF from ``docs/report`` (pdflatex + bibtex, figures from ``docs/figures``).

The report uses the competition's official template (ACM ``acmart``, ``sigconf``), which needs the Linux
Libertine, newtx and Inconsolata fonts (Debian package ``texlive-fonts-extra``) and, through ``hyperxmp``,
``binhex.tex`` (``texlive-plain-generic``). Installs TeX Live with apt-get when ``pdflatex`` is missing
(Linux, e.g. a Kaggle kernel), compiles in a temporary copy and writes ``main.pdf`` and ``main.log`` to
``--out``.

The repository keeps placeholders in the author block of ``main.tex``, so that it holds no personal data;
``--author``, ``--institution``, ``--city``, ``--country`` and ``--email`` fill them in the build copy.

Usage:
    python docs/report/build_pdf.py --out build/
    python docs/report/build_pdf.py --out build/ --author "First Last" --institution "University" \
        --city City --country Country --email first.last@example.org
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ["texlive-latex-base", "texlive-latex-recommended", "texlive-latex-extra", "texlive-publishers",
            "texlive-fonts-recommended", "texlive-fonts-extra", "texlive-plain-generic"]
AUTHOR_PLACEHOLDERS = {"author": "[Author name]", "institution": "[Institution (optional)]", "city": "[City]",
                       "country": "[Country]", "email": "[e-mail]"}


def fill_author(tex: str, values: dict) -> tuple[str, list[str]]:
    """Replace the author-block placeholders of ``main.tex`` by the given values (LaTeX text; missing or
    empty values keep their placeholder). Returns the new text and the placeholders that are left."""
    for key, placeholder in AUTHOR_PLACEHOLDERS.items():
        if values.get(key):
            tex = tex.replace(placeholder, values[key])
    return tex, [p for p in AUTHOR_PLACEHOLDERS.values() if p in tex]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    for key, placeholder in AUTHOR_PLACEHOLDERS.items():
        ap.add_argument(f"--{key}", default=None, help=f"replaces {placeholder} in the author block")
    args = ap.parse_args()
    if shutil.which("pdflatex") is None:
        env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
        subprocess.run(["apt-get", "update", "-qq"], check=True, env=env)
        subprocess.run(["apt-get", "install", "-y", "-qq", "--no-install-recommends", *PACKAGES], check=True, env=env)
    work = Path(tempfile.mkdtemp(prefix="report_"))
    shutil.copytree(ROOT / "docs" / "report", work / "report")
    shutil.copytree(ROOT / "docs" / "figures", work / "figures")
    cwd = work / "report"
    tex, left = fill_author((cwd / "main.tex").read_text(encoding="utf-8"), vars(args))
    (cwd / "main.tex").write_text(tex, encoding="utf-8")
    latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
    for cmd in (latex, ["bibtex", "main"], latex, latex):
        subprocess.run(cmd, cwd=cwd, stdout=subprocess.DEVNULL)
    args.out.mkdir(parents=True, exist_ok=True)
    log = (cwd / "main.log").read_text(errors="replace") if (cwd / "main.log").exists() else ""
    (args.out / "main.log").write_text(log)
    if (cwd / "main.pdf").exists():
        shutil.copy(cwd / "main.pdf", args.out / "main.pdf")
    pages = re.search(r"Output written on main\.pdf \((\d+) pages?", log)
    warnings = [ln for ln in log.splitlines()
                if ln.startswith(("LaTeX Warning", "Class acmart Warning", "Overfull", "Underfull", "!"))]
    print(f"pages: {pages.group(1) if pages else 'no PDF'}; {len(warnings)} warnings / errors", flush=True)
    if left:
        print("  author placeholders left (see --author etc.): " + ", ".join(left), flush=True)
    for ln in warnings[:40]:
        print("  " + ln)


if __name__ == "__main__":
    main()
