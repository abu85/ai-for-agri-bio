"""
Build the course labs.

The single source of truth for every lab is a pair of plain scripts in
"percent" format (the same format Jupytext, VS Code and Spyder understand):

    labs/python/NN_name.py
    labs/r/NN_name.R

This tool turns them into:

1. Clean Jupyter notebooks for Google Colab   -> notebooks/{python,r}/NN_name.ipynb
2. Executed copies rendered as Markdown with
   outputs and figures, for the website       -> _labs/NN_name_{python,r}.md
                                                 figures/labs/*.png

Usage (from the repository root):

    python tools/build_labs.py                 # notebooks + run all offline labs
    python tools/build_labs.py --no-execute    # notebooks only, fast
    python tools/build_labs.py --only 03       # one lab

Executing needs Python with requirements.txt installed, R with the packages in
install.R, and the R Jupyter kernel (IRkernel::installspec()).
"""

import argparse
import base64
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
REPO = "YOUR-GITHUB-USERNAME/ai-for-agri-bio"

# id, executed when building the site? (False = needs internet/GPU, run in Colab)
LABS = [
    ("02_know_your_data", True),
    ("03_first_models", True),
    ("04_honest_evaluation", True),
    ("05_unsupervised", True),
    ("06_neural_networks", True),
    ("07a_plant_disease_images", False),
    ("07b_kmer_taxonomy", True),
    ("07c_protein_language_models", False),
    ("08_llm_verification", True),
]

LANGS = {
    "python": dict(ext=".py", kernel="python3", display="Python 3", language="python"),
    "r": dict(ext=".R", kernel="ir", display="R", language="R"),
}

MAX_OUTPUT_LINES = 40


def parse_percent(path):
    """Split a percent-format script into (kind, source) cells."""
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        lines = buf[:]
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if kind == "markdown":
            lines = [re.sub(r"^# ?", "", ln) for ln in lines]
        if lines:
            cells.append((kind, "\n".join(lines)))

    for line in path.read_text().splitlines():
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def make_notebook(cells, lang):
    spec = LANGS[lang]
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {"name": spec["kernel"], "display_name": spec["display"], "language": spec["language"]},
        "language_info": {"name": spec["language"]},
        "colab": {"provenance": []},
    }
    for kind, src in cells:
        nb.cells.append(new_markdown_cell(src) if kind == "markdown" else new_code_cell(src))
    return nb


def badges(lab_id):
    base = f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks"
    gh = f"https://github.com/{REPO}/blob/main"
    return (
        f'::: {{.lab-links}}\n'
        f'[Open Python notebook in Colab]({base}/python/{lab_id}.ipynb){{.btn .btn-sm .btn-primary}} '
        f'[Open R notebook in Colab]({base}/r/{lab_id}.ipynb){{.btn .btn-sm .btn-primary}} '
        f'[Python script]({gh}/labs/python/{lab_id}.py){{.btn .btn-sm .btn-outline-secondary}} '
        f'[R script]({gh}/labs/r/{lab_id}.R){{.btn .btn-sm .btn-outline-secondary}}\n'
        f':::\n'
    )


def text_block(text):
    lines = text.rstrip("\n").splitlines()
    if len(lines) > MAX_OUTPUT_LINES:
        lines = lines[:MAX_OUTPUT_LINES] + [f"... ({len(lines) - MAX_OUTPUT_LINES} more lines)"]
    body = "\n".join(lines)
    return f"```{{.text .lab-output}}\n{body}\n```\n"


def strip_ansi(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def render_markdown(nb, lab_id, lang, executed):
    """Turn a notebook (executed or not) into a Markdown fragment for the site."""
    fig_dir = ROOT / "figures" / "labs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for old in fig_dir.glob(f"{lab_id}_{lang}_*.png"):
        old.unlink()
    out, fig_n = [], 0
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            lines = cell.source.splitlines()
            if lines and lines[0].startswith("# "):
                lines = lines[1:]  # the page has its own title
            md = "\n".join(re.sub(r"^(#{2,5}) ", r"#\1 ", ln) for ln in lines)
            out.append(md.strip() + "\n")
            continue
        out.append(f"```{LANGS[lang]['language'].lower()}\n{cell.source}\n```\n")
        if not executed:
            continue
        for o in cell.get("outputs", []):
            if o.output_type == "stream":
                if o.name == "stderr":
                    continue  # package start-up messages
                out.append(text_block(strip_ansi(o.text)))
            elif o.output_type == "error":
                raise RuntimeError(f"{lab_id} ({lang}) failed:\n" + strip_ansi("\n".join(o.traceback)))
            elif o.output_type in ("execute_result", "display_data"):
                data = o.data
                if "image/png" in data:
                    fig_n += 1
                    name = f"{lab_id}_{lang}_{fig_n:02d}.png"
                    (fig_dir / name).write_bytes(base64.b64decode(data["image/png"]))
                    out.append(f"![](/figures/labs/{name}){{.lab-figure}}\n")
                elif lang == "python" and "text/html" in data and "<table" in data["text/html"]:
                    out.append(f"```{{=html}}\n<div class=\"lab-table\">{data['text/html']}</div>\n```\n")
                elif lang == "r" and "text/markdown" in data:
                    md = re.sub(r" &lt;\w+&gt;", "", data["text/markdown"])  # column type tags
                    md = re.sub(r"^A (data\.frame|matrix): .*\n", "", md.strip() + "\n", flags=re.M)
                    out.append("::: {.lab-table}\n" + md.strip() + "\n:::\n")
                elif "text/plain" in data:
                    out.append(text_block(strip_ansi(data["text/plain"])))
    if not executed:
        out.insert(0, "::: {.callout-note}\nThis lab needs internet access (and ideally a GPU), so it is shown "
                      "without outputs. Run it in Google Colab with the button above.\n:::\n")
    return "\n".join(out)


def execute(nb, lang):
    from nbclient import NotebookClient

    client = NotebookClient(
        nb,
        kernel_name=LANGS[lang]["kernel"],
        timeout=1800,
        resources={"metadata": {"path": str(ROOT / "notebooks" / lang)}},
    )
    client.execute()
    return nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-execute", action="store_true")
    ap.add_argument("--only", help="build labs whose id starts with this")
    ap.add_argument("--lang", choices=list(LANGS))
    args = ap.parse_args()

    (ROOT / "_labs").mkdir(exist_ok=True)
    for lab_id, offline in LABS:
        if args.only and not lab_id.startswith(args.only):
            continue
        (ROOT / "_labs" / f"{lab_id}_links.md").write_text(badges(lab_id))
        for lang, spec in LANGS.items():
            if args.lang and lang != args.lang:
                continue
            src = ROOT / "labs" / lang / f"{lab_id}{spec['ext']}"
            if not src.exists():
                print(f"  missing {src.relative_to(ROOT)}", file=sys.stderr)
                continue
            nb = make_notebook(parse_percent(src), lang)
            nbformat.write(nb, ROOT / "notebooks" / lang / f"{lab_id}.ipynb")
            run = offline and not args.no_execute
            if run:
                print(f"running {lab_id} ({lang}) ...", flush=True)
                nb = execute(nb, lang)
            frag = ROOT / "_labs" / f"{lab_id}_{lang}.md"
            if run or not offline or not frag.exists():
                frag.write_text(render_markdown(nb, lab_id, lang, executed=run))
            print(f"built {lab_id} ({lang})")


if __name__ == "__main__":
    main()
