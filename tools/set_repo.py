"""
Point every link in the course at your GitHub repository.

The Colab buttons, data download links and site URL use the placeholder
YOUR-GITHUB-USERNAME/ai-for-agri-bio. Run this once after creating your repo:

    python tools/set_repo.py your-username/your-repo-name

then commit the changes.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER_USER = "YOUR-GITHUB-USERNAME"
PLACEHOLDER_REPO = "ai-for-agri-bio"
SUFFIXES = {".qmd", ".md", ".yml", ".yaml", ".py", ".R", ".ipynb"}
SKIP_DIRS = {".git", "_site", ".quarto"}


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("usage: python tools/set_repo.py your-username/your-repo-name")
    user, repo = sys.argv[1].split("/", 1)
    replacements = [
        (f"{PLACEHOLDER_USER}/{PLACEHOLDER_REPO}", f"{user}/{repo}"),
        (f"{PLACEHOLDER_USER}.github.io/{PLACEHOLDER_REPO}", f"{user.lower()}.github.io/{repo}"),
    ]
    changed = 0
    for path in ROOT.rglob("*"):
        if path.suffix not in SUFFIXES or SKIP_DIRS & set(path.parts) or path.name == "set_repo.py":
            continue
        text = path.read_text(encoding="utf-8")
        new = text
        for old, rep in replacements:
            new = new.replace(old, rep)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed += 1
    print(f"Updated {changed} files to point at {user}/{repo}.")


if __name__ == "__main__":
    main()
