#!/usr/bin/env python3
"""
One-command local launcher for Abstract Lens.

  python run_local.py

It makes sure the prebuilt index is in place, then starts the web app at
http://localhost:5000 (open that in a browser).

Requirements
------------
1. Python 3.11+ with the runtime deps:
       pip install -r requirements-run.txt
2. Your Anthropic API key, so the assistant can answer:
       Windows PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
       macOS/Linux:         export ANTHROPIC_API_KEY="sk-ant-..."
   (Without it, only the fragment-search mode works.)
3. The prebuilt index (the licensed abstract corpus — NOT in this public repo).
   Get `index.zip` from the study authors and place it in this folder, OR set a
   GitHub token that can read the private index repo:
       $env:GH_TOKEN = "github_pat_..."     # optional auto-download

Optional
--------
   $env:APP_PASSWORD = "your-password"   # gate the app (recommended if you expose it)
"""

import os
import sys
import zipfile
import subprocess
from pathlib import Path

# Avoid Windows console (cp1252) crashes on any non-ASCII output.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT   = Path(__file__).resolve().parent
RAG    = ROOT / "rag"
CHROMA = RAG / "chromadb"
META   = RAG / "meta_index.json"
ZIP    = ROOT / "index.zip"

INDEX_REPO = os.environ.get("INDEX_REPO", "JaspaNHS/abstract-lens-index")
INDEX_TAG  = os.environ.get("INDEX_TAG", "index-v1")


def _extract(zip_path: Path):
    print(f"Extracting {zip_path.name} into rag/ ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAG)


def _download_private_index(token: str):
    import requests
    print(f"Fetching index from private release {INDEX_REPO}@{INDEX_TAG} ...")
    r = requests.get(
        f"https://api.github.com/repos/{INDEX_REPO}/releases/tags/{INDEX_TAG}",
        headers={"Authorization": f"token {token}"}, timeout=30)
    r.raise_for_status()
    asset = next(a for a in r.json()["assets"] if a["name"] == "index.zip")
    with requests.get(asset["url"], stream=True, timeout=120,
                      headers={"Authorization": f"token {token}",
                               "Accept": "application/octet-stream"}) as resp:
        resp.raise_for_status()
        with open(ZIP, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
    print("Downloaded index.zip")


def ensure_index():
    if CHROMA.exists() and META.exists():
        print("Index already present - good.")
        return
    if ZIP.exists():
        _extract(ZIP)
    elif os.environ.get("GH_TOKEN"):
        _download_private_index(os.environ["GH_TOKEN"])
        _extract(ZIP)
    else:
        sys.exit(
            "\nERROR: the prebuilt index was not found.\n"
            "  Abstract Lens needs the abstract corpus index, which is not in this public\n"
            "  repository (licensed content). Do ONE of:\n"
            "    (a) get `index.zip` from the study authors and put it in this folder, or\n"
            "    (b) set GH_TOKEN to a token that can read the private index repo.\n"
            "  Then run `python run_local.py` again.\n")


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("WARNING: ANTHROPIC_API_KEY is not set — synthesis is disabled (search still works).")
    ensure_index()
    print("\nStarting Abstract Lens -> open http://localhost:5000 in your browser.\n")
    subprocess.run([sys.executable, "app.py"], cwd=RAG)


if __name__ == "__main__":
    main()
