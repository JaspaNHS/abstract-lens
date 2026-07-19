# Abstract Lens

A citation-grounded conversational assistant over the **8,255 abstracts of the 67th ASH
Annual Meeting** (Blood 2025;146 Supplement 1). Ask a question and it answers **only**
from what the abstracts actually say — with a citation on every claim — and declines when
the corpus does not contain the answer. Retrieval is re-ranked by presentation tier
(Plenary > Oral > Poster > Online-only).

> ⚠️ **Licensed corpus.** The abstract texts and the prebuilt search index are **not**
> included in this repository (they are covered by Elsevier's text-and-data-mining
> licence). To run the app you need the index — see step 3 below.

---

## Run it locally (quick start)

You need **Python 3.11+** and an **Anthropic API key**.

**1. Get the code**
```bash
git clone https://github.com/JaspaNHS/abstract-lens.git
cd abstract-lens
```

**2. Install the runtime dependencies**
```bash
pip install -r requirements-run.txt
```

**3. Get the index** (the licensed corpus — not in this repo)
Obtain `index.zip` from the study authors and place it in the project folder
(next to `run_local.py`). *(Authorized collaborators can instead set a `GH_TOKEN`
that can read the private index repo, and it will download automatically.)*

**4. Set your Anthropic API key**
- Windows PowerShell: `$env:ANTHROPIC_API_KEY = "sk-ant-..."`
- macOS / Linux: `export ANTHROPIC_API_KEY="sk-ant-..."`

Get a key at https://console.anthropic.com → API Keys (each question costs ~1–3 cents).

**5. Run it**
```bash
python run_local.py
```
Then open **http://localhost:5000** in your browser.

That's it. `run_local.py` unpacks the index the first time and starts the app.

> Optional: set `APP_PASSWORD` to require a password (recommended if you share the link,
> e.g. over a tunnel). Without it, the app is open on your machine only.

---

## What it does

- **Synthesize** — answers a question from the retrieved abstracts, with clickable `[n]`
  citations, and asks you to narrow down vague questions.
- **Search fragments** — direct semantic search over the abstracts.
- Every cited source shows its presentation tier and ASH communication number
  (`Blood 2025;146(S1):<page>`).

## How it was built (pipeline)

The `rag/` scripts reproduce the corpus and index from scratch (needs an Elsevier
Developer API key):

| Step | Script |
|---|---|
| Scrape the issue TOC (with section tags) | `scrape_toc_selenium.py` |
| Download abstract PDFs via the Elsevier API | `download_blood_pdfs.py` |
| Extract text | `rag/01_process_pdfs.py` |
| Build the vector index (ChromaDB + ONNX MiniLM) | `rag/02_build_index.py` |
| Per-article metadata / session tiers | `rag/build_metadata.py` |
| Recover any body-less abstracts | `rag/recover_missing.py` |
| Cited synthesis layer (Claude) | `rag/synthesize.py` |
| Web app | `rag/app.py` |

Evaluation: `rag/evaluate.py` (grounding + citation faithfulness) and
`rag/adversarial_test.py` (hallucination stress test).

Methods, results and figures: [`docs/METHODS_AND_RESULTS.md`](docs/METHODS_AND_RESULTS.md).

## Notes

- Embeddings run on CPU (ChromaDB's built-in ONNX MiniLM); no GPU needed.
- Synthesis model: `claude-opus-4-8`.
- The app is a research prototype. Answers are limited to the ASH 2025 abstract corpus,
  may be incomplete, and are **not medical advice**.
