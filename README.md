# Blood ASH 2025 — RAG Assistant

An OpenEvidence-style retrieval-augmented generation (RAG) system over the abstracts
of **Blood, Vol. 146, Supplement S1** (the 67th ASH Annual Meeting, 2025).

It downloads every abstract PDF through the **official Elsevier Developer API**, indexes
them locally, and answers questions **grounded only in those abstracts**, with a citation
for every claim. If a question is too vague, it asks for clarification instead of guessing.

> ⚠️ **Access & licensing.** PDFs are downloaded only via the official Elsevier Article
> Retrieval API using your own developer key, subject to Elsevier's terms. The PDFs and
> generated indexes are **not** included in this repository (see `.gitignore`).

---

## Pipeline

```
┌────────────────────┐   ┌──────────────────┐   ┌─────────────────────┐   ┌──────────────────┐
│ 1. Download PDFs   │ → │ 2. Extract text  │ → │ 3. Build vector     │ → │ 4. Ask / search  │
│ Elsevier API       │   │ two modes:       │   │ index (ChromaDB +   │   │ Claude synthesis │
│ + TOC scrape       │   │ with/without figs│   │ ONNX embeddings)    │   │ with citations   │
└────────────────────┘   └──────────────────┘   └─────────────────────┘   └──────────────────┘
```

**Two corpora are built in parallel:**
- `with_figs` — full text including figure captions and tables
- `no_figs` — body text only, figure/table blocks removed

---

## Setup

```bash
pip install -r requirements.txt
```

Set your API keys as environment variables (never hardcode them):

```powershell
# Elsevier Developer key — https://dev.elsevier.com
$env:ELSEVIER_API_KEY = '...'

# Anthropic key (synthesis layer) — https://console.anthropic.com
$env:ANTHROPIC_API_KEY = 'sk-ant-...'
```

---

## Usage

### 1. Download the abstract PDFs

```bash
python scrape_toc_selenium.py     # scrape the TOC → manifest.json (handles login)
python download_blood_pdfs.py     # download every PDF via the Elsevier API
```

### 2. Process + index

```bash
cd rag
python 01_process_pdfs.py         # extract text in both modes
python 02_build_index.py          # embed + store in ChromaDB
```

### 3. Query

```bash
# Command line
python 03_query.py                # interactive fragment search
python synthesize.py              # interactive cited synthesis (needs ANTHROPIC_API_KEY)

# Web app
python app.py                     # → http://localhost:5000
```

---

## Web app

`http://localhost:5000` offers two actions:

- **Synthesize answer** — Claude (`claude-opus-4-8`) reads the retrieved fragments and
  writes a 2–3 paragraph answer grounded **only** in the abstracts, with clickable `[n]`
  citations that jump to the source. Vague questions trigger a clarification request.
- **Search fragments** — direct semantic search returning the raw matching chunks.

Each corpus (`with figures` / `without figures`) is selectable.

---

## Files

| File | Purpose |
|---|---|
| `scrape_toc_selenium.py` | Scrape the ScienceDirect TOC → `manifest.json` |
| `download_blood_pdfs.py` | Download PDFs via the Elsevier Article Retrieval API |
| `rag/01_process_pdfs.py` | Extract text in both modes (PyMuPDF) |
| `rag/02_build_index.py` | Build the ChromaDB vector index (ONNX embeddings) |
| `rag/03_query.py` | Command-line fragment search |
| `rag/synthesize.py` | Cited synthesis layer (Claude) |
| `rag/app.py` | Flask web app |
| `rag/templates/index.html` | Web UI |

---

## Notes

- Embeddings use ChromaDB's built-in ONNX MiniLM model (no PyTorch required).
- The synthesis model is `claude-opus-4-8`; cost is roughly 1–3 cents per question.
- ~8,255 abstracts → ~13,700 chunks per corpus.
