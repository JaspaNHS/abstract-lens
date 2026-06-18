# Abstract Lens — Methods & Results (working draft)

> Working document to support writing the Methods and Results sections.
> All numbers below are **measured**, not estimated. Where a result is a technical
> sanity check rather than expert-validated accuracy, it is flagged as such.
> The expert-validated accuracy study (with a clinician-curated question set) is the
> planned next phase and is **not** covered here.

---

## 1. Methods

### 1.1 Corpus acquisition
- **Source:** *Blood*, Vol. 146, Supplement S1 — the abstracts of the 67th ASH Annual
  Meeting (2025), accessed through the official **Elsevier Article Retrieval API**
  (developer key) under the text-and-data-mining (TDM) licence.
- **Index of contents (TOC):** the issue TOC on ScienceDirect was traversed with a
  scripted browser (Selenium) to enumerate every abstract and its PII identifier. Each
  abstract was additionally tagged with the **presentation tier** it appears under in the
  TOC: *Plenary Scientific Session → Oral → Poster → Online Publication Only* (the
  meeting's order of prominence).
- **Retrieval:** each abstract PDF was downloaded via the Article Retrieval API by PII.
- **Pipeline integrity:** abstract counts were identical at every stage
  (TOC enumeration = PDF download = text processing = **8,255**), i.e. no silent loss.

### 1.2 Text extraction (parsing)
- Full text was extracted from each PDF with **PyMuPDF**.
- A quality-control scan over all 8,255 records measured: broken/replacement characters,
  presence of the abstract body, and word counts.
- **Body-recovery step:** 116 records (1.4%) whose downloaded PDF contained only a page
  header (no abstract body) were re-fetched from the API's full-text field
  (`coredata/dc:description`) and reinserted, bringing body-less records to **0**.
- *Note on tables/figures:* these abstracts contain no machine-extractable data tables
  (vector-table detection returned 0; the only embedded images are journal logos,
  ~49×59 pt). A text-only vs. text+tables comparison was therefore **not** pursued; for
  this corpus, tables add no information beyond the abstract text.

### 1.3 Indexing (vector store)
- Each abstract's text was embedded with a **MiniLM-L6-v2** sentence-embedding model
  (384-dim, ONNX runtime; no GPU required) and stored in a local **ChromaDB** vector
  store (cosine similarity).
- Retrieval unit: abstract text (long abstracts are split into ≤600-word chunks with
  overlap; results are de-duplicated to one fragment per abstract at query time).

### 1.4 Question answering (retrieval-augmented synthesis)
- For each question, the system retrieves the most similar fragments, **re-ranks them by
  presentation tier** (Plenary > Oral > Poster > Publication-only), and de-duplicates to
  distinct abstracts.
- The retrieved fragments are passed to a large language model
  (**Claude Opus 4.8**, `claude-opus-4-8`) under a strict system prompt that requires the
  model to:
  1. answer **only** from the supplied fragments (no external knowledge);
  2. attach a bracketed citation `[n]` to **every** factual claim;
  3. **decline / request clarification** when the question is too vague or the fragments
     do not contain the answer;
  4. scale answer length to the breadth of the question;
  5. weight evidence by presentation tier.
- A post-processing step keeps only the sources actually cited, renumbers them
  contiguously, and strips any out-of-range citation markers.
- Each answer is shown with: the cited abstracts (ASH communication number,
  `Blood 2025;146(Supplement 1):<page>`, and presentation tier), and a **coverage line**
  reporting how many abstracts matched above a relevance floor, by tier — making explicit
  that the answer reflects the closest matches, not an exhaustive review.

### 1.5 Interface and deployment
- A lightweight web interface supports multi-turn (follow-up) questions and a direct
  fragment-search mode.
- Verbatim source text shown to the user is capped at **200 characters** per snippet
  (Elsevier TDM compliance); synthesized answers are paraphrased, not quoted.
- For the evaluation phase the application is exposed over an authenticated HTTPS tunnel;
  every submitted question and its answer are logged for review.

### 1.6 Technical evaluation (this phase)
Two automated, ground-truth-free test suites were run against the live system:
- **Grounding / format suite** — checks that answers carry valid contiguous citations,
  are in English, and that vague queries trigger clarification; an LLM judge scores
  citation faithfulness (fraction of cited claims actually supported by the cited
  fragment).
- **Hallucination stress test** — adversarial probes: fabricated trials/drugs not in ASH
  2025, false-premise questions, and ambiguous cross-domain questions, plus an answerable
  control. Pass criterion: the system declines / corrects / asks for clarification rather
  than fabricating.

---

## 2. Results

### 2.1 Corpus and parsing
- **8,255 abstracts** ingested with no silent loss across the pipeline.
- Presentation-tier composition (**Figure 1**): 6 Plenary, 1,092 Oral, 5,330 Poster,
  1,827 Publication-only.
- Parse quality: **0** records with broken/replacement characters, **0** missing the
  abstract body after recovery (from 116 before recovery), median **725 words** per
  abstract.

### 2.2 Grounding and citation faithfulness *(exploratory queries — not expert-validated)*
On a set of exploratory queries:
- Valid, contiguous citations with no out-of-range (hallucinated) markers: **8/8**.
- Answers in English: **8/8**.
- Vague queries correctly handled with a clarification request: **8/8**.
- **Citation faithfulness** (LLM-judge; fraction of cited claims supported by the cited
  fragment): **92–100%** across answerable queries (**Figure 3**), all above the 90%
  threshold.

### 2.3 Hallucination stress test
The system showed correct grounding behaviour on **8/8** adversarial probes
(**Figure 2**):
- Fabricated trials/drugs not in the corpus: **3/3** declined.
- False-premise questions: **2/2** premise rejected.
- Ambiguous cross-domain questions: **2/2** clarification requested.
- Answerable control: **1/1** answered with citations.

No fabricated answers and no out-of-range citations were observed across the suite.

### 2.4 Summary
The corpus is complete and cleanly parsed, retrieval is tier-aware, and the system
adheres to its grounding constraints under both normal and adversarial questioning. These
are **technical-integrity results**; clinical accuracy will be assessed in the planned
expert-validation phase using a clinician-curated question set with reference answers.

---

## 3. Figures

| File | Caption |
|---|---|
| `figures/fig1_corpus_tiers.png` | **Figure 1.** Corpus composition by ASH 2025 presentation tier (n = 8,255). |
| `figures/fig2_stress_test.png` | **Figure 2.** Hallucination stress test: correct grounding behaviour on all adversarial categories (8/8). |
| `figures/fig3_faithfulness.png` | **Figure 3.** Citation faithfulness on exploratory queries (LLM-judge); all queries ≥ 90%. |

---

## 4. Reproducibility notes
- Embedding model: `BAAI`-style MiniLM-L6-v2 via ChromaDB's ONNX runtime (deterministic,
  CPU-only).
- Synthesis model: `claude-opus-4-8` (adaptive thinking, effort = medium).
- Retrieval: 40 candidates → tier re-rank → top 12 distinct abstracts as context.
- Relevance score reported in the UI is cosine similarity (0–1); it is a semantic-match
  score, not a measure of clinical importance.
- Code and pipeline scripts: see the repository (`scrape_toc_selenium.py`,
  `download_blood_pdfs.py`, `rag/01_process_pdfs.py`, `rag/02_build_index.py`,
  `rag/build_metadata.py`, `rag/recover_missing.py`, `rag/synthesize.py`,
  `rag/evaluate.py`, `rag/adversarial_test.py`, `rag/app.py`).

> ⚠️ **Caveat to keep in the paper:** all accuracy-related numbers here are from
> exploratory, non-expert-validated queries and an automated LLM judge. They establish
> that the system is technically sound and well-grounded — not that its answers are
> clinically correct. State this limitation explicitly.
