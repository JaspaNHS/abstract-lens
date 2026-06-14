"""
OpenEvidence-style RAG synthesis layer.
Retrieves chunks from ChromaDB and uses Claude to synthesize an answer
based EXCLUSIVELY on the retrieved abstracts, with citations.

Rules:
  - Only uses information present in the retrieved fragments (no external knowledge)
  - Cites every claim with [n] pointing to the source
  - If the query is too vague, asks for more context instead of answering
  - Maximum 2-3 paragraphs
"""

import os
import re
import sys
import anthropic
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

CITE_RE = re.compile(r"\[(\d+)\]")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL    = "claude-opus-4-8"
DB_PATH  = str(Path(__file__).parent / "chromadb")
N_CHUNKS = 12   # fragments retrieved as context

SYSTEM_PROMPT = """You are a biomedical research assistant that answers questions \
based EXCLUSIVELY on the fragments of scientific abstracts provided to you \
(from the journal Blood, Vol. 146, Supplement S1 — the abstracts of the 67th ASH \
Annual Meeting, 2025).

STRICT RULES:

1. ONLY use information contained in the provided FRAGMENTS. Do not add external \
knowledge, do not fill in with what you know from other sources, do not speculate. \
If the fragments do not contain the answer, say so explicitly: "The retrieved \
abstracts do not contain enough information about this."

2. CITE every claim with a [n] marker whose number is the index of the fragment that \
supports it. ONLY cite fragment numbers that actually appear in the provided list \
(if you were given fragments [1] through [12], never write [13] or any number outside \
that range). A claim may carry several citations, e.g. [1][3]. Only cite a fragment if \
it genuinely supports the claim — do not attach a citation to a fragment that is not \
actually about what you are stating. Citing is mandatory for every factual claim.

3. If the user's query is TOO VAGUE or broad to give a useful answer (for example \
"cancer", "treatment", "studies"), do NOT try to answer. Instead, politely ask them \
to narrow it down: which disease, which drug, which population, which outcome they \
are interested in. Be specific about what information would help.

4. CALIBRATE LENGTH to the breadth of the question:
   - Narrow/specific question (one drug, one outcome) → 1-3 concise paragraphs.
   - Broad question spanning several studies or subtopics → a more comprehensive \
answer (up to ~6 paragraphs), organized by subtopic, synthesizing across the relevant \
fragments. Be thorough but stay grounded and cited; never pad.
   Do not repeat the question. Go straight to the evidence.

5. ALWAYS answer in ENGLISH, regardless of the language of the user's question.

6. Do not invent numerical data, drug names, or results. If a value is not in the \
fragments, do not provide it.

7. STICK TO WHAT THE FRAGMENT STATES. Do not add evaluative or editorializing \
language that the fragment itself does not contain (e.g. do not call a result \
"highly effective", "impressive", or "promising" unless the fragment uses such \
wording). Report specific numbers (response rates, thresholds, percentages, doses) \
ONLY when that exact figure appears in the fragment you cite. When a fragment only \
describes what a study measured or set out to do — without stating the result — say \
that the result is not reported, rather than implying an outcome. Never attach a \
citation [n] to a claim that fragment n does not directly support."""


def load_collections():
    ef = ONNXMiniLM_L6_V2()
    client = chromadb.PersistentClient(path=DB_PATH)
    cols = {}
    for key, name in {"with_figs": "blood_with_figs", "no_figs": "blood_no_figs"}.items():
        try:
            cols[key] = client.get_collection(name, embedding_function=ef)
        except Exception:
            pass
    return cols


def retrieve(collection, query: str, n: int = N_CHUNKS) -> list[dict]:
    res = collection.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append({
            "text":  doc,
            "title": meta.get("title", "Untitled"),
            "doi":   meta.get("doi", ""),
            "pii":   meta.get("pii", ""),
            "score": round(1 - dist, 3),
        })
    return out


def build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """Builds the numbered context block and the source list."""
    # Deduplicate by article, keeping the best fragment of each
    seen = {}
    for c in chunks:
        key = c["doi"] or c["pii"]
        if key not in seen:
            seen[key] = c
    sources = list(seen.values())

    lines = []
    for i, s in enumerate(sources, 1):
        url = f"https://doi.org/{s['doi']}" if s["doi"] else \
              f"https://www.sciencedirect.com/science/article/pii/{s['pii']}"
        lines.append(
            f"[{i}] Title: {s['title']}\n"
            f"    Source: {url}\n"
            f"    Fragment: {s['text'][:2500]}"
        )
    return "\n\n".join(lines), sources


def reconcile_citations(answer: str, all_sources: list[dict]) -> tuple[str, list[dict], list[int]]:
    """
    Keep only the sources actually cited in the answer, renumber them contiguously
    by order of first appearance, and rewrite the [n] markers to match.

    Returns (rewritten_answer, cited_sources, invalid_citations).
    - cited_sources: only the sources referenced in the text, renumbered 1..k,
      each annotated with its new "num".
    - invalid_citations: any [n] in the text that did not map to a real fragment
      (n out of range) — these are hallucinated citations and are stripped from
      the text.
    """
    n_sources = len(all_sources)
    cited_in_order: list[int] = []
    invalid: list[int] = []

    for m in CITE_RE.finditer(answer):
        old = int(m.group(1))
        if 1 <= old <= n_sources:
            if old not in cited_in_order:
                cited_in_order.append(old)
        else:
            if old not in invalid:
                invalid.append(old)

    # old (1-indexed) -> new (1-indexed, contiguous)
    remap = {old: i + 1 for i, old in enumerate(cited_in_order)}

    def _sub(m):
        old = int(m.group(1))
        if old in remap:
            return f"[{remap[old]}]"
        return ""  # strip hallucinated / out-of-range citations

    rewritten = CITE_RE.sub(_sub, answer)
    # tidy up any doubled spaces left by stripped markers
    rewritten = re.sub(r" {2,}", " ", rewritten).replace(" .", ".").replace(" ,", ",")

    cited_sources = []
    for old in cited_in_order:
        s = dict(all_sources[old - 1])
        s["num"] = remap[old]
        cited_sources.append(s)

    return rewritten, cited_sources, invalid


def synthesize(query: str, mode: str = "with_figs", client=None, cols=None):
    """Returns (answer, cited_sources, invalid_citations)."""
    if cols is None:
        cols = load_collections()
    if client is None:
        client = anthropic.Anthropic()

    collection = cols.get(mode) or next(iter(cols.values()))
    chunks = retrieve(collection, query)
    context, all_sources = build_context(chunks)

    user_message = (
        f"RETRIEVED FRAGMENTS:\n\n{context}\n\n"
        f"{'='*60}\n"
        f"USER QUESTION: {query}\n\n"
        f"Answer following the rules. Cite with [n] using only the fragment numbers "
        f"above. If it is too vague, ask for more detail."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=2200,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_answer = "".join(b.text for b in response.content if b.type == "text")
    answer, cited_sources, invalid = reconcile_citations(raw_answer, all_sources)
    return answer, cited_sources, invalid


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set the ANTHROPIC_API_KEY environment variable")
        print("  PowerShell:  $env:ANTHROPIC_API_KEY = 'sk-ant-...'")
        return

    cols = load_collections()
    client = anthropic.Anthropic()

    print("=" * 64)
    print("  Synthetic RAG — Blood Vol.146 Suppl.S1 (OpenEvidence-style)")
    print("  Type your question. 'exit' to quit.")
    print("  Commands: /mode with_figs | /mode no_figs")
    print("=" * 64)

    mode = "with_figs"
    while True:
        try:
            q = input(f"\n[{mode}] Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        if q.startswith("/mode "):
            m = q.split()[1]
            if m in cols:
                mode = m
                print(f"  Mode: {mode}")
            continue

        print("\n  Synthesizing...\n")
        answer, sources, invalid = synthesize(q, mode, client, cols)
        print(answer)
        if invalid:
            print(f"\n  [!] Stripped {len(invalid)} hallucinated citation(s): {invalid}")
        print("\n" + "-" * 64)
        print("CITED SOURCES:")
        for s in sources:
            url = f"https://doi.org/{s['doi']}" if s["doi"] else \
                  f"https://www.sciencedirect.com/science/article/pii/{s['pii']}"
            print(f"  [{s['num']}] {s['title'][:70]}\n      {url}")


if __name__ == "__main__":
    main()
