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
import sys
import anthropic
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

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

2. CITE every claim with a [n] marker corresponding to the number of the fragment \
used. A single claim may carry several citations [1][3]. Citing is mandatory.

3. If the user's query is TOO VAGUE or broad to give a useful answer (for example \
"cancer", "treatment", "studies"), do NOT try to answer. Instead, politely ask them \
to narrow it down: which disease, which drug, which population, which outcome they \
are interested in. Be specific about what information would help.

4. Be CONCISE: maximum 2-3 paragraphs. Do not repeat the question. Go straight to \
the evidence.

5. Answer in the SAME language as the user's question.

6. Do not invent numerical data, drug names, or results. If a value is not in the \
fragments, do not provide it."""


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
            f"    Fragment: {s['text'][:1200]}"
        )
    return "\n\n".join(lines), sources


def synthesize(query: str, mode: str = "with_figs", client=None, cols=None):
    if cols is None:
        cols = load_collections()
    if client is None:
        client = anthropic.Anthropic()

    collection = cols.get(mode) or next(iter(cols.values()))
    chunks = retrieve(collection, query)
    context, sources = build_context(chunks)

    user_message = (
        f"RETRIEVED FRAGMENTS:\n\n{context}\n\n"
        f"{'='*60}\n"
        f"USER QUESTION: {query}\n\n"
        f"Answer following the rules. Cite with [n]. If it is too vague, ask for "
        f"more detail."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = "".join(b.text for b in response.content if b.type == "text")
    return answer, sources


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
        answer, sources = synthesize(q, mode, client, cols)
        print(answer)
        print("\n" + "-" * 64)
        print("SOURCES:")
        for i, s in enumerate(sources, 1):
            url = f"https://doi.org/{s['doi']}" if s["doi"] else \
                  f"https://www.sciencedirect.com/science/article/pii/{s['pii']}"
            print(f"  [{i}] {s['title'][:70]}\n      {url}")


if __name__ == "__main__":
    main()
