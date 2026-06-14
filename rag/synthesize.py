"""
OpenEvidence-style RAG synthesis layer.
Retrieves chunks from ChromaDB and uses Claude to synthesize an answer
based EXCLUSIVELY on the retrieved abstracts, with citations.

Features:
  - Grounded answers with mandatory [n] citations; no external knowledge.
  - Asks for clarification when the query is too vague.
  - Length scales with the breadth of the question.
  - Session-tier re-ranking: Plenary > Oral > Poster > Online-Publication-Only,
    so higher-importance abstracts surface first (data from meta_index.json).
  - Conversational follow-ups via a message history.
  - Coverage stats so the user can see how much of the corpus matched.
"""

import os
import re
import sys
import json
import anthropic
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

CITE_RE = re.compile(r"\[(\d+)\]")

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL    = "claude-opus-4-8"
DB_PATH  = str(Path(__file__).parent / "chromadb")
META_PATH = Path(__file__).parent / "meta_index.json"

N_CANDIDATES = 40   # fragments pulled from the vector store before re-ranking
N_CHUNKS     = 12   # distinct abstracts kept as context after re-ranking
RELEVANCE_FLOOR = 0.32   # cosine similarity below this is treated as "weak match"

# Session importance order and the bonus added to the cosine score when re-ranking.
# Relevance still dominates; the bonus only decides near-ties and sinks pub-only.
TIER_BONUS = {"plenary": 0.08, "oral": 0.04, "poster": 0.0, "pubonly": -0.06,
              "regular": 0.0, "unknown": 0.0}
TIER_LABEL = {"plenary": "Plenary", "oral": "Oral", "poster": "Poster",
              "pubonly": "Publication-only", "regular": "Session", "unknown": "Session"}

SYSTEM_PROMPT = """You are a biomedical research assistant that answers questions \
based EXCLUSIVELY on the fragments of scientific abstracts provided to you \
(from the journal Blood, Vol. 146, Supplement S1 — the abstracts of the 67th ASH \
Annual Meeting, 2025). Each fragment is tagged with its presentation tier \
(Plenary > Oral > Poster > Publication-only), reflecting the meeting's importance order.

STRICT RULES:

1. ONLY use information contained in the provided FRAGMENTS. Do not add external \
knowledge, do not fill in with what you know from other sources, do not speculate. \
If the fragments do not contain the answer, say so explicitly: "The retrieved \
abstracts do not contain enough information about this."

2. CITE every claim with a [n] marker whose number is the index of the fragment that \
supports it. ONLY cite fragment numbers that actually appear in the provided list \
(if you were given fragments [1] through [12], never write [13] or any number outside \
that range). A claim may carry several citations, e.g. [1][3]. Only cite a fragment if \
it genuinely supports the claim. Citing is mandatory for every factual claim.

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
wording). Report specific numbers ONLY when that exact figure appears in the fragment \
you cite. When a fragment only describes what a study measured — without stating the \
result — say the result is not reported. Never attach a citation [n] to a claim that \
fragment n does not directly support.

8. WEIGH EVIDENCE BY TIER. Lead with Plenary and Oral findings; treat Poster and \
especially Publication-only abstracts as lower-strength. When an important claim rests \
only on Poster or Publication-only fragments, note that briefly so the reader can judge \
the strength of the evidence.

9. CONVERSATION. If earlier turns are present, treat the new question as a follow-up on \
the same topic: build on what was already discussed, do not repeat it, and ground the \
new answer in the freshly provided fragments."""


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


def load_meta() -> dict:
    if META_PATH.exists():
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    return {}


def retrieve(collection, query: str, meta: dict, n_candidates=N_CANDIDATES) -> list[dict]:
    """Retrieve candidates, attach session metadata, and re-rank by tier."""
    res = collection.query(
        query_texts=[query],
        n_results=n_candidates,
        include=["documents", "metadatas", "distances"],
    )
    cand = []
    for doc, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        pii = m.get("pii", "")
        info = meta.get(pii, {})
        stype = info.get("session_type", "unknown")
        score = round(1 - dist, 3)
        cand.append({
            "text":  doc,
            "title": m.get("title", "Untitled"),
            "doi":   info.get("doi") or m.get("doi", ""),
            "pii":   pii,
            "page":  info.get("page"),
            "session_type": stype,
            "tier_label": TIER_LABEL.get(stype, "Session"),
            "score": score,
            "adj":   round(score + TIER_BONUS.get(stype, 0.0), 4),
        })
    # Deduplicate by article (keep best-adjusted fragment), then sort by adjusted score
    best = {}
    for c in cand:
        key = c["doi"] or c["pii"]
        if key not in best or c["adj"] > best[key]["adj"]:
            best[key] = c
    ranked = sorted(best.values(), key=lambda x: -x["adj"])
    return ranked


def coverage_stats(ranked: list[dict]) -> dict:
    """Summarize how much of the corpus matched, by tier and relevance."""
    strong = [c for c in ranked if c["score"] >= RELEVANCE_FLOOR]
    by_tier = {}
    for c in strong:
        by_tier[c["session_type"]] = by_tier.get(c["session_type"], 0) + 1
    return {
        "candidates_total": len(ranked),
        "above_floor": len(strong),
        "by_tier": by_tier,
        "floor": RELEVANCE_FLOOR,
    }


def build_context(sources: list[dict]) -> str:
    lines = []
    for i, s in enumerate(sources, 1):
        loc = f"Blood 2025;146(S1):{s['page']}" if s.get("page") else (s.get("doi") or s["pii"])
        lines.append(
            f"[{i}] ({s['tier_label']} session · {loc}) {s['title']}\n"
            f"    {s['text'][:2500]}"
        )
    return "\n\n".join(lines)


def reconcile_citations(answer: str, all_sources: list[dict]):
    """Keep only cited sources, renumber contiguously, strip out-of-range markers."""
    n = len(all_sources)
    cited_in_order, invalid = [], []
    for m in CITE_RE.finditer(answer):
        old = int(m.group(1))
        if 1 <= old <= n:
            if old not in cited_in_order:
                cited_in_order.append(old)
        elif old not in invalid:
            invalid.append(old)

    remap = {old: i + 1 for i, old in enumerate(cited_in_order)}

    def _sub(m):
        old = int(m.group(1))
        return f"[{remap[old]}]" if old in remap else ""

    rewritten = CITE_RE.sub(_sub, answer)
    rewritten = re.sub(r" {2,}", " ", rewritten).replace(" .", ".").replace(" ,", ",")

    cited_sources = []
    for old in cited_in_order:
        s = dict(all_sources[old - 1])
        s["num"] = remap[old]
        cited_sources.append(s)
    return rewritten, cited_sources, invalid


def synthesize(query: str, history=None, client=None, cols=None, meta=None, mode="with_figs"):
    """
    Returns (answer, cited_sources, invalid_citations, coverage).
    history: optional list of {"q": str, "a": str} prior turns (oldest first).
    """
    if cols is None:
        cols = load_collections()
    if client is None:
        client = anthropic.Anthropic()
    if meta is None:
        meta = load_meta()

    collection = cols.get(mode) or next(iter(cols.values()))

    # For follow-ups, resolve anaphora by retrieving on the last question + the new one
    retrieval_query = query
    if history:
        retrieval_query = f"{history[-1]['q']} {query}"

    ranked = retrieve(collection, retrieval_query, meta)
    cov = coverage_stats(ranked)
    sources = ranked[:N_CHUNKS]
    context = build_context(sources)

    # Build message history (clean Q&A turns) + current turn with fresh fragments
    messages = []
    for turn in (history or []):
        messages.append({"role": "user", "content": turn["q"]})
        messages.append({"role": "assistant", "content": turn["a"]})

    user_message = (
        f"RETRIEVED FRAGMENTS:\n\n{context}\n\n"
        f"{'='*60}\n"
        f"USER QUESTION: {query}\n\n"
        f"Answer following the rules. Cite with [n] using only the fragment numbers "
        f"above. If it is too vague, ask for more detail."
    )
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model=MODEL,
        max_tokens=2200,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    raw = "".join(b.text for b in response.content if b.type == "text")
    answer, cited_sources, invalid = reconcile_citations(raw, sources)
    return answer, cited_sources, invalid, cov


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set the ANTHROPIC_API_KEY environment variable")
        return

    cols = load_collections()
    meta = load_meta()
    client = anthropic.Anthropic()

    print("=" * 64)
    print("  Synthetic RAG — Blood Vol.146 Suppl.S1 (OpenEvidence-style)")
    print("  Type your question. 'exit' to quit. 'new' to reset the thread.")
    print("=" * 64)

    history = []
    while True:
        try:
            q = input(f"\n[turn {len(history)+1}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "new":
            history = []
            print("  (conversation reset)")
            continue

        print("\n  Synthesizing...\n")
        answer, sources, invalid, cov = synthesize(q, history, client, cols, meta)
        print(answer)
        if invalid:
            print(f"\n  [!] stripped hallucinated citations: {invalid}")
        print("\n  " + "-" * 60)
        print(f"  Coverage: {cov['above_floor']} abstracts matched above relevance "
              f"{cov['floor']} (by tier: {cov['by_tier']})")
        print("  CITED SOURCES:")
        for s in sources_cited(sources):
            loc = f"Blood 2025;146(S1):{s['page']}" if s.get("page") else s["pii"]
            doi = f" doi:{s['doi']}" if s.get("doi") else ""
            print(f"    [{s['num']}] ({s['tier_label']}) {s['title'][:60]}  {loc}{doi}")
        history.append({"q": q, "a": answer})


def sources_cited(sources):
    return sources


if __name__ == "__main__":
    main()
