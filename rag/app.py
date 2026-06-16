"""
Web interface for the Blood Vol.146 Suppl.S1 RAG system.
Run:   python app.py
Open:  http://localhost:5000
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from synthesize import (
    synthesize as rag_synthesize,
    load_meta,
    TIER_LABEL,
)

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

DB_PATH = str(Path(__file__).parent / "chromadb")
CORPUS  = "blood_with_figs"   # figures/no-figures toggle removed — always with figures

app = Flask(__name__)

print("Loading ONNX embedding and ChromaDB...")
ef     = ONNXMiniLM_L6_V2()
client = chromadb.PersistentClient(path=DB_PATH)
cols   = {}
try:
    cols["with_figs"] = client.get_collection(CORPUS, embedding_function=ef)
    print(f"  corpus: {cols['with_figs'].count():,} chunks")
except Exception as e:
    print(f"  corpus not available: {e}")

meta = load_meta()
print(f"  metadata: {len(meta):,} articles with session tags")

anthropic_client = None
if HAS_ANTHROPIC and os.environ.get("ANTHROPIC_API_KEY"):
    anthropic_client = anthropic.Anthropic()
    print("  [synthesis] Claude connected (claude-opus-4-8)")
else:
    print("  [synthesis] ANTHROPIC_API_KEY not set — search only")
print("Ready.\n")


def source_url(doi: str, pii: str) -> str:
    if doi:
        return f"https://doi.org/{doi}"
    return f"https://www.sciencedirect.com/science/article/pii/{pii}"


@app.route("/")
def index():
    n = cols["with_figs"].count() if "with_figs" in cols else 0
    return render_template("index.html", n_chunks=n, synthesis=bool(anthropic_client))


@app.route("/ask", methods=["POST"])
def ask():
    """RAG synthesis: answer grounded only in the abstracts, with citations."""
    if not anthropic_client:
        return jsonify({"error": "Synthesis unavailable: ANTHROPIC_API_KEY is not set on the server."}), 503

    data    = request.json or {}
    query   = (data.get("query") or "").strip()
    history = data.get("history") or []   # [{"q":..., "a":...}, ...]

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # sanitize history
    clean_hist = [{"q": str(t.get("q", "")), "a": str(t.get("a", ""))}
                  for t in history if t.get("q") and t.get("a")][-6:]

    try:
        answer, sources, invalid, cov = rag_synthesize(
            query, clean_hist, anthropic_client, cols, meta
        )
    except Exception as e:
        return jsonify({"error": f"Synthesis error: {e}"}), 500

    src_list = []
    for s in sources:
        loc = f"Blood 2025;146(S1):{s['page']}" if s.get("page") else ""
        src_list.append({
            "num":   s["num"],
            "title": s["title"],
            "url":   source_url(s.get("doi", ""), s.get("pii", "")),
            "doi":   s.get("doi", ""),
            "page":  s.get("page"),
            "locator": loc,
            "tier":  s.get("tier_label", "Session"),
            "session_type": s.get("session_type", "unknown"),
            "score": s.get("score", 0),
        })

    return jsonify({
        "answer": answer,
        "sources": src_list,
        "invalid_citations": invalid,
        "coverage": cov,
    })


@app.route("/search", methods=["POST"])
def search():
    """Direct semantic fragment search (with figures corpus)."""
    data  = request.json or {}
    query = (data.get("query") or "").strip()
    top_k = min(int(data.get("top_k", 8)), 20)
    if not query:
        return jsonify({"error": "Empty query"}), 400
    if "with_figs" not in cols:
        return jsonify({"error": "Corpus not available"}), 503

    res = cols["with_figs"].query(
        query_texts=[query], n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for doc, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        pii = m.get("pii", "")
        info = meta.get(pii, {})
        doi = info.get("doi") or m.get("doi", "")
        stype = info.get("session_type", "unknown")
        results.append({
            "score":   round(1 - dist, 3),
            "title":   m.get("title", "Untitled"),
            "url":     source_url(doi, pii),
            "doi":     doi,
            "tier":    TIER_LABEL.get(stype, "Session"),
            # TDM compliance: never display more than 200 verbatim characters
            "excerpt": doc.strip()[:200],
        })
    return jsonify({"results": results, "total": len(results)})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
