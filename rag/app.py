"""
Interfaz web para el sistema RAG de Blood Vol.146 Suppl.S1
Ejecutar: python app.py
Abrir: http://localhost:5000
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Importar la capa de síntesis (Claude)
from synthesize import synthesize as rag_synthesize, SYSTEM_PROMPT

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

DB_PATH = str(Path(__file__).parent / "chromadb")

COLLECTIONS = {
    "with_figs": "blood_with_figs",
    "no_figs":   "blood_no_figs",
}

app = Flask(__name__)

print("Loading ONNX embedding and ChromaDB...")
ef     = ONNXMiniLM_L6_V2()
client = chromadb.PersistentClient(path=DB_PATH)
cols   = {}
for key, name in COLLECTIONS.items():
    try:
        cols[key] = client.get_collection(name, embedding_function=ef)
        print(f"  [{key}] {cols[key].count():,} chunks")
    except Exception as e:
        print(f"  [{key}] Not available: {e}")

# Claude client (optional — synthesis only)
anthropic_client = None
if HAS_ANTHROPIC and os.environ.get("ANTHROPIC_API_KEY"):
    anthropic_client = anthropic.Anthropic()
    print("  [synthesis] Claude connected (claude-opus-4-8)")
else:
    print("  [synthesis] ANTHROPIC_API_KEY not set — search only")
print("Ready.\n")


def make_citation(meta: dict) -> dict:
    doi   = meta.get("doi", "")
    pii   = meta.get("pii", "")
    title = meta.get("title", "Sin título")
    url   = f"https://doi.org/{doi}" if doi else f"https://www.sciencedirect.com/science/article/pii/{pii}"
    return {"title": title, "url": url, "doi": doi, "pii": pii}


@app.route("/")
def index():
    stats = {k: cols[k].count() if k in cols else 0 for k in COLLECTIONS}
    return render_template("index.html", stats=stats, synthesis=bool(anthropic_client))


@app.route("/ask", methods=["POST"])
def ask():
    """RAG synthesis: answer based only on the abstracts, with citations."""
    if not anthropic_client:
        return jsonify({"error": "Synthesis unavailable: ANTHROPIC_API_KEY is not set on the server."}), 503

    data  = request.json
    query = (data.get("query") or "").strip()
    mode  = data.get("mode", "with_figs")
    if mode == "both":
        mode = "with_figs"

    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        answer, sources = rag_synthesize(query, mode, anthropic_client, cols)
    except Exception as e:
        return jsonify({"error": f"Synthesis error: {e}"}), 500

    src_list = []
    for s in sources:
        url = f"https://doi.org/{s['doi']}" if s.get("doi") else \
              f"https://www.sciencedirect.com/science/article/pii/{s['pii']}"
        src_list.append({
            "title": s["title"],
            "url":   url,
            "doi":   s.get("doi", ""),
            "score": s.get("score", 0),
        })

    return jsonify({"answer": answer, "sources": src_list})


@app.route("/search", methods=["POST"])
def search():
    data  = request.json
    query = (data.get("query") or "").strip()
    mode  = data.get("mode", "with_figs")
    top_k = min(int(data.get("top_k", 5)), 20)

    if not query:
        return jsonify({"error": "Empty query"}), 400

    modes_to_query = list(cols.keys()) if mode == "both" else [mode]
    all_results = []

    for m in modes_to_query:
        if m not in cols:
            continue
        res = cols[m].query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            score = round(1 - dist, 3)
            cit   = make_citation(meta)
            all_results.append({
                "score":   score,
                "mode":    m,
                "title":   cit["title"],
                "url":     cit["url"],
                "doi":     cit["doi"],
                "excerpt": doc[:500].strip(),
                "chunk":   meta.get("chunk", 0),
            })

    # Si ambos modos, deduplicar por DOI/PII y quedarse con mejor score
    if mode == "both":
        seen = {}
        for r in sorted(all_results, key=lambda x: -x["score"]):
            key = r["doi"] or r["url"]
            if key not in seen:
                seen[key] = r
        all_results = sorted(seen.values(), key=lambda x: -x["score"])[:top_k]

    return jsonify({"results": all_results, "total": len(all_results)})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
