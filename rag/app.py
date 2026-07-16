"""
Web interface for the Blood Vol.146 Suppl.S1 RAG system.
Run:   python app.py
Open:  http://localhost:5000
"""

import os
import sys
import json
import time
import datetime
from collections import deque, defaultdict
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response
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

# Access gate: when APP_PASSWORD is set, every request needs HTTP Basic auth.
# Leave it unset for local development (no gate). Username is ignored.
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
# Validation logging: every question (and its answer) is appended here.
LOG_PATH = Path(__file__).parent / "query_log.jsonl"

# Cost protection for the public tunnel.
MAX_QUERY_CHARS   = 1000                    # reject pathologically long questions
RATE_LIMIT_N      = 20                       # max synthesis calls ...
RATE_LIMIT_WINDOW = 60                       # ... per IP per this many seconds
MAX_DAILY_ASKS    = int(os.environ.get("MAX_DAILY_ASKS", "0"))  # 0 = unlimited
_hits = defaultdict(deque)                   # ip -> timestamps
_daily = {"date": None, "count": 0}

app = Flask(__name__)


def _client_ip():
    return request.headers.get("CF-Connecting-IP") or request.remote_addr or "?"


def _rate_limited(ip: str) -> bool:
    now = time.time()
    dq = _hits[ip]
    while dq and now - dq[0] > RATE_LIMIT_WINDOW:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_N:
        return True
    dq.append(now)
    return False


def _daily_exceeded() -> bool:
    if MAX_DAILY_ASKS <= 0:
        return False
    today = datetime.date.today().isoformat()
    if _daily["date"] != today:
        _daily["date"], _daily["count"] = today, 0
    if _daily["count"] >= MAX_DAILY_ASKS:
        return True
    _daily["count"] += 1
    return False


@app.before_request
def _gate():
    if not APP_PASSWORD:
        return  # open (local dev)
    auth = request.authorization
    if auth and auth.password == APP_PASSWORD:
        return
    return Response(
        "Abstract Lens — authentication required.", 401,
        {"WWW-Authenticate": 'Basic realm="Abstract Lens"'},
    )


def log_query(query, n_turn, answer, sources, cov):
    """Append a Q&A record for the validation phase."""
    try:
        rec = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "turn": n_turn,
            "query": query,
            "n_sources": len(sources),
            "sources": [{"num": s["num"], "tier": s["tier"],
                         "loc": s.get("locator", ""), "title": s["title"]}
                        for s in sources],
            "coverage": cov,
            "answer": answer,
        }
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"  [log] failed: {e}")

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
    if len(query) > MAX_QUERY_CHARS:
        return jsonify({"error": f"Question too long (max {MAX_QUERY_CHARS} characters)."}), 400
    if _rate_limited(_client_ip()):
        return jsonify({"error": "Too many requests — please wait a moment and try again."}), 429
    if _daily_exceeded():
        return jsonify({"error": "Daily question limit reached for this deployment."}), 429

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

    log_query(query, len(clean_hist) + 1, answer, src_list, cov)

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
    # PORT is provided by the host (Hugging Face Spaces uses 7860); defaults to 5000 locally.
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
