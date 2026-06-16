"""
Recovers abstracts whose downloaded PDF contained only a page header (no body).
For each such record, fetches the full abstract text from the Elsevier Article
Retrieval API (dc:description), rebuilds its chunk files, updates meta_index.json,
and re-embeds it in both ChromaDB collections.

Run:  ELSEVIER_API_KEY=... ANTHROPIC_API_KEY not needed here
      python recover_missing.py
"""

import os, re, sys, io, json, glob, time
import requests
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

API_KEY = os.environ.get("ELSEVIER_API_KEY", "")
if not API_KEY:
    sys.exit("ERROR: set ELSEVIER_API_KEY")

DB_PATH    = str(Path(__file__).parent / "chromadb")
META_PATH  = Path(__file__).parent / "meta_index.json"
WF_DIR     = Path("chunks_with_figs")
NF_DIR     = Path("chunks_no_figs")
CHUNK_SIZE = 600
OVERLAP    = 80
WORD_FLOOR = 60   # records below this are treated as body-less


def clean(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def find_bodyless() -> list[str]:
    piis = []
    for f in sorted(glob.glob(str(WF_DIR / "*.json"))):
        c = json.loads(Path(f).read_text(encoding="utf-8"))
        if not c:
            continue
        if len(" ".join(x["text"] for x in c).split()) < WORD_FLOOR:
            piis.append(c[0]["pii"])
    return piis


def fetch_body(pii: str) -> dict:
    r = requests.get(
        f"https://api.elsevier.com/content/article/pii/{pii}",
        headers={"X-ELS-APIKey": API_KEY, "Accept": "application/json"},
        timeout=30,
    )
    if r.status_code != 200:
        return {}
    cd = r.json().get("full-text-retrieval-response", {}).get("coredata", {})
    return {
        "title": clean(cd.get("dc:title", "")),
        "body":  clean(cd.get("dc:description", "")),
        "page":  cd.get("prism:startingPage", ""),
    }


def chunk_text(text: str, pii: str, title: str, mode: str) -> list[dict]:
    words = text.split()
    out, start, idx = [], 0, 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        out.append({
            "id": f"{pii}_{mode}_{idx}", "text": " ".join(words[start:end]),
            "pii": pii, "doi": "", "title": title[:120], "mode": mode, "chunk": idx,
        })
        idx += 1
        if end == len(words):
            break
        start = end - OVERLAP
    return out


def main():
    piis = find_bodyless()
    print(f"Body-less records to recover: {len(piis)}")

    ef = ONNXMiniLM_L6_V2()
    client = chromadb.PersistentClient(path=DB_PATH)
    col_wf = client.get_collection("blood_with_figs", embedding_function=ef)
    col_nf = client.get_collection("blood_no_figs", embedding_function=ef)
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}

    recovered = failed = 0
    for i, pii in enumerate(piis, 1):
        info = fetch_body(pii)
        body, title, page = info.get("body", ""), info.get("title", ""), info.get("page", "")
        if not body or len(body.split()) < WORD_FLOOR:
            failed += 1
            print(f"  [{i}/{len(piis)}] still empty: {pii} ({len(body.split())} words)")
            time.sleep(0.3)
            continue

        # New chunk text = title + recovered body (so it is self-describing)
        full = f"{title}. {body}" if title else body
        wf_chunks = chunk_text(full, pii, title, "with_figs")
        nf_chunks = chunk_text(full, pii, title, "no_figs")
        (WF_DIR / f"{pii}.json").write_text(json.dumps(wf_chunks, ensure_ascii=False), encoding="utf-8")
        (NF_DIR / f"{pii}.json").write_text(json.dumps(nf_chunks, ensure_ascii=False), encoding="utf-8")

        # Update meta_index (keep authoritative session_type/tier; refresh page/title)
        m = meta.get(pii, {})
        if page and str(page).strip().isdigit():
            m["page"] = int(str(page).strip())
        m["title"] = title[:120]
        m.setdefault("session_type", "poster")
        m.setdefault("tier_rank", 2)
        m.setdefault("doi", "")
        m.setdefault("category", None)
        meta[pii] = m

        # Re-embed in ChromaDB: delete old chunks for this pii, add new
        for col, chunks in ((col_wf, wf_chunks), (col_nf, nf_chunks)):
            try:
                col.delete(where={"pii": pii})
            except Exception:
                pass
            col.add(
                documents=[c["text"] for c in chunks],
                ids=[c["id"] for c in chunks],
                metadatas=[{"pii": c["pii"], "doi": "", "title": c["title"],
                            "mode": c["mode"], "chunk": c["chunk"]} for c in chunks],
            )

        recovered += 1
        if i % 20 == 0 or i == len(piis):
            print(f"  [{i}/{len(piis)}] recovered so far: {recovered}")
        time.sleep(0.3)

    META_PATH.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Recovered: {recovered}  Still empty: {failed}")
    print(f"Collections now: with_figs={col_wf.count():,}  no_figs={col_nf.count():,}")


if __name__ == "__main__":
    main()
