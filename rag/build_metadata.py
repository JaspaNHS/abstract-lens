"""
Builds meta_index.json: a per-article metadata lookup used for citation display
and session-type re-ranking.

For every abstract (keyed by PII) it records:
  - session_type : "plenary" | "regular" | "pubonly"   (parsed from the header line)
  - page         : the Blood 146 (2025) <N> locator (unique abstract number)
  - category     : the ASH session-category number (e.g. 642 — shared by a topic)
  - doi, title   : carried from the chunk metadata

Session type is reliably detectable for plenary (header "PLENARY SCIENTIFIC SESSION")
and publication-only (header "ONLINE PUBLICATION ONLY"). Everything else is a regular
oral/poster session — oral vs poster is NOT distinguishable from the abstract text
(they are interleaved by topic), so they share the "regular" tier.

Run once after processing PDFs:  python build_metadata.py
"""

import re
import sys
import json
import glob
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHUNKS_DIR = Path("chunks_with_figs")
MANIFEST   = Path("../pdfs_blood_146_S1/manifest.json")
OUT        = Path("meta_index.json")

PAGE_RE = re.compile(r"Blood 146 \(2025\)\s+(\d+)")
CAT_RE  = re.compile(r"67th ASH Annual Meeting Abstracts\s+(?:PLENARY SCIENTIFIC SESSION\s+|ONLINE PUBLICATION ONLY\s+)?(\d+[A-Z]?)\.")

# Importance ranking used for re-ranking retrieved fragments
TIER_RANK = {"plenary": 4, "oral": 3, "poster": 2, "pubonly": 1, "regular": 2, "unknown": 2}


def classify_from_text(text: str) -> str:
    head = text[:200]
    if "PLENARY SCIENTIFIC SESSION" in head:
        return "plenary"
    if "ONLINE PUBLICATION ONLY" in head:
        return "pubonly"
    return "regular"


def load_manifest_sections() -> dict:
    """pii -> authoritative section from the scraped TOC (if available)."""
    if not MANIFEST.exists():
        return {}
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out = {}
    for a in data:
        sec = a.get("section")
        if a.get("pii") and sec and sec != "unknown":
            out[a["pii"]] = sec
    return out


def main():
    manifest_sec = load_manifest_sections()
    if manifest_sec:
        print(f"Using TOC section tags from manifest for {len(manifest_sec):,} articles")
    else:
        print("No section tags in manifest — falling back to text classification "
              "(plenary/regular/pubonly only; oral vs poster not separable)")

    files = sorted(glob.glob(str(CHUNKS_DIR / "*.json")))
    index = {}
    counts = {}
    with_doi = 0

    for f in files:
        chunks = json.loads(Path(f).read_text(encoding="utf-8"))
        if not chunks:
            continue
        first = chunks[0]
        text  = first["text"]
        pii   = first["pii"]
        if not pii:
            continue

        # Authoritative TOC section if we have it; else infer from text
        stype = manifest_sec.get(pii) or classify_from_text(text)
        counts[stype] = counts.get(stype, 0) + 1

        pm = PAGE_RE.search(text[:200])
        cm = CAT_RE.search(text[:200])
        doi = first.get("doi", "")
        if doi:
            with_doi += 1

        index[pii] = {
            "session_type": stype,
            "tier_rank": TIER_RANK.get(stype, 2),
            "page":     int(pm.group(1)) if pm else None,
            "category": cm.group(1) if cm else None,
            "doi":      doi,
            "title":    first.get("title", ""),
        }

    OUT.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT} with {len(index):,} articles")
    print(f"  session types: {counts}")
    print(f"  with DOI: {with_doi:,} / {len(index):,}")


if __name__ == "__main__":
    main()
