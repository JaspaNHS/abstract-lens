"""
PASO 1 — Extrae texto de los PDFs en dos modos:
  - with_figs : texto completo incluyendo pies de figura y tablas
  - no_figs   : solo párrafos de texto, sin bloques de figura/tabla
Guarda chunks JSON en rag/chunks_with_figs/ y rag/chunks_no_figs/
"""

import sys, json, re
from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDF_DIR    = Path("../pdfs_blood_146_S1")
MANIFEST   = PDF_DIR / "manifest.json"
OUT_BASE   = Path(".")
CHUNK_SIZE = 600    # palabras por chunk aprox.
OVERLAP    = 80     # palabras de solapamiento

MODES = {
    "with_figs": OUT_BASE / "chunks_with_figs",
    "no_figs"  : OUT_BASE / "chunks_no_figs",
}
for p in MODES.values():
    p.mkdir(exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_blocks(pdf_path: Path) -> tuple[list[str], list[str]]:
    """
    Devuelve (all_blocks, text_only_blocks).
    all_blocks  : todo el texto de la página (incluye pies de figura/tabla)
    text_only   : solo bloques de párrafo, excluye bloques junto a imágenes
    """
    doc = fitz.open(pdf_path)
    all_blocks, text_only = [], []

    for page in doc:
        # get_image_info() devuelve dicts con "bbox" — más fiable que get_images()
        img_rects = [fitz.Rect(info["bbox"]) for info in page.get_image_info()
                     if info.get("bbox")]

        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, bno, btype = block
            if btype != 0:   # 0 = texto, 1 = imagen
                continue
            t = clean(text)
            if not t or len(t) < 15:
                continue

            all_blocks.append(t)

            # Comprobar si el bloque está adyacente a alguna imagen
            brect = fitz.Rect(x0, y0, x1, y1)
            near_image = any(
                brect.intersects(fitz.Rect(ir.x0-30, ir.y0-30, ir.x1+30, ir.y1+30))
                for ir in img_rects
            )
            if not near_image:
                text_only.append(t)

    doc.close()
    return all_blocks, text_only


def chunk_text(blocks: list[str], meta: dict, mode: str) -> list[dict]:
    """Divide los bloques en chunks con solapamiento y adjunta metadata."""
    words = " ".join(blocks).split()
    chunks = []
    start = 0
    cidx = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        text = " ".join(words[start:end])
        chunks.append({
            "id"   : f"{meta['pii']}_{mode}_{cidx}",
            "text" : text,
            "pii"  : meta["pii"],
            "doi"  : meta["doi"],
            "title": meta["title"],
            "mode" : mode,
            "chunk": cidx,
        })
        cidx += 1
        if end == len(words):
            break
        start = end - OVERLAP
    return chunks


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    with open(MANIFEST, encoding="utf-8") as f:
        articles = json.load(f)

    # Indexar artículos por PII para lookup rápido
    meta_by_pii = {a["pii"]: a for a in articles if a.get("pii")}

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    print(f"PDFs a procesar: {len(pdf_files)}")

    ok = err = skipped = 0

    for pdf_path in tqdm(pdf_files, desc="Procesando"):
        # Extraer PII del nombre de archivo o buscarlo en manifest
        stem = pdf_path.stem  # e.g. "0001_Titulo..."
        # Buscar en el manifest por posición o nombre
        idx_str = stem.split("_")[0]
        try:
            idx = int(idx_str) - 1
            art = articles[idx] if 0 <= idx < len(articles) else None
        except Exception:
            art = None

        if not art or not art.get("pii"):
            skipped += 1
            continue

        meta = {"pii": art["pii"], "doi": art.get("doi", ""), "title": art.get("title", "")}
        out_wf = MODES["with_figs"] / f"{art['pii']}.json"
        out_nf = MODES["no_figs"]   / f"{art['pii']}.json"

        if out_wf.exists() and out_nf.exists():
            skipped += 1
            continue

        try:
            all_blocks, text_only = extract_blocks(pdf_path)
            if not all_blocks:
                skipped += 1
                continue

            chunks_wf = chunk_text(all_blocks, meta, "with_figs")
            chunks_nf = chunk_text(text_only or all_blocks, meta, "no_figs")

            out_wf.write_text(json.dumps(chunks_wf, ensure_ascii=False), encoding="utf-8")
            out_nf.write_text(json.dumps(chunks_nf, ensure_ascii=False), encoding="utf-8")
            ok += 1
        except Exception as e:
            tqdm.write(f"  Error {pdf_path.name}: {e}")
            err += 1

    print(f"\nOK: {ok} | Errores: {err} | Saltados: {skipped}")
    print("Listo. Ejecuta: python 02_build_index.py")


if __name__ == "__main__":
    main()
