"""
PASO 2 — Crea embeddings y los almacena en ChromaDB.
Usa el embedding ONNX integrado de ChromaDB (sin PyTorch/torchvision).
Genera dos colecciones:
  blood_with_figs  — chunks con figuras
  blood_no_figs    — chunks solo texto
"""

import sys, json
from pathlib import Path
from tqdm import tqdm
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAG_DIR    = Path(".")
BATCH_SIZE = 64

MODES = {
    "with_figs": ("chunks_with_figs", "blood_with_figs"),
    "no_figs"  : ("chunks_no_figs",   "blood_no_figs"),
}


def load_all_chunks(chunks_dir: Path) -> list[dict]:
    chunks = []
    for f in sorted(chunks_dir.glob("*.json")):
        try:
            chunks.extend(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return chunks


def build_collection(ef, client, chunks: list[dict], collection_name: str):
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    texts     = [c["text"]  for c in chunks]
    ids       = [c["id"]    for c in chunks]
    metadatas = [{
        "pii"  : c["pii"],
        "doi"  : c["doi"],
        "title": c["title"][:200],
        "mode" : c["mode"],
        "chunk": c["chunk"],
    } for c in chunks]

    print(f"  Indexando {len(texts):,} chunks en '{collection_name}'...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  Batches"):
        col.add(
            documents=texts[i:i+BATCH_SIZE],
            ids=ids[i:i+BATCH_SIZE],
            metadatas=metadatas[i:i+BATCH_SIZE],
        )

    print(f"  Listo: {col.count():,} chunks indexados.\n")


def main():
    print("Inicializando embedding ONNX (MiniLM-L6-v2)...")
    ef = ONNXMiniLM_L6_V2()

    db_path = str(RAG_DIR / "chromadb")
    client  = chromadb.PersistentClient(path=db_path)
    print(f"ChromaDB en: {db_path}\n")

    for mode_key, (chunks_dir_name, col_name) in MODES.items():
        chunks_dir = RAG_DIR / chunks_dir_name
        if not chunks_dir.exists():
            print(f"[{mode_key}] No encontrado: {chunks_dir}")
            continue

        print(f"[{mode_key}] Cargando chunks de {chunks_dir_name}/...")
        chunks = load_all_chunks(chunks_dir)
        print(f"  {len(chunks):,} chunks cargados.")

        if not chunks:
            print("  Sin chunks. Saltando.\n")
            continue

        build_collection(ef, client, chunks, col_name)

    print("Índice construido. Ejecuta: python 03_query.py")


if __name__ == "__main__":
    main()
