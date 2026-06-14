"""
PASO 3 — Interfaz de consulta RAG con citas.
Uso:
  python 03_query.py --query "efficacy of CAR-T in myeloma" --mode with_figs --top 5
  python 03_query.py  (modo interactivo)
"""

import sys, argparse
from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DB_PATH = str(Path(__file__).parent / "chromadb")

COLLECTIONS = {
    "with_figs": "blood_with_figs",
    "no_figs"  : "blood_no_figs",
}


def format_citation(meta: dict) -> str:
    doi   = meta.get("doi", "")
    title = meta.get("title", "Sin título")
    pii   = meta.get("pii", "")
    ref   = f"https://doi.org/{doi}" if doi else f"https://www.sciencedirect.com/science/article/pii/{pii}"
    return f'"{title}" — Blood 2025;146(Suppl 1) | {ref}'


def query_rag(
    query: str,
    mode: str = "with_figs",
    top_k: int = 5,
    model=None,
    collection=None,
) -> list[dict]:
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        score = round(1 - dist, 4)   # similitud coseno (1 = idéntico)
        hits.append({"text": doc, "meta": meta, "score": score})

    return hits


def print_results(query: str, hits: list[dict], mode: str):
    print(f"\n{'='*70}")
    print(f"  Consulta : {query}")
    print(f"  Modo     : {mode}")
    print(f"  Resultados: {len(hits)}")
    print(f"{'='*70}\n")

    for i, hit in enumerate(hits, 1):
        print(f"[{i}] Relevancia: {hit['score']:.3f}")
        print(f"    Cita: {format_citation(hit['meta'])}")
        print(f"    Chunk #{hit['meta'].get('chunk', '?')}:")
        # Mostrar hasta 400 chars del chunk
        excerpt = hit["text"][:400].replace("\n", " ")
        if len(hit["text"]) > 400:
            excerpt += "..."
        print(f"    {excerpt}")
        print()


def interactive_loop(model, collections: dict):
    print("\n" + "="*70)
    print("  RAG — Blood Vol.146 Suppl.S1")
    print("  Escribe tu pregunta o 'salir' para terminar.")
    print("  Comandos: /modo with_figs | /modo no_figs | /top N")
    print("="*70 + "\n")

    mode  = "with_figs"
    top_k = 5
    col   = collections[mode]

    while True:
        try:
            line = input(f"[{mode}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaliendo.")
            break

        if not line:
            continue
        if line.lower() in ("salir", "exit", "quit"):
            print("Saliendo.")
            break
        if line.startswith("/modo "):
            new_mode = line.split()[1]
            if new_mode in collections:
                mode = new_mode
                col  = collections[mode]
                print(f"  Modo cambiado a: {mode}")
            else:
                print(f"  Modos disponibles: {list(collections.keys())}")
            continue
        if line.startswith("/top "):
            try:
                top_k = int(line.split()[1])
                print(f"  Top-K cambiado a: {top_k}")
            except ValueError:
                print("  Uso: /top N")
            continue

        hits = query_rag(line, mode, top_k, ef, col)
        print_results(line, hits, mode)


def main():
    parser = argparse.ArgumentParser(description="RAG — Blood 2025 ASH Abstracts")
    parser.add_argument("--query", "-q", type=str, default=None, help="Consulta en texto libre")
    parser.add_argument("--mode",  "-m", type=str, default="with_figs",
                        choices=["with_figs", "no_figs"], help="Modo de indexado")
    parser.add_argument("--top",   "-n", type=int, default=5, help="Número de resultados")
    parser.add_argument("--both",  action="store_true", help="Consultar en ambos modos")
    args = parser.parse_args()

    print("Cargando base de datos y embedding ONNX...")
    ef     = ONNXMiniLM_L6_V2()
    client = chromadb.PersistentClient(path=DB_PATH)

    collections = {}
    for mode_key, col_name in COLLECTIONS.items():
        try:
            collections[mode_key] = client.get_collection(col_name, embedding_function=ef)
            n = collections[mode_key].count()
            print(f"  [{mode_key}] {col_name}: {n:,} chunks")
        except Exception as e:
            print(f"  [{mode_key}] No disponible ({e}). Ejecuta 02_build_index.py primero.")

    if not collections:
        print("No hay índices disponibles.")
        return

    if args.query:
        modes = list(collections.keys()) if args.both else [args.mode]
        for m in modes:
            if m in collections:
                hits = query_rag(args.query, m, args.top, ef, collections[m])
                print_results(args.query, hits, m)
    else:
        interactive_loop(ef, collections)


if __name__ == "__main__":
    main()
