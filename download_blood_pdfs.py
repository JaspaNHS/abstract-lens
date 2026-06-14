"""
PDF downloader - Blood Vol.146 Suppl.S1
Strategy:
  1. Scrape the ScienceDirect TOC to get PIIs/DOIs
  2. Download PDFs via the official Elsevier Article Retrieval API
"""

import os
import re
import sys
import time
import json
import requests
from pathlib import Path

# Force UTF-8 output to avoid crashes on Unicode characters (β, γ, etc.)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from bs4 import BeautifulSoup

# Elsevier Developer API key — read from the environment (never hardcode).
#   PowerShell:  $env:ELSEVIER_API_KEY = '...'
API_KEY  = os.environ.get("ELSEVIER_API_KEY", "")
if not API_KEY:
    sys.exit("ERROR: set the ELSEVIER_API_KEY environment variable.")
TOC_URL  = "https://www.sciencedirect.com/journal/blood/vol/146/suppl/S1"
OUTPUT_DIR = Path("pdfs_blood_146_S1")
OUTPUT_DIR.mkdir(exist_ok=True)

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

API_HEADERS = {
    "X-ELS-APIKey": API_KEY,
    "Accept": "application/pdf",
}


# ── 1. Obtener lista de artículos ────────────────────────────────────────────

def fetch_toc_articles() -> list[dict]:
    """Scraping del TOC de ScienceDirect para extraer PIIs y títulos."""
    print(f"Descargando TOC desde: {TOC_URL}")
    session = requests.Session()
    articles = []
    page = 0

    while True:
        params = {"page": page} if page > 0 else {}
        resp = session.get(TOC_URL, headers=BROWSER_HEADERS, params=params, timeout=30)
        print(f"  Página {page}: HTTP {resp.status_code}")

        if resp.status_code != 200:
            print(f"  Error al obtener TOC: {resp.status_code}")
            break

        soup = BeautifulSoup(resp.text, "lxml")

        # Extraer PIIs desde los enlaces de artículo
        found_in_page = 0
        for a in soup.select("a[href*='/science/article/pii/']"):
            href = a.get("href", "")
            m = re.search(r"/science/article/pii/([A-Z0-9]+)", href)
            if not m:
                continue
            pii = m.group(1)
            # Evitar duplicados
            if any(art["pii"] == pii for art in articles):
                continue
            # Intentar obtener título del contexto
            title_el = a.find("span", class_=re.compile("title|article-title", re.I))
            title = title_el.get_text(strip=True) if title_el else a.get_text(strip=True)
            if not title:
                title = pii
            articles.append({"pii": pii, "doi": "", "title": title[:120]})
            found_in_page += 1

        print(f"    Encontrados en esta página: {found_in_page}")

        # Verificar si hay más páginas
        next_btn = soup.select_one("a.next-link, li.next a, [aria-label='Next page']")
        if not next_btn or found_in_page == 0:
            break

        page += 1
        time.sleep(1)

    print(f"  Total PIIs extraídos: {len(articles)}")
    return articles


def enrich_with_api(articles: list[dict]) -> list[dict]:
    """Enriquece los artículos con DOI via Elsevier Article Retrieval API."""
    print("\nEnriqueciendo con DOIs via API...")
    json_headers = {**API_HEADERS, "Accept": "application/json"}

    for i, art in enumerate(articles, 1):
        pii = art["pii"]
        url = f"https://api.elsevier.com/content/article/pii/{pii}"
        try:
            r = requests.get(url, headers=json_headers, timeout=15)
            if r.status_code == 200:
                data = r.json().get("full-text-retrieval-response", {})
                coredata = data.get("coredata", {})
                doi = coredata.get("prism:doi", "")
                title = coredata.get("dc:title", art["title"])
                art["doi"]   = doi
                art["title"] = title[:120]
                print(f"  [{i}/{len(articles)}] OK: {title[:60]}")
            else:
                print(f"  [{i}/{len(articles)}] No metadata ({r.status_code}): {pii}")
        except Exception as e:
            print(f"  [{i}/{len(articles)}] Error: {e}")
        time.sleep(0.3)

    return articles


# ── 2. Descargar PDFs ────────────────────────────────────────────────────────

def sanitize(name: str, max_len: int = 80) -> str:
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip(". ")[:max_len] or "sin_titulo"


def download_pdf(article: dict, index: int) -> bool:
    pii   = article.get("pii", "")
    doi   = article.get("doi", "")
    title = article.get("title", pii)

    filename = f"{index:04d}_{sanitize(title)}.pdf"
    filepath = OUTPUT_DIR / filename

    if filepath.exists() and filepath.stat().st_size > 10_000:
        return None  # ya existe, saltar sin sleep ni conteo

    urls_to_try = []
    if pii:
        urls_to_try.append(f"https://api.elsevier.com/content/article/pii/{pii}")
    if doi:
        urls_to_try.append(f"https://api.elsevier.com/content/article/doi/{doi}")

    for url in urls_to_try:
        try:
            r = requests.get(url, headers=API_HEADERS, stream=True, timeout=30)
            ct = r.headers.get("Content-Type", "")
            if r.status_code == 200 and "pdf" in ct:
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
                kb = filepath.stat().st_size // 1024
                print(f"  [{index}] OK ({kb} KB): {title[:60]}")
                return True
            elif r.status_code == 401:
                print(f"  [{index}] Sin acceso institucional (401): {title[:60]}")
                return False
            elif r.status_code == 200 and "html" in ct:
                # Guardar HTML como indicador de acceso denegado sin token
                print(f"  [{index}] Devuelve HTML (requiere IP institucional): {title[:60]}")
                return False
            elif r.status_code == 404:
                continue
            else:
                print(f"  [{index}] HTTP {r.status_code}: {title[:60]}")
        except Exception as e:
            print(f"  [{index}] Error de red: {e}")

    return False


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Elsevier PDF Downloader  —  Blood Vol.146 Suppl.S1")
    print("=" * 65)

    manifest_path = OUTPUT_DIR / "manifest.json"

    # Reusar manifiesto si ya existe
    if manifest_path.exists():
        print(f"Usando manifiesto existente: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as f:
            articles = json.load(f)
    else:
        articles = fetch_toc_articles()
        if not articles:
            print("\nNo se encontraron artículos en el TOC.")
            print("ScienceDirect puede requerir un navegador completo (JavaScript).")
            print("Considera usar el modo alternativo con Selenium.")
            return

        articles = enrich_with_api(articles)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        print(f"Manifiesto guardado: {manifest_path} ({len(articles)} artículos)")

    print(f"\nDescargando PDFs ({len(articles)} artículos)...")
    print("-" * 65)

    success = failed = 0
    for i, art in enumerate(articles, 1):
        try:
            result = download_pdf(art, i)
            if result is None:
                pass  # ya existía, sin sleep
            elif result is True:
                success += 1
                time.sleep(1)
            else:
                failed += 1
                time.sleep(1)
        except Exception as e:
            print(f"  [{i}] Error inesperado, continuando: {type(e).__name__}")
            failed += 1
            time.sleep(1)

    print("\n" + "=" * 65)
    print(f"Completado: {success} PDFs descargados, {failed} no disponibles")
    print(f"Directorio: {OUTPUT_DIR.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
