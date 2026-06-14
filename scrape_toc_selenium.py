"""
Extrae los PIIs/DOIs del TOC de Blood Vol.146 Suppl.S1 usando Selenium.
Guarda el resultado en manifest.json dentro de pdfs_blood_146_S1/
"""

import re
import json
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

TOC_URL   = "https://www.sciencedirect.com/journal/blood/vol/146/suppl/S1"
OUT_DIR   = Path("pdfs_blood_146_S1")
OUT_DIR.mkdir(exist_ok=True)
MANIFEST  = OUT_DIR / "manifest.json"

PII_RE = re.compile(r"/science/article/(?:pii/|abs/pii/)([A-Z0-9]{17,})")


def make_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1400,900")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    return webdriver.Chrome(options=opts)


def extract_articles(driver) -> list[dict]:
    articles = []
    seen_piis = set()

    def scrape_current_page():
        # Esperar a que carguen los artículos
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "a[href*='/science/article/pii/']")
                )
            )
        except Exception:
            pass

        time.sleep(2)  # espera extra para JS dinámico

        links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/science/article/pii/']")
        new_found = 0
        for a in links:
            href = a.get_attribute("href") or ""
            m = PII_RE.search(href)
            if not m:
                continue
            pii = m.group(1)
            if pii in seen_piis:
                continue
            seen_piis.add(pii)
            # Intentar obtener título del enlace o su contenedor
            title = a.text.strip()
            if not title:
                try:
                    parent = a.find_element(By.XPATH, "./ancestor::h3[1] | ./ancestor::h2[1]")
                    title = parent.text.strip()
                except Exception:
                    title = pii
            articles.append({"pii": pii, "doi": "", "title": title[:120]})
            new_found += 1
        return new_found

    print(f"Abriendo: {TOC_URL}")
    driver.get(TOC_URL)
    time.sleep(4)

    # Si hay redirección a login, esperar que el usuario se loguee
    if "login" in driver.current_url.lower() or "signin" in driver.current_url.lower():
        print("\n*** Se requiere login. Inicia sesión en el navegador que se abrió. ***")
        print("*** Presiona ENTER aquí cuando hayas iniciado sesión. ***")
        input()
        driver.get(TOC_URL)
        time.sleep(4)

    page = 1
    while True:
        print(f"  Página {page}: extrayendo artículos...")
        found = scrape_current_page()
        print(f"    Nuevos en esta página: {found}")

        # Buscar botón "Siguiente página"
        try:
            next_btn = driver.find_element(
                By.CSS_SELECTOR,
                "a.next-link, li.pagination-next a, [aria-label='Next page'], "
                "button[aria-label='Next page'], a[aria-label='Next']"
            )
            if next_btn.is_displayed() and next_btn.is_enabled():
                next_btn.click()
                page += 1
                time.sleep(2)
            else:
                break
        except Exception:
            break

    return articles


def main():
    print("=" * 60)
    print("Scraping TOC — Blood Vol.146 Suppl.S1")
    print("=" * 60)

    driver = make_driver()
    try:
        articles = extract_articles(driver)
    finally:
        driver.quit()

    if not articles:
        print("\nNo se encontraron artículos. El sitio puede requerir login.")
        return

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"\nGuardados {len(articles)} artículos en {MANIFEST}")
    print("Ahora ejecuta: python download_blood_pdfs.py")


if __name__ == "__main__":
    main()
