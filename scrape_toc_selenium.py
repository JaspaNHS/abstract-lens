"""
Extracts PIIs/DOIs from the Blood Vol.146 Suppl.S1 TOC using Selenium, and tags
each article with the ScienceDirect section header it appears under:
  Plenary Scientific Session > Oral > Poster > Online Publication Only
(the meeting's importance order). Saves manifest.json in pdfs_blood_146_S1/.
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

# Canonical section labels in importance order, normalized for matching
SECTION_LABELS = {
    "plenary scientific session": "plenary",
    "oral": "oral",
    "poster": "poster",
    "online publication only": "pubonly",
}

# Walk the TOC body in document order, emitting section headers and article PIIs
# interleaved, so each PII can be attributed to the section header above it.
ORDERED_WALK_JS = r"""
const LABELS = ["Plenary Scientific Session","Oral","Poster","Online Publication Only"];
const out = [];
const all = document.querySelectorAll('h1,h2,h3,h4,a');
for (const el of all) {
  const tag = el.tagName.toLowerCase();
  if (tag === 'a') {
    const href = el.getAttribute('href') || '';
    const m = href.match(/\/science\/article\/(?:abs\/)?pii\/([A-Z0-9]{17,})/);
    if (m) out.push({t:'a', v:m[1], title:(el.textContent||'').trim().slice(0,160)});
  } else {
    const txt = (el.textContent || '').trim();
    if (LABELS.includes(txt)) out.push({t:'h', v:txt});
  }
}
return out;
"""


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
    current_section = "unknown"          # carried ACROSS pages
    section_counts = {}

    def scrape_current_page():
        nonlocal current_section
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "a[href*='/science/article/pii/']")
                )
            )
        except Exception:
            pass

        time.sleep(2)  # extra wait for dynamic JS

        items = driver.execute_script(ORDERED_WALK_JS) or []
        new_found = 0
        for it in items:
            if it["t"] == "h":
                current_section = SECTION_LABELS.get(it["v"].strip().lower(), current_section)
            else:  # article link
                pii = it["v"]
                if pii in seen_piis:
                    continue
                seen_piis.add(pii)
                title = (it.get("title") or pii)[:120]
                articles.append({
                    "pii": pii, "doi": "", "title": title,
                    "section": current_section,
                })
                section_counts[current_section] = section_counts.get(current_section, 0) + 1
                new_found += 1
        return new_found

    print(f"Opening: {TOC_URL}")
    driver.get(TOC_URL)
    time.sleep(4)

    if "login" in driver.current_url.lower() or "signin" in driver.current_url.lower():
        print("\n*** Login required. Sign in on the browser window that opened. ***")
        print("*** Press ENTER here once you are signed in. ***")
        input()

    # JS-driven "Next page" button. Click via JS after scrolling it into view;
    # stop when the button is gone or disabled, or when no new articles appear.
    page = 1
    while True:
        found = scrape_current_page()
        print(f"  Page {page}: +{found}  (section={current_section}, total={len(articles)})")

        next_btn = None
        for sel in ["a[aria-label='Next page']", "button[aria-label='Next page']",
                    "a.next-link", "li.pagination-next a"]:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            if els:
                next_btn = els[0]
                break

        if not next_btn:
            break
        disabled = (next_btn.get_attribute("aria-disabled") == "true"
                    or next_btn.get_attribute("disabled") is not None)
        if disabled:
            break

        try:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", next_btn)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", next_btn)
            page += 1
            time.sleep(2.5)
        except Exception as e:
            print(f"    pagination stopped: {e}")
            break

        if page > 120:   # safety cap
            break

    print(f"\n  Section breakdown: {section_counts}")
    return articles


def main():
    print("=" * 60)
    print("Scraping TOC — Blood Vol.146 Suppl.S1 (with section tags)")
    print("=" * 60)

    driver = make_driver()
    try:
        articles = extract_articles(driver)
    finally:
        driver.quit()

    if not articles:
        print("\nNo articles found. The site may require login.")
        return

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(articles)} articles to {MANIFEST}")
    print("Next: python rag/build_metadata.py   (then rebuild as needed)")


if __name__ == "__main__":
    main()
