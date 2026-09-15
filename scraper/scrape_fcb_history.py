"""
Scrapes the FC Barcelona official "Decade by Decade" history section.

The index page (INDEX_URL) links out to ~12 individual decade sub-pages
(e.g. /en/card/643900/1950-1961-the-kubala-era). This script:
  1. Fetches the index page and discovers those decade links.
  2. Visits each decade page and extracts the main article text
     (using trafilatura, which strips nav/footer/menu boilerplate).
  3. Saves each decade as a separate .txt file + a manifest.json.

Run from the project root:
    python scraper/scrape_fcb_history.py
"""

import json
import time
from pathlib import Path

import requests
import trafilatura
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw/fcbarcelona")
INDEX_URL = "https://www.fcbarcelona.com/en/club/history/decade-by-decade"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
        "BarcaRagProject/1.0 (educational graduation project)"
    )
}
REQUEST_DELAY_SECONDS = 1.0  # be polite to the server


def get_decade_links() -> dict[str, str]:
    """Find every /en/card/... link on the index page (one per decade)."""
    resp = requests.get(INDEX_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    links: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/en/card/" in href:
            full_url = href if href.startswith("http") else f"https://www.fcbarcelona.com{href}"
            title = a.get_text(strip=True) or full_url.rstrip("/").split("/")[-1]
            links[full_url] = title
    return links


def scrape_page(url: str) -> str | None:
    """Download a decade page and extract the main article text."""
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        # fallback if trafilatura's own fetcher is blocked
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        downloaded = resp.text

    return trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching index page: {INDEX_URL}")
    links = get_decade_links()
    print(f"Found {len(links)} decade pages\n")

    manifest = []
    for url, title in links.items():
        print(f"Scraping: {title}\n  -> {url}")
        try:
            text = scrape_page(url)
        except requests.RequestException as exc:
            print(f"  ERROR fetching page: {exc}")
            continue

        if not text or len(text) < 100:
            print("  WARNING: little or no text extracted, skipping")
            continue

        slug = url.rstrip("/").split("/")[-1]
        out_path = RAW_DIR / f"{slug}.txt"
        out_path.write_text(text, encoding="utf-8")
        manifest.append(
            {"title": title, "url": url, "file": str(out_path), "char_count": len(text)}
        )
        print(f"  Saved {len(text)} chars -> {out_path}")
        time.sleep(REQUEST_DELAY_SECONDS)

    manifest_path = RAW_DIR / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone. Saved {len(manifest)}/{len(links)} pages. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
