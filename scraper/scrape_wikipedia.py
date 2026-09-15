"""
Scrapes the three Wikipedia pages used as sources for the Barca RAG assistant.

Note: the two Wikipedia URLs given in the assignment brief were duplicates
(both pointed at History_of_FC_Barcelona). The correct stats/season-by-season
page is List_of_FC_Barcelona_seasons, which is what's used below.

Run from the project root:
    python scraper/scrape_wikipedia.py
"""

import json
from pathlib import Path

import trafilatura

RAW_DIR = Path("data/raw/wikipedia")

PAGES = {
    "history_of_fc_barcelona": "https://en.wikipedia.org/wiki/History_of_FC_Barcelona",
    "list_of_fc_barcelona_seasons": "https://en.wikipedia.org/wiki/List_of_FC_Barcelona_seasons",
    "camp_nou": "https://en.wikipedia.org/wiki/Camp_Nou",
}


def scrape_page(url: str) -> str | None:
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise RuntimeError(f"Could not fetch {url}")
    # include_tables=True matters here: the seasons page's value is its table
    return trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for name, url in PAGES.items():
        print(f"Scraping: {name}\n  -> {url}")
        try:
            text = scrape_page(url)
        except RuntimeError as exc:
            print(f"  ERROR: {exc}")
            continue

        if not text or len(text) < 100:
            print("  WARNING: little or no text extracted, skipping")
            continue

        out_path = RAW_DIR / f"{name}.txt"
        out_path.write_text(text, encoding="utf-8")
        manifest.append(
            {"title": name, "url": url, "file": str(out_path), "char_count": len(text)}
        )
        print(f"  Saved {len(text)} chars -> {out_path}")

    manifest_path = RAW_DIR / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone. Saved {len(manifest)}/{len(PAGES)} pages. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
