"""
Cleans the raw scraped .txt files and writes them into data/processed/,
ready to be picked up by notebooks/rag_pipeline.ipynb for chunking.

This also produces the manifest that answers Phase 2.1's question:
"how many documents, what formats, which files failed to parse?"

Run from the project root, after both scrapers:
    python scraper/clean_text.py
"""

import json
import re
from pathlib import Path

RAW_DIRS = [Path("data/raw/fcbarcelona"), Path("data/raw/wikipedia")]
PROCESSED_DIR = Path("data/processed")

MIN_USABLE_CHARS = 100  # below this, treat the file as failed/empty


def clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)      # strip Wikipedia-style [1] [2] refs
    text = re.sub(r"[ \t]+", " ", text)       # collapse repeated spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)    # collapse excess blank lines
    return text.strip()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    failed = []

    for raw_dir in RAW_DIRS:
        if not raw_dir.exists():
            print(f"Skipping missing directory: {raw_dir} (run the matching scraper first)")
            continue

        source_group = raw_dir.name  # "fcbarcelona" or "wikipedia"
        for txt_file in sorted(raw_dir.glob("*.txt")):
            raw_text = txt_file.read_text(encoding="utf-8")
            cleaned = clean_text(raw_text)

            if len(cleaned) < MIN_USABLE_CHARS:
                print(f"FAILED (too short): {txt_file}")
                failed.append(str(txt_file))
                continue

            out_name = f"{source_group}__{txt_file.stem}.txt"
            out_path = PROCESSED_DIR / out_name
            out_path.write_text(cleaned, encoding="utf-8")

            manifest.append(
                {
                    "source_group": source_group,
                    "original_file": str(txt_file),
                    "processed_file": str(out_path),
                    "char_count": len(cleaned),
                }
            )
            print(f"Cleaned {txt_file.name} -> {out_path.name} ({len(cleaned)} chars)")

    summary = {
        "documents_processed": len(manifest),
        "documents_failed": len(failed),
        "failed_files": failed,
        "documents": manifest,
    }
    manifest_path = PROCESSED_DIR / "_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nDone. {len(manifest)} processed, {len(failed)} failed.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
