"""
ingest_all.py — Batch Ingestion Script (LlamaIndex Native v2.0)
===============================================================
Reads all PDF files from data/manuals/ and ingests them into ChromaDB using LlamaIndex.
In v2.0, ingestion is 100% TEXT-ONLY (zero VLM calls upfront). Visual pages are pre-rendered
and flagged for query-time processing.

Usage:
  python ingest_all.py                     # ingest all manuals
  python ingest_all.py --force             # re-ingest even if already stored
"""

import argparse
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MANUALS_PATH
from src.pdf_parser import get_page_count
from src.engine import SearchEngine


def ingest_manual(
    pdf_path: str,
    store: SearchEngine,
    verbose: bool = False,
) -> dict:
    """
    Runs Stages 1-4 for a single PDF manual via LlamaIndex text-only v2.0 ingestion.
    """
    manual_name = Path(pdf_path).stem
    result = {
        "manual":   manual_name,
        "path":     pdf_path,
        "status":   "pending",
        "pages":    0,
        "chunks":   0,
        "error":    None,
    }

    try:
        t0 = time.time()
        pages = get_page_count(pdf_path)
        result["pages"] = pages

        count = store.ingest_manual(pdf_path, force=True)
        result["chunks"] = count

        elapsed = time.time() - t0
        result["status"] = "ok"
        print(f"  [OK] {manual_name}: {pages} pages | {count} chunks | {elapsed:.1f}s")

    except FileNotFoundError:
        result["status"] = "missing"
        result["error"]  = "File not found"
        print(f"  [FAIL] {manual_name}: File not found at {pdf_path}")
    except Exception as e:
        result["status"] = "error"
        result["error"]  = str(e)
        print(f"  [FAIL] {manual_name}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch ingest all manuals into ChromaDB using LlamaIndex.")
    parser.add_argument("--force", action="store_true",
                        help="Re-ingest all manuals even if already stored.")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip manuals already in ChromaDB (default: True).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show progress details.")
    parser.add_argument("--manuals-dir", default=MANUALS_PATH,
                        help=f"Directory containing PDFs (default: {MANUALS_PATH}).")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ParseOS -- LlamaIndex Batch Manual Ingestion (v2.0)")
    print("=" * 60)

    manuals_dir = Path(args.manuals_dir)
    if not manuals_dir.exists():
        print(f"\n[FAIL] Manuals directory not found: {manuals_dir}")
        sys.exit(1)

    pdfs = sorted(manuals_dir.glob("*.pdf"))
    if not pdfs:
        print(f"\n[FAIL] No PDF files found in: {manuals_dir}")
        sys.exit(1)

    print(f"  Found {len(pdfs)} PDF file(s) in {manuals_dir}\n")

    store = SearchEngine()

    results  = []
    t_start  = time.time()

    for pdf in pdfs:
        manual_name = pdf.stem
        skip        = (not args.force) and store.manual_exists(manual_name)

        if skip:
            count = store.get_manual_chunk_count(manual_name)
            print(f"  - {manual_name}: already stored ({count} chunks) -- skipping")
            results.append({"manual": manual_name, "status": "skipped", "chunks": count})
            continue

        result = ingest_manual(str(pdf), store, verbose=args.verbose)
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    ok_count   = sum(1 for r in results if r["status"] == "ok")
    skip_count = sum(1 for r in results if r["status"] == "skipped")
    err_count  = sum(1 for r in results if r["status"] in ("error", "missing"))

    print("\n" + "-" * 60)
    print(f"  Ingestion complete in {total_time:.1f}s")
    print(f"  [OK] Ingested : {ok_count}")
    print(f"  - Skipped     : {skip_count}")
    print(f"  [FAIL] Errors : {err_count}")
    print(f"\n  Total chunks in DB: {store.total_chunks()}")
    print(f"  Manuals stored    : {store.list_manuals()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
