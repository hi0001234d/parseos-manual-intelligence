"""
ingest_all.py — Batch Ingestion Script (LlamaIndex Native v2.0 + v3.0 Vocab)
=============================================================================
Reads all PDF files from data/manuals/ and ingests them into ChromaDB using LlamaIndex.
In v2.0, ingestion is 100% TEXT-ONLY (zero VLM calls upfront). Visual pages are pre-rendered
and flagged for query-time processing.

v3.0 addition: After each successful ingestion, technical vocabulary is automatically
extracted from the manual text and saved to knowledge_layer/manual_metadata/.

Usage:
  python ingest_all.py                     # ingest all manuals (skip existing)
  python ingest_all.py --force             # re-ingest even if already stored
  python ingest_all.py --build-vocab       # generate/refresh vocabulary for all stored manuals
                                           # (no re-ingestion — uses stored ChromaDB chunks)
  python ingest_all.py --force --build-vocab  # full re-ingest + rebuild vocabulary
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MANUALS_PATH
from src.pdf_parser import get_page_count
from src.engine import SearchEngine
from src.manual_metadata import extract_technical_terms, save_manual_metadata


def build_vocab_for_manual(manual_name: str, store: SearchEngine) -> int:
    """
    Extracts and persists the technical vocabulary for a single manual
    using its already-stored ChromaDB chunks (no PDF re-parsing needed).

    Returns the number of terms extracted. Returns 0 on failure.
    """
    full_text = store.get_manual_text(manual_name)
    if not full_text.strip():
        print(f"  [WARN] No stored text found for '{manual_name}'. Skipping vocab build.")
        return 0

    terms = extract_technical_terms(full_text)
    out_path = save_manual_metadata(manual_name, terms)
    print(f"  [Vocab] '{manual_name}': {len(terms)} terms → {out_path.name}")
    return len(terms)


def ingest_manual(
    pdf_path: str,
    store: SearchEngine,
    verbose: bool = False,
    build_vocab: bool = True,
) -> dict:
    """
    Runs Stages 1-4 for a single PDF manual via LlamaIndex text-only v2.0 ingestion.
    v3.0: After ingestion, extracts and saves per-manual technical vocabulary.
    """
    manual_name = Path(pdf_path).stem
    result: dict[str, Any] = {
        "manual":   manual_name,
        "path":     pdf_path,
        "status":   "pending",
        "pages":    0,
        "chunks":   0,
        "vocab":    0,
        "error":    None,
    }

    try:
        t0 = time.time()
        pages = get_page_count(pdf_path)
        result["pages"] = pages

        count = store.ingest_manual(pdf_path, force=True)
        result["chunks"] = count

        # ── v3.0: Build per-manual vocabulary after ingestion ─────────────
        if build_vocab:
            vocab_count = build_vocab_for_manual(manual_name, store)
            result["vocab"] = vocab_count

        elapsed = time.time() - t0
        result["status"] = "ok"
        vocab_info = f" | {result['vocab']} vocab terms" if build_vocab else ""
        print(f"  [OK] {manual_name}: {pages} pages | {count} chunks{vocab_info} | {elapsed:.1f}s")

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
    parser = argparse.ArgumentParser(
        description="Batch ingest all manuals into ChromaDB using LlamaIndex."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-ingest all manuals even if already stored.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip manuals already in ChromaDB (default: True).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show progress details.",
    )
    parser.add_argument(
        "--manuals-dir", default=MANUALS_PATH,
        help=f"Directory containing PDFs (default: {MANUALS_PATH}).",
    )
    parser.add_argument(
        "--build-vocab", action="store_true",
        help=(
            "Extract and persist technical vocabulary for all manuals already stored "
            "in ChromaDB. Does NOT re-ingest. Use together with --force to re-ingest "
            "AND rebuild vocabulary."
        ),
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  ParseOS -- LlamaIndex Batch Manual Ingestion (v2.0 + v3.0 Vocab)")
    print("=" * 60)

    store = SearchEngine()

    # ── Backfill mode: build vocab for already-ingested manuals ──────────────
    if args.build_vocab and not args.force:
        existing_manuals = store.list_manuals()
        if not existing_manuals:
            print("\n[WARN] No manuals found in ChromaDB. Ingest manuals first.")
            sys.exit(1)

        print(f"\n  Building vocabulary for {len(existing_manuals)} stored manual(s)...\n")
        t_start = time.time()
        total_terms = 0
        for manual_name in existing_manuals:
            vocab_count = build_vocab_for_manual(manual_name, store)
            total_terms += vocab_count

        total_time = time.time() - t_start
        print("\n" + "-" * 60)
        print(f"  Vocabulary build complete in {total_time:.1f}s")
        print(f"  Manuals processed : {len(existing_manuals)}")
        print(f"  Total terms saved : {total_terms}")
        print("=" * 60 + "\n")
        return

    # ── Normal ingestion mode ─────────────────────────────────────────────────
    manuals_dir = Path(args.manuals_dir)
    if not manuals_dir.exists():
        print(f"\n[FAIL] Manuals directory not found: {manuals_dir}")
        sys.exit(1)

    pdfs = sorted(manuals_dir.glob("*.pdf"))
    if not pdfs:
        print(f"\n[FAIL] No PDF files found in: {manuals_dir}")
        sys.exit(1)

    print(f"  Found {len(pdfs)} PDF file(s) in {manuals_dir}\n")

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

        result = ingest_manual(
            str(pdf),
            store,
            verbose=args.verbose,
            build_vocab=True,   # always extract vocab during fresh ingestion
        )
        results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    ok_count   = sum(1 for r in results if r.get("status") == "ok")
    skip_count = sum(1 for r in results if r.get("status") == "skipped")
    err_count  = sum(1 for r in results if r.get("status") in ("error", "missing"))
    total_vocab = sum(r.get("vocab", 0) for r in results if r.get("status") == "ok")

    print("\n" + "-" * 60)
    print(f"  Ingestion complete in {total_time:.1f}s")
    print(f"  [OK] Ingested : {ok_count}")
    print(f"  - Skipped     : {skip_count}")
    print(f"  [FAIL] Errors : {err_count}")
    print(f"\n  Total chunks in DB  : {store.total_chunks()}")
    print(f"  Manuals stored      : {store.list_manuals()}")
    if ok_count > 0:
        print(f"  Vocab terms saved   : {total_vocab} (across {ok_count} ingested manuals)")
    print("=" * 60 + "\n")

    print("  Tip: run  python ingest_all.py --build-vocab  to generate")
    print("       vocabulary metadata for manuals that were skipped above.\n")


if __name__ == "__main__":
    main()
