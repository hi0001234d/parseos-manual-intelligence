"""
parseos_pipeline.py — Master Pipeline Runner & Interactive Chat (v2.0)
======================================================================
CLI runner driving the complete 7-stage ParseOS SOP Translation pipeline.

Usage Examples:
  # Full run: ingest + query
  python parseos_pipeline.py --manual "data/manuals/TECO Westinghouse Motor.pdf" --query "motor overheating procedure" --category manufacturing

  # Ingest only (Stages 1-4, text-only v2.0)
  python parseos_pipeline.py --manual "data/manuals/TECO Westinghouse Motor.pdf" --ingest-only

  # Query only (Stages 5-7)
  python parseos_pipeline.py --query "motor bearing overheating maintenance" --category manufacturing --query-only

  # Interactive Chat Mode
  python parseos_pipeline.py --chat
"""

import argparse
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MANUALS_PATH
from src.pdf_parser import get_page_count
from src.engine import SearchEngine
from src.sop_extractor import resolve_visual_context, extract_sop
from src.knowledge_layer import save_to_knowledge_layer
from src.chat_formatter import format_sop_output


def run_pipeline(
    manual_path: str | None = None,
    query: str | None = None,
    category: str = "general",
    ingest_only: bool = False,
    query_only: bool = False,
    force_ingest: bool = False,
    filter_manual: str | None = None,
) -> dict | None:
    """
    Executes the ParseOS SOP Pipeline according to mode flags.
    """
    engine = SearchEngine()

    # Mode 1: Ingest only
    if ingest_only:
        if not manual_path:
            raise ValueError("--manual path required for --ingest-only mode.")
        print(f"\n[Pipeline] Running Ingestion (Stages 1-4 text-only v2.0) for: {manual_path}")
        chunks_count = engine.ingest_manual(manual_path, force=force_ingest)
        print(f"  [OK] Ingestion complete! {chunks_count} chunk nodes stored in ChromaDB.\n")
        return None

    # Mode 2: Query only or Full Run
    if query_only or (manual_path and query):
        if manual_path:
            manual_name = Path(manual_path).stem
            filter_manual = manual_name
            if force_ingest or not engine.manual_exists(manual_name):
                print(f"[Pipeline] Ingesting manual first: {manual_path}")
                engine.ingest_manual(manual_path, force=force_ingest)
            else:
                print(f"[Pipeline] Using existing ingested index for manual: {manual_name}")

        if not query:
            raise ValueError("--query string required for query execution.")

        print(f"\n[Pipeline Stage 5] Running vector similarity search for query: '{query}'")
        retrieved_chunks = engine.search(query, top_k=3, filter_manual=filter_manual)

        print(f"  [OK] Retrieved {len(retrieved_chunks)} relevant chunk(s).")
        for idx, r in enumerate(retrieved_chunks, 1):
            print(f"    Chunk {idx}: page {r.page_num} in '{r.manual_name}' (similarity: {r.similarity:.3f}, visual={r.has_visual_content})")

        # Stage 6a: Conditional Visual Processing
        print(f"\n[Pipeline Stage 6a] Checking for retrieved visual pages (max cap = 3)…")
        image_context = resolve_visual_context(retrieved_chunks, user_query=query)

        # Stage 6b: LLM SOP Extraction
        print(f"\n[Pipeline Stage 6b] Extracting structured SOP JSON via LLM reasoning…")
        source_manual_name = filter_manual or (retrieved_chunks[0].manual_name if retrieved_chunks else "industrial_manual")
        sop_data = extract_sop(query=query, retrieved_chunks=retrieved_chunks, image_context=image_context, category=category)

        # Stage 7: Knowledge Layer Storage
        print(f"\n[Pipeline Stage 7] Exporting to ParseOS Knowledge Layer…")
        kl_path, kl_entry = save_to_knowledge_layer(sop_data, manual_name=source_manual_name, category=category, query=query)

        # Render Formatted Output
        formatted_output = format_sop_output(sop_data, knowledge_id=kl_entry["knowledge_id"])
        print(formatted_output)

        return kl_entry

    raise ValueError("Invalid CLI argument combination. Provide --manual, --query, or --chat.")


def run_interactive_chat():
    """Runs interactive CLI chat mode allowing user to query ingested manuals iteratively."""
    engine = SearchEngine()
    print("\n" + "=" * 70)
    print("  ParseOS Manual Intelligence Engine -- Interactive Chat Mode (v2.0)")
    print("=" * 70)

    manuals = engine.list_manuals()
    print(f"Knowledge Base Status: {engine.total_chunks()} total chunks across {len(manuals)} manual(s).")
    if manuals:
        print("Available manuals:", ", ".join(manuals))
    else:
        print("[WARN] Knowledge base is currently empty! Please ingest manuals first using `python ingest_all.py`.")

    print("\nType your question (e.g. 'motor overheating procedure') or 'exit' / 'quit' to stop.\n")

    selected_manual = None

    while True:
        try:
            prompt_str = f"ParseOS [{selected_manual or 'ALL MANUALS'}] > "
            user_input = input(prompt_str).strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("\nExiting ParseOS Chat Mode. Goodbye!")
                break

            if user_input.lower().startswith("use "):
                target = user_input[4:].strip()
                if target.lower() in ("all", "clear", "none"):
                    selected_manual = None
                    print("  Switched target manual filter to: ALL MANUALS")
                else:
                    selected_manual = target
                    print(f"  Switched target manual filter to: {selected_manual}")
                continue

            # Run query through pipeline
            run_pipeline(query=user_input, filter_manual=selected_manual, query_only=True)

        except KeyboardInterrupt:
            print("\nExiting ParseOS Chat Mode.")
            break
        except Exception as err:
            print(f"  [WARN] Error executing query: {err}\n")


def main():
    parser = argparse.ArgumentParser(description="ParseOS SOP Translation Master Pipeline (v2.0)")
    parser.add_argument("--manual", help="Path to source PDF manual file.")
    parser.add_argument("--query", help="User query string for procedure extraction.")
    parser.add_argument("--category", default="general", help="Machine category classification.")
    parser.add_argument("--ingest-only", action="store_true", help="Run ingestion (Stages 1-4) only.")
    parser.add_argument("--query-only", action="store_true", help="Run query retrieval & extraction (Stages 5-7) only.")
    parser.add_argument("--force", action="store_true", help="Force re-ingest manual even if already stored.")
    parser.add_argument("--chat", action="store_true", help="Launch interactive CLI chat mode.")

    args = parser.parse_args()

    if args.chat:
        run_interactive_chat()
    else:
        run_pipeline(
            manual_path=args.manual,
            query=args.query,
            category=args.category,
            ingest_only=args.ingest_only,
            query_only=args.query_only,
            force_ingest=args.force,
        )


if __name__ == "__main__":
    main()
