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
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import MANUALS_PATH
from src.pdf_parser import get_page_count
from src.engine import SearchEngine
from src.sop_extractor import resolve_visual_context, extract_sop
from src.knowledge_layer import save_to_knowledge_layer
from src.chat_formatter import format_sop_output, format_orchestrated_response
from src.manual_metadata import load_all_known_vocab
from src.response_orchestrator import orchestrate_response


def run_pipeline(
    manual_path: str | None = None,
    query: str | None = None,
    category: str = "general",
    ingest_only: bool = False,
    query_only: bool = False,
    force_ingest: bool = False,
    filter_manual: str | None = None,
    save_output: bool = True,
    interactive_chat: bool = True,
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

    # Mode 2: Query execution (if --query or --query-only or full run with --manual)
    if query_only or query or manual_path:
        if manual_path:
            # Check if manual_path points to a PDF file on disk
            if os.path.exists(manual_path) and os.path.isfile(manual_path):
                manual_name = Path(manual_path).stem
                filter_manual = manual_name
                if force_ingest or not engine.manual_exists(manual_name):
                    print(f"[Pipeline] Ingesting manual first: {manual_path}")
                    engine.ingest_manual(manual_path, force=force_ingest)
                else:
                    print(f"[Pipeline] Using existing ingested index for manual: {manual_name}")
            else:
                # DEF-3: Early rejection for file-like paths that don't exist on disk.
                # A path is "file-like" if it has a file extension or contains path separators.
                looks_like_file = (
                    "." in Path(manual_path).name
                    or "/" in manual_path
                    or "\\" in manual_path
                )
                if looks_like_file:
                    # Check whether it matches an already-indexed manual name
                    available_manuals = engine.list_manuals()
                    matched = next(
                        (m for m in available_manuals if manual_path.lower() in m.lower()),
                        None,
                    )
                    if matched is None:
                        raise ValueError(
                            f"Manual not found: '{manual_path}' does not exist on disk "
                            f"and is not indexed in the knowledge base.\n"
                            f"Available manuals: {', '.join(available_manuals) if available_manuals else 'none (ingest first)'}."
                        )
                    filter_manual = matched
                    print(f"[Pipeline] Filtering search by indexed manual: '{filter_manual}'")
                else:
                    # Plain substring filter (no path separators / extension)
                    available_manuals = engine.list_manuals()
                    matched = next(
                        (m for m in available_manuals if manual_path.lower() in m.lower()),
                        None,
                    )
                    filter_manual = matched or manual_path
                    print(f"[Pipeline] Filtering search by manual: '{filter_manual}'")

        if not query:
            raise ValueError("--query string required for query execution.")

        # ── Manual selection prompt (CLI mode, no --manual specified) ────────
        # When the user hasn't pinned a specific manual, do a quick broad scan
        # to surface the most relevant manuals and let them pick one.
        if filter_manual is None:
            broad_hits = engine.search(query, top_k=15)
            if broad_hits:
                seen: list[str] = []
                for r in broad_hits:
                    if r.manual_name not in seen:
                        seen.append(r.manual_name)

                if len(seen) > 1:
                    print(f"\n  [ParseOS] Multiple manuals may contain relevant information for this query.")
                    print(f"  Please select the manual you want to search:\n")
                    for idx, m in enumerate(seen, 1):
                        print(f"    {idx}. {m}")
                    print(f"    {len(seen) + 1}. Search ALL manuals")

                    while True:
                        selection = input(f"\n  Enter a number (1-{len(seen) + 1}): ").strip()
                        if selection.isdigit():
                            choice = int(selection)
                            if 1 <= choice <= len(seen):
                                filter_manual = seen[choice - 1]
                                print(f"  -> Targeting manual: '{filter_manual}'\n")
                                break
                            elif choice == len(seen) + 1:
                                print("  -> Searching ALL manuals.\n")
                                break
                        print(f"  Invalid selection. Please enter a number between 1 and {len(seen) + 1}.")
                else:
                    # Only one manual matched — auto-select it
                    filter_manual = seen[0]
                    print(f"  -> Auto-selected the only relevant manual: '{filter_manual}'\n")

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
        kl_path, kl_entry = None, None
        if save_output:
            # DEF-3: Skip saving when evidence is insufficient and no chunks were retrieved
            skip_save = (
                sop_data.get("is_insufficient", False)
                and len(retrieved_chunks) == 0
            )
            if skip_save:
                print("  [INFO] Skipping Knowledge Layer save — no chunks retrieved for this manual path.")
                kl_entry = {"knowledge_id": "N/A"}
            else:
                kl_path, kl_entry = save_to_knowledge_layer(sop_data, manual_name=source_manual_name, category=category, query=query)
        else:
            print("  [INFO] --no-save active: skipping Knowledge Layer output.")
            kl_entry = {"knowledge_id": "N/A"}

        # Stage 6c: Response Orchestration
        print(f"\n[Pipeline Stage 6c] Building orchestrated response (short answer + options)...")
        combined_text = "\n\n".join(r.chunk_text for r in retrieved_chunks)
        result_meta   = [
            {"manual": r.manual_name, "page_number": r.page_num}
            for r in retrieved_chunks
        ]
        orchestrated = orchestrate_response(
            query=query,
            sop_data=sop_data,
            retrieved_text=combined_text,
            result_metadata=result_meta,
        )

        # Render Formatted Output
        # interactive=False for --query CLI mode (prints both options in full)
        format_orchestrated_response(orchestrated, interactive=interactive_chat)

        return kl_entry

    raise ValueError("Invalid CLI argument combination. Provide --manual, --query, or --chat.")


def _is_valid_query(text: str) -> tuple[bool, str]:
    """
    Validates that the user's input looks like an industrial procedure query.
    Returns (is_valid, rejection_reason).

    Rejects:
      - Shell / CLI commands (git, ls, cd, python, pip, …)
      - Single-word inputs that carry no procedural intent
      - Inputs shorter than 5 characters
      - Numeric-only or symbol-only strings
      - Irrelevant / conversational queries containing zero recognized industrial equipment terms or model codes
    """
    stripped = text.strip()
    lower = stripped.lower()

    # ── Layer 1: Structural checks ───────────────────────────────────────────
    if len(stripped) < 5:
        return False, "Query is too short. Please describe a procedure or problem."

    tokens = stripped.split()

    SHELL_COMMANDS = {
        "git", "ls", "cd", "pwd", "cp", "mv", "rm", "mkdir", "touch", "cat",
        "python", "python3", "pip", "pip3", "conda", "npm", "node", "curl",
        "wget", "docker", "kubectl", "ssh", "scp", "chmod", "chown", "sudo",
        "ps", "kill", "top", "htop", "df", "du", "echo", "export", "set",
        "dir", "cls", "type", "del", "copy", "ren", "ipconfig", "ping",
    }
    first_word = tokens[0].lower()
    if first_word in SHELL_COMMANDS:
        return False, (
            f"'{tokens[0]}' looks like a shell command, not a query.\n"
            "  ParseOS Chat Mode only accepts industrial procedure queries.\n"
            "  Example queries:\n"
            "    • motor overheating procedure\n"
            "    • bearing lubrication steps for ABB IRB\n"
            "    • emergency stop reset procedure Siemens PLC\n"
            "    • pump seal replacement maintenance"
        )

    if len(tokens) == 1:
        if not re.match(r'^[a-zA-Z]{4,}$', stripped):
            return False, (
                "Please enter a more descriptive query.\n"
                "  Example: 'motor overheating procedure' or 'valve seal inspection steps'"
            )
        return False, (
            f"'{stripped}' is too vague. Please describe the procedure or issue you need help with.\n"
            "  Example: 'spindle warm-up procedure' or 'bearing replacement steps'"
        )

    word_chars = re.sub(r'[^a-zA-Z]', '', stripped)
    if len(word_chars) < 4:
        return False, "Query must contain descriptive words, not just numbers or symbols."

    # ── Layer 2: Conversational / Personal intent detection ──────────────────
    CONVERSATIONAL_OPENERS = [
        "i want to", "i need to", "i would like to", "i'd like to",
        "i want a", "i need a", "i am going to", "i'm going to",
        "i like", "i love", "i hate", "i feel", "i think",
        "tell me about yourself", "who are you", "what are you",
        "how are you", "what is your name", "what can you do",
        "can you help me with", "do you know", "do you like",
        "what do you", "why do you", "are you a", "what you can",
        "let's talk", "let me test", "i want to test", "what is",
        "i want to try", "i want to check", "i want to see",
        "just testing", "testing testing", "hello", "hi there",
        "good morning", "good evening", "good afternoon",
        "what's up", "whats up", "hey parseos", "hey there",
    ]

    # Industrial signal terms — core industrial hardware, parameters, procedures & domains
    INDUSTRIAL_SIGNALS = {
        # Equipment & Hardware Components
        "motor", "motors", "pump", "pumps", "valve", "valves", "bearing", "bearings",
        "spindle", "spindles", "seal", "seals", "gear", "gears", "belt", "belts",
        "shaft", "shafts", "compressor", "compressors", "conveyor", "robot", "robotics",
        "plc", "plcs", "sensor", "sensors", "actuator", "actuators", "relay", "relays",
        "fuse", "fuses", "brake", "brakes", "encoder", "encoders", "servo", "servos",
        "hydraulic", "pneumatic", "coolant", "lubricant", "lubrication", "gasket",
        "coupling", "flange", "impeller", "solenoid", "contactor", "inverter",
        "drive", "drives", "controller", "controllers", "cylinder", "cylinders",
        "piston", "pistons", "nozzle", "nozzles", "orifice", "manifold", "regulator",
        "transformer", "turbine", "turbines", "generator", "generators", "boiler",
        "boilers", "chiller", "chillers", "transducer", "transmitter", "gauge",
        "meter", "breaker", "switch", "cable", "cables", "terminal", "wire", "harness",
        "fan", "fans", "blower", "filter", "filters", "strainer", "separator",
        "tank", "vessel", "exchanger", "condenser", "evaporator", "clutch",
        "pulley", "sprocket", "chain", "screw", "bolt", "housing", "casing",
        "rotor", "stator", "armature", "winding", "coil", "diaphragm", "vane",
        "lobe", "rod", "bushing", "sleeve", "liner", "disk", "blade", "axis",
        "gripper", "tool", "fixture", "chuck", "collet", "vise",
        # Industrial Domains & Disciplines
        "milling", "cnc", "lathe", "turning", "machining", "welding", "piping",
        "electrical", "mechanical", "instrumentation", "automation", "scada",
        "hmi", "telemetry", "avionics", "aerospace", "construction", "engineering",
        # Procedural & Operational Actions
        "overheating", "maintenance", "replacement", "inspection", "calibration",
        "troubleshoot", "troubleshooting", "procedure", "startup", "shutdown",
        "fault", "alarm", "error", "reset", "install", "installation",
        "replace", "repair", "diagnose", "diagnostics", "commissioning",
        "decommission", "overload", "vibration", "leakage", "pressure", "temperature",
        "torque", "voltage", "current", "frequency", "flow", "clearance", "alignment",
        "tension", "wiring", "grounding", "schematic", "circuit", "panel", "cabinet",
        "fieldbus", "profibus", "profinet", "ethernet", "modbus", "canbus",
        "lockout", "tagout", "isolation", "interlock", "purge", "flush", "drain",
        "vent", "bleed", "dismount", "safety", "hazard", "caution", "warning",
        # Manual & Technical Keywords
        "sop", "manual", "manuals", "specification", "drawing", "datasheet",
        "handbook", "handbooks", "checklist", "handbook", "handbooks",
    }

    query_words = {w.lower() for w in re.findall(r'\b[a-zA-Z]{3,}\b', stripped)}
    has_industrial_term = bool(query_words & INDUSTRIAL_SIGNALS)

    # Check for model / manual codes pattern (e.g. S7-1500, IRB-6700, EM_385-1-1, FAA-H-8083, CRN32, GA37)
    has_model_code = bool(re.search(r'\b[A-Za-z0-9]*\d[A-Za-z0-9\-_]*\b', stripped)) or bool(re.search(r'\b[a-zA-Z]{2,}[\-_][a-zA-Z0-9\-_]+\b', stripped))

    has_conversational_opener = any(lower.startswith(phrase) for phrase in CONVERSATIONAL_OPENERS)

    if has_conversational_opener and not (has_industrial_term or has_model_code):
        return False, (
            "That doesn't look like an industrial procedure query.\n"
            "  ParseOS is a technical SOP extraction engine for industrial manuals.\n"
            "  Please describe a machine problem, procedure, or maintenance task.\n"
            "  Example queries:\n"
            "    • motor overheating procedure\n"
            "    • bearing lubrication steps for ABB IRB 6700\n"
            "    • emergency stop reset procedure Siemens S7 PLC\n"
            "    • pump seal replacement — Grundfos CR-CRN"
        )

    # ── Layer 3: Technical Relevance Guard ────────────────────────────────────
    if not (has_industrial_term or has_model_code):
        return False, (
            "Query does not appear to contain any industrial terms or equipment identifiers recognized by ParseOS.\n"
            "  Please describe a machine problem, maintenance task, or procedure.\n"
            "  Example queries:\n"
            "    • motor overheating procedure\n"
            "    • valve seal inspection steps\n"
            "    • Siemens PLC fault reset procedure\n"
            "    • compressor startup sequence after emergency stop"
        )

    return True, ""


# ── DEF-1: Safety intent guard ────────────────────────────────────────────────

# Bypass / defeat verbs — describes the intent to circumvent a safety system
_UNSAFE_VERBS: frozenset[str] = frozenset({
    "bypass", "bypassing", "bypassed",
    "disable", "disabling", "disabled",
    "override", "overriding", "overridden",
    "circumvent", "circumventing",
    "defeat", "defeating", "defeated",
    "skip", "skipping",
    "ignore", "ignoring",
    "force", "forcing",
    "suppress", "suppressing",
    "deactivate", "deactivating",
    "trick", "tricking",
    "shortcut",
})

# Safety system nouns that must not be bypassed
_SAFETY_NOUNS: frozenset[str] = frozenset({
    "protective stop", "protection stop",
    "safety stop", "emergency stop", "e-stop", "estop",
    "interlock", "interlocks",
    "safety relay", "safety relays",
    "safety circuit", "safety circuits",
    "guard", "guarding",
    "lockout", "tagout", "loto",
    "safety limit", "safety limits",
    "safeguard", "safeguards",
    "safe torque off", "sto",
    "safety function", "safety functions",
    "safety gate", "safety gates",
    "light curtain", "light curtains",
    "door interlock",
})


def _is_unsafe_intent(text: str) -> tuple[bool, str]:
    """
    DEF-1: Detects queries that request bypassing, disabling, or overriding
    an industrial safety system.

    Returns (is_unsafe: bool, refusal_message: str).
    A True result must prevent retrieval and LLM generation entirely.
    """
    lower = text.lower()
    tokens = set(re.findall(r'\b[a-z]+\b', lower))

    has_unsafe_verb = bool(tokens & _UNSAFE_VERBS)
    has_safety_noun = any(noun in lower for noun in _SAFETY_NOUNS)

    if has_unsafe_verb and has_safety_noun:
        matched_verb = next(v for v in _UNSAFE_VERBS if v in tokens)
        matched_noun = next(n for n in _SAFETY_NOUNS if n in lower)
        return True, (
            f"\n  [ParseOS SAFETY REFUSAL]\n"
            f"  This system cannot provide instructions to '{matched_verb}' a '{matched_noun}'.\n"
            f"  Bypassing or disabling industrial safety systems is UNSAFE and contrary to\n"
            f"  ISO 13849, IEC 62061, and machine-specific safety standards.\n\n"
            f"  If you believe this safety system has triggered incorrectly:\n"
            f"    • Consult the official service manual for the safety reset procedure.\n"
            f"    • Contact a qualified safety engineer or the manufacturer.\n"
            f"    • Do NOT attempt to override safety systems without proper authorisation.\n"
        )

    return False, ""




def run_interactive_chat():
    """Runs interactive CLI chat mode allowing user to query ingested manuals iteratively."""
    engine = SearchEngine()
    print("\n" + "=" * 70)
    print("  ParseOS Manual Intelligence Engine -- Interactive Chat Mode (v2.0)")
    print("=" * 70)

    total_chunks = engine.total_chunks()
    manuals = engine.list_manuals()
    print(f"Knowledge Base Status: {total_chunks} total chunks across {len(manuals)} manual(s).")
    if manuals:
        print("Available manuals:", ", ".join(manuals))
    elif total_chunks == 0:
        # DEF-2: Only warn "empty" when the chunk count is also zero.
        # list_manuals() can return [] under certain ChromaDB metadata query
        # conditions even when chunks exist; checking total_chunks() avoids
        # the contradictory "179747 chunks across 0 manuals" display.
        print("[WARN] Knowledge base is currently empty! Please ingest manuals first using `python ingest_all.py`.")
    else:
        print("[INFO] Manual metadata index is rebuilding or unavailable; chunk data is present.")

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

            # ── Safety intent guard (DEF-1) ──────────────────────────────────
            # Must run BEFORE _is_valid_query to ensure bypass attempts are
            # refused even for otherwise well-formed industrial queries.
            unsafe, safety_msg = _is_unsafe_intent(user_input)
            if unsafe:
                print(safety_msg)
                continue

            # ── Query validation ─────────────────────────────────────────────
            valid, reason = _is_valid_query(user_input)
            if not valid:
                print(f"\n  [ParseOS] Invalid query: {reason}\n")
                continue

            # If no manual is selected, suggest some based on relevance
            current_target = selected_manual
            if current_target is None:
                retrieved = engine.search(user_input, top_k=15)
                if retrieved:
                    suggested_manuals = []
                    for r in retrieved:
                        if r.manual_name not in suggested_manuals:
                            suggested_manuals.append(r.manual_name)
                    
                    # Auto-select if a suggested manual name appears explicitly in the query
                    auto_selected = None
                    query_lower = user_input.lower()
                    for m in suggested_manuals:
                        if m.lower() in query_lower:
                            auto_selected = m
                            break
                    
                    if auto_selected:
                        current_target = auto_selected
                        print(f"  -> Detected manual '{current_target}' in query. Auto-selecting...")
                    elif suggested_manuals:
                        print("\n  [ParseOS] To provide the most accurate SOP, please select a specific manual for this query:")
                        for idx, m in enumerate(suggested_manuals, 1):
                            print(f"    {idx}. {m}")
                        
                        selection = input(f"\n  Enter a number (1-{len(suggested_manuals)}) or 'c' to cancel: ").strip()
                        if selection.lower() in ('c', 'cancel', 'quit', 'exit'):
                            print("  Query cancelled.\n")
                            continue
                        
                        try:
                            choice_idx = int(selection) - 1
                            if 0 <= choice_idx < len(suggested_manuals):
                                current_target = suggested_manuals[choice_idx]
                                print(f"  -> Targeting manual: {current_target}")
                            else:
                                print(f"  Invalid selection. Expected a number between 1 and {len(suggested_manuals)}.\n")
                                continue
                        except ValueError:
                            print("  Invalid input. Please enter a valid number.\n")
                            continue

            # Run query through pipeline
            result = run_pipeline(query=user_input, filter_manual=current_target, query_only=True, interactive_chat=True)

            # In chat mode the pipeline already printed the orchestrated response.
            # run_pipeline returns kl_entry (the knowledge layer dict); nothing
            # extra to render here — format_orchestrated_response was called
            # inside run_pipeline with interactive=False.  If you want interactive
            # option selection in chat mode, swap interactive=False → True in
            # run_pipeline(), or add the call here after result is returned.

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
    # DEF-5: --no-save skips Stage 7 Knowledge Layer output
    parser.add_argument("--no-save", action="store_true", help="Execute query without writing Knowledge Layer JSON output.")

    args = parser.parse_args()

    # Default to interactive chat mode if no CLI options specified
    if args.chat or not (args.manual or args.query or args.ingest_only or args.query_only):
        run_interactive_chat()
    else:
        run_pipeline(
            manual_path=args.manual,
            query=args.query,
            category=args.category,
            ingest_only=args.ingest_only,
            query_only=args.query_only,
            force_ingest=args.force,
            save_output=not args.no_save,
        )

    """ ---------- DEBUG MODE ----------
        debug_manual = r"data/manuals/Maintenance-manual-v3.2.1-web.pdf"
        debug_query = "motor overheating procedure"

        run_pipeline(
            manual_path=debug_manual,
            query=debug_query,
            category="manufacturing",
            ingest_only=False,
            query_only=False,
            force_ingest=False,
    )
    """

if __name__ == "__main__":
    main()
