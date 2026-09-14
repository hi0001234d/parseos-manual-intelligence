"""
chat_formatter.py — Terminal Output Renderer for ParseOS SOP Engine (v2.0)
=========================================================================
Formats extracted SOP JSON data into rich, readable terminal UI output.

v2.1 — Added format_orchestrated_response() for Stage 6c orchestrated
        responses (short answer + Option 1 / Option 2 interactive flow).
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

try:
    from colorama import Fore, Style, init  # type: ignore[import-untyped]
    init(autoreset=True)
    COLOR_ENABLED = True
except ImportError:
    COLOR_ENABLED = False
    class Fore:
        CYAN = YELLOW = RED = GREEN = MAGENTA = BLUE = WHITE = RESET = ""
    class Style:
        BRIGHT = RESET_ALL = ""


def format_sop_output(sop_data: dict, knowledge_id: str | None = None) -> str:
    """Formats an SOP dictionary into a clean terminal report string."""
    is_insufficient = sop_data.get("is_insufficient", False)
    title = sop_data.get("procedure_title", "Standard Operating Procedure")
    machine = sop_data.get("machine_type", "Industrial System")
    duration = sop_data.get("estimated_duration", "N/A")
    steps = sop_data.get("steps", [])
    warnings = sop_data.get("safety_warnings", [])
    related_topics = sop_data.get("related_topics_found", [])
    recommendation = sop_data.get("recommendation", "")

    lines = []
    lines.append("\n" + "=" * 70)

    if is_insufficient or sop_data.get("status") == "INSUFFICIENT_EVIDENCE":
        coverage_info = sop_data.get("coverage_info", {})
        cov = coverage_info.get("coverage", "N/A")
        missing_crit = sop_data.get("missing_critical", coverage_info.get("missing_critical", []))
        pages = sop_data.get("retrieved_pages", [])
        msg = sop_data.get("message", "The retrieved manual sections do not contain sufficient evidence to generate a reliable SOP for this query.")

        lines.append(f"{Style.BRIGHT}{Fore.RED}INSUFFICIENT EVIDENCE{Style.RESET_ALL}")
        lines.append("=" * 70)
        lines.append(f"{Style.BRIGHT}Query:{Style.RESET_ALL} {sop_data.get('query_context', title)}")
        lines.append(f"{Style.BRIGHT}Coverage:{Style.RESET_ALL} {cov}")
        lines.append(f"{Style.BRIGHT}Missing Critical Terms:{Style.RESET_ALL} {Fore.YELLOW}{missing_crit}{Style.RESET_ALL}")
        lines.append(f"{Style.BRIGHT}Retrieved Pages:{Style.RESET_ALL} {Fore.CYAN}{pages}{Style.RESET_ALL}")
        lines.append(f"\n{Fore.RED}{msg}{Style.RESET_ALL}")

        if related_topics:
            lines.append(f"\n{Style.BRIGHT}{Fore.CYAN}[RELATED TOPICS FOUND IN RETRIEVED PAGES]:{Style.RESET_ALL}")
            for t in related_topics:
                lines.append(f"  - {Fore.CYAN}{t}{Style.RESET_ALL}")

        if recommendation:
            lines.append(f"\n{Style.BRIGHT}{Fore.GREEN}[RECOMMENDATION]:{Style.RESET_ALL}")
            lines.append(f"  * {Fore.GREEN}{recommendation}{Style.RESET_ALL}")

        lines.append("\n" + "=" * 70 + "\n")
        return "\n".join(lines)

    lines.append(f"{Style.BRIGHT}{Fore.CYAN}SOP: {title.upper()}{Style.RESET_ALL}")
    if knowledge_id:
        lines.append(f"{Fore.BLUE}ID: {knowledge_id}{Style.RESET_ALL}")
    lines.append(f"{Fore.WHITE}Machine Type: {machine} | Est. Duration: {duration}{Style.RESET_ALL}")
    lines.append("=" * 70)

    if warnings:
        lines.append(f"\n{Style.BRIGHT}{Fore.YELLOW}[SAFETY WARNINGS & PRECAUTIONS]:{Style.RESET_ALL}")
        for w in warnings:
            lines.append(f"  * {Fore.YELLOW}{w}{Style.RESET_ALL}")

    lines.append(f"\n{Style.BRIGHT}{Fore.GREEN}[PROCEDURE STEPS] ({len(steps)} total):{Style.RESET_ALL}")
    lines.append("-" * 70)

    for step in steps:
        s_num = step.get("step_number", 1)
        action = step.get("action", "")
        obj = step.get("object", "")
        cond = step.get("condition", "")
        risk = step.get("risk_level", "low").lower()
        tool = step.get("required_tool", "")

        page = step.get("source_page")
        evidence = step.get("evidence", "")

        if risk == "high":
            risk_badge = f"{Fore.RED}[HIGH RISK]{Style.RESET_ALL}"
        elif risk == "medium":
            risk_badge = f"{Fore.YELLOW}[MED RISK]{Style.RESET_ALL}"
        else:
            risk_badge = f"{Fore.GREEN}[LOW RISK]{Style.RESET_ALL}"

        page_tag = f" {Fore.CYAN}(Page {page}){Style.RESET_ALL}" if page else ""
        lines.append(f"\n  {Style.BRIGHT}Step {s_num}:{Style.RESET_ALL} {risk_badge}{page_tag} {Style.BRIGHT}{action}{Style.RESET_ALL} -> {obj}")
        if cond:
            lines.append(f"          {Fore.WHITE}Condition: {cond}{Style.RESET_ALL}")
        if tool and tool.lower() != "none":
            lines.append(f"          {Fore.MAGENTA}Tool Required: {tool}{Style.RESET_ALL}")
        if evidence:
            lines.append(f"          {Fore.BLUE}Evidence: \"{evidence[:120]}\"{Style.RESET_ALL}")

    # ── Coverage Gaps Section ────────────────────────────────────────────────
    coverage_gaps = sop_data.get("coverage_gaps", [])
    if coverage_gaps:
        lines.append(f"\n{Style.BRIGHT}{Fore.YELLOW}[COVERAGE GAPS]{Style.RESET_ALL}")
        lines.append("-" * 70)
        for gap in coverage_gaps:
            status = gap.get("status", "not_found")
            topic = gap.get("topic", "Unknown topic")
            note = gap.get("note", "")
            searched = gap.get("searched_pages", [])
            searched_str = ", ".join(str(p) for p in searched) if searched else "N/A"

            if status == "not_found":
                lines.append(f"\n  {Fore.RED}\u26a0 NOT FOUND{Style.RESET_ALL}")
            else:
                lines.append(f"\n  {Fore.YELLOW}\u25d0 PARTIALLY COVERED{Style.RESET_ALL}")

            lines.append(f"    {Style.BRIGHT}Topic:{Style.RESET_ALL} {topic}")
            lines.append(f"    {Fore.CYAN}Searched pages:{Style.RESET_ALL} {searched_str}")
            if note:
                lines.append(f"    {Fore.WHITE}Note:{Style.RESET_ALL} {note}")
            lines.append(f"    {Fore.GREEN}Recommendation:{Style.RESET_ALL} Search additional manual sections or consult the full service manual.")

    lines.append("\n" + "=" * 70 + "\n")
    return "\n".join(lines)


# ── Stage 6c: Orchestrated Response Renderer ─────────────────────────────────

def _render_reference_card(elem: dict, expanded: bool = False) -> str:
    """
    Renders the Option 2 clickable reference card.

    Collapsed (default) — shows only title + page.
    Expanded (after user action) — reveals reference_text.
    """
    title = elem.get("display_title", "Manual Reference")
    page  = elem.get("page_number")
    manual = elem.get("source_manual", "Manual")
    ref_text = elem.get("reference_text", "")

    page_display = f"Page {page}" if page else "Page N/A"
    border = "─" * 52

    lines = []
    lines.append(f"  ┌{border}┐")
    lines.append(f"  │  {Fore.CYAN}📄  {Style.BRIGHT}{title:<45}{Style.RESET_ALL}  │")
    lines.append(f"  │      {Fore.WHITE}{page_display} — {manual:<35}{Style.RESET_ALL}│")

    if not expanded:
        lines.append(f"  │{' ' * 40}{Fore.GREEN}[Press Enter to view]{Style.RESET_ALL} │")
        lines.append(f"  └{border}┘")
    else:
        lines.append(f"  │{'─' * 52}│")
        # Word-wrap reference text at ~50 chars per line
        import textwrap
        wrapped = textwrap.wrap(f'"{ref_text}"', width=50)
        for line in wrapped:
            lines.append(f"  │  {Fore.WHITE}{line:<50}{Style.RESET_ALL}  │")
        lines.append(f"  └{border}┘")

    return "\n".join(lines)


def format_orchestrated_response(
    orchestrated: dict,
    interactive: bool = True,
) -> None:
    """
    Renders a Stage 6c orchestrated response to the terminal.

    Args:
        orchestrated : dict returned by orchestrate_response()
        interactive  : If True (chat mode), prompts user to pick an option
                       and expands Option 2 on demand.
                       If False (CLI mode), prints both options in full.
    """
    if not orchestrated:
        return

    query        = orchestrated.get("query", "")
    short_answer = orchestrated.get("short_answer", "")
    options      = orchestrated.get("options")

    print("\n" + "=" * 70)
    print(f"{Style.BRIGHT}{Fore.CYAN}ParseOS Response{Style.RESET_ALL}")
    if query:
        print(f"{Fore.WHITE}Query: {query}{Style.RESET_ALL}")
    print("=" * 70)

    # ── Short direct answer ───────────────────────────────────────────────
    print(f"\n{Style.BRIGHT}{Fore.GREEN}▶  Direct Answer:{Style.RESET_ALL}")
    print(f"   {short_answer}")

    if not options:
        # INSUFFICIENT_EVIDENCE path
        print(f"\n{Fore.YELLOW}No procedure options available for this query.{Style.RESET_ALL}")
        print("\n" + "=" * 70 + "\n")
        return

    opt1 = options.get("option_1", {})
    opt2 = options.get("option_2", {})

    if not interactive:
        # ── Non-interactive (CLI --query mode): print both in full ────────
        print(f"\n{Style.BRIGHT}{Fore.CYAN}Option 1 — {opt1.get('label', 'Explain in Detail')}:{Style.RESET_ALL}")
        print("-" * 70)
        print(opt1.get("content", ""))

        print(f"\n{Style.BRIGHT}{Fore.CYAN}Option 2 — {opt2.get('label', 'Reference to Actual Document')}:{Style.RESET_ALL}")
        elem = opt2.get("clickable_element", {})
        print(_render_reference_card(elem, expanded=False))
        print()
        print(_render_reference_card(elem, expanded=True))
        print("\n" + "=" * 70 + "\n")
        return

    # ── Interactive (chat mode): prompt for option choice ─────────────────
    import sys
    
    opt1_available = True
    opt2_available = True

    while opt1_available or opt2_available:
        lines_printed = 0
        
        print(f"\n{Style.BRIGHT}Choose an option:{Style.RESET_ALL}")
        lines_printed += 2
        
        if opt1_available:
            print(f"  {Fore.CYAN}1.{Style.RESET_ALL} {opt1.get('label', 'Explain in Detail')}")
            lines_printed += 1
            
        if opt2_available:
            print(f"  {Fore.CYAN}2.{Style.RESET_ALL} {opt2.get('label', 'Reference to Actual Document')}")
            lines_printed += 1
            
        print()
        lines_printed += 1
        
        try:
            valid_choices = []
            if opt1_available: valid_choices.append("1")
            if opt2_available: valid_choices.append("2")
            prompt_str = f"  Enter {' or '.join(valid_choices)} (or press Enter to skip): "
            choice = input(prompt_str).strip()
            lines_printed += 1
        except (EOFError, KeyboardInterrupt):
            print()
            return
            
        # Erase the menu lines
        sys.stdout.write(f"\033[{lines_printed}A") 
        sys.stdout.write("\033[J")
        sys.stdout.flush()

        if not choice:
            print(f"  {Fore.WHITE}Skipped option selection.{Style.RESET_ALL}")
            break
            
        if choice == "1" and opt1_available:
            print(f"\n{Style.BRIGHT}{Fore.GREEN}[Detailed SOP]{Style.RESET_ALL}")
            print("-" * 70)
            print(opt1.get("content", ""))
            opt1_available = False

        elif choice == "2" and opt2_available:
            elem = opt2.get("clickable_element", {})
            print(f"\n{Style.BRIGHT}{Fore.CYAN}[Document Reference]{Style.RESET_ALL}")
            print(_render_reference_card(elem, expanded=True))
            opt2_available = False
            
    print("\n" + "=" * 70 + "\n")
