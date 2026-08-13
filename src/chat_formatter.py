"""
chat_formatter.py — Terminal Output Renderer for ParseOS SOP Engine (v2.0)
=========================================================================
Formats extracted SOP JSON data into rich, readable terminal UI output.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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

    lines.append("\n" + "=" * 70 + "\n")
    return "\n".join(lines)
