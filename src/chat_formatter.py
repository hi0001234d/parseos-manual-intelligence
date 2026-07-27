"""
chat_formatter.py — Terminal Output Renderer for ParseOS SOP Engine (v2.0)
=========================================================================
Formats extracted SOP JSON data into rich, readable terminal UI output.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from colorama import Fore, Style, init
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
    title = sop_data.get("procedure_title", "Standard Operating Procedure")
    machine = sop_data.get("machine_type", "Industrial System")
    duration = sop_data.get("estimated_duration", "N/A")
    steps = sop_data.get("steps", [])
    warnings = sop_data.get("safety_warnings", [])

    lines = []
    lines.append("\n" + "=" * 70)
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

        if risk == "high":
            risk_badge = f"{Fore.RED}[HIGH RISK]{Style.RESET_ALL}"
        elif risk == "medium":
            risk_badge = f"{Fore.YELLOW}[MED RISK]{Style.RESET_ALL}"
        else:
            risk_badge = f"{Fore.GREEN}[LOW RISK]{Style.RESET_ALL}"

        lines.append(f"\n  {Style.BRIGHT}Step {s_num}:{Style.RESET_ALL} {risk_badge} {Style.BRIGHT}{action}{Style.RESET_ALL} -> {obj}")
        if cond:
            lines.append(f"          {Fore.WHITE}Condition: {cond}{Style.RESET_ALL}")
        if tool and tool.lower() != "none":
            lines.append(f"          {Fore.MAGENTA}Tool Required: {tool}{Style.RESET_ALL}")

    lines.append("\n" + "=" * 70 + "\n")
    return "\n".join(lines)
