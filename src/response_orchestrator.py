"""
response_orchestrator.py — Stage 6c: Response Orchestrator (v1.0)
=================================================================
Wraps the extracted SOP (Stage 6b) into a smart, structured 3-part response:

  [1] SHORT ANSWER     — 2-3 sentence direct answer to the query
  [2] OPTION 1         — "Explain in Detail"  → full formatted SOP steps
  [3] OPTION 2         — "Reference to Actual Document"
                          → clickable card: section title + page number visible,
                            reference_text HIDDEN until user expands it.

IMPORTANT — UI CONTRACT FOR OPTION 2:
  reference_text is present inside the response object but must NOT be
  rendered by the UI until the user explicitly clicks / expands the card.
  This applies for terminal (press Enter to expand) and any future web UI.

EDGE CASES:
  - SOP is INSUFFICIENT_EVIDENCE  → returns short_answer + options: null
  - LLM unavailable               → graceful offline fallback (no crash)
  - page_number missing           → shows "Page N/A" (no crash)
  - reference JSON parse fails    → falls back to first 500 chars of retrieved text
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from src.config import GROQ_API_KEY, GROQ_MODEL
from src.api_retry import retry_with_backoff


# ── Provider Setup (mirrors sop_extractor pattern) ───────────────────────────

def _get_llm_client():
    """Returns an OpenAI-compat Groq client, or None if key is not set."""
    if not GROQ_API_KEY:
        return None, None
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
        return client, GROQ_MODEL
    except Exception:
        return None, None


def _chat(client, model: str, messages: list[dict], temperature: float = 0.2, max_tokens: int = 800) -> str:
    """Calls the LLM and returns the text content. Returns '' on any failure."""
    try:
        def _call():
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        response = retry_with_backoff(_call, max_retries=2, initial_delay=2.0)
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception as exc:
        print(f"  [WARN] [Stage 6c] LLM call failed: {exc}")
        return ""


# ── Part 1 — Short Direct Answer ─────────────────────────────────────────────

def _generate_short_answer(
    query: str,
    sop_data: dict,
    client,
    model: str,
) -> str:
    """Generates a 2-3 sentence direct answer to the query using the SOP data."""
    prompt = (
        f"A technician asked: '{query}'\n\n"
        f"Based on this extracted procedure:\n{json.dumps(sop_data, indent=2)}\n\n"
        "Write a direct answer in EXACTLY 2-3 sentences.\n"
        "Rules:\n"
        "- Be specific. Mention the most critical action and the highest risk if present.\n"
        "- Do NOT list steps. Just answer directly in plain prose.\n"
        "- Do NOT start with 'Based on the procedure' or similar meta-phrasing.\n"
        "- Output ONLY the answer sentences, nothing else."
    )
    raw = _chat(client, model, [{"role": "user", "content": prompt}], temperature=0.2, max_tokens=200)
    if raw:
        return raw
    # Offline fallback: synthesize from SOP
    return _offline_short_answer(query, sop_data)


def _offline_short_answer(query: str, sop_data: dict) -> str:
    """Generates a short answer offline when LLM is unavailable."""
    title = sop_data.get("procedure_title", f"Procedure for {query}")
    steps = sop_data.get("steps", [])
    warnings = sop_data.get("safety_warnings", [])

    first_action = ""
    highest_risk = "unknown"
    if steps:
        first_action = f"{steps[0].get('action', '')} {steps[0].get('object', '')}".strip()
        risk_levels = [s.get("risk_level", "low") for s in steps]
        if "high" in risk_levels:
            highest_risk = "HIGH"
        elif "medium" in risk_levels:
            highest_risk = "MEDIUM"
        else:
            highest_risk = "LOW"

    answer = f"This procedure covers: {title}."
    if first_action:
        answer += f" The first critical action is to {first_action.lower()}."
    if warnings:
        answer += f" Risk level is {highest_risk} — always follow safety precautions before starting."
    return answer


# ── Part 2 — Option 1: Formatted Detail SOP ──────────────────────────────────

def _format_option1_content(sop_data: dict) -> str:
    """Formats the full SOP into readable Option 1 content (plain text)."""
    lines = []
    lines.append(f"Procedure: {sop_data.get('procedure_title', 'N/A')}")
    lines.append(f"Machine:   {sop_data.get('machine_type', 'N/A')}")
    lines.append(f"Duration:  {sop_data.get('estimated_duration', 'N/A')}")
    lines.append("")

    for step in sop_data.get("steps", []):
        action = step.get("action", "")
        obj = step.get("object", "")
        cond = step.get("condition", "")
        tool = step.get("required_tool", "")
        risk = step.get("risk_level", "low").upper()
        page = step.get("source_page")
        evidence = step.get("evidence", "")

        page_tag = f" [Page {page}]" if page else ""
        lines.append(f"Step {step.get('step_number', '?')}:{page_tag} {action} {obj}")
        if cond:
            lines.append(f"   When:      {cond}")
        if tool and tool.lower() not in ("none", "null", ""):
            lines.append(f"   Tool:      {tool}")
        lines.append(f"   Risk:      {risk}")
        if evidence:
            lines.append(f"   Evidence:  \"{evidence[:100]}\"")
        lines.append("")

    warnings = sop_data.get("safety_warnings", [])
    if warnings:
        lines.append("Safety Warnings:")
        for w in warnings:
            lines.append(f"  ⚠  {w}")

    coverage_gaps = sop_data.get("coverage_gaps", [])
    if coverage_gaps:
        lines.append("")
        lines.append("Coverage Gaps:")
        for g in coverage_gaps:
            status = g.get("status", "not_found").upper()
            topic = g.get("topic", "Unknown")
            note = g.get("note", "")
            lines.append(f"  [{status}] {topic}")
            if note:
                lines.append(f"           {note}")

    return "\n".join(lines)


# ── Part 3 — Option 2: Reference Extraction ───────────────────────────────────

def _extract_reference_data(
    query: str,
    retrieved_text: str,
    result_metadata: list[dict],
    client,
    model: str,
) -> dict:
    """
    Extracts section_title + reference_text from the retrieved manual text.
    Also resolves page_number and source_manual from result_metadata.
    Returns a clickable_element dict.
    """
    # ── Resolve page number and source manual from top result ─────────────
    page_number = None
    source_manual = "Manual"
    if result_metadata and len(result_metadata) > 0:
        top_meta = result_metadata[0]
        # Support both "page_number" and legacy "page" key
        page_number = top_meta.get("page_number") or top_meta.get("page")
        source_manual = top_meta.get("manual", top_meta.get("source_manual", "Manual"))

    # ── LLM: extract section title + reference text ───────────────────────
    reference_prompt = (
        f"From the manual text below, find the section most relevant to this query: '{query}'\n\n"
        f"Manual text:\n---\n{retrieved_text[:3000]}\n---\n\n"
        "Return ONLY a valid JSON object with exactly these two fields:\n"
        "{\n"
        '  "section_title": "the heading or section name of the most relevant part. '
        'If no heading exists, write a short descriptive title based on the content (never leave empty or write N/A)",\n'
        '  "reference_text": "the exact original paragraph from the manual that most directly answers the query. '
        'CRITICAL: Keep this under 500 characters. Do not include the entire manual section. Just the most relevant 3-4 sentences."\n'
        "}\n"
        "Return only JSON. No extra text."
    )

    section_title = "Manual Reference"
    reference_text = retrieved_text[:500] if retrieved_text else "No reference text available."

    if client:
        raw = _chat(client, model, [{"role": "user", "content": reference_prompt}], temperature=0.0, max_tokens=1000)
        if raw:
            clean = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            clean = re.sub(r"\s*```$", "", clean).strip()
            try:
                ref_data = json.loads(clean)
                section_title = ref_data.get("section_title") or section_title
                reference_text = ref_data.get("reference_text") or reference_text
            except json.JSONDecodeError as err:
                print(f"  [WARN] [Stage 6c] Reference JSON parse failed ({err}). Attempting regex rescue...")
                
                # Regex fallback for truncated JSON
                title_match = re.search(r'"section_title"\s*:\s*"([^"]+)"', clean, re.IGNORECASE)
                if title_match:
                    section_title = title_match.group(1).strip()
                
                ref_match = re.search(r'"reference_text"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)', clean, re.IGNORECASE | re.DOTALL)
                if ref_match:
                    ref_text_rescued = ref_match.group(1).strip()
                    # Clean up escaped newlines
                    ref_text_rescued = ref_text_rescued.replace('\\n', '\n').replace('\\"', '\"')
                    reference_text = ref_text_rescued + " [Truncated]"

    return {
        "display_title": section_title,
        "page_number": page_number,         # None → UI shows "Page N/A"
        "source_manual": source_manual,
        "reference_text": reference_text,   # HIDDEN until user clicks/expands
    }


# ── Main Orchestrator ─────────────────────────────────────────────────────────

def orchestrate_response(
    query: str,
    sop_data: dict,
    retrieved_text: str,
    result_metadata: list[dict],
) -> dict:
    """
    Stage 6c: Orchestrates a smart 3-part response from the extracted SOP.

    Args:
        query           : Original user query string.
        sop_data        : Structured SOP dict returned by extract_sop() (Stage 6b).
        retrieved_text  : Combined text of all retrieved chunks (joined by newlines).
        result_metadata : List of metadata dicts from search results.
                          Each dict should have keys: "manual" and "page_number" (or "page").
                          Example: [{"manual": "weg_motor", "page_number": 19}, ...]

    Returns:
        Orchestrated response dict with keys:
          query, short_answer, options (or None for INSUFFICIENT_EVIDENCE).

    UI CONTRACT — Option 2:
        clickable_element.reference_text MUST be hidden by default.
        Only reveal it after an explicit user click / expand action.
    """
    print("  [Stage 6c] Building orchestrated response…")

    # ── Edge case: INSUFFICIENT_EVIDENCE ─────────────────────────────────────
    if sop_data.get("is_insufficient") or sop_data.get("status") == "INSUFFICIENT_EVIDENCE":
        print("  [Stage 6c] SOP is INSUFFICIENT_EVIDENCE — returning minimal response.")
        return {
            "query": query,
            "short_answer": (
                "This procedure was not found in the available manuals. "
                "The retrieved sections do not contain sufficient evidence to generate a reliable SOP for this query. "
                "Please consult the full service manual or try a more specific query."
            ),
            "options": None,
        }

    client, model = _get_llm_client()
    if not client:
        print("  [WARN] [Stage 6c] No LLM provider available. Generating offline orchestration.")

    # ── Step 1: Short direct answer ───────────────────────────────────────────
    print("  [Stage 6c] Step 1/3 — Generating short direct answer…")
    if client is not None and model is not None:
        short_answer = _generate_short_answer(query, sop_data, client, model)
    else:
        short_answer = _offline_short_answer(query, sop_data)

    # ── Step 2: Format Option 1 (detailed SOP) ───────────────────────────────
    print("  [Stage 6c] Step 2/3 — Formatting Option 1 (detailed SOP)…")
    option1_content = _format_option1_content(sop_data)

    # ── Step 3: Extract reference data for Option 2 ───────────────────────────
    print("  [Stage 6c] Step 3/3 — Extracting reference card data for Option 2…")
    if client is not None and model is not None:
        clickable_element = _extract_reference_data(query, retrieved_text, result_metadata, client, model)
    else:
        clickable_element = _extract_reference_data(query, retrieved_text, result_metadata, None, "")

    print("  [OK] [Stage 6c] Orchestration complete.")

    return {
        "query": query,
        "short_answer": short_answer,
        "options": {
            "option_1": {
                "label": "Explain in Detail",
                "content": option1_content,
                "sop_data": sop_data,           # raw SOP for programmatic use
            },
            "option_2": {
                "label": "Reference to Actual Document",
                "clickable_element": clickable_element,
                # reference_text lives inside clickable_element but must stay
                # HIDDEN in the UI until the user explicitly expands the card.
            },
        },
    }
