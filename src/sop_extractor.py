"""
sop_extractor.py — Stage 6a & Stage 6b: SOP Extraction Engine (v2.0)
====================================================================
Stage 6a: Conditional Visual Processing (Query-Time Vision with Hard Cap & Cache)
Stage 6b: LLM SOP Reasoning (Extracts structured JSON SOP steps)
Includes multi-provider fallback on Rate Limits (429) & Not Found (404),
plus offline fallback when API quotas are fully exhausted.
"""

import os
import json
import base64
import io
import re
from dataclasses import dataclass, field
import time
from typing import Any, Sequence
from pathlib import Path
from PIL import Image
from openai import OpenAI

import requests  # type: ignore[import-untyped]


from src.config import (
    IMAGE_HEAVY_THRESHOLD,
    VLM_CACHE_PATH,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    LLM_MODEL,
    GROQ_MODEL,
    INGEST_VLM_MODEL,
    PREDICT_API_URL,
    SUBTOPIC_NOT_FOUND_THRESHOLD,
    SUBTOPIC_PARTIAL_THRESHOLD,
)
from src.api_retry import retry_with_backoff
from src.engine import SearchResult


# ── Predict Endpoint Client ───────────────────────────────────────────────────

def predict_image_object(image_path: str, text: str, endpoint: str = PREDICT_API_URL) -> dict | str:
    """
    Calls local visual prediction API endpoint (e.g. http://192.168.0.128/predict)
    using multipart/form-data with 'image' file and 'text' prompt fields.
    
    Equivalent to:
      curl -X POST http://192.168.0.128/predict -F "image=@<image_path>" -F "text=<text>"
    """
    target_endpoint = endpoint or PREDICT_API_URL
    try:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/png")}
            data = {"text": text}
            response = requests.post(target_endpoint, files=files, data=data, timeout=30)
            response.raise_for_status()
            try:
                return response.json()
            except Exception:
                return response.text
    except Exception as err:
        print(f"  [WARN] Predict API call to {target_endpoint} failed: {err}")
        return {}


def extract_predict_text(res: Any) -> str:
    """
    Extracts prediction/analysis string from response payload returned by local predict API.
    Handles dict with common keys ('prediction', 'extracted_data', 'result', 'description',
    'output', 'text', 'message'), string payloads, or nested structures.
    """
    if not res:
        return ""
    if isinstance(res, str):
        return res.strip()
    if isinstance(res, dict):
        for key in ("prediction", "extracted_data", "result", "description", "output", "text", "message"):
            if key in res and isinstance(res[key], str) and res[key].strip():
                return res[key].strip()
        parts = []
        for k, v in res.items():
            if isinstance(v, (str, dict, list)) and v:
                parts.append(f"{k}: {v}")
        if parts:
            return "\n".join(parts)
        return json.dumps(res)
    return str(res)



# ── Multi-Provider Helper ─────────────────────────────────────────────────────

def _get_available_providers(is_vision: bool = False) -> list[dict]:
    """
    Returns list of configured API provider configurations in order of priority:
    1. Groq API     (llama-3.3-70b-versatile via OpenAI-compat endpoint — for text reasoning)
    2. Gemini API   (gemini-2.0-flash via OpenAI-compat endpoint)
    3. OpenRouter API (OpenAI-compat endpoint)
    4. OpenAI API   (native OpenAI endpoint)
    """
    providers = []

    # ── Groq via OpenAI-compat endpoint ───────────────────────────────────────
    if GROQ_API_KEY and not is_vision:
        providers.append({
            "name": "groq",
            "client": OpenAI(
                api_key=GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1"
            ),
            "model": GROQ_MODEL,
            "provider_type": "openai",
        })

    # ── Gemini via OpenAI-compat endpoint ─────────────────────────────────────
    if GEMINI_API_KEY:
        providers.append({
            "name": "gemini (2.0-flash)",
            "client": OpenAI(
                api_key=GEMINI_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            ),
            "model": "gemini-2.0-flash",
            "provider_type": "openai",
        })

    # ── OpenRouter ────────────────────────────────────────────────────────────
    if OPENROUTER_API_KEY:
        providers.append({
            "name": "openrouter",
            "client": OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1"
            ),
            "model": f"openai/{INGEST_VLM_MODEL if is_vision else LLM_MODEL}",
            "provider_type": "openai",
        })

    # ── OpenAI ────────────────────────────────────────────────────────────────
    if OPENAI_API_KEY:
        providers.append({
            "name": "openai",
            "client": OpenAI(api_key=OPENAI_API_KEY),
            "model": INGEST_VLM_MODEL if is_vision else LLM_MODEL,
            "provider_type": "openai",
        })

    return providers


def _extract_content(response: Any, provider_type: str = "openai") -> str:
    """
    Normalizes response text extraction across OpenAI-compatible providers.
    """
    if hasattr(response, "choices") and response.choices and response.choices[0].message.content:
        return response.choices[0].message.content
    return ""


# ── Stage 6a: Conditional Visual Processing ───────────────────────────────────

def resolve_visual_context(
    retrieved_chunks: Sequence[SearchResult | dict],
    user_query: str = "",
    max_images: int = IMAGE_HEAVY_THRESHOLD,
) -> list[str]:
    """
    Stage 6a: Primary Image Processing using Local Visual Predictor API (POST /predict with image + expectation text).
    Extracts image data/analysis and feeds it to LLM alongside text retrieval chunks.
    Falls back gracefully to cloud VLM models if local API endpoint is unreachable.
    """
    os.makedirs(VLM_CACHE_PATH, exist_ok=True)
    
    # Standardize items
    visual_items = []
    for item in retrieved_chunks:
        if isinstance(item, SearchResult):
            has_vis = item.has_visual_content
            img_path = item.image_path
            page_num = item.page_num
            manual = item.manual_name
        else:
            has_vis = item.get("has_visual_content", False)
            img_path = item.get("image_path", "")
            page_num = item.get("page", -1)
            manual = item.get("manual", "unknown")

        if has_vis and img_path and os.path.exists(img_path):
            visual_items.append({
                "manual": manual,
                "page_num": page_num,
                "image_path": img_path,
            })

    # Page-level deduplication by (manual, page_num)
    seen_pages = set()
    unique_visual_pages = []
    for item in visual_items:
        key = (item["manual"], item["page_num"])
        if key not in seen_pages:
            seen_pages.add(key)
            unique_visual_pages.append(item)

    # Hard cap limit enforcement
    selected_pages = unique_visual_pages[:max_images]
    print(f"  [Stage 6a] Retrieved {len(visual_items)} visual chunks -> {len(selected_pages)} unique page(s) capped (max {max_images}).")

    if not selected_pages:
        return []

    image_transcriptions = []

    for page in selected_pages:
        manual_name = page["manual"]
        page_num = page["page_num"]
        img_path = page["image_path"]

        # Check disk cache
        cache_filename = f"{manual_name}_page_{page_num}.json"
        cache_file_path = os.path.join(VLM_CACHE_PATH, cache_filename)

        if os.path.exists(cache_file_path):
            try:
                with open(cache_file_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    print(f"  [OK] [Stage 6a] Loaded disk VLM cache for {manual_name} page {page_num}")
                    image_transcriptions.append(cached_data["transcription"])
                    continue
            except Exception as e:
                print(f"  [WARN] Failed to read VLM cache file {cache_file_path}: {e}")

        # Construct expectation prompt for Local Visual Predictor API
        expectation_prompt = (
            f"Analyze attached page image from manual '{manual_name}' (page {page_num}).\n"
            f"User query context: '{user_query}'\n\n"
            f"Expected outcome: Extract and transcribe all relevant diagram details, tables, spec values, "
            f"warning notes, or schematics needed to answer the procedure query: '{user_query}'."
        )

        transcribed = False

        # ── Primary Choice: Local Visual Predictor API ───────────────────────
        print(f"  [Stage 6a] Calling Local Visual Predictor API ({PREDICT_API_URL}) for {manual_name} page {page_num}…")
        local_res = predict_image_object(img_path, text=expectation_prompt, endpoint=PREDICT_API_URL)
        extracted_text = extract_predict_text(local_res)

        if extracted_text:
            formatted_transcription = f"--- Visual Context (Local Predictor API) from {manual_name} (Page {page_num}) ---\n{extracted_text}"
            image_transcriptions.append(formatted_transcription)

            # Save to disk cache
            with open(cache_file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "manual": manual_name,
                    "page_num": page_num,
                    "transcription": formatted_transcription,
                    "source": "local_predictor_api"
                }, f, indent=2)

            transcribed = True
            print(f"  [OK] [Stage 6a] Local Visual Predictor API succeeded for {manual_name} page {page_num}")
        else:
            print(f"  [WARN] [Stage 6a] Local Visual Predictor API unreachable or empty response. Falling back to Cloud VLM providers...")

        # ── Fallback Choice: Cloud VLM Providers ──────────────────────────────
        if not transcribed:
            providers = _get_available_providers(is_vision=True)
            if not providers:
                print("  [WARN] [Stage 6a] No Cloud VLM API keys configured. Falling back gracefully to text-only.")
            else:
                try:
                    with Image.open(img_path) as img:
                        buffered = io.BytesIO()
                        img.save(buffered, format="PNG")
                        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                    prompt = (
                        f"You are an industrial manual page parser. Analyze the attached page image from '{manual_name}'.\n"
                        f"User query context: '{user_query}'\n\n"
                        f"Perform layout-aware visual transcription:\n"
                        f"1. Transcribe tables into clean Markdown tables, focusing on values/specs relevant to the query.\n"
                        f"2. Transcribe mathematical formulas into LaTeX notation.\n"
                        f"3. Describe key engineering diagrams, warnings, or schematics in concise 1-2 sentence bracketed notes.\n"
                        f"Output ONLY clean Markdown transcriptions."
                    )

                    for p in providers:
                        p_name = p["name"]
                        client = p["client"]
                        model = p["model"]
                        print(f"  [Stage 6a] Calling Cloud VLM ({p_name}/{model}) for {manual_name} page {page_num}…")

                        try:
                            p_type = p.get("provider_type", "openai")

                            def _call_vlm():
                                return client.chat.completions.create(
                                    model=model,
                                    messages=[{
                                        "role": "user",
                                        "content": [
                                            {"type": "text", "text": prompt},
                                            {
                                                "type": "image_url",
                                                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                                            }
                                        ]
                                    }],
                                    temperature=0.1,
                                    max_tokens=1500,
                                )

                            res = retry_with_backoff(_call_vlm, max_retries=1, initial_delay=2.0)
                            transcription = _extract_content(res, p_type).strip()

                            if transcription:
                                formatted_transcription = f"--- Visual Context ({p_name}) from {manual_name} (Page {page_num}) ---\n{transcription}"
                                image_transcriptions.append(formatted_transcription)

                                # Save to disk cache
                                with open(cache_file_path, "w", encoding="utf-8") as f:
                                    json.dump({
                                        "manual": manual_name,
                                        "page_num": page_num,
                                        "transcription": formatted_transcription,
                                        "source": p_name
                                    }, f, indent=2)

                                transcribed = True
                                break

                        except Exception as p_err:
                            print(f"  [WARN] Provider {p_name} failed: {p_err}. Trying next option…")

                except Exception as exc:
                    print(f"  [WARN] [Stage 6a] Image reading failed for page {page_num} ({exc}). Falling back gracefully.")

        if not transcribed:
            print(f"  [WARN] [Stage 6a] All VLM options failed for page {page_num}. Falling back gracefully to text-only.")

    return image_transcriptions


# ── Cross-Domain Contamination Validator & Domain Grounding ─────────────────

MACHINE_DOMAIN_FORBIDDEN_TERMS = {
    "robotics": ["impeller", "cnc g-code", "spindle warm-up", "cutting fluid", "valve seat"],
    "cnc_milling": ["robot", "remastering", "teach pendant", "joint axis", "servo arm", "end effector", "collaborative robot", "manipulator arm"],
    "pumps": ["robot", "remastering", "teach pendant", "joint axis", "cnc g-code", "spindle warm-up", "end effector"],
    "motors": ["robot", "remastering", "teach pendant", "joint axis", "end effector", "impeller cavity", "cnc g-code"],
    "valves": ["robot", "remastering", "teach pendant", "joint axis", "end effector", "cnc g-code", "spindle speed"],
    "plc": ["robot", "remastering", "teach pendant", "joint axis", "end effector", "cutting fluid", "spindle warm-up"],
}


def detect_machine_domain(manual_name: str) -> str:
    """Detects target industrial machine domain from manual name for cross-domain validation."""
    name_lower = manual_name.lower()
    if any(k in name_lower for k in ("abb", "irb", "ur5", "universal robot", "robot")):
        return "robotics"
    if any(k in name_lower for k in ("haas", "mill", "cnc", "lathe")):
        return "cnc_milling"
    if any(k in name_lower for k in ("pump", "grundfos", "centrifugal")):
        return "pumps"
    if any(k in name_lower for k in ("motor", "teco", "westinghouse", "weg")):
        return "motors"
    if any(k in name_lower for k in ("valve", "fisher")):
        return "valves"
    if any(k in name_lower for k in ("plc", "siemens", "s7")):
        return "plc"
    return "general"
    
# ── Base filler words ────────────────────────────────────────────────────────
# Words that carry zero evidence value and are always stripped before coverage
# is computed, regardless of query domain.
STOP_WORDS = {
    "the", "and", "for", "with", "from", "into",
    "after", "before", "using", "use", "manual",
    "machine", "page", "procedure", "operation"
}

# ── Conversational / query-structural terms ───────────────────────────────────
# Words that appear in how an operator phrases a question, NOT in manual text.
# Stripping these prevents natural-language queries from failing coverage checks.
# Example: "what steps should be followed after an emergency stop" — every
# bolded word is query structure, not technical manual evidence.
CONVERSATIONAL_TERMS = {
    # question words
    "what", "which", "where", "when", "why", "how",
    # modal / auxiliary verbs
    "should", "would", "could", "must", "will", "can",
    # process / task nouns common in queries but rare in chunk text
    "steps", "step", "recovery", "recover", "restarting", "restart",
    "referenced", "reference", "continuing", "continue", "following",
    "followed", "performed", "perform", "addressed", "ensure", "ensuring",
    "verify", "verifying", "needed", "needs", "need", "requires", "required",
    "appear", "appears", "immediately", "correctly", "properly",
    # role / personnel words
    "operator", "technician", "maintenance", "user", "personnel",
    # temporal / logical connectives
    "during", "while", "once", "then", "also", "only", "still",
    "longer", "already", "again", "now", "out",
    # query-structural adjectives
    "scheduled", "new", "newly", "safely", "safe", "ready",
    "clear", "cleared", "appropriate", "correct", "proper",
    # state / condition words common in conversational queries
    "state", "condition", "situation", "status", "case",
    "including", "related", "possible", "likely",
}

# Merge conversational terms into stop-word set so they are excluded at
# tokenisation, before any coverage calculation.
STOP_WORDS |= CONVERSATIONAL_TERMS

# ── Universal safety-critical component nouns (Layer 1 — hard-fail) ──────────
#
# These terms name PHYSICAL COMPONENTS that appear across virtually every
# industrial domain. If any of these appear in the query but are ABSENT from
# the retrieved evidence, the validator will immediately reject the query as
# INSUFFICIENT_EVIDENCE.
#
# Design rules:
#   • Only include terms that are cross-domain (appear in CNC, pumps, robots,
#     PLCs, motors, valves, and future domains alike).
#   • Domain-specific terms (impeller, coolant, manifold, solenoid, pneumatic,
#     actuator, lubrication) are deliberately excluded here — they are now
#     captured automatically per-manual by manual_metadata.py and used only
#     as coverage signals (Layer 2), not hard-fail triggers.
#   • "pressure" excluded: acts as symptom descriptor in pump/valve queries
#     ("does not build pressure") rather than a named component noun.
UNIVERSAL_CRITICAL: frozenset[str] = frozenset({
    "bearing", "spindle", "servo", "encoder", "hydraulic",
    "pump", "valve", "motor", "relay", "fuse", "sensor",
    "brake", "gasket", "seal", "shaft",
})

# Backward-compatibility alias — any internal code still referencing
# CRITICAL_TERMS will continue to work without changes.
CRITICAL_TERMS = UNIVERSAL_CRITICAL


# Presentation / UI words — excluded from coverage calculation
# (users often ask for "diagram" or "panel" which may not appear verbatim in chunk text)
PRESENTATION_TERMS = {
    "diagram", "figure", "image", "illustration",
    "panel", "screen", "display", "photo", "chart",
    "picture", "schematic"
}

# Generic action/task words — excluded from coverage calculation
# (e.g. "startup", "operation" are query intent words, not evidence tokens)
ACTION_TERMS = {
    "procedure", "operation", "startup", "shutdown",
    "inspection", "diagnostic", "install",
    "fails", "fail", "return", "check",
    "start", "stop", "run", "set", "get"
    # NOTE: "reset" intentionally excluded — it is a technical expectation in
    # industrial contexts (CPU MRES, servo reset, safety reset) and should
    # remain in coverage scoring to verify evidence relevance.
}


# ── Domain-Critical Terms for Sub-Topic Coverage Weighting ───────────────────
# When these terms appear in a sub-topic's term set but are ALL absent from
# evidence, the sub-topic is forced to "not_found" regardless of ratio score.
# This prevents generic filler words (e.g. "requirements") from inflating
# coverage for sub-topics whose core concepts are completely missing.
DOMAIN_CRITICAL_TERMS: frozenset[str] = frozenset({
    "lockout", "tagout", "isolation", "deenergize", "disconnect",
    "loto", "energize", "energized", "interlock", "grounding",
})

# ── Compound Phrases — False Conjunction Safeguard ───────────────────────────
# Phrases that look like conjunctions but are compound noun phrases.
# If any of these appear verbatim in the query, the conjunction is NOT split.
COMPOUND_PHRASES: frozenset[str] = frozenset({
    "installation and operation",
    "operation and maintenance",
    "safety and handling",
    "care and maintenance",
    "setup and configuration",
    "start and stop",
    "lock and tag",
    "test and measurement",
    "input and output",
    "supply and return",
})

# ── Conjunction pattern for sub-topic splitting ──────────────────────────────
# NOTE: 'or' is deliberately excluded — it typically joins alternatives within
# a single topic ("lockout or isolation requirements") rather than separating
# distinct topics.  Only 'and', commas, 'as well as', and 'including' trigger
# sub-topic decomposition.
_CONJUNCTION_PATTERN = re.compile(
    r"\s*(?:,?\s+and\s+|,\s+|\s+as well as\s+|\s+including\s+)\s*",
    re.IGNORECASE,
)


@dataclass
class QuerySubTopic:
    """Represents one sub-topic extracted from a multi-topic query."""
    label: str
    terms: set[str] = field(default_factory=set)


def decompose_query_topics(query: str) -> list[QuerySubTopic]:
    """
    Splits a multi-topic query into distinct sub-topics using conjunction
    detection. Applies compound-phrase safeguards and modifier propagation.

    Design:
    - Conjunction-level splitting only (no semantic parsing).
    - Compound phrases (e.g. "installation and operation") are protected.
    - Trailing modifiers shared across sub-topics are propagated.
      e.g. "electrical and mechanical safety warnings"
        -> ["electrical safety warnings", "mechanical safety warnings"]

    Returns:
        List of QuerySubTopic instances. Single-topic queries return a list
        with one element (no decomposition applied).
    """
    # ── Strip leading action/intent verbs common in queries ───────────────
    # e.g. "Extract all electrical safety..." -> "electrical safety..."
    query_clean = re.sub(
        r"^(extract|list|find|show|get|identify|describe|what are the|what are)\s+(all\s+)?",
        "", query, flags=re.IGNORECASE,
    ).strip()
    if not query_clean:
        query_clean = query.strip()

    # ── Strip trailing meta-phrases ──────────────────────────────────────
    # e.g. "...mentioned in the manual" -> remove
    query_clean = re.sub(
        r"\s+(mentioned|found|listed|described|specified|stated|given|provided)"
        r"\s+(in|within|on|from)\s+(the\s+)?(manual|document|pdf|page)s?\.?$",
        "", query_clean, flags=re.IGNORECASE,
    ).strip()

    # ── Compound phrase protection ───────────────────────────────────────
    query_lower = query_clean.lower()
    for phrase in COMPOUND_PHRASES:
        if phrase in query_lower:
            # This query contains a compound noun phrase -- don't split it
            terms = _tokenize_subtopic(query_clean)
            return [QuerySubTopic(label=query_clean.strip(), terms=terms)]

    # ── Conjunction splitting ────────────────────────────────────────────
    fragments = _CONJUNCTION_PATTERN.split(query_clean)
    fragments = [f.strip() for f in fragments if f.strip()]

    if len(fragments) <= 1:
        terms = _tokenize_subtopic(query_clean)
        return [QuerySubTopic(label=query_clean.strip(), terms=terms)]

    # ── Modifier propagation ─────────────────────────────────────────────
    # Detect shared trailing modifier in the last fragment.
    # e.g. fragments = ["electrical", "mechanical safety warnings"]
    #   -> last_terms = ["mechanical", "safety", "warnings"]
    #   -> modifier candidates = ["safety", "warnings"] (non-STOP words from end)
    # If a preceding fragment has <=1 content term, propagate the modifier.
    last_terms = re.findall(r"\b[a-zA-Z]{3,}\b", fragments[-1])
    modifier_words: list[str] = []
    for word in reversed(last_terms):
        if word.lower() not in STOP_WORDS:
            modifier_words.insert(0, word)
        else:
            break

    subtopics: list[QuerySubTopic] = []
    for i, frag in enumerate(fragments):
        content_words = [
            w for w in re.findall(r"\b[a-zA-Z]{3,}\b", frag)
            if w.lower() not in STOP_WORDS
        ]

        # If fragment is too thin (<=1 content term) and it's not the last,
        # propagate modifier from the last fragment.
        if len(content_words) < 2 and i < len(fragments) - 1 and modifier_words:
            # Avoid duplicating modifier words already in the fragment
            existing_lower = {w.lower() for w in content_words}
            additions = [w for w in modifier_words if w.lower() not in existing_lower]
            enriched_label = frag + " " + " ".join(additions)
            terms = _tokenize_subtopic(enriched_label)
            subtopics.append(QuerySubTopic(label=enriched_label.strip(), terms=terms))
        else:
            terms = _tokenize_subtopic(frag)
            # Skip fragments where ALL terms were stop-words
            if terms:
                subtopics.append(QuerySubTopic(label=frag.strip(), terms=terms))

    # Fallback: if decomposition produced nothing, return single topic
    if not subtopics:
        terms = _tokenize_subtopic(query_clean)
        return [QuerySubTopic(label=query_clean.strip(), terms=terms)]

    return subtopics


def _tokenize_subtopic(text: str) -> set[str]:
    """Tokenize a sub-topic label, stripping STOP_WORDS."""
    raw = {t.lower() for t in re.findall(r"\b[a-zA-Z]{3,}\b", text)}
    filtered = raw - STOP_WORDS
    return filtered if filtered else raw


def score_subtopic_coverage(
    subtopic: QuerySubTopic,
    evidence_blob: str,
    retrieved_pages: list[int],
) -> dict:
    """
    Score a single sub-topic's coverage against the evidence blob.

    Uses critical-term weighting: if the sub-topic contains any
    DOMAIN_CRITICAL_TERMS and NONE of them match the evidence, the sub-topic
    is forced to 'not_found' regardless of ratio score.

    Returns dict with keys: label, coverage, status, matched, missing,
    matched_pages, searched_pages, note.
    """
    terms = subtopic.terms - PRESENTATION_TERMS - ACTION_TERMS
    if not terms:
        terms = subtopic.terms  # fallback: keep all if everything was excluded

    matched = sorted([t for t in terms if t in evidence_blob])
    missing = sorted([t for t in terms if t not in evidence_blob])

    ratio = len(matched) / max(len(terms), 1)

    # ── Critical-term weighting ──────────────────────────────────────────
    critical_in_subtopic = terms & DOMAIN_CRITICAL_TERMS
    if critical_in_subtopic and not any(t in evidence_blob for t in critical_in_subtopic):
        # Domain-critical terms are ALL absent -> force not_found
        status = "not_found"
        ratio = 0.0
        note = (
            f"No explicit {subtopic.label} content was identified in the "
            f"retrieved sections (pages {', '.join(str(p) for p in retrieved_pages)})."
        )
    elif ratio < SUBTOPIC_NOT_FOUND_THRESHOLD:
        status = "not_found"
        note = (
            f"No explicit {subtopic.label} content was identified in the "
            f"retrieved sections (pages {', '.join(str(p) for p in retrieved_pages)})."
        )
    elif ratio < SUBTOPIC_PARTIAL_THRESHOLD:
        status = "partial"
        note = (
            f"Only partial evidence for {subtopic.label} was found. "
            f"Missing concepts: {', '.join(missing)}."
        )
    else:
        status = "covered"
        note = ""

    return {
        "label": subtopic.label,
        "coverage": round(ratio, 2),
        "status": status,
        "matched": matched,
        "missing": missing,
        "matched_pages": [],  # populated downstream by caller if needed
        "searched_pages": retrieved_pages,
        "note": note,
    }


def build_coverage_gap_advisory(
    subtopic_scores: list[dict],
) -> str:
    """
    Builds a COVERAGE GAP ADVISORY block for injection into the LLM user
    prompt. Only includes sub-topics with status 'not_found' or 'partial'.

    Returns empty string if all sub-topics are fully covered.
    """
    gaps = [s for s in subtopic_scores if s["status"] in ("not_found", "partial")]
    if not gaps:
        return ""

    lines = [
        "--- COVERAGE GAP ADVISORY ---",
        "The following query sub-topics have INSUFFICIENT or ZERO evidence "
        "in the retrieved context:",
        "",
    ]
    for g in gaps:
        status_label = "NOT FOUND" if g["status"] == "not_found" else "PARTIAL"
        lines.append(
            f"- \"{g['label']}\" ({status_label}, coverage: {g['coverage']:.2f}, "
            f"missing terms: {', '.join(g['missing'])})"
        )

    lines.append("")
    lines.append(
        "For sub-topics marked NOT FOUND, you MUST:\n"
        "- NOT generate steps or claims about them\n"
        "- Include a \"coverage_gaps\" array in your JSON output listing each "
        "unsupported sub-topic with keys: \"topic\", \"status\", \"searched_pages\", \"note\"\n"
        "- State explicitly that these topics were not found in the retrieved sections"
    )
    lines.append(
        "\nFor sub-topics marked PARTIAL, generate steps only for the "
        "concepts that ARE present in the evidence, and note missing concepts "
        "in the coverage_gaps array."
    )
    lines.append("--- END COVERAGE GAP ADVISORY ---")
    return "\n".join(lines)


def validate_topic_evidence(
    query: str,
    evidence_texts: list[str],
    manual_vocab: set[str] | None = None,
) -> tuple[bool, dict]:
    """
    Validate whether retrieved evidence actually supports the query topic.

    v3.0 two-tier vocabulary architecture
    ======================================
    Tier 0 — STOP_WORDS (includes CONVERSATIONAL_TERMS):
        Stripped at tokenisation. Structural/filler words that appear in the
        question but carry no evidence expectation:
        e.g. "should", "recovery", "technician", "referenced", "restarting".

    Tier 1 — UNIVERSAL_CRITICAL (hard-fail):
        Cross-domain physical component nouns. If any appear in the query but
        are absent from evidence, reject immediately.
        e.g. "spindle", "bearing", "pump", "valve".
        Domain-specific terms (impeller, coolant, airend …) are NOT in this
        set — they come from manual_vocab (Tier 2).

    Tier 2 — manual_vocab (coverage signal only — NOT hard-fail):
        Dynamically extracted per-manual vocabulary (from manual_metadata.py).
        Terms found in both the query and manual_vocab are guaranteed to
        participate in coverage scoring even if they were in ACTION_TERMS.
        Their absence from evidence HURTS the coverage score but does NOT
        trigger an immediate hard-fail.
        e.g. for an Atlas Copco manual: {"airend", "aftercooler", "condensate"}

    Tier 3 — PRESENTATION_TERMS + ACTION_TERMS (soft-exclude):
        Words excluded from coverage scoring because they appear in queries
        but not necessarily in manual text chunks.
        Exception: terms explicitly in manual_vocab are protected from this
        exclusion — if the manual itself uses a word as a technical noun, it
        should count toward coverage even if it looks like an action word.

    Soft-fail rule: reject only when BOTH coverage < 0.35 AND a critical term
    is also missing. A lone low-coverage score (caused by conversational
    phrasing after Tier-0 stripping) is NOT sufficient to block the query.

    Args:
        query:        The user's natural-language query string.
        evidence_texts: List of retrieved chunk texts from the vector store.
        manual_vocab: Dynamic vocabulary loaded from manual metadata JSON
                      (output of get_combined_vocab()). Pass None to operate
                      in UNIVERSAL_CRITICAL-only mode (safe fallback).

    Returns:
        (is_valid: bool, coverage_info: dict)
    """
    # Tokenise, strip Tier-0 stop + conversational words
    query_terms = {
        t.lower()
        for t in re.findall(r"\b[a-zA-Z]{3,}\b", query)
        if t.lower() not in STOP_WORDS
    }
    # Edge-case: if every word was stripped, keep them all
    if not query_terms:
        query_terms = {t.lower() for t in re.findall(r"\b[a-zA-Z]{3,}\b", query)}

    evidence_blob = " ".join(evidence_texts).lower()

    # ── Full match logging (before any exclusion, for debug visibility) ───────
    matched = sorted([t for t in query_terms if t in evidence_blob])
    missing = sorted([t for t in query_terms if t not in evidence_blob])

    effective_terms = query_terms - PRESENTATION_TERMS

    # ── Tier 1: Hard-fail — ONLY on UNIVERSAL_CRITICAL terms ─────────────────
    # manual_vocab terms are coverage signals, never hard-fail triggers.
    critical_in_query = effective_terms & UNIVERSAL_CRITICAL
    missing_critical = sorted([t for t in critical_in_query if t not in evidence_blob])

    # ── Tier 2 + 3: Coverage scoring ─────────────────────────────────────────
    # Base coverage set: exclude pure action/presentation words.
    # Exception: if a term is in manual_vocab, restore it even if ACTION_TERMS
    # would have excluded it — the manual itself uses it as a technical noun.
    dynamic_vocab = (manual_vocab or set()) - STOP_WORDS
    query_in_dynamic = effective_terms & dynamic_vocab   # domain-specific query signals

    base_coverage_terms = effective_terms - ACTION_TERMS - PRESENTATION_TERMS
    # Restore dynamic-vocab terms that were caught by ACTION_TERMS exclusion
    coverage_terms = base_coverage_terms | query_in_dynamic

    matched_effective = [t for t in coverage_terms if t in evidence_blob]
    coverage = len(matched_effective) / max(len(coverage_terms), 1)

    # ── Sub-topic decomposition and per-topic scoring ────────────────────────
    subtopics = decompose_query_topics(query)
    # Pages are populated downstream in extract_sop from retrieved_chunks

    subtopic_scores: list[dict] = []
    if len(subtopics) > 1:
        for st in subtopics:
            st_score = score_subtopic_coverage(st, evidence_blob, [])
            subtopic_scores.append(st_score)

    info = {
        "coverage": round(coverage, 2),
        "matched": matched,
        "missing": missing,
        "missing_critical": missing_critical,
        "dynamic_signals": sorted(query_in_dynamic),
        "subtopic_coverage": subtopic_scores,
    }

    # Hard-fail: a universal safety-critical component is absent from evidence
    if missing_critical:
        return False, info

    # Soft-fail: only when coverage is very low AND a critical term is also
    # missing. A lone low-coverage score (caused by conversational phrasing
    # after Tier-0 stripping) is NOT sufficient to block the query.
    if coverage < 0.35 and missing_critical:
        return False, info

    return True, info


def build_insufficient_evidence_response(
    query: str,
    manual_name: str,
    retrieved_chunks: Sequence[SearchResult | dict],
    matched_keywords: list[str],
    missing_keywords: list[str],
    missing_critical: list[str] | None = None,
    coverage_info: dict | None = None,
) -> dict:
    """Constructs a structured production-grade INSUFFICIENT EVIDENCE response."""
    valid_pages = []
    page_summaries = []
    for c in retrieved_chunks:
        if isinstance(c, SearchResult):
            valid_pages.append(c.page_num)
            snippet = c.chunk_text[:120].replace("\n", " ").strip()
            page_summaries.append(f"Page {c.page_num}: {snippet}…")
        elif isinstance(c, dict):
            page_num = c.get("page", -1)
            valid_pages.append(page_num)
            snippet = c.get("chunk_text", "")[:120].replace("\n", " ").strip()
            page_summaries.append(f"Page {page_num}: {snippet}…")

    unique_pages = sorted(list(set(valid_pages)))
    pages_str = ", ".join(str(p) for p in unique_pages) if unique_pages else "N/A"

    return {
        "status": "INSUFFICIENT_EVIDENCE",
        "procedure_title": f"INSUFFICIENT EVIDENCE: Diagnostic Procedure Not Found in Manual",
        "machine_type": f"{manual_name}",
        "is_insufficient": True,
        "query_context": query,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "missing_critical": missing_critical or [],
        "coverage_info": coverage_info or {"coverage": 0.0, "matched": matched_keywords, "missing": missing_keywords, "missing_critical": missing_critical or []},
        "retrieved_pages": unique_pages,
        "steps": [],
        "message": "The retrieved manual sections do not contain sufficient evidence to generate a reliable SOP for this query.",
        "safety_warnings": [
            f"The retrieved sections of '{manual_name}' (Pages {pages_str}) do not contain sufficient evidence for '{query}'.",
            f"Missing required critical terms in retrieved context: {missing_critical if missing_critical else missing_keywords}."
        ],
        "related_topics_found": page_summaries,
        "recommendation": f"Consult the official '{manual_name}' service manual or diagnostic troubleshooting section for '{query}'.",
        "estimated_duration": "N/A",
    }


def validate_and_sanitize_sop(
    sop_data: dict,
    manual_name: str,
    retrieved_chunks: Sequence[SearchResult | dict],
    query: str = "",
    subtopic_scores: list[dict] | None = None,
) -> dict:
    """
    Validates extracted SOP JSON against cross-domain contamination and ensures
    evidence/source page mapping. Performs Insufficient Evidence Detection.

    v3.1: Injects coverage_gaps from pre-computed subtopic analysis when the
    LLM omits them. Qualifies the procedure title when not_found topics exist.
    """
    domain = detect_machine_domain(manual_name)
    forbidden = MACHINE_DOMAIN_FORBIDDEN_TERMS.get(domain, [])

    # Check for explicit LLM insufficiency declarations
    title_lower = sop_data.get("procedure_title", "").lower()
    if "insufficient" in title_lower or "not contain" in title_lower or "no specific procedure" in title_lower:
        sop_data["is_insufficient"] = True
        return sop_data

    # Extract valid pages from retrieved chunks
    valid_pages = []
    for c in retrieved_chunks:
        if isinstance(c, SearchResult):
            valid_pages.append(c.page_num)
        elif isinstance(c, dict):
            valid_pages.append(c.get("page", -1))

    fallback_page = valid_pages[0] if valid_pages else 1

    sanitized_steps = []
    for step in sop_data.get("steps", []):
        step_text = json.dumps(step).lower()
        # Reject step if forbidden domain term is present
        if any(term in step_text for term in forbidden):
            matched = [term for term in forbidden if term in step_text]
            print(f"  [WARN] Sanitizer rejected cross-domain contamination step ({matched}): '{step.get('action')} {step.get('object')}'")
            continue

        # Ensure source_page and evidence fields
        if not step.get("source_page") or step.get("source_page") not in valid_pages:
            step["source_page"] = fallback_page
        if not step.get("evidence"):
            step["evidence"] = f"Extracted from manual context (Page {step['source_page']})"

        sanitized_steps.append(step)

    # Re-index step numbers sequentially
    for idx, s in enumerate(sanitized_steps, 1):
        s["step_number"] = idx

    sop_data["steps"] = sanitized_steps

    # Filter safety warnings for forbidden terms
    sanitized_warnings = []
    for w in sop_data.get("safety_warnings", []):
        if not any(term in w.lower() for term in forbidden):
            sanitized_warnings.append(w)
        else:
            print(f"  [WARN] Sanitizer removed cross-domain warning: '{w}'")

    sop_data["safety_warnings"] = sanitized_warnings

    # ── Coverage gap fallback injection ──────────────────────────────────────
    # If subtopic analysis detected gaps but the LLM omitted coverage_gaps,
    # inject them from the pre-computed scores as a safety net.
    if subtopic_scores:
        gaps_from_scores = [
            {
                "topic": s["label"],
                "status": s["status"],
                "searched_pages": s["searched_pages"],
                "note": s["note"],
            }
            for s in subtopic_scores
            if s["status"] in ("not_found", "partial")
        ]

        llm_gaps = sop_data.get("coverage_gaps", [])
        if gaps_from_scores and not llm_gaps:
            # LLM didn't produce coverage_gaps — inject from analysis
            sop_data["coverage_gaps"] = gaps_from_scores
            print(f"  [Sanitizer] Injected {len(gaps_from_scores)} coverage gap(s) from subtopic analysis.")
        elif gaps_from_scores and llm_gaps:
            # Merge: keep LLM gaps but add any missing ones from analysis
            llm_topics = {g.get("topic", "").lower() for g in llm_gaps}
            for g in gaps_from_scores:
                if g["topic"].lower() not in llm_topics:
                    llm_gaps.append(g)
            sop_data["coverage_gaps"] = llm_gaps

        # ── Title qualification (only for not_found topics) ──────────────
        not_found_topics = [
            s["label"] for s in subtopic_scores
            if s["status"] == "not_found"
        ]
        if not_found_topics:
            current_title = sop_data.get("procedure_title", "")
            # Only qualify if not already qualified
            if "Not Found" not in current_title:
                short = ", ".join(
                    topic.title() for topic in not_found_topics[:2]
                )
                sop_data["procedure_title"] = f"{current_title} ({short} Not Found)"

        # Persist subtopic scores for downstream use
        sop_data["subtopic_coverage"] = subtopic_scores

    return sop_data


def extract_sop(
    query: str,
    retrieved_chunks: Sequence[SearchResult | dict],
    image_context: list[str] | None = None,
    category: str = "general",
) -> dict:
    """
    Stage 6b: Takes retrieved text chunks + optional Stage 6a image context,
    invokes LLM, and returns structured JSON SOP. Uses multi-provider fallback,
    and returns offline fallback if all API quotas are exhausted.

    v3.0: Loads dynamic per-manual vocabulary from manual_metadata.py and
    passes it to validate_topic_evidence() as a coverage signal (Layer 2).
    Falls back gracefully to UNIVERSAL_CRITICAL-only mode if no metadata found.
    """
    from src.manual_metadata import get_combined_vocab  # local import avoids circular deps

    first_chunk = retrieved_chunks[0] if retrieved_chunks else None
    target_manual_name = (
        first_chunk.manual_name if isinstance(first_chunk, SearchResult)
        else (first_chunk.get("manual", "industrial_manual") if isinstance(first_chunk, dict) else "industrial_manual")
    )

    # ── Pre-LLM Topic Evidence Validation Pass ────────────────────────────────
    evidence_texts = []
    for c in retrieved_chunks:
        if isinstance(c, SearchResult):
            evidence_texts.append(c.chunk_text)
        elif isinstance(c, dict):
            evidence_texts.append(c.get("chunk_text", ""))

    if image_context:
        evidence_texts.extend(image_context)

    # v3.0 — load dynamic vocabulary for the retrieved manuals
    # Fails silently (returns empty set) if metadata files don't exist yet.
    dynamic_vocab = get_combined_vocab(retrieved_chunks)

    is_valid, coverage_info = validate_topic_evidence(
        query=query,
        evidence_texts=evidence_texts,
        manual_vocab=dynamic_vocab,
    )

    # ── Populate subtopic searched_pages from retrieved chunks ─────────────
    subtopic_scores = coverage_info.get("subtopic_coverage", [])
    retrieved_pages = sorted(set(
        c.page_num if isinstance(c, SearchResult) else c.get("page", -1)
        for c in retrieved_chunks
    ))
    for st_score in subtopic_scores:
        st_score["searched_pages"] = retrieved_pages
        # Regenerate note with actual page numbers (initial scoring used [])
        pages_str = ", ".join(str(p) for p in retrieved_pages)
        if st_score["status"] == "not_found" and st_score.get("note"):
            st_score["note"] = (
                f"No explicit {st_score['label']} content was identified in the "
                f"retrieved sections (pages {pages_str})."
            )
        elif st_score["status"] == "partial" and st_score.get("note"):
            st_score["note"] = (
                f"Only partial evidence for {st_score['label']} was found. "
                f"Missing concepts: {', '.join(st_score.get('missing', []))}."
            )

    print(
        f"  [Stage 6b] Topic Evidence Coverage: {coverage_info['coverage']} "
        f"(Matched: {coverage_info['matched']}, Missing: {coverage_info['missing']}, "
        f"Missing Critical: {coverage_info['missing_critical']})"
    )

    # ── Log subtopic decomposition if multi-topic ─────────────────────────
    if subtopic_scores:
        for st_s in subtopic_scores:
            status_icon = {"not_found": "\u2717", "partial": "\u25d0", "covered": "\u2713"}.get(st_s["status"], "?")
            print(
                f"  [Stage 6b] Sub-topic '{st_s['label']}': "
                f"{status_icon} {st_s['status']} (coverage: {st_s['coverage']:.2f}, "
                f"matched: {st_s['matched']}, missing: {st_s['missing']})"
            )

    if not is_valid:
        print(f"  [WARN] Pre-LLM Validation Failed! Missing critical terms: {coverage_info['missing_critical']} (Coverage: {coverage_info['coverage']}). Rejecting prior to LLM call.")
        return build_insufficient_evidence_response(
            query=query,
            manual_name=target_manual_name,
            retrieved_chunks=retrieved_chunks,
            matched_keywords=coverage_info["matched"],
            missing_keywords=coverage_info["missing"],
            missing_critical=coverage_info["missing_critical"],
            coverage_info=coverage_info,
        )

    providers = _get_available_providers(is_vision=False)
    if not providers:
        print("  [WARN] No active LLM API provider keys found. Using offline text extraction fallback.")
        fallback = _generate_offline_fallback_sop(query, retrieved_chunks)
        return validate_and_sanitize_sop(fallback, target_manual_name, retrieved_chunks, query=query)

    # Format text context
    text_pieces = []
    for c in retrieved_chunks:
        if isinstance(c, SearchResult):
            text_pieces.append(f"[Page {c.page_num} | {c.manual_name}]\n{c.chunk_text}")
        else:
            text_pieces.append(f"[Page {c.get('page', -1)} | {c.get('manual', 'manual')}]\n{c.get('chunk_text', '')}")

    text_context = "\n\n".join(text_pieces)
    visual_context = "\n\n".join(image_context) if image_context else "None"

    system_prompt = (
        f"You are ParseOS, an expert industrial AI intelligence engine.\n"
        f"Your task is to extract a strictly grounded, step-by-step Standard Operating Procedure (SOP) "
        f"EXCLUSIVELY from the provided retrieved manual context for '{target_manual_name}'.\n\n"
        "STRICT GROUNDING & DOMAIN SAFETY RULES:\n"
        f"1. STRICT DOMAIN ISOLATION: Extract SOP steps ONLY for '{target_manual_name}'. "
        "Do NOT introduce components, safety warnings, or terms from unrelated machine domains "
        "(e.g., do NOT mention robots, remastering, teach pendants, joint axes, or end effectors when parsing a mill, pump, or motor manual).\n"
        "2. EVIDENCE TRACEABILITY: Every step MUST be directly traceable to the retrieved context. Include the exact "
        "source page number ('source_page') and a brief supporting text quote/excerpt ('evidence') from the context.\n"
        "3. NO HALLUCINATED PROCEDURES: If the retrieved context does NOT contain a specific procedure for the query, "
        "explicitly state 'The retrieved manual sections do not contain a complete procedure for this query' in procedure_title "
        "and leave steps empty [] rather than inventing speculative maintenance steps.\n"
        "4. COVERAGE GAP HONESTY: If a COVERAGE GAP ADVISORY is present in the user context, "
        "you MUST NOT generate unsupported procedures for those topics. Instead, include a "
        "'coverage_gaps' array in your JSON output for any sub-topics that lack evidence.\n"
        "5. STRICT JSON FORMAT: Output MUST be valid JSON adhering strictly to the schema below without markdown wrappers or codeblocks.\n\n"
        "JSON SCHEMA:\n"
        "{\n"
        '  "procedure_title": "Concise Procedure Title (only for topics WITH evidence)",\n'
        '  "machine_type": "Specific Machine / System Type",\n'
        '  "steps": [\n'
        "    {\n"
        '      "step_number": 1,\n'
        '      "action": "Clear imperative verb (e.g. Shut down, Disconnect, Inspect)",\n'
        '      "object": "Target component or system",\n'
        '      "condition": "Precondition, threshold, or timing rule",\n'
        '      "risk_level": "low" | "medium" | "high",\n'
        '      "required_tool": "Required tool or device (or none)",\n'
        '      "source_page": 144,\n'
        '      "evidence": "Brief supporting quote or excerpt from the context"\n'
        "    }\n"
        "  ],\n"
        '  "coverage_gaps": [\n'
        "    {\n"
        '      "topic": "Sub-topic that lacks evidence",\n'
        '      "status": "not_found",\n'
        '      "searched_pages": [8, 9, 367],\n'
        '      "note": "Brief explanation of what was not found"\n'
        "    }\n"
        "  ],\n"
        '  "safety_warnings": ["Warning statement 1", "Warning statement 2"],\n'
        '  "estimated_duration": "Estimated time (e.g. 15-30 minutes)"\n'
        "}\n"
    )

    # ── Build coverage gap advisory for user prompt injection ─────────────
    coverage_advisory = build_coverage_gap_advisory(subtopic_scores)

    user_prompt = (
        f"Target Manual: {target_manual_name}\n"
        f"Category: {category}\n"
        f"User Query: {query}\n\n"
    )
    if coverage_advisory:
        user_prompt += f"{coverage_advisory}\n\n"
    user_prompt += (
        f"--- MANUAL TEXT CONTEXT ---\n"
        f"{text_context}\n\n"
        f"--- VISUAL TABLE & DIAGRAM CONTEXT ---\n"
        f"{visual_context}\n\n"
        f"Extract the structured SOP JSON now:"
    )

    last_error = None
    for p in providers:
        p_name = p["name"]
        client = p["client"]
        model = p["model"]
        print(f"  [Stage 6b] Generating structured SOP JSON via {p_name}/{model}…")

        try:
            p_type = p.get("provider_type", "openai")

            # OpenAI-compat API call (Groq, Gemini, OpenRouter, OpenAI)
            def _call_llm():
                return client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                )

            response = retry_with_backoff(_call_llm, max_retries=1, initial_delay=2.0)
            raw_content = _extract_content(response, p_type).strip()
            sop_data = parse_json_response(raw_content)

            # Apply domain-safety validation and evidence coverage check
            sanitized_sop = validate_and_sanitize_sop(
                sop_data, target_manual_name, retrieved_chunks,
                query=query, subtopic_scores=subtopic_scores,
            )
            return sanitized_sop

        except Exception as exc:
            last_error = exc
            print(f"  [WARN] Stage 6b failed with provider {p_name} ({exc}). Trying next option…")

    print(f"  [WARN] All LLM API providers failed or exhausted quota ({last_error}). Generating offline fallback SOP.")
    fallback = _generate_offline_fallback_sop(query, retrieved_chunks)
    return validate_and_sanitize_sop(
        fallback, target_manual_name, retrieved_chunks,
        query=query, subtopic_scores=subtopic_scores,
    )


def _generate_offline_fallback_sop(query: str, retrieved_chunks: Sequence[SearchResult | dict]) -> dict:
    """Generates structured fallback SOP from retrieved text chunks when API quota is exhausted."""
    first_chunk = retrieved_chunks[0] if retrieved_chunks else None
    manual_name = first_chunk.manual_name if isinstance(first_chunk, SearchResult) else "Industrial Manual"
    page_num = first_chunk.page_num if isinstance(first_chunk, SearchResult) else 1

    if isinstance(first_chunk, SearchResult):
        text_excerpt = first_chunk.chunk_text[:300]
    elif isinstance(first_chunk, dict):
        text_excerpt = str(first_chunk.get("text", "Refer to manual documentation."))[:300]
    else:
        text_excerpt = "Refer to manual documentation."

    return {
        "procedure_title": f"Procedure for {query.title()}",
        "machine_type": f"System in {manual_name}",
        "steps": [
            {
                "step_number": 1,
                "action": "Inspect & Isolate",
                "object": f"Machine component per manual section (Page {page_num})",
                "condition": "Prior to starting maintenance procedure",
                "risk_level": "medium",
                "required_tool": "Standard Maintenance Toolkit",
                "source_page": page_num,
                "evidence": text_excerpt[:150],
            },
            {
                "step_number": 2,
                "action": "Execute Procedure",
                "object": f"Follow manual instructions: {text_excerpt[:100]}…",
                "condition": "Ensure system power is safely disconnected",
                "risk_level": "high",
                "required_tool": "Calibrated Measuring Device / Safety Gear",
                "source_page": page_num,
                "evidence": text_excerpt[100:250] if len(text_excerpt) > 100 else text_excerpt,
            }
        ],
        "safety_warnings": [
            "Always verify power supply isolation before beginning work.",
            "Displaying grounded text-chunk fallback SOP."
        ],
        "estimated_duration": "30-60 minutes",
    }


def parse_json_response(content: str) -> dict:
    """Helper to parse raw JSON output, stripping markdown code block wrappers if present."""
    clean = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean).strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Trailing comma cleanup fallback
        clean_fixed = re.sub(r",\s*([}\]])", r"\1", clean)
        try:
            return json.loads(clean_fixed)
        except json.JSONDecodeError as err:
            print(f"  [WARN] JSON Parsing failed ({err}). Returning structured error fallback.")
            return {
                "procedure_title": "Extracted SOP Procedure",
                "machine_type": "Industrial Equipment",
                "steps": [
                    {
                        "step_number": 1,
                        "action": "Review procedure raw output",
                        "object": "Manual Text",
                        "condition": "Parsing note",
                        "risk_level": "medium",
                        "required_tool": "None"
                    }
                ],
                "safety_warnings": ["Always follow standard electrical and physical safety guidelines."],
                "estimated_duration": "Unknown",
                "raw_text": content,
            }
