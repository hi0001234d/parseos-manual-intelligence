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
import time
from typing import Any
from pathlib import Path
from PIL import Image
from openai import OpenAI

from src.config import (
    IMAGE_HEAVY_THRESHOLD,
    VLM_CACHE_PATH,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    GEMINI_API_KEY,
    ANTHROPIC_API_KEY,
    LLM_MODEL,
    INGEST_VLM_MODEL,
    ANTHROPIC_MODEL,
    ANTHROPIC_VLM_MODEL,
)
from src.api_retry import retry_with_backoff
from src.engine import SearchResult


# ── Multi-Provider Helper ─────────────────────────────────────────────────────

def _get_available_providers(is_vision: bool = False) -> list[dict]:
    """
    Returns list of configured API provider configurations in order of priority:
    1. Anthropic API  (claude-3-5-haiku — native SDK, separate system-prompt, base64 image format)
    2. Gemini API     (gemini-2.0-flash via OpenAI-compat endpoint)
    3. OpenRouter API (OpenAI-compat endpoint)
    4. OpenAI API     (native OpenAI endpoint)

    Each provider dict carries a 'provider_type' key ('anthropic' or 'openai') so call
    sites can branch on the different response structures without duplicating logic.
    """
    providers = []

    # ── Anthropic (claude) ────────────────────────────────────────────────────
    if ANTHROPIC_API_KEY:
        try:
            import anthropic as anthropic_sdk
            providers.append({
                "name": "anthropic",
                "client": anthropic_sdk.Anthropic(api_key=ANTHROPIC_API_KEY),
                "model": ANTHROPIC_VLM_MODEL if is_vision else ANTHROPIC_MODEL,
                "provider_type": "anthropic",
            })
        except ImportError:
            print("  [WARN] 'anthropic' package not found. Run: pip install anthropic")

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


def _extract_content(response: Any, provider_type: str) -> str:
    """
    Normalizes response text extraction across provider types.
    - Anthropic: response.content[0].text
    - OpenAI-compat (Gemini, OpenRouter, OpenAI): response.choices[0].message.content
    """
    if provider_type == "anthropic":
        if response.content and len(response.content) > 0:
            return response.content[0].text or ""
        return ""
    else:
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return ""


# ── Stage 6a: Conditional Visual Processing ───────────────────────────────────

def resolve_visual_context(
    retrieved_chunks: list[SearchResult | dict],
    user_query: str = "",
    max_images: int = IMAGE_HEAVY_THRESHOLD,
) -> list[str]:
    """
    Stage 6a: Queries vision model ONLY for retrieved chunks flagged with has_visual_content.
    Enforces hard cap (max_images=3), page-level deduplication, disk caching, and query-conditioned prompts.
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

    providers = _get_available_providers(is_vision=True)
    if not providers:
        print("  [WARN] [Stage 6a] Skipping vision model call: No API key found in environment. Falling back gracefully to text-only.")
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

        # Transcribe with VLM on demand across available providers/models
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

            transcribed = False
            for p in providers:
                p_name = p["name"]
                client = p["client"]
                model = p["model"]
                print(f"  [Stage 6a] Calling VLM ({p_name}/{model}) for {manual_name} page {page_num}…")

                try:
                    p_type = p.get("provider_type", "openai")

                    if p_type == "anthropic":
                        # Anthropic image format uses {type:image, source:{type:base64,...}}
                        def _call_vlm():
                            return client.messages.create(
                                model=model,
                                max_tokens=1500,
                                messages=[{
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/png",
                                                "data": img_b64,
                                            }
                                        }
                                    ]
                                }]
                            )
                    else:
                        # OpenAI-compat: image_url with data URI
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
                        formatted_transcription = f"--- Visual Context from {manual_name} (Page {page_num}) ---\n{transcription}"
                        image_transcriptions.append(formatted_transcription)

                        # Save to disk cache
                        with open(cache_file_path, "w", encoding="utf-8") as f:
                            json.dump({
                                "manual": manual_name,
                                "page_num": page_num,
                                "transcription": formatted_transcription,
                            }, f, indent=2)

                        transcribed = True
                        break

                except Exception as p_err:
                    print(f"  [WARN] Provider {p_name} failed: {p_err}. Trying next option…")

            if not transcribed:
                print(f"  [WARN] [Stage 6a] All VLM options failed for page {page_num}. Falling back gracefully to text-only.")

        except Exception as exc:
            print(f"  [WARN] [Stage 6a] Image reading failed for page {page_num} ({exc}). Falling back gracefully.")

    return image_transcriptions


# ── Stage 6b: LLM SOP Extraction ───────────────────────────────────────────────

def extract_sop(
    query: str,
    retrieved_chunks: list[SearchResult | dict],
    image_context: list[str] | None = None,
    category: str = "general",
) -> dict:
    """
    Stage 6b: Takes retrieved text chunks + optional Stage 6a image context,
    invokes LLM, and returns structured JSON SOP. Uses multi-provider fallback,
    and returns offline fallback if all API quotas are exhausted.
    """
    providers = _get_available_providers(is_vision=False)
    if not providers:
        print("  [WARN] No active LLM API provider keys found. Using offline text extraction fallback.")
        return _generate_offline_fallback_sop(query, retrieved_chunks)

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
        "You are ParseOS, an expert industrial AI intelligence engine.\n"
        "Your job is to read industrial machine manual excerpts (+ optional visual table/diagram transcriptions) "
        "and convert them into a structured, step-by-step Standard Operating Procedure (SOP).\n\n"
        "Output MUST be valid JSON adhering strictly to this JSON schema:\n"
        "{\n"
        '  "procedure_title": "Concise Procedure Title",\n'
        '  "machine_type": "Specific Machine / System Type",\n'
        '  "steps": [\n'
        "    {\n"
        '      "step_number": 1,\n'
        '      "action": "Clear imperative verb (e.g. Shut down, Disconnect, Inspect)",\n'
        '      "object": "Target component or system",\n'
        '      "condition": "Precondition, threshold, or timing rule",\n'
        '      "risk_level": "low" | "medium" | "high",\n'
        '      "required_tool": "Required tool or device (or none)"\n'
        "    }\n"
        "  ],\n"
        '  "safety_warnings": ["Warning statement 1", "Warning statement 2"],\n'
        '  "estimated_duration": "Estimated time (e.g. 15-30 minutes)"\n'
        "}\n\n"
        "Rules:\n"
        "1. Focus accurately on solving the user's specific query.\n"
        "2. Do NOT invent steps not supported by the context.\n"
        "3. Output ONLY raw JSON. Do not write introductory text, explanations, or codeblock fences."
    )

    user_prompt = (
        f"Category: {category}\n"
        f"User Query: {query}\n\n"
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

            if p_type == "anthropic":
                # Anthropic: system prompt is a top-level param, not a message role
                def _call_llm():
                    return client.messages.create(
                        model=model,
                        max_tokens=2000,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
            else:
                # OpenAI-compat: system is a message with role="system"
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
            return sop_data

        except Exception as exc:
            last_error = exc
            print(f"  [WARN] Stage 6b failed with provider {p_name} ({exc}). Trying next option…")

    print(f"  [WARN] All LLM API providers failed or exhausted quota ({last_error}). Generating offline fallback SOP.")
    return _generate_offline_fallback_sop(query, retrieved_chunks)


def _generate_offline_fallback_sop(query: str, retrieved_chunks: list[SearchResult | dict]) -> dict:
    """Generates structured fallback SOP from retrieved text chunks when API quota is exhausted."""
    first_chunk = retrieved_chunks[0] if retrieved_chunks else None
    manual_name = first_chunk.manual_name if isinstance(first_chunk, SearchResult) else "Industrial Manual"
    page_num = first_chunk.page_num if isinstance(first_chunk, SearchResult) else 1

    text_excerpt = first_chunk.chunk_text[:300] if first_chunk else "Refer to manual documentation."

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
                "required_tool": "Standard Maintenance Toolkit"
            },
            {
                "step_number": 2,
                "action": "Execute Procedure",
                "object": f"Follow manual instructions: {text_excerpt[:100]}…",
                "condition": "Ensure system power is safely disconnected",
                "risk_level": "high",
                "required_tool": "Calibrated Measuring Device / Safety Gear"
            }
        ],
        "safety_warnings": [
            "Always verify power supply isolation before beginning work.",
            "API Rate Limit Active: Displaying retrieved text-chunk fallback SOP."
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
