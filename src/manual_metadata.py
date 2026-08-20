"""
manual_metadata.py — Dynamic Per-Manual Vocabulary Extraction & Persistence
===========================================================================
Implements the v3.0 adaptive vocabulary architecture for ParseOS.

Architectural role:
  - NEW MANUAL ingested → extract_technical_terms() → save_manual_metadata()
  - QUERY time         → get_combined_vocab() loads vocab for retrieved manuals
  - VALIDATOR receives  → manual_vocab (coverage signal, NOT hard-fail)

Two-tier vocabulary contract:
  UNIVERSAL_CRITICAL (sop_extractor.py) → hard-fail if missing from evidence
  manual_vocab       (this module)       → coverage / evidence scoring signal only

Term extraction uses four complementary strategies to handle the breadth of
industrial nomenclature:
  1. Adaptive-frequency regular words   (threshold scales with document size)
  2. Uppercase abbreviation detection   (MRES, PLC, CRN, CPU, STOP, RUN …)
  3. Model-code pattern detection       (S7-1200, IRB-120, UR5e, GA37 …)
  4. Hyphenated compound-term detection (tool-changer, servo-drive …)
"""

from __future__ import annotations

import json
import typing
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ── Storage directory ─────────────────────────────────────────────────────────
MANUAL_METADATA_DIR: Path = (
    Path(__file__).resolve().parent.parent
    / "knowledge_layer"
    / "manual_metadata"
)

# ── Extended English stop-words (industrial context) ─────────────────────────
# Words that carry no domain-specific signal value and must be excluded before
# frequency counting so they don't inflate or pollute the vocabulary.
EXTRACTION_STOP_WORDS: frozenset[str] = frozenset({
    # articles / prepositions / conjunctions
    "the", "and", "for", "with", "from", "into", "after", "before", "using",
    "use", "page", "this", "that", "these", "those", "will", "shall", "must",
    "also", "only", "note", "see", "all", "not", "are", "may", "can", "has",
    "have", "been", "was", "were", "is", "be", "do", "does", "did", "but",
    "if", "or", "as", "at", "to", "in", "of", "on", "by", "an", "a",
    # question / relative words
    "when", "where", "which", "who", "how", "what", "any", "each", "more",
    "such", "than", "then", "its", "it", "they", "their", "there", "here",
    "both", "between", "through", "during", "while",
    # generic verbs / actions that appear everywhere in manuals
    "ensure", "set", "make", "take", "get", "keep", "allow", "check",
    "remove", "install", "connect", "disconnect", "replace", "press",
    "start", "stop", "run", "open", "close", "turn", "enable", "disable",
    # structural / navigational words
    "shown", "figure", "table", "section", "chapter", "refer", "following",
    "described", "above", "below", "right", "left", "result", "required",
    "according", "possible", "properly", "manually", "automatically",
    "always", "never", "minimum", "maximum", "length", "width", "height",
    "number", "version", "type", "example", "however", "without", "within",
    "should", "other", "used", "being", "provide", "provided", "include",
    "included", "based", "given", "shown", "item", "items", "must", "may",
    "can", "would", "could",
    # extremely common English words that slip past length filters
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "first", "last", "next", "back", "into", "onto", "from", "over",
    "under", "again", "then", "just", "also", "well", "very", "only",
    "each", "both", "some", "most", "more", "same", "own", "out", "new",
    "off", "now", "not", "but", "too", "way", "per", "via", "etc",
    "fig", "rev", "doc", "ref", "ser", "std", "max", "min",
})

# Abbreviations that look like meaningful caps but are common English
_COMMON_CAPS_EXCLUSIONS: frozenset[str] = frozenset({
    "I", "A", "THE", "AND", "FOR", "OR", "IS", "IN", "OF", "AT",
    "ON", "BY", "DO", "BE", "TO", "IF", "AS", "AN", "IT", "WE",
    "US", "NO", "SO", "UP", "OK", "EG", "IE", "VS", "RE", "etc",
})


# ── Adaptive frequency threshold ──────────────────────────────────────────────

def _adaptive_min_freq(word_count: int) -> int:
    """
    Returns the minimum term frequency required for inclusion in the vocabulary.
    Scales with document length so rare-but-important terms in short manuals
    are not silently discarded.

    Bands:
      < 5 000 words  → 2   (short manual, e.g. 5-15 pages)
      < 20 000 words → 3   (medium manual, e.g. 30-80 pages)
      ≥ 20 000 words → 5   (large manual, e.g. 100+ pages)
    """
    if word_count < 5_000:
        return 2
    if word_count < 20_000:
        return 3
    return 5


# ── Technical term extraction ─────────────────────────────────────────────────

def extract_technical_terms(text: str) -> set[str]:
    """
    Extracts a domain-specific technical vocabulary from raw manual text.

    Uses four complementary extraction strategies:

    Strategy 1 — Adaptive-frequency regular words
        Lowercase alphabetic tokens of length ≥ 3, filtered through an
        extended stop-word list and frequency-thresholded by document size.
        Catches: spindle, coolant, impeller, airend, aftercooler …

    Strategy 2 — Uppercase abbreviations
        Tokens in ALL-CAPS of length 2-6 that are not common English words.
        Catches: MRES, CPU, PLC, CRN, STOP, RUN, ACK, BLKF …

    Strategy 3 — Model-number / alphanumeric codes
        Tokens that start with 1-4 uppercase letters followed by a digit
        (and optionally more alphanum/hyphen characters).
        Catches: S7-1200, IRB-120, UR5e, GA37, EZ-E, CR-N …

    Strategy 4 — Hyphenated compound terms
        Lowercase hyphen-joined words of total length ≥ 6.
        Catches: servo-drive, tool-changer, anti-clockwise, safety-relay …

    Returns a set[str] of candidate technical terms (all lowercase).
    """
    # ── Strategy 1: Frequency-ranked regular words ───────────────────────────
    words = re.findall(r'\b[a-zA-Z][a-zA-Z]{2,}\b', text.lower())
    words = [w for w in words if w not in EXTRACTION_STOP_WORDS]
    freq = Counter(words)
    total_word_count = len(words)
    min_freq = _adaptive_min_freq(total_word_count)
    freq_terms: set[str] = {w for w, c in freq.items() if c >= min_freq}

    # ── Strategy 2: Uppercase abbreviations ──────────────────────────────────
    raw_abbrevs = re.findall(r'\b[A-Z]{2,6}\b', text)
    abbrevs: set[str] = {
        a.lower()
        for a in raw_abbrevs
        if a not in _COMMON_CAPS_EXCLUSIONS
    }

    # ── Strategy 3: Model-number / alphanumeric codes ────────────────────────
    model_codes: set[str] = set()
    for token in re.findall(r'\b[A-Z]{1,4}[\d][\w\-]{0,10}\b', text):
        if len(token) >= 3:
            model_codes.add(token.lower())

    # ── Strategy 4: Hyphenated compound terms ────────────────────────────────
    compound_terms: set[str] = {
        c
        for c in re.findall(r'\b[a-z]+-[a-z]+\b', text.lower())
        if len(c) >= 6  # exclude very short ones like "re-do", "non-stop"
    }

    # ── Union, then basic sanity filtering ───────────────────────────────────
    all_terms = freq_terms | abbrevs | model_codes | compound_terms
    # Keep only tokens that look like valid identifiers
    all_terms = {
        t for t in all_terms
        if re.match(r'^[a-zA-Z0-9][\w\-]*$', t) and len(t) >= 3
        and not t.isdigit()
    }

    return all_terms


# ── Domain auto-detection (mirrors sop_extractor.py for consistency) ──────────

def detect_domain(manual_name: str) -> str:
    """Infers the industrial domain from the manual name string."""
    n = manual_name.lower()
    if any(k in n for k in ("abb", "irb", "ur5", "universal robot", "robot")):
        return "robotics"
    if any(k in n for k in ("haas", "mill", "cnc", "lathe")):
        return "cnc_milling"
    if any(k in n for k in ("pump", "grundfos", "centrifugal", "crn")):
        return "pumps"
    if any(k in n for k in ("motor", "teco", "westinghouse", "weg")):
        return "motors"
    if any(k in n for k in ("valve", "fisher")):
        return "valves"
    if any(k in n for k in ("plc", "siemens", "s7")):
        return "plc"
    return "general"


# ── Persistence ───────────────────────────────────────────────────────────────

def save_manual_metadata(
    manual_name: str,
    terms: set[str],
    domain: str | None = None,
) -> Path:
    """
    Persists the extracted vocabulary to
    ``knowledge_layer/manual_metadata/<manual_name>.json``.

    JSON schema::

        {
          "manual":          "haas_mill_operator_manual",
          "domain":          "cnc_milling",
          "ingested_at":     "2026-08-07T12:00:00",
          "technical_terms": ["axis", "chuck", "coolant", "offset", "spindle"],
          "term_count":      5
        }

    Returns the path to the written file.
    """
    MANUAL_METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Make filename filesystem-safe
    safe_name = re.sub(r'[^\w\-]', '_', manual_name)
    file_path = MANUAL_METADATA_DIR / f"{safe_name}.json"

    payload = {
        "manual":          manual_name,
        "domain":          domain or detect_domain(manual_name),
        "ingested_at":     datetime.now(timezone.utc).isoformat(),
        "technical_terms": sorted(terms),
        "term_count":      len(terms),
    }

    with open(file_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    return file_path


# ── Loading & query-time combination ─────────────────────────────────────────

def load_manual_vocab(manual_name: str) -> set[str]:
    """
    Loads the stored dynamic vocabulary for ``manual_name``.

    Returns an empty set if no metadata file exists — callers are safe to
    use this as a fallback without raising exceptions.
    """
    safe_name = re.sub(r'[^\w\-]', '_', manual_name)
    file_path = MANUAL_METADATA_DIR / f"{safe_name}.json"

    if not file_path.exists():
        return set()

    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return set(data.get("technical_terms", []))
    except Exception:
        return set()


def get_combined_vocab(retrieved_chunks: typing.Sequence[typing.Any]) -> set[str]:
    """
    Loads and unions the dynamic technical vocabulary for **all** manuals
    referenced in ``retrieved_chunks``.

    Accepts both ``SearchResult`` objects (with ``.manual_name``) and plain
    ``dict`` objects (with ``"manual"`` or ``"manual_name"`` key).

    Returns an empty set when no metadata files are found — the validator
    must degrade gracefully to ``UNIVERSAL_CRITICAL``-only mode in that case.
    """
    manual_names: set[str] = set()
    for chunk in retrieved_chunks:
        if hasattr(chunk, "manual_name"):
            manual_names.add(chunk.manual_name)
        elif isinstance(chunk, dict):
            name = chunk.get("manual") or chunk.get("manual_name", "")
            if name:
                manual_names.add(name)

    combined: set[str] = set()
    for name in manual_names:
        vocab = load_manual_vocab(name)
        if vocab:
            print(f"  [Vocab] Loaded {len(vocab)} dynamic terms for '{name}'")
        else:
            print(f"  [Vocab] No metadata found for '{name}' — using universal terms only")
        combined |= vocab

    return combined


def load_all_known_vocab() -> set[str]:
    """
    Loads and unions all dynamic technical terms across ALL ingested manual metadata files
    stored in ``knowledge_layer/manual_metadata/``.
    """
    if not MANUAL_METADATA_DIR.exists():
        return set()

    combined: set[str] = set()
    for json_file in MANUAL_METADATA_DIR.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            combined.update(data.get("technical_terms", []))
        except Exception:
            pass

    return combined

