"""
config.py — ParseOS SOP Engine
Central configuration: all constants loaded from environment / .env file.
Import this module anywhere you need a setting.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Locate and load .env from the project root ─────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── Paths ───────────────────────────────────────────────────────────────────
CHROMA_PATH     = str(_PROJECT_ROOT / "chroma_storage")
KNOWLEDGE_PATH  = str(_PROJECT_ROOT / "knowledge_layer" / "sops")
MANUALS_PATH    = str(_PROJECT_ROOT / "data" / "manuals")
PAGE_IMAGES_PATH = str(_PROJECT_ROOT / "data" / "page_images")
VLM_CACHE_PATH   = str(_PROJECT_ROOT / "data" / "vlm_cache")

# ── Embedding model ─────────────────────────────────────────────────────────
EMBED_MODEL     = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# ── LLM ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o")
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
INGEST_VLM_MODEL = os.getenv("INGEST_VLM_MODEL", "gpt-4o-mini")
USE_LAYOUT_PARSER = os.getenv("USE_LAYOUT_PARSER", "True").lower() in ("true", "1", "yes")

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Search ───────────────────────────────────────────────────────────────────
TOP_K_RESULTS   = int(os.getenv("TOP_K_RESULTS", "3"))

# ── Vision Cap ───────────────────────────────────────────────────────────────
IMAGE_HEAVY_THRESHOLD = int(os.getenv("IMAGE_HEAVY_THRESHOLD", "3"))

# ── Sub-Topic Coverage Gap Thresholds ────────────────────────────────────────
# Used by decompose_query_topics / score_subtopic_coverage in sop_extractor.py
# to classify each sub-topic's evidence coverage:
#   coverage < NOT_FOUND  → "not_found"
#   NOT_FOUND ≤ coverage < PARTIAL → "partial"
#   coverage ≥ PARTIAL    → "covered"
SUBTOPIC_NOT_FOUND_THRESHOLD = float(os.getenv("SUBTOPIC_NOT_FOUND_THRESHOLD", "0.20"))
SUBTOPIC_PARTIAL_THRESHOLD   = float(os.getenv("SUBTOPIC_PARTIAL_THRESHOLD", "0.50"))

# ── Local Visual Predictor API Endpoint ───────────────────────────────────────
PREDICT_API_URL = os.getenv("PREDICT_API_URL", "http://192.168.0.128/predict")

# ── ChromaDB collection name ─────────────────────────────────────────────────
CHROMA_COLLECTION = "manual_knowledge"

