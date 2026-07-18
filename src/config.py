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
KNOWLEDGE_PATH  = str(_PROJECT_ROOT / "knowledge_layer")
MANUALS_PATH    = str(_PROJECT_ROOT / "data" / "manuals")
PAGE_IMAGES_PATH = str(_PROJECT_ROOT / "data" / "page_images")

# ── Embedding model ─────────────────────────────────────────────────────────
EMBED_MODEL     = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# ── LLM ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o")
INGEST_VLM_MODEL = os.getenv("INGEST_VLM_MODEL", "gpt-4o-mini")
USE_LAYOUT_PARSER = os.getenv("USE_LAYOUT_PARSER", "True").lower() in ("true", "1", "yes")

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Search ───────────────────────────────────────────────────────────────────
TOP_K_RESULTS   = int(os.getenv("TOP_K_RESULTS", "3"))

# ── ChromaDB collection name ─────────────────────────────────────────────────
CHROMA_COLLECTION = "manual_knowledge"
