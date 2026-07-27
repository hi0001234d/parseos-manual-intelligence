"""
pdf_parser.py — Stage 1: PDF Text Extraction & Visual Content Detection (v2.0)
==============================================================================
Reads every page of an industrial manual PDF and returns clean text alongside metadata.
Supports text-based PDFs and image-based (scanned) PDFs via Tesseract OCR fallback.

v2.0 Architectural Change (Deferred-Vision Architecture)
-------------------------------------------------------
- Ingestion is TEXT-ONLY. No VLM API calls occur during ingestion.
- Pages containing tables, diagrams, or formulas are detected and flagged using
  the `looks_like_table_diagram_or_formula` heuristic with a confidence score (0.0 to 1.0).
- High-resolution raster images of pages are rendered and cached in `data/page_images/`
  so they can be passed to the VLM on demand at query time (Stage 6a).
"""

import os
import re
from typing import Any
import fitz  # PyMuPDF

from src.config import PAGE_IMAGES_PATH

# ── OCR configuration ────────────────────────────────────────────────────────
OCR_CHAR_THRESHOLD: int = 30
OCR_RENDER_DPI: int = 300
OCR_DEFAULT_LANG: str = "eng"

_TESSERACT_ENV_PATH = os.environ.get("TESSERACT_CMD")
_TESSERACT_DEFAULT_WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ── Public helpers ────────────────────────────────────────────────────────────

def is_tesseract_available() -> bool:
    """Returns True if the Tesseract-OCR binary can be found."""
    try:
        import pytesseract
        _configure_tesseract(pytesseract)
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def looks_like_table_diagram_or_formula(page: fitz.Page, text: str) -> tuple[bool, float, dict]:
    """
    Analyzes a PyMuPDF page and its extracted text to detect visual elements
    (tables, diagrams, engineering drawings, formulas), while discounting
    decorative header/footer logos, page borders, and line dividers.

    Returns:
        tuple[bool, float, dict]:
            - has_visual_content: bool flag (True if confidence >= 0.40)
            - confidence: float score in [0.0, 1.0]
            - details: dict detailing detected features
    """
    image_count = len(page.get_images())
    drawing_count = len(page.get_drawings())
    char_count = len(text.strip())

    # Text density heuristics
    low_text_density = char_count < 400
    high_text_density = char_count > 800

    # Keyword indicators in text (requiring specific table/fig/spec captions)
    caption_keywords = [
        r"\btable\s+\d+", r"\bfigure\s+\d+", r"\bfig\.\s*\d+", r"\bdiagram\s+\d+",
        r"\bschematic\b", r"\bpinout\b", r"\bwiring diagram\b", r"\bformula\b"
    ]
    caption_matches = [kw for kw in caption_keywords if re.search(kw, text, re.IGNORECASE)]

    general_keywords = [
        r"\btable\b", r"\bfigure\b", r"\bfig\.", r"\bdiagram\b", r"\bequation\b",
        r"\bchart\b", r"\bformula\b", r"\bgraph\b"
    ]
    general_matches = [kw for kw in general_keywords if re.search(kw, text, re.IGNORECASE)]

    # Math/Formula indicators
    math_symbols = [r"=", r"\+", r"\-", r"\*", r"/", r"±", r"Δ", r"Σ", r"∫", r"√", r"≈", r"≤", r"≥", r"∞"]
    math_count = sum(len(re.findall(re.escape(sym), text)) for sym in math_symbols)

    # Markdown or structural table signals (pipes, grid patterns)
    table_formatting = len(re.findall(r"\|", text)) > 2 or len(re.findall(r"\t", text)) > 3

    # Compute heuristic confidence score
    score = 0.0

    if image_count > 0:
        # Require images to be substantive if text density is high
        if low_text_density:
            score += 0.35
        elif not high_text_density:
            score += 0.20
        else:
            score += 0.05  # likely small header/footer logo

    if drawing_count > 10:
        if low_text_density:
            score += 0.30
        else:
            score += 0.10  # page border / divider line

    if caption_matches:
        score += 0.30
    elif general_matches:
        score += 0.15

    if math_count > 6:
        score += 0.25

    if table_formatting:
        score += 0.30

    # Discount dense narrative text pages without explicit captions/tables
    if high_text_density and not (caption_matches or table_formatting or math_count > 6 or (image_count > 0 and low_text_density)):
        score = max(0.0, score - 0.25)

    confidence = min(1.0, round(score, 2))
    has_visual = confidence >= 0.40 or (image_count > 0 and low_text_density)

    details = {
        "image_count": image_count,
        "drawing_count": drawing_count,
        "char_count": char_count,
        "low_text_density": low_text_density,
        "high_text_density": high_text_density,
        "caption_matches": caption_matches,
        "math_count": math_count,
        "table_formatting": table_formatting,
        "confidence": confidence,
    }

    return has_visual, confidence, details


# ── Render & Cache Helpers ───────────────────────────────────────────────────

def _render_page_to_pil(page: fitz.Page) -> Any:
    """Renders a fitz Page to a high-resolution PIL Image."""
    from PIL import Image
    scale = OCR_RENDER_DPI / 72
    mat = fitz.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _cache_page_image(page: fitz.Page, path: str) -> None:
    """Caches page layout image to file path for query-time VLM processing."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            img = _render_page_to_pil(page)
            img.save(path, "PNG")
    except Exception as e:
        print(f"  ⚠ Failed to cache page image to {path}: {e}")


# ── Main Extraction Functions ─────────────────────────────────────────────────

def extract_text_with_metadata(
    pdf_path: str,
    ocr_fallback: bool = True,
    ocr_lang: str = OCR_DEFAULT_LANG,
) -> list[dict]:
    """
    Extracts text from each page together with per-page metadata.
    In Stage 1 v2.0, ingestion is strictly TEXT-ONLY. Pages are evaluated for
    visual content and pre-rendered to PNG files for query-time processing.
    """
    _validate_pdf_path(pdf_path)

    doc = fitz.open(pdf_path)
    pages = []
    
    manual_name = os.path.splitext(os.path.basename(pdf_path))[0]
    manual_image_dir = os.path.join(PAGE_IMAGES_PATH, manual_name)
    os.makedirs(manual_image_dir, exist_ok=True)

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        ocr_used = False

        # OCR fallback for scanned pages
        if (not text or len(text) < OCR_CHAR_THRESHOLD) and ocr_fallback:
            ocr_text = _ocr_page(page, lang=ocr_lang)
            if ocr_text:
                text = ocr_text
                ocr_used = True

        # Heuristic visual check
        has_visual, confidence, details = looks_like_table_diagram_or_formula(page, text)

        # Pre-render page image if visual content is detected or OCR was used
        page_image_path = os.path.join(manual_image_dir, f"page_{page_num + 1}.png")
        if has_visual or ocr_used or len(text) < OCR_CHAR_THRESHOLD:
            _cache_page_image(page, page_image_path)

        pages.append(
            {
                "page_num": page_num + 1,
                "text": text,
                "char_count": len(text),
                "has_text": bool(text),
                "ocr_used": ocr_used,
                "has_visual_content": has_visual,
                "visual_confidence": confidence,
                "visual_details": details,
                "image_path": page_image_path if os.path.exists(page_image_path) else "",
            }
        )

    doc.close()
    return pages


def extract_text_from_pdf(
    pdf_path: str,
    ocr_fallback: bool = True,
    ocr_lang: str = OCR_DEFAULT_LANG,
) -> str:
    """Extracts all text from a PDF file as a single flat string."""
    pages = extract_text_with_metadata(pdf_path, ocr_fallback=ocr_fallback, ocr_lang=ocr_lang)
    full_text = ""
    for p in pages:
        tag = " [OCR]" if p.get("ocr_used") else (" [VISUAL]" if p.get("has_visual_content") else "")
        full_text += f"\n--- Page {p['page_num']}{tag} ---\n"
        full_text += p["text"] + "\n"
    return full_text


def get_page_count(pdf_path: str) -> int:
    """Returns total number of pages in a PDF."""
    _validate_pdf_path(pdf_path)
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


# ── Internal Helpers ─────────────────────────────────────────────────────────

def _ocr_page(page: fitz.Page, lang: str = OCR_DEFAULT_LANG) -> str:
    """Renders a fitz Page to a high-resolution image and extracts text via Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        return ""

    try:
        _configure_tesseract(pytesseract)
        scale = OCR_RENDER_DPI / 72
        mat = fitz.Matrix(scale, scale)
        pixmap = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        raw = pytesseract.image_to_string(img, lang=lang)
        return raw.strip()
    except Exception as exc:
        print(f"  ⚠ OCR failed on page: {exc}")
        return ""


def _configure_tesseract(pytesseract_module) -> None:
    if _TESSERACT_ENV_PATH:
        pytesseract_module.pytesseract.tesseract_cmd = _TESSERACT_ENV_PATH
    elif os.path.isfile(_TESSERACT_DEFAULT_WIN):
        pytesseract_module.pytesseract.tesseract_cmd = _TESSERACT_DEFAULT_WIN


def _validate_pdf_path(pdf_path: str) -> None:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: '{pdf_path}'")
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"Expected a .pdf file, got: '{pdf_path}'")
