"""
pdf_parser.py — Stage 1: PDF Text Extraction
Reads every page of an industrial manual PDF and returns clean text.
Supports both text-based and image-based (scanned) PDFs via OCR fallback.

OCR Fallback Strategy
---------------------
For every page, native text extraction (PyMuPDF) is tried first.
If a page yields fewer than OCR_CHAR_THRESHOLD characters, the page is
rendered to a high-resolution image and passed through Tesseract OCR.

Requirements for OCR:
    pip install pytesseract Pillow

    Additionally, the Tesseract-OCR binary must be installed:
    Windows : https://github.com/UB-Mannheim/tesseract/wiki
              Default path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe
    Linux   : sudo apt install tesseract-ocr
    macOS   : brew install tesseract

    Set the TESSERACT_CMD environment variable to override the binary path.
"""

import os
import base64
import io
from typing import Any
import fitz  # PyMuPDF
from openai import OpenAI

from src.config import (
    USE_LAYOUT_PARSER,
    INGEST_VLM_MODEL,
    PAGE_IMAGES_PATH,
    OPENAI_API_KEY,
    OPENROUTER_API_KEY,
    GEMINI_API_KEY,
)

# ── OCR configuration ────────────────────────────────────────────────────────

# Pages with fewer characters than this trigger OCR.
OCR_CHAR_THRESHOLD: int = 30

# DPI used when rendering a page to an image for OCR (higher = better quality).
OCR_RENDER_DPI: int = 300

# Default Tesseract language(s). Use '+' to combine, e.g. "eng+deu".
OCR_DEFAULT_LANG: str = "eng"

# Allow overriding the Tesseract binary via an environment variable.
_TESSERACT_ENV_PATH = os.environ.get("TESSERACT_CMD")
_TESSERACT_DEFAULT_WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ── Public helpers ────────────────────────────────────────────────────────────

def is_tesseract_available() -> bool:
    """
    Returns True if the Tesseract-OCR binary can be found.

    Checks (in order):
      1. TESSERACT_CMD environment variable.
      2. Default Windows install path.
      3. System PATH (Linux / macOS).
    """
    try:
        import pytesseract
        _configure_tesseract(pytesseract)
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


# ── Main extraction functions ─────────────────────────────────────────────────

def _render_page_to_pil(page: fitz.Page) -> Any:
    """Renders a fitz Page to a high-resolution PIL Image."""
    from PIL import Image
    scale = OCR_RENDER_DPI / 72
    mat = fitz.Matrix(scale, scale)
    pixmap = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _cache_page_image(page: fitz.Page, path: str) -> None:
    """Caches page layout image to file path."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        img = _render_page_to_pil(page)
        img.save(path, "PNG")
    except Exception as e:
        print(f"  ⚠ Failed to cache page image to {path}: {e}")


def _vlm_transcribe_page(img, page_num: int, manual_name: str) -> str:
    """Sends page image to VLM (GPT-4o-mini or configured model) for layout-aware transcription."""
    key = GEMINI_API_KEY or OPENROUTER_API_KEY or OPENAI_API_KEY
    if not key:
        print("  ⚠ VLM Skipped: No API key found in environment.")
        return ""

    try:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        if GEMINI_API_KEY:
            client = OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            model = INGEST_VLM_MODEL
            if model in ("gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo"):
                model = "gemini-2.5-flash"
        elif key.startswith("sk-or-v1-"):
            client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
            model = INGEST_VLM_MODEL
            if model in ("gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo"):
                model = f"openai/{model}"
        else:
            client = OpenAI(api_key=key)
            model = INGEST_VLM_MODEL

        prompt = (
            "You are an industrial manual page parser. Analyze the attached page image.\n"
            "Perform layout-aware transcription:\n"
            "1. Transcribe the paragraph text natively and accurately.\n"
            "2. Transcribe any mathematical equations/formulas into LaTeX (e.g. `$m_e = P_e / (c_p \\Delta T)$`). Include a brief textual explanation of what the formula computes if not fully self-explanatory in the surrounding text.\n"
            "3. Convert any tables into clean Markdown tables, and prepend a 1-sentence summary describing the table.\n"
            "4. For any diagrams, engineering drawings, or warnings/symbols, insert a placeholder in brackets describing them in detail, e.g. `[Diagram: 1-2 sentence description of the diagram contents and symbols]`.\n"
            "Output ONLY the transcribed Markdown content. Do not write any preamble, intro, or wrap in markdown block fences."
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )
        content = response.choices[0].message.content
        if content:
            return content.strip()
    except Exception as exc:
        print(f"  ⚠ VLM transcription failed on page {page_num}: {exc}")
    
    return ""


def extract_text_from_pdf(
    pdf_path: str,
    ocr_fallback: bool = True,
    ocr_lang: str = OCR_DEFAULT_LANG,
) -> str:
    """
    Extracts all text from a PDF file as a single flat string.
    """
    pages = extract_text_with_metadata(pdf_path, ocr_fallback=ocr_fallback, ocr_lang=ocr_lang)
    full_text = ""
    for p in pages:
        tag = " [OCR]" if p.get("ocr_used") else (" [VLM]" if p.get("layout_parsed") else "")
        full_text += f"\n--- Page {p['page_num']}{tag} ---\n"
        full_text += p["text"] + "\n"
    return full_text


def extract_text_with_metadata(
    pdf_path: str,
    ocr_fallback: bool = True,
    ocr_lang: str = OCR_DEFAULT_LANG,
) -> list[dict]:
    """
    Extracts text from each page together with per-page metadata.
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
        layout_parsed = False

        page_image_path = os.path.join(manual_image_dir, f"page_{page_num + 1}.png")

        has_images = len(page.get_images()) > 0 or len(page.get_drawings()) > 0
        needs_layout = USE_LAYOUT_PARSER and (has_images or len(text) < OCR_CHAR_THRESHOLD)

        if needs_layout:
            _cache_page_image(page, page_image_path)
            from PIL import Image
            try:
                img = Image.open(page_image_path)
                vlm_text = _vlm_transcribe_page(img, page_num + 1, manual_name)
                if vlm_text:
                    text = vlm_text
                    layout_parsed = True
            except Exception as e:
                print(f"  ⚠ Failed to render/transcribe page {page_num + 1} with VLM: {e}")

        if (not text or len(text) < OCR_CHAR_THRESHOLD) and ocr_fallback and not layout_parsed:
            ocr_text = _ocr_page(page, lang=ocr_lang)
            if ocr_text:
                text = ocr_text
                ocr_used = True

        if USE_LAYOUT_PARSER and not os.path.exists(page_image_path):
            _cache_page_image(page, page_image_path)

        pages.append(
            {
                "page_num":   page_num + 1,
                "text":       text,
                "char_count": len(text),
                "has_text":   bool(text),
                "ocr_used":   ocr_used,
                "layout_parsed": layout_parsed,
                "image_path":  page_image_path if os.path.exists(page_image_path) else "",
            }
        )

    doc.close()
    return pages


def get_page_count(pdf_path: str) -> int:
    """Returns the total number of pages in a PDF."""
    _validate_pdf_path(pdf_path)
    doc = fitz.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


# ── Internal helpers ─────────────────────────────────────────────────────────

def _ocr_page(page: fitz.Page, lang: str = OCR_DEFAULT_LANG) -> str:
    """
    Renders a fitz Page to a high-resolution image and extracts text via OCR.

    Args:
        page: A ``fitz.Page`` object.
        lang: Tesseract language code(s).

    Returns:
        OCR-extracted text string, or an empty string if OCR is unavailable
        or the page yields no text.

    Notes:
        Requires ``pytesseract`` and ``Pillow`` to be installed, plus the
        Tesseract-OCR binary on the system.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        print(
            f"  ⚠  OCR skipped: {exc}. "
            f"Run `pip install pytesseract Pillow` to enable OCR support."
        )
        return ""

    try:
        _configure_tesseract(pytesseract)

        # Render at high DPI: scale = DPI / 72 (PDF base resolution)
        scale = OCR_RENDER_DPI / 72
        mat = fitz.Matrix(scale, scale)
        pixmap = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)

        # Convert pixmap bytes to a PIL Image
        img = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)

        # Run Tesseract
        raw = pytesseract.image_to_string(img, lang=lang)
        return raw.strip()

    except Exception as exc:
        print(f"  ⚠  OCR failed on page: {exc}")
        return ""


def _configure_tesseract(pytesseract_module) -> None:
    """
    Points pytesseract at the Tesseract binary.

    Priority:
      1. TESSERACT_CMD environment variable.
      2. Default Windows install location (if it exists).
      3. Assume it is on PATH (Linux / macOS).
    """
    if _TESSERACT_ENV_PATH:
        pytesseract_module.pytesseract.tesseract_cmd = _TESSERACT_ENV_PATH
    elif os.path.isfile(_TESSERACT_DEFAULT_WIN):
        pytesseract_module.pytesseract.tesseract_cmd = _TESSERACT_DEFAULT_WIN
    # Otherwise, rely on system PATH.


def _validate_pdf_path(pdf_path: str) -> None:
    """Raises descriptive errors for invalid PDF paths."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found: '{pdf_path}'\n"
            f"Make sure the file is placed in data/manuals/"
        )
    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(
            f"Expected a .pdf file, got: '{pdf_path}'"
        )


# ── Quick standalone test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/manuals/weg_motor_manual.pdf"
    print(f"\n[pdf_parser] Testing on: {path}")
    print(f"  Tesseract available: {is_tesseract_available()}")
    print(f"  Page count: {get_page_count(path)}")

    print("\n[Mode 1] extract_text_from_pdf (with OCR fallback):")
    text = extract_text_from_pdf(path, ocr_fallback=True)
    print(f"  Total characters extracted: {len(text):,}")
    print(f"\n--- First 1000 characters ---")
    print(text[:1000])

    print("\n[Mode 2] extract_text_with_metadata (with OCR fallback):")
    pages = extract_text_with_metadata(path, ocr_fallback=True)
    for p in pages[:5]:  # show first 5 pages
        ocr_tag = " [OCR]" if p["ocr_used"] else ""
        print(f"  Page {p['page_num']}{ocr_tag}: {p['char_count']} chars, has_text={p['has_text']}")
