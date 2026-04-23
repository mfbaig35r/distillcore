"""Vision-based OCR using OpenAI GPT-4o-mini for scanned PDF pages."""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

from .client import get_client

logger = logging.getLogger(__name__)

OCR_PROMPT = (
    "Transcribe ALL text from this scanned document page exactly as printed. "
    "Preserve paragraph breaks, line breaks, and formatting structure. "
    "Do not summarize, interpret, paraphrase, or correct any text. "
    "If a word is illegible, write [illegible]. "
    "Return only the transcribed text, nothing else."
)


def ocr_page(image: "Image.Image", api_key: str = "") -> str:  # noqa: F821
    """OCR a single page image using GPT-4o-mini vision."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    client = get_client(api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            }
        ],
        max_tokens=4096,
        temperature=0,
    )

    return response.choices[0].message.content or ""


def ocr_pdf_pages(
    pdf_path: str | Path,
    page_numbers: list[int],
    dpi: int = 200,
    api_key: str = "",
) -> dict[int, str]:
    """OCR specific pages of a PDF. Returns {page_number: text}.

    page_numbers are 1-indexed to match pdfplumber convention.
    """
    from pdf2image import convert_from_path

    pdf_path = Path(pdf_path)
    results: dict[int, str] = {}

    for page_num in page_numbers:
        logger.info(f"OCR page {page_num} of {pdf_path.name}")
        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            first_page=page_num,
            last_page=page_num,
        )
        if images:
            text = ocr_page(images[0], api_key=api_key)
            results[page_num] = text

    return results
