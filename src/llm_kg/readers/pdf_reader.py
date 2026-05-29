from __future__ import annotations

import base64
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from llm_kg.config import Settings
from llm_kg.models import Document
from llm_kg.readers.common import build_document

PageMode = Literal["native_text", "timeout_placeholder", "failed_placeholder", "ocr_text"]


@dataclass
class PageExtraction:
    page_number: int
    text: str
    mode: PageMode
    error: str | None = None
    ocr_provider: str | None = None


def read_pdf(path: Path, settings: Settings | None = None) -> Document:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Reading .pdf requires the pypdf package.") from exc

    settings = settings or Settings.from_env()
    reader = PdfReader(str(path), strict=False)
    page_count = len(reader.pages)
    timeout = float(os.getenv("LLM_KG_PDF_PAGE_TIMEOUT_SECONDS", "2"))

    if page_count > 20:
        pages = [_extract_page_text_with_timeout(path, index, timeout) for index in range(page_count)]
    else:
        pages = [_extract_page_text_inline(reader, index) for index in range(page_count)]

    warnings: list[str] = []
    pages = _apply_ocr_fallback(path, pages, settings, warnings)
    content = "\n\n".join(_format_page(page) for page in pages if page.text.strip())
    metadata = _coverage_metadata(pages, warnings)
    if not _has_real_text(pages):
        raise ValueError(f"Source is empty: PDF text extraction failed for all pages in {path}")
    return build_document(path, "pdf", content, metadata=metadata)


def _extract_page_text_inline(reader: Any, page_index: int) -> PageExtraction:
    try:
        text = reader.pages[page_index].extract_text() or ""
        return PageExtraction(page_index + 1, text, "native_text")
    except Exception as exc:
        return PageExtraction(
            page_number=page_index + 1,
            text=f"[Page {page_index + 1} text extraction failed: {exc}]",
            mode="failed_placeholder",
            error=str(exc),
        )


def _extract_page_text_with_timeout(path: Path, page_index: int, timeout_seconds: float) -> PageExtraction:
    queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=1)
    process = multiprocessing.Process(target=_extract_page_text_worker, args=(str(path), page_index, queue))
    process.start()
    process.join(timeout_seconds)
    page_number = page_index + 1
    if process.is_alive():
        process.terminate()
        process.join()
        message = f"Page {page_number} text extraction timed out after {timeout_seconds:g}s"
        return PageExtraction(page_number, f"[{message}]", "timeout_placeholder", error=message)
    if process.exitcode != 0 or queue.empty():
        message = f"Page {page_number} text extraction failed"
        return PageExtraction(page_number, f"[{message}]", "failed_placeholder", error=message)
    ok, payload = queue.get()
    if ok:
        return PageExtraction(page_number, payload, "native_text")
    message = f"Page {page_number} text extraction failed: {payload}"
    return PageExtraction(page_number, f"[{message}]", "failed_placeholder", error=str(payload))


def _extract_page_text_worker(path: str, page_index: int, queue: multiprocessing.Queue) -> None:
    try:
        from pypdf import PdfReader

        reader = PdfReader(path, strict=False)
        queue.put((True, reader.pages[page_index].extract_text() or ""))
    except Exception as exc:
        queue.put((False, str(exc)))


def _apply_ocr_fallback(
    path: Path,
    pages: list[PageExtraction],
    settings: Settings,
    warnings: list[str],
) -> list[PageExtraction]:
    candidates = [page for page in pages if page.mode in {"timeout_placeholder", "failed_placeholder"}]
    if not candidates or settings.ocr_provider == "none":
        return pages
    if settings.ocr_provider != "openai":
        warnings.append(f"Unsupported OCR provider: {settings.ocr_provider}")
        return pages
    if not os.getenv("OPENAI_API_KEY"):
        warnings.append("OpenAI OCR requested but OPENAI_API_KEY is not set.")
        return pages

    ocr_client = _OpenAIVisionOCR(model=settings.ocr_model or settings.openai_model, timeout=settings.ocr_timeout_seconds)
    remaining = max(settings.ocr_max_pages, 0)
    updated: list[PageExtraction] = []
    for page in pages:
        if page.mode not in {"timeout_placeholder", "failed_placeholder"} or remaining <= 0:
            updated.append(page)
            continue
        try:
            image_bytes = _render_pdf_page(path, page.page_number)
            text = ocr_client.extract_text(image_bytes, page.page_number)
            if text.strip():
                updated.append(
                    PageExtraction(
                        page_number=page.page_number,
                        text=text.strip(),
                        mode="ocr_text",
                        error=page.error,
                        ocr_provider="openai",
                    )
                )
            else:
                warnings.append(f"OpenAI OCR returned no text for page {page.page_number}.")
                updated.append(page)
        except Exception as exc:
            warnings.append(f"OpenAI OCR failed for page {page.page_number}: {exc}")
            updated.append(page)
        remaining -= 1
    skipped = len(candidates) - min(len(candidates), settings.ocr_max_pages)
    if skipped > 0:
        warnings.append(f"OCR max page cap skipped {skipped} timeout/failed pages.")
    return updated


class _OpenAIVisionOCR:
    def __init__(self, model: str, timeout: int) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.timeout = timeout

    def extract_text(self, image_bytes: bytes, page_number: int) -> str:
        image_url = f"data:image/png;base64,{base64.b64encode(image_bytes).decode('ascii')}"
        prompt = (
            "Extract all visible text from this planning PDF page. Preserve useful labels, addresses, "
            "APNs, zoning references, sheet titles, dimensions, and notes. Return plain text only. "
            "Do not infer text that is not visible."
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"{prompt}\n\nPDF page number: {page_number}"},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            timeout=self.timeout,
        )
        return response.output_text


def _render_pdf_page(path: Path, page_number: int) -> bytes:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("OpenAI Vision OCR fallback requires pymupdf to render PDF pages.") from exc

    with fitz.open(path) as doc:
        page = doc.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        return pixmap.tobytes("png")


def _format_page(page: PageExtraction) -> str:
    provider = f" | provider={page.ocr_provider}" if page.ocr_provider else ""
    return f"[Page {page.page_number} | {page.mode}{provider}]\n{page.text.strip()}"


def _has_real_text(pages: list[PageExtraction]) -> bool:
    return any(page.mode in {"native_text", "ocr_text"} and page.text.strip() for page in pages)


def _coverage_metadata(pages: list[PageExtraction], warnings: list[str]) -> dict[str, Any]:
    counts = {
        "total_pages": len(pages),
        "native_pages": sum(1 for page in pages if page.mode == "native_text" and page.text.strip()),
        "ocr_pages": sum(1 for page in pages if page.mode == "ocr_text" and page.text.strip()),
        "timeout_pages": sum(1 for page in pages if page.mode == "timeout_placeholder"),
        "failed_pages": sum(1 for page in pages if page.mode == "failed_placeholder"),
        "text_pages": sum(1 for page in pages if page.mode in {"native_text", "ocr_text"} and page.text.strip()),
    }
    return {
        "pdf_page_coverage": counts,
        "pdf_page_status": [
            {
                "page_number": page.page_number,
                "mode": page.mode,
                "ocr_provider": page.ocr_provider,
                "error": page.error,
                "char_count": len(page.text),
            }
            for page in pages
        ],
        "warnings": warnings,
    }
