"""Turn JPG, PNG, and PDF uploads into page images."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

SUPPORTED = {".jpg", ".jpeg", ".png", ".pdf"}
PDF_SCALE = 300 / 72


class IngestError(ValueError):
    pass


def load_document(content: bytes, filename: str) -> list[Image.Image]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise IngestError(f"Desteklenmeyen dosya türü: {suffix or filename}")
    if suffix == ".pdf":
        return _load_pdf(content)
    return [_load_raster(content)]


def _load_raster(content: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(content)) as image:
            image = ImageOps.exif_transpose(image)
            return image.convert("RGB")
    except Exception as exc:
        raise IngestError(f"Görüntü açılamadı: {exc}") from exc


def _load_pdf(content: bytes) -> list[Image.Image]:
    import pypdfium2 as pdfium

    pages: list[Image.Image] = []
    try:
        document = pdfium.PdfDocument(content)
    except Exception as exc:
        raise IngestError(f"PDF açılamadı: {exc}") from exc
    try:
        if len(document) == 0:
            raise IngestError("PDF sayfa içermiyor")
        for page in document:
            bitmap = page.render(scale=PDF_SCALE)
            pages.append(bitmap.to_pil().convert("RGB"))
    finally:
        document.close()
    return pages
