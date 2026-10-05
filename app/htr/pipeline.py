"""Page image to ordered Ottoman line transcriptions."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.config import Settings
from app.htr.geometry import as_points, crop_polygon
from app.htr.recognize import prediction_text, recognize_line_image, recognize_segmentation
from app.htr.segment import segment_page
from app.utils.turkish import to_turkish


def is_line_image(image: Image.Image) -> bool:
    """True for a pre-cropped text line. Those must not go through page segmentation."""
    width, height = image.size
    if height <= 0 or width < height:
        return False
    if height <= 140:
        return True
    return height <= 320 and width >= height * 2


def _shown(script: str) -> str:
    turkish = to_turkish(script)
    return turkish or script


def read_page_image(
    image: Image.Image,
    settings: Settings,
    imagename: str = "page",
    model_path: Path | None = None,
) -> list[dict]:
    rgb = image.convert("RGB")
    if is_line_image(rgb):
        script = recognize_line_image(rgb, settings, model_path=model_path)
        width, height = rgb.size
        return [
            {
                "line_index": 0,
                "text": _shown(script),
                "text_script": script,
                "baseline": [(0, height // 2), (max(0, width - 1), height // 2)],
                "boundary": [
                    (0, 0),
                    (max(0, width - 1), 0),
                    (max(0, width - 1), max(0, height - 1)),
                    (0, max(0, height - 1)),
                    (0, 0),
                ],
                "crop": rgb,
                "confidence": None,
            }
        ]
    segmentation = segment_page(rgb, settings, imagename=imagename)
    if not getattr(segmentation, "lines", None):
        return []
    records = recognize_segmentation(rgb, segmentation, settings, model_path=model_path)
    lines: list[dict] = []
    for index, record in enumerate(records):
        baseline = as_points(getattr(record, "baseline", None))
        boundary = as_points(getattr(record, "boundary", None))
        extracted = getattr(record, "image", None)
        if isinstance(extracted, Image.Image):
            crop = extracted.convert("RGB")
        else:
            crop = crop_polygon(rgb, boundary, baseline)
        confidences = list(getattr(record, "confidences", []) or [])
        confidence = sum(confidences) / len(confidences) if confidences else None
        script = prediction_text(record)
        lines.append(
            {
                "line_index": index,
                "text": _shown(script),
                "text_script": script,
                "baseline": baseline,
                "boundary": boundary,
                "crop": crop,
                "confidence": confidence,
            }
        )
    return lines
