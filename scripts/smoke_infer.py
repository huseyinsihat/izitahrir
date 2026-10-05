"""Load models and run recognition plus segmentation on a synthetic page."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

from app.config import get_settings
from app.htr.pipeline import read_page_image
from app.htr.recognize import recognize_line_image
from app.utils import configure_stdio


def synthetic_page() -> Image.Image:
    image = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(image)
    for top in (80, 180, 280, 380):
        draw.line((80, top, 1100, top), fill="black", width=3)
    return image


def synthetic_line() -> Image.Image:
    image = Image.new("RGB", (640, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 28, 620, 52), fill="black")
    return image


def main() -> None:
    configure_stdio()
    settings = get_settings()
    line = synthetic_line()
    text = recognize_line_image(line, settings)
    print("line_prediction:", text)
    page_lines = read_page_image(synthetic_page(), settings, imagename="synthetic")
    print("page_lines:", len(page_lines))
    for item in page_lines:
        print(f"{item['line_index']}: {item['text']}")
    if not isinstance(text, str):
        raise SystemExit("Satır tanıma metin döndürmedi")
    if len(page_lines) < 1:
        raise SystemExit("Sayfa segmentasyonu satır döndürmedi")
    print("smoke_ok")


if __name__ == "__main__":
    main()
