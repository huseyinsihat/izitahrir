"""Cropped Ottoman lines are read as one line and shown in Turkish."""

import time
from pathlib import Path

import pytest
from PIL import Image

from app.config import get_settings, reset_settings
from app.htr.pipeline import read_page_image

ROOT = Path(__file__).resolve().parents[1]
RECOGNIZER = ROOT / "modeller" / "kraken" / "kraken-ppocrv6-medium" / "medium.safetensors"

pytestmark = pytest.mark.skipif(not RECOGNIZER.is_file(), reason="PP-OCRv6 medium is not on disk")


def test_cropped_lines_return_turkish_quickly(isolated_home):
    config_path = isolated_home / "config.yaml"
    config = config_path.read_text(encoding="utf-8")
    config = config.replace(
        "modeller/kraken/kraken-ppocrv6-medium/medium.safetensors",
        RECOGNIZER.as_posix(),
    )
    config_path.write_text(config, encoding="utf-8")
    reset_settings()
    settings = get_settings()

    started = time.perf_counter()
    with Image.open(ROOT / "modeller" / "testveri" / "osmanlica-nesih" / "01.png") as image:
        nesih = read_page_image(image.convert("RGB"), settings)
    assert len(nesih) == 1
    assert nesih[0]["crop"].size == (1014, 59)
    turkish = nesih[0]["text"].lower()
    assert "bir" in turkish
    assert "kelimenin" in turkish
    assert not any("\u0600" <= char <= "\u06FF" for char in nesih[0]["text"])
    assert nesih[0]["text_script"]

    with Image.open(ROOT / "modeller" / "testveri" / "osmanlica-elyazisi" / "01.png") as image:
        handwriting = read_page_image(image.convert("RGB"), settings)
    assert len(handwriting) == 1
    assert "kahve" in handwriting[0]["text"].lower()
    assert "iskemleleri" in handwriting[0]["text"].lower()
    assert time.perf_counter() - started < 60
