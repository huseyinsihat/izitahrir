"""End-to-end read, correction, and export when the models are present."""

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.config import get_settings, reset_settings
from app.service import HTRService

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "muharaf_rec_best.mlmodel"
SEGMENTER = ROOT / "models" / "blla.mlmodel"

pytestmark = pytest.mark.skipif(
    not (MODEL.is_file() and SEGMENTER.is_file()),
    reason="recognition or segmentation model is not downloaded",
)


def test_read_correct_and_export(tmp_path, monkeypatch):
    config = (ROOT / "config.yaml").read_text(encoding="utf-8")
    recognition = MODEL.as_posix()
    segmenter = SEGMENTER.as_posix()
    config = config.replace(
        "modeller/kraken/kraken-ppocrv6-medium/medium.safetensors",
        recognition,
        1,
    )
    config = config.replace("models/blla.mlmodel", segmenter, 1)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config, encoding="utf-8")
    monkeypatch.setenv("TARIHHTR_ROOT", str(tmp_path))
    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    reset_settings()

    page = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(page)
    for top in (80, 180, 280, 380):
        draw.line((80, top, 1100, top), fill="black", width=3)
    image_path = tmp_path / "TT075.png"
    page.save(image_path)

    service = HTRService(get_settings())
    payload = service.ingest(image_path.name, image_path.read_bytes())
    payload = service.read_document(payload["document"]["id"])
    assert payload["lines"], "segmentasyon satır döndürmedi"
    line = payload["lines"][0]
    saved = service.save_ground_truth(line["id"], "قرية", "reviewed")
    assert saved["text_ota"] == "قرية"
    assert saved["verification_status"] == "reviewed"
    assert saved["source"] == "user_correction"
    name, body, _media = service.export(payload["document"]["id"], "pagexml")
    assert name.endswith(".xml")
    assert "قرية".encode("utf-8") in body
    text_name, text_body, _media = service.export(payload["document"]["id"], "txt")
    assert text_body.decode("utf-8").strip() == "قرية"
    assert text_name.endswith(".txt")
