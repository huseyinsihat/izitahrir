from PIL import Image, ImageDraw

from app.data.catalog import Catalog
from app.data.validate import validate_line, validation_report
from app.utils.unicode_text import normalize_ota


def _image(path, size, color):
    Image.new("RGB", size, color).save(path)


def test_accepts_wide_line_and_rejects_bad_records(tmp_path):
    good = tmp_path / "good.png"
    tall = tmp_path / "tall.png"
    blank = tmp_path / "blank.png"
    image = Image.new("RGB", (200, 40), "white")
    ImageDraw.Draw(image).rectangle((5, 10, 190, 28), fill="black")
    image.save(good)
    _image(tall, (40, 120), (10, 10, 10))
    _image(blank, (200, 40), "white")
    catalog = Catalog(tmp_path / "catalog.sqlite")
    lines = [
        _row("ok", str(good), "قرية فلان", "abc"),
        _row("empty", str(good), "  ", "def"),
        _row("tall", str(tall), "قرية", "ghi"),
        _row("blank", str(blank), "قرية", "jkl"),
        _row("missing", str(tmp_path / "nope.png"), "قرية", "mno"),
        _row("dup", str(good), "قرية فلان", "abc"),
    ]
    report = validation_report(lines, catalog)
    rejected = {item["line_id"]: item["issues"] for item in report["rejected_lines"]}
    assert "ok" not in rejected
    assert "empty_transcription" in rejected["empty"]
    assert "crop_not_line_like" in rejected["tall"]
    assert "blank_image" in rejected["blank"]
    assert "missing_image" in rejected["missing"]
    assert "duplicate" in rejected["dup"]
    assert normalize_ota(lines[0]["text_ota"]) == "قرية فلان"


def test_bidi_corruption_is_rejected(tmp_path):
    path = tmp_path / "line.png"
    image = Image.new("RGB", (180, 36), "white")
    ImageDraw.Draw(image).rectangle((4, 8, 170, 24), fill="black")
    image.save(path)
    issues = validate_line(_row("bidi", str(path), "قرية\u202e", "hash"))
    assert "bidi_override" in issues


def _row(line_id, image_path, text, digest):
    return {
        "id": line_id,
        "page_id": "p",
        "image_path": image_path,
        "image_sha256": digest,
        "text_ota": text,
        "script": "siyakat",
        "source": "user_correction",
        "split": "train",
        "verification_status": "reviewed",
        "duplicate_of": None,
    }
