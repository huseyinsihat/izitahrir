from PIL import Image

from app.preprocessing.pages import load_document


def test_png_jpg_and_pdf(tmp_path):
    image = Image.new("RGB", (80, 30), "white")
    png = tmp_path / "a.png"
    jpg = tmp_path / "a.jpg"
    pdf = tmp_path / "a.pdf"
    image.save(png)
    image.save(jpg, quality=90)
    image.save(pdf, "PDF")
    assert len(load_document(png.read_bytes(), "a.png")) == 1
    assert len(load_document(jpg.read_bytes(), "a.jpg")) == 1
    pages = load_document(pdf.read_bytes(), "a.pdf")
    assert len(pages) == 1
    assert pages[0].mode == "RGB"
    assert pages[0].size[0] > 0
