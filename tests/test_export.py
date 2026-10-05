from app.htr.export import document_to_json, lines_to_txt, page_to_pagexml


def test_txt_and_json_keep_logical_order():
    pages = [{"id": "p", "page_index": 0, "image_path": "a.png", "split": "train", "width": 100, "height": 40}]
    lines = [
        {
            "id": "l",
            "page_id": "p",
            "line_index": 0,
            "page_index": 0,
            "image_path": "l.png",
            "text_ota": "قرية",
            "text_pred": "قريه",
            "script": "siyakat",
            "source": "user_correction",
            "split": "train",
            "verification_status": "reviewed",
            "baseline": [[90, 30], [10, 30]],
            "boundary": [[0, 0], [100, 0], [100, 40], [0, 40], [0, 0]],
        }
    ]
    assert lines_to_txt(lines).strip() == "قرية"
    body = document_to_json({"id": "d", "filename": "TT075.png"}, pages, lines)
    assert "قرية" in body
    xml = page_to_pagexml(pages[0], lines, "0000.png")
    assert "قرية" in xml
    assert 'readingDirection="right-to-left"' in xml
    assert "90,30 10,30" in xml
