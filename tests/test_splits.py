import pytest

from app.data.catalog import Catalog
from app.data.splits import apply_split_rules, find_conflicts, resolve_split


def test_rule_conflict():
    rules = {
        "train_pages": ["TT075:0"],
        "validation_pages": ["TT075:0"],
        "test_pages": [],
        "external_test_documents": [],
    }
    assert find_conflicts(rules)


def test_page_rules_and_external_defter(tmp_path):
    catalog = Catalog(tmp_path / "catalog.sqlite")
    catalog.create_document("d1", "TT075.png", "TT075", "2026-01-01T00:00:00+00:00")
    catalog.create_document("d2", "TT080.png", "TT080", "2026-01-01T00:00:00+00:00")
    pages = [
        ("p0", "d1", 0),
        ("p1", "d1", 1),
        ("p2", "d1", 2),
        ("p3", "d2", 0),
    ]
    for page_id, doc_id, index in pages:
        catalog.insert_page(
            {
                "id": page_id,
                "document_id": doc_id,
                "page_index": index,
                "image_path": str(tmp_path / f"{page_id}.png"),
                "width": 10,
                "height": 10,
                "split": "unassigned",
            }
        )
    rules = {
        "train_pages": ["TT075:0"],
        "validation_pages": ["TT075:1"],
        "test_pages": ["TT075:2"],
        "external_test_documents": ["TT080"],
    }
    assigned = apply_split_rules(catalog, rules)
    assert assigned == {
        "p0": "train",
        "p1": "validation",
        "p2": "test",
        "p3": "external_test",
    }
    catalog.insert_auxiliary_line(_line("l0", "d1", "p0", "train"))
    catalog.insert_auxiliary_line(_line("l1", "d1", "p0", "train"))
    catalog.set_page_split("p0", "train")
    splits = {line["id"]: line["split"] for line in catalog.list_lines(page_id="p0")}
    assert splits == {"l0": "train", "l1": "train"}


def test_same_page_cannot_resolve_twice():
    page = {"id": "p", "document_id": "d", "page_index": 0}
    document = {"stem": "TT075"}
    rules = {
        "train_pages": ["TT075"],
        "validation_pages": ["TT075:0"],
        "test_pages": [],
    }
    with pytest.raises(ValueError):
        resolve_split(page, document, rules)


def _line(line_id, doc_id, page_id, split):
    return {
        "id": line_id,
        "document_id": doc_id,
        "page_id": page_id,
        "line_index": 0 if line_id == "l0" else 1,
        "image_path": "x.png",
        "image_sha256": None,
        "text_pred": None,
        "text_ota": "قرية",
        "script": "siyakat",
        "source": "user_correction",
        "split": split,
        "verification_status": "reviewed",
        "baseline": [[1, 1], [2, 1]],
        "boundary": [[0, 0], [3, 0], [3, 2], [0, 2], [0, 0]],
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
