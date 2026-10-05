from PIL import Image
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.config import get_settings
from app.data.catalog import Catalog


def test_health_upload_and_ground_truth(tmp_path):
    client = TestClient(create_app())
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    image = Image.new("RGB", (120, 40), "white")
    image_path = tmp_path / "TT075.png"
    image.save(image_path)
    with image_path.open("rb") as handle:
        response = client.post("/documents", files={"file": ("TT075.png", handle, "image/png")})
    assert response.status_code == 200
    document_id = response.json()["document_id"]
    assert response.json()["pages"] == 1

    settings = get_settings()
    catalog = Catalog(settings.catalog_db)
    page = catalog.list_pages(document_id)[0]
    line_path = tmp_path / "line.png"
    Image.new("RGB", (120, 30), "black").save(line_path)
    catalog.insert_auxiliary_line(
        {
            "id": "line1",
            "document_id": document_id,
            "page_id": page["id"],
            "line_index": 0,
            "image_path": str(line_path),
            "image_sha256": "hash-1",
            "text_pred": "قريه",
            "text_ota": "قريه",
            "script": "siyakat",
            "source": "htr_prediction",
            "split": page["split"],
            "verification_status": "unverified",
            "baseline": [[100, 20], [0, 20]],
            "boundary": [[0, 0], [120, 0], [120, 30], [0, 30], [0, 0]],
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
    )
    saved = client.post(
        "/lines/line1/ground-truth",
        json={"text_ota": "قرية", "verification_status": "expert_verified"},
    )
    assert saved.status_code == 200
    assert saved.json()["text_ota"] == "قرية"
    assert saved.json()["source"] == "user_correction"
    assert saved.json()["verification_status"] == "expert_verified"

    exported = client.get(f"/documents/{document_id}/export", params={"format": "txt"})
    assert exported.status_code == 200
    assert exported.content.decode("utf-8").strip() == "قرية"
    xml = client.get(f"/documents/{document_id}/export", params={"format": "pagexml"})
    assert xml.status_code == 200
    assert "قرية".encode("utf-8") in xml.content

    missing = client.get("/documents/yok")
    assert missing.status_code == 404
