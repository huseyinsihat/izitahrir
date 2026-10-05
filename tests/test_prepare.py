from PIL import Image, ImageDraw

from app.config import get_settings, reset_settings
from app.service import HTRService
from app.training.prepare import prepare_training


def test_prepare_keeps_pages_together(tmp_path, monkeypatch):
    config = f"""
paths:
  models_dir: models
  ground_truth_dir: datasets/ground_truth
  documents_dir: datasets/ground_truth/documents
  catalog_db: datasets/ground_truth/catalog.sqlite
  outputs_dir: outputs
  training_dir: outputs/training
  evaluation_dir: outputs/evaluation
models:
  recognition: models/muharaf_rec_best.mlmodel
  recognition_finetuned: outputs/training/best.safetensors
  segmentation: models/blla.mlmodel
  muharaf_url: https://example.invalid/m
  muharaf_md5: abc
  blla_url: https://example.invalid/b
  blla_md5: def
inference:
  text_direction: horizontal-rl
  device: cpu
  precision: 32-true
  batch_size: 1
  num_line_workers: 0
data:
  training_verification_statuses: [reviewed, expert_verified]
  training_sources: [user_correction, tahrir]
  auxiliary_sources: [openiti_makhzan]
  default_script: siyakat
  default_source: user_correction
  prediction_source: htr_prediction
splits:
  train_pages: ["TT075:0"]
  validation_pages: ["TT075:1"]
  test_pages: ["TT075:2"]
  external_test_documents: ["TT080"]
training:
  epochs: 1
  batch_size: 1
  resize: add
  quit: fixed
  lag: 1
  min_epochs: 1
  learning_rate: 0.0001
  min_reviewed_lines: 2
  weights_format: safetensors
  normalization: NFC
  normalize_whitespace: false
  augment: false
  num_workers: 1
unicode:
  normalization: NFC
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config, encoding="utf-8")
    monkeypatch.setenv("CONFIG_PATH", str(config_path))
    reset_settings()
    settings = get_settings()
    service = HTRService(settings)
    service.catalog.create_document("d1", "TT075.png", "TT075", "2026-01-01T00:00:00+00:00")
    service.catalog.create_document("d2", "TT080.png", "TT080", "2026-01-01T00:00:00+00:00")
    page_ids = []
    for doc_id, indexes in (("d1", (0, 1, 2)), ("d2", (0,))):
        for index in indexes:
            page_id = f"{doc_id}-{index}"
            image_path = settings.documents_dir / f"{page_id}.png"
            image = Image.new("RGB", (240, 60), "white")
            ImageDraw.Draw(image).rectangle((8, 16, 220, 40), fill="black")
            image.save(image_path)
            service.catalog.insert_page(
                {
                    "id": page_id,
                    "document_id": doc_id,
                    "page_index": index,
                    "image_path": str(image_path),
                    "width": 240,
                    "height": 60,
                    "split": "unassigned",
                }
            )
            line_path = settings.documents_dir / f"{page_id}-line.png"
            image.crop((0, 10, 240, 50)).save(line_path)
            service.catalog.insert_auxiliary_line(
                {
                    "id": f"line-{page_id}",
                    "document_id": doc_id,
                    "page_id": page_id,
                    "line_index": 0,
                    "image_path": str(line_path),
                    "image_sha256": f"sha-{page_id}",
                    "text_pred": "قريه",
                    "text_ota": "قرية",
                    "script": "siyakat",
                    "source": "user_correction" if doc_id == "d1" else "tahrir",
                    "split": "unassigned",
                    "verification_status": "reviewed",
                    "baseline": [[220, 40], [10, 40]],
                    "boundary": [[0, 0], [240, 0], [240, 60], [0, 60], [0, 0]],
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            )
            page_ids.append(page_id)
    summary = prepare_training(settings)
    assert summary["format_type"] == "page"
    assert summary["line_counts"] == {"train": 1, "validation": 1}
    train_manifest = (settings.training_dir / "train.lst").read_text(encoding="utf-8")
    val_manifest = (settings.training_dir / "validation.lst").read_text(encoding="utf-8")
    test_manifest = (settings.training_dir / "test.lst").read_text(encoding="utf-8")
    assert "d1-0.xml" in train_manifest
    assert "d1-1.xml" in val_manifest
    assert "d1-2.xml" in test_manifest
    assert "d1-0" not in val_manifest
    assert "TT080" not in train_manifest + val_manifest + test_manifest
    xml = (settings.training_dir / "data" / "train" / "d1-0.xml").read_text(encoding="utf-8")
    assert "قرية" in xml
