"""Compare the base recognizer with a fine-tuned model on held-out lines."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.config import Settings
from app.evaluation.metrics import aggregate
from app.htr.recognize import recognize_line_image
from app.service import HTRService
from app.utils.unicode_text import normalize_ota


def _pairs_for_model(lines: list[dict], settings: Settings, model_path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in lines:
        with Image.open(line["image_path"]) as image:
            hypothesis = recognize_line_image(image.convert("RGB"), settings, model_path=model_path)
        reference = normalize_ota(line.get("text_ota") or "")
        pairs.append((reference, hypothesis))
    return pairs


def evaluate(settings: Settings) -> dict:
    service = HTRService(settings)
    service.apply_splits()
    eligible = [
        line
        for line in service.catalog.list_lines()
        if line["verification_status"] in settings.training_statuses
        and line["source"] in settings.training_sources
        and line["split"] in {"test", "external_test"}
        and line.get("text_ota")
        and Path(line["image_path"]).is_file()
    ]
    report: dict = {"models": {}, "splits": {"test": {}, "external_test": {}}}
    model_paths = {"base": settings.recognition_model}
    if settings.finetuned_model.is_file():
        model_paths["finetuned"] = settings.finetuned_model
    else:
        empty = {"cer": None, "wer": None, "lines": 0, "model": None, "note": "İnce ayar modeli yok"}
        report["models"]["finetuned"] = empty
        for split in ("test", "external_test"):
            report["splits"][split]["finetuned"] = {
                "cer": None,
                "wer": None,
                "lines": 0,
            }

    grouped = {
        split: [line for line in eligible if line["split"] == split]
        for split in ("test", "external_test")
    }
    for name, path in model_paths.items():
        per_split_pairs = {
            split: _pairs_for_model(grouped[split], settings, path) for split in grouped
        }
        combined = [pair for pairs in per_split_pairs.values() for pair in pairs]
        scores = aggregate(combined)
        scores["model"] = str(path)
        report["models"][name] = scores
        for split, pairs in per_split_pairs.items():
            report["splits"][split][name] = aggregate(pairs)

    settings.evaluation_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.evaluation_dir / "report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["path"] = str(destination)
    return report
