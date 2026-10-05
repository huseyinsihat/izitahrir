"""Build Kraken training manifests from verified ground truth."""

from __future__ import annotations

import os
from pathlib import Path

from app.config import Settings
from app.data.validate import validate_line
from app.htr.export import write_pagexml_file
from app.service import HTRService
from app.utils.unicode_text import normalize_ota


def prepare_training(settings: Settings) -> dict:
    service = HTRService(settings)
    report = service.dataset_report()
    accepted_ids = {item["line_id"] for item in report["accepted_lines"]}
    lines = [
        line
        for line in service.catalog.list_lines()
        if line["id"] in accepted_ids and line["split"] in {"train", "validation", "test"}
    ]
    by_page: dict[str, list[dict]] = {}
    for line in lines:
        by_page.setdefault(line["page_id"], []).append(line)

    use_page_xml = True
    warnings: list[str] = []
    for page_id, page_lines in by_page.items():
        page = service.catalog.get_page(page_id)
        if page is None or not Path(page["image_path"]).is_file():
            use_page_xml = False
            warnings.append(f"{page_id}: sayfa görüntüsü yok, path formatına düşülecek")
            continue
        if any(not line.get("baseline") or not line.get("boundary") for line in page_lines):
            use_page_xml = False
            warnings.append(
                f"{page_id}: baseline veya sınır yok. Path formatı baseline segmentasyonla uyumsuzdur."
            )

    out_dir = settings.training_dir / "data"
    if out_dir.exists():
        for child in list(out_dir.rglob("*")):
            if child.is_file():
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests = {"train": [], "validation": [], "test": []}
    format_type = "page" if use_page_xml and lines else "path"
    if format_type == "path" and lines:
        warnings.append(
            "Eğitim path formatında. Bu formatla ince ayar yapılan model, sayfa baseline segmentasyonuyla uyumsuz olabilir."
        )

    for page_id, page_lines in by_page.items():
        page = service.catalog.get_page(page_id)
        assert page is not None
        split = page["split"]
        if split not in manifests:
            continue
        if format_type == "page":
            xml_path = out_dir / split / f"{page_id}.xml"
            image_name = _relative_image(xml_path, Path(page["image_path"]))
            prepared = []
            for line in page_lines:
                item = dict(line)
                item["text_ota"] = normalize_ota(line["text_ota"] or "")
                prepared.append(item)
            write_pagexml_file(xml_path, page, prepared, image_name)
            manifests[split].append(str(xml_path))
        else:
            for line in page_lines:
                issues = validate_line(line)
                if issues:
                    continue
                target = out_dir / split / f"{line['id']}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(line["image_path"]).read_bytes())
                gt = target.with_suffix(".gt.txt")
                gt.write_text(normalize_ota(line["text_ota"] or "") + "\n", encoding="utf-8")
                manifests[split].append(str(target))

    manifest_paths = {}
    for split, entries in manifests.items():
        path = settings.training_dir / f"{split}.lst"
        path.write_text("".join(f"{entry}\n" for entry in entries), encoding="utf-8")
        manifest_paths[split] = str(path)

    accepted = report["accepted_lines"]
    train_ok = sum(1 for item in accepted if item["split"] == "train")
    val_ok = sum(1 for item in accepted if item["split"] == "validation")
    summary = {
        "format_type": format_type,
        "warnings": warnings,
        "counts": {split: len(entries) for split, entries in manifests.items()},
        "line_counts": {"train": train_ok, "validation": val_ok},
        "manifests": manifest_paths,
        "validation": report,
        "ready": train_ok > 0 and val_ok > 0 and (train_ok + val_ok) >= int(settings.training["min_reviewed_lines"]),
    }
    return summary


def _relative_image(xml_path: Path, image_path: Path) -> str:
    relative = os.path.relpath(image_path, start=xml_path.parent)
    return Path(relative).as_posix()
