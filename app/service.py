"""Document ingest, recognition, correction, and export."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from app.config import Settings, get_settings
from app.data.catalog import Catalog, CatalogError
from app.data.splits import apply_split_rules
from app.data.validate import validation_report
from app.htr.export import document_to_json, lines_to_txt, write_pagexml_file
from app.htr.models import ModelMissingError, require_file
from app.preprocessing.pages import IngestError, load_document
from app.utils import new_id, sha256_bytes
from app.utils.unicode_text import normalize_ota


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class HTRService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.catalog = Catalog(self.settings.catalog_db)

    def ingest(self, filename: str, content: bytes) -> dict:
        pages = load_document(content, filename)
        doc_id = new_id()
        stem = Path(filename).stem
        created = utcnow()
        self.catalog.create_document(doc_id, filename, stem, created)
        doc_dir = self.settings.documents_dir / doc_id
        stored_pages = []
        for index, image in enumerate(pages):
            page_id = new_id()
            image_path = doc_dir / "pages" / f"{index:04d}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path, format="PNG")
            page = {
                "id": page_id,
                "document_id": doc_id,
                "page_index": index,
                "image_path": str(image_path),
                "width": image.width,
                "height": image.height,
                "split": "unassigned",
            }
            self.catalog.insert_page(page)
            stored_pages.append(page)
        self.apply_splits()
        return self.document_payload(doc_id)

    def apply_splits(self) -> dict[str, str]:
        return apply_split_rules(self.catalog, self.settings.splits)

    def read_document(self, document_id: str, model_path: Path | None = None) -> dict:
        document = self._require_document(document_id)
        model = Path(model_path or self.settings.recognition_model)
        require_file(model)
        from app.htr.pipeline import read_page_image

        for page in self.catalog.list_pages(document["id"]):
            with Image.open(page["image_path"]) as image:
                rgb = image.convert("RGB")
                recognized = read_page_image(
                    rgb,
                    self.settings,
                    imagename=page["image_path"],
                    model_path=model,
                )
            self._store_lines(document, page, recognized)
        return self.document_payload(document_id)

    def _store_lines(self, document: dict, page: dict, recognized: list[dict]) -> None:
        now = utcnow()
        line_dir = self.settings.documents_dir / document["id"] / "lines"
        line_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in recognized:
            line_id = new_id()
            crop_path = line_dir / f"{line_id}.png"
            crop = item["crop"].convert("RGB")
            crop.save(crop_path, format="PNG")
            raw = crop_path.read_bytes()
            script = normalize_ota(item.get("text_script") or item.get("text") or "")
            shown = normalize_ota(item.get("text") or "") or script
            rows.append(
                {
                    "id": line_id,
                    "document_id": document["id"],
                    "page_id": page["id"],
                    "line_index": item["line_index"],
                    "image_path": str(crop_path),
                    "image_sha256": sha256_bytes(raw),
                    "text_pred": script,
                    "text_ota": shown,
                    "script": self.settings.default_script,
                    "source": self.settings.prediction_source,
                    "split": page["split"],
                    "verification_status": "unverified",
                    "baseline": item["baseline"],
                    "boundary": item["boundary"],
                    "created_at": now,
                    "updated_at": now,
                }
            )
        self.catalog.replace_page_lines(page["id"], rows)

    def save_ground_truth(self, line_id: str, text_ota: str, verification_status: str) -> dict:
        text = normalize_ota(text_ota)
        if not text:
            raise CatalogError("Transkripsiyon boş olamaz")
        return self.catalog.save_ground_truth(
            line_id=line_id,
            text_ota=text,
            verification_status=verification_status,
            source=self.settings.default_source,
            updated_at=utcnow(),
        )

    def document_payload(self, document_id: str) -> dict:
        document = self._require_document(document_id)
        pages = self.catalog.list_pages(document_id)
        lines = self.catalog.list_lines(document_id=document_id)
        return {"document": document, "pages": pages, "lines": lines}

    def export(self, document_id: str, export_format: str) -> tuple[str, bytes, str]:
        payload = self.document_payload(document_id)
        document = payload["document"]
        pages = payload["pages"]
        lines = payload["lines"]
        if export_format == "txt":
            body = lines_to_txt(self._lines_with_page_index(lines, pages)).encode("utf-8")
            return f"{document['stem']}.txt", body, "text/plain; charset=utf-8"
        if export_format == "json":
            body = document_to_json(document, pages, lines).encode("utf-8")
            return f"{document['stem']}.json", body, "application/json; charset=utf-8"
        if export_format == "pagexml":
            return self._export_pagexml(document, pages, lines)
        raise CatalogError(f"Bilinmeyen format: {export_format}")

    def _export_pagexml(self, document: dict, pages: list[dict], lines: list[dict]) -> tuple[str, bytes, str]:
        by_page: dict[str, list[dict]] = {}
        for line in lines:
            by_page.setdefault(line["page_id"], []).append(line)
        if len(pages) == 1:
            page = pages[0]
            image_name = Path(page["image_path"]).resolve().as_posix()
            xml_path = self.settings.outputs_dir / "exports" / f"{document['id']}.xml"
            write_pagexml_file(xml_path, page, by_page.get(page["id"], []), image_name)
            return f"{document['stem']}.xml", xml_path.read_bytes(), "application/xml; charset=utf-8"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for page in pages:
                image_name = Path(page["image_path"]).name
                xml_path = self.settings.outputs_dir / "exports" / document["id"] / f"{page['page_index']:04d}.xml"
                write_pagexml_file(xml_path, page, by_page.get(page["id"], []), image_name)
                archive.write(xml_path, arcname=xml_path.name)
        name = f"{document['stem']}_pagexml.zip"
        return name, buffer.getvalue(), "application/zip"

    def dataset_report(self) -> dict:
        self.apply_splits()
        lines = self.catalog.list_lines()
        trainable = [
            line
            for line in lines
            if line["verification_status"] in self.settings.training_statuses
            and line["source"] in self.settings.training_sources
        ]
        report = validation_report(trainable, self.catalog)
        report["page_splits"] = {
            page["id"]: page["split"] for page in self.catalog.list_pages()
        }
        report["split_counts"] = _count(trainable, "split")
        report["status_counts"] = _count(lines, "verification_status")
        self._assert_page_split_integrity(lines)
        return report

    def _assert_page_split_integrity(self, lines: list[dict]) -> None:
        pages = {page["id"]: page["split"] for page in self.catalog.list_pages()}
        for line in lines:
            if line["split"] != pages[line["page_id"]]:
                raise CatalogError(
                    f"Sayfa {line['page_id']} satırları farklı bölümlere düşmüş"
                )

    def _require_document(self, document_id: str) -> dict:
        document = self.catalog.get_document(document_id)
        if document is None:
            raise CatalogError(f"Belge yok: {document_id}")
        return document

    @staticmethod
    def _lines_with_page_index(lines: list[dict], pages: list[dict]) -> list[dict]:
        index = {page["id"]: page["page_index"] for page in pages}
        enriched = []
        for line in lines:
            item = dict(line)
            item["page_index"] = index.get(line["page_id"], 0)
            enriched.append(item)
        return enriched


def _count(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row[key]] = counts.get(row[key], 0) + 1
    return counts


__all__ = ["HTRService", "CatalogError", "IngestError", "ModelMissingError"]
