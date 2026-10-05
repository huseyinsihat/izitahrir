"""Import local OpenITI MAKHZAN Ottoman line images as auxiliary data.

This data is mostly naskh. It is stored with source=openiti_makhzan and is
excluded from the default training filter.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from app.config import get_settings
from app.data.catalog import Catalog
from app.utils import configure_stdio, new_id, sha256_file
from app.utils.unicode_text import normalize_ota


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Yerel MAKHZAN satırlarını yardımcı veri olarak içe aktar")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if not args.directory.is_dir():
        raise SystemExit(f"Klasör yok: {args.directory}")
    settings = get_settings()
    catalog = Catalog(settings.catalog_db)
    now = utcnow()
    imported = 0
    for image_path in sorted(args.directory.rglob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        gt_path = image_path.with_suffix(".gt.txt")
        if not gt_path.is_file():
            stem_gt = image_path.parent / f"{image_path.name}.gt.txt"
            gt_path = stem_gt if stem_gt.is_file() else gt_path
        if not gt_path.is_file():
            continue
        text = normalize_ota(gt_path.read_text(encoding="utf-8"))
        if not text:
            continue
        doc_id = new_id()
        page_id = new_id()
        line_id = new_id()
        with Image.open(image_path) as image:
            width, height = image.size
        catalog.create_document(doc_id, image_path.name, image_path.stem, now)
        catalog.insert_page(
            {
                "id": page_id,
                "document_id": doc_id,
                "page_index": 0,
                "image_path": str(image_path.resolve()),
                "width": width,
                "height": height,
                "split": "unassigned",
            }
        )
        catalog.insert_auxiliary_line(
            {
                "id": line_id,
                "document_id": doc_id,
                "page_id": page_id,
                "line_index": 0,
                "image_path": str(image_path.resolve()),
                "image_sha256": sha256_file(image_path),
                "text_pred": None,
                "text_ota": text,
                "script": "nesih",
                "source": "openiti_makhzan",
                "split": "unassigned",
                "verification_status": "reviewed",
                "baseline": None,
                "boundary": None,
                "created_at": now,
                "updated_at": now,
            }
        )
        imported += 1
    print(f"İçe aktarılan yardımcı satır: {imported}. Varsayılan eğitime dahil edilmez.")


if __name__ == "__main__":
    main()
