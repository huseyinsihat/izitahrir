"""HTTP API for upload, recognition, correction, and export."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.config import get_settings
from app.data.catalog import CatalogError
from app.htr.models import ModelMissingError
from app.preprocessing.pages import IngestError
from app.service import HTRService


class GroundTruthUpdate(BaseModel):
    text_ota: str = Field(min_length=1)
    verification_status: Literal["unverified", "reviewed", "expert_verified"] = "reviewed"


def create_app() -> FastAPI:
    app = FastAPI(title="Tarihhtr", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        settings = get_settings()
        return {
            "status": "ok",
            "recognition_model": settings.recognition_model.is_file(),
            "segmentation_model": settings.segmentation_model.is_file(),
            "finetuned_model": settings.finetuned_model.is_file(),
        }

    @app.post("/documents")
    async def upload_document(file: UploadFile = File(...)) -> dict:
        content = await file.read()
        try:
            payload = HTRService().ingest(file.filename or "belge.png", content)
        except IngestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"document_id": payload["document"]["id"], "pages": len(payload["pages"])}

    @app.post("/documents/{document_id}/read")
    def read_document(document_id: str) -> dict:
        try:
            payload = HTRService().read_document(document_id)
        except CatalogError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ModelMissingError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _public_document(payload)

    @app.get("/documents/{document_id}")
    def get_document(document_id: str) -> dict:
        try:
            payload = HTRService().document_payload(document_id)
        except CatalogError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _public_document(payload)

    @app.get("/documents/{document_id}/pages/{page_id}/lines")
    def page_lines(document_id: str, page_id: str) -> dict:
        service = HTRService()
        try:
            service.document_payload(document_id)
        except CatalogError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        page = service.catalog.get_page(page_id)
        if page is None or page["document_id"] != document_id:
            raise HTTPException(status_code=404, detail="Sayfa yok")
        return {"page": page, "lines": service.catalog.list_lines(page_id=page_id)}

    @app.post("/lines/{line_id}/ground-truth")
    def save_ground_truth(line_id: str, body: GroundTruthUpdate) -> dict:
        try:
            return HTRService().save_ground_truth(line_id, body.text_ota, body.verification_status)
        except CatalogError as exc:
            status = 404 if "yok" in str(exc) else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc

    @app.get("/documents/{document_id}/export")
    def export_document(document_id: str, format: Literal["txt", "json", "pagexml"] = "txt") -> Response:
        try:
            filename, body, media_type = HTRService().export(document_id, format)
        except CatalogError as exc:
            status = 404 if "yok" in str(exc) else 400
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        return Response(
            content=body,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
        )

    return app


def _public_document(payload: dict) -> dict:
    return {
        "document_id": payload["document"]["id"],
        "filename": payload["document"]["filename"],
        "pages": [
            {
                "page_id": page["id"],
                "page_index": page["page_index"],
                "split": page["split"],
                "width": page["width"],
                "height": page["height"],
            }
            for page in payload["pages"]
        ],
        "lines": payload["lines"],
    }


app = create_app()
