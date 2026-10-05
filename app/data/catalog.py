"""SQLite catalog for documents, pages, and line ground truth."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.config import VALID_SPLITS, VALID_STATUSES

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    stem TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    page_index INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    split TEXT NOT NULL DEFAULT 'unassigned',
    UNIQUE(document_id, page_index)
);

CREATE TABLE IF NOT EXISTS lines (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    page_id TEXT NOT NULL REFERENCES pages(id),
    line_index INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    image_sha256 TEXT,
    text_pred TEXT,
    text_ota TEXT,
    script TEXT NOT NULL,
    source TEXT NOT NULL,
    split TEXT NOT NULL DEFAULT 'unassigned',
    verification_status TEXT NOT NULL DEFAULT 'unverified',
    baseline_json TEXT,
    boundary_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(page_id, line_index)
);

CREATE INDEX IF NOT EXISTS idx_lines_page ON lines(page_id);
CREATE INDEX IF NOT EXISTS idx_lines_status ON lines(verification_status);
CREATE INDEX IF NOT EXISTS idx_lines_split ON lines(split);
"""


class CatalogError(ValueError):
    pass


class Catalog:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=5)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        return con

    def create_document(self, doc_id: str, filename: str, stem: str, created_at: str) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO documents (id, filename, stem, created_at) VALUES (?, ?, ?, ?)",
                (doc_id, filename, stem, created_at),
            )

    def insert_page(self, page: dict) -> None:
        self._check_split(page["split"])
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO pages (id, document_id, page_index, image_path, width, height, split)
                VALUES (:id, :document_id, :page_index, :image_path, :width, :height, :split)
                """,
                page,
            )

    def replace_page_lines(self, page_id: str, rows: list[dict]) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM lines WHERE page_id = ?", (page_id,))
            for row in rows:
                self._check_split(row["split"])
                self._check_status(row["verification_status"])
                payload = dict(row)
                payload["baseline_json"] = json.dumps(row.get("baseline"), ensure_ascii=False)
                payload["boundary_json"] = json.dumps(row.get("boundary"), ensure_ascii=False)
                con.execute(
                    """
                    INSERT INTO lines (
                        id, document_id, page_id, line_index, image_path, image_sha256,
                        text_pred, text_ota, script, source, split, verification_status,
                        baseline_json, boundary_json, created_at, updated_at
                    ) VALUES (
                        :id, :document_id, :page_id, :line_index, :image_path, :image_sha256,
                        :text_pred, :text_ota, :script, :source, :split, :verification_status,
                        :baseline_json, :boundary_json, :created_at, :updated_at
                    )
                    """,
                    payload,
                )

    def get_document(self, doc_id: str) -> dict | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def list_pages(self, document_id: str | None = None) -> list[dict]:
        query = "SELECT * FROM pages"
        params: tuple = ()
        if document_id:
            query += " WHERE document_id = ? ORDER BY page_index"
            params = (document_id,)
        else:
            query += " ORDER BY document_id, page_index"
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_page(self, page_id: str) -> dict | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM pages WHERE id = ?", (page_id,)).fetchone()
        return dict(row) if row else None

    def list_lines(self, document_id: str | None = None, page_id: str | None = None) -> list[dict]:
        query = "SELECT * FROM lines"
        params: list = []
        if page_id:
            query += " WHERE page_id = ?"
            params.append(page_id)
        elif document_id:
            query += " WHERE document_id = ?"
            params.append(document_id)
        query += " ORDER BY page_id, line_index"
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
        return [self._decode_line(row) for row in rows]

    def get_line(self, line_id: str) -> dict | None:
        with self._connect() as con:
            row = con.execute("SELECT * FROM lines WHERE id = ?", (line_id,)).fetchone()
        return self._decode_line(row) if row else None

    def find_duplicate(self, image_sha256: str, text_ota: str, exclude_id: str | None = None) -> dict | None:
        query = """
            SELECT * FROM lines
            WHERE image_sha256 = ? AND text_ota = ? AND id != ?
        """
        with self._connect() as con:
            row = con.execute(query, (image_sha256, text_ota, exclude_id or "")).fetchone()
        return self._decode_line(row) if row else None

    def set_page_split(self, page_id: str, split: str) -> None:
        self._check_split(split)
        with self._connect() as con:
            con.execute("UPDATE pages SET split = ? WHERE id = ?", (split, page_id))
            con.execute("UPDATE lines SET split = ? WHERE page_id = ?", (split, page_id))

    def save_ground_truth(
        self,
        line_id: str,
        text_ota: str,
        verification_status: str,
        source: str,
        updated_at: str,
    ) -> dict:
        self._check_status(verification_status)
        line = self.get_line(line_id)
        if line is None:
            raise CatalogError(f"Satır yok: {line_id}")
        duplicate = self.find_duplicate(line["image_sha256"] or "", text_ota, exclude_id=line_id)
        if duplicate and text_ota:
            raise CatalogError(f"Aynı görüntü ve metin zaten kayıtlı: {duplicate['id']}")
        with self._connect() as con:
            con.execute(
                """
                UPDATE lines
                SET text_ota = ?, verification_status = ?, source = ?, updated_at = ?
                WHERE id = ?
                """,
                (text_ota, verification_status, source, updated_at, line_id),
            )
        saved = self.get_line(line_id)
        assert saved is not None
        return saved

    def insert_auxiliary_line(self, row: dict) -> None:
        self._check_split(row["split"])
        self._check_status(row["verification_status"])
        payload = dict(row)
        payload["baseline_json"] = json.dumps(row.get("baseline"), ensure_ascii=False)
        payload["boundary_json"] = json.dumps(row.get("boundary"), ensure_ascii=False)
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO lines (
                    id, document_id, page_id, line_index, image_path, image_sha256,
                    text_pred, text_ota, script, source, split, verification_status,
                    baseline_json, boundary_json, created_at, updated_at
                ) VALUES (
                    :id, :document_id, :page_id, :line_index, :image_path, :image_sha256,
                    :text_pred, :text_ota, :script, :source, :split, :verification_status,
                    :baseline_json, :boundary_json, :created_at, :updated_at
                )
                """,
                payload,
            )

    @staticmethod
    def _check_split(split: str) -> None:
        if split not in VALID_SPLITS:
            raise CatalogError(f"Geçersiz split: {split}")

    @staticmethod
    def _check_status(status: str) -> None:
        if status not in VALID_STATUSES:
            raise CatalogError(f"Geçersiz verification_status: {status}")

    @staticmethod
    def _decode_line(row: sqlite3.Row) -> dict:
        item = dict(row)
        item["baseline"] = json.loads(item.pop("baseline_json") or "null")
        item["boundary"] = json.loads(item.pop("boundary_json") or "null")
        return item
