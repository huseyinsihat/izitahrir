"""Isolate every test from the project catalog."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import reset_settings

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    config = (ROOT / "config.yaml").read_text(encoding="utf-8")
    (tmp_path / "config.yaml").write_text(config, encoding="utf-8")
    monkeypatch.setenv("TARIHHTR_ROOT", str(tmp_path))
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "config.yaml"))
    reset_settings()
    yield tmp_path
    reset_settings()
