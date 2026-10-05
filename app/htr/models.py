"""Kraken model cache."""

from __future__ import annotations

from pathlib import Path

_SEGMENTERS: dict[str, object] = {}
_RECOGNIZERS: dict[str, object] = {}


class ModelMissingError(FileNotFoundError):
    pass


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise ModelMissingError(
            f"Model yok: {path}. Önce `python scripts/download_models.py` çalıştırın."
        )
    return path


def load_segmenter(path: Path):
    key = str(path.resolve())
    cached = _SEGMENTERS.get(key)
    if cached is not None:
        return cached
    require_file(path)
    from kraken.tasks import SegmentationTaskModel

    model = SegmentationTaskModel.load_model(key)
    _SEGMENTERS[key] = model
    return model


def load_recognizer(path: Path):
    key = str(path.resolve())
    cached = _RECOGNIZERS.get(key)
    if cached is not None:
        return cached
    require_file(path)
    from kraken.tasks import RecognitionTaskModel

    model = RecognitionTaskModel.load_model(key)
    _RECOGNIZERS[key] = model
    return model


def clear_model_cache() -> None:
    _SEGMENTERS.clear()
    _RECOGNIZERS.clear()
