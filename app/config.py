"""Load config.yaml and environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

VALID_SPLITS = ("train", "validation", "test", "external_test", "unassigned")
VALID_STATUSES = ("unverified", "reviewed", "expert_verified")


@dataclass(frozen=True)
class Settings:
    root: Path
    config_path: Path
    models_dir: Path
    recognition_model: Path
    muharaf_model: Path
    finetuned_model: Path
    segmentation_model: Path
    ground_truth_dir: Path
    documents_dir: Path
    catalog_db: Path
    outputs_dir: Path
    training_dir: Path
    evaluation_dir: Path
    text_direction: str
    device: str
    precision: str
    batch_size: int
    num_line_workers: int
    training_statuses: tuple[str, ...]
    training_sources: tuple[str, ...]
    auxiliary_sources: tuple[str, ...]
    default_script: str
    default_source: str
    prediction_source: str
    splits: dict
    training: dict
    muharaf_url: str
    muharaf_md5: str
    blla_url: str
    blla_md5: str
    recognition_url: str
    recognition_md5: str
    api_port: int
    gradio_port: int
    openrouter_api_key: str
    openrouter_model: str

    def accelerator_kwargs(self) -> dict:
        device = self.device
        if device == "cpu":
            accelerator, resolved = "cpu", 1
        elif device.startswith("cuda"):
            index = device.split(":", 1)[1] if ":" in device else "0"
            accelerator, resolved = "gpu", [int(index)]
        else:
            accelerator, resolved = "auto", "auto"
        return {
            "accelerator": accelerator,
            "device": resolved,
            "precision": self.precision,
        }

    def thread_count(self) -> int:
        return max(1, min(8, os.cpu_count() or 4))


def project_root() -> Path:
    env_root = os.environ.get("TARIHHTR_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[1]


def _abs(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_settings() -> Settings:
    load_dotenv(project_root() / ".env", override=False)
    root = project_root()
    config_path = Path(os.environ.get("CONFIG_PATH", root / "config.yaml"))
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    paths = raw["paths"]
    models = raw["models"]
    inference = raw["inference"]
    data = raw["data"]
    device = os.environ.get("TARIHHTR_DEVICE", inference["device"])

    settings = Settings(
        root=root,
        config_path=config_path,
        models_dir=_abs(root, paths["models_dir"]),
        recognition_model=_abs(root, models["recognition"]),
        muharaf_model=_abs(root, models.get("muharaf", "models/muharaf_rec_best.mlmodel")),
        finetuned_model=_abs(root, models["recognition_finetuned"]),
        segmentation_model=_abs(root, models["segmentation"]),
        ground_truth_dir=_abs(root, paths["ground_truth_dir"]),
        documents_dir=_abs(root, paths["documents_dir"]),
        catalog_db=_abs(root, paths["catalog_db"]),
        outputs_dir=_abs(root, paths["outputs_dir"]),
        training_dir=_abs(root, paths["training_dir"]),
        evaluation_dir=_abs(root, paths["evaluation_dir"]),
        text_direction=inference["text_direction"],
        device=device,
        precision=inference["precision"],
        batch_size=int(inference["batch_size"]),
        num_line_workers=int(inference["num_line_workers"]),
        training_statuses=tuple(data["training_verification_statuses"]),
        training_sources=tuple(data["training_sources"]),
        auxiliary_sources=tuple(data.get("auxiliary_sources", [])),
        default_script=data["default_script"],
        default_source=data["default_source"],
        prediction_source=data["prediction_source"],
        splits=dict(raw["splits"]),
        training=dict(raw["training"]),
        muharaf_url=models["muharaf_url"],
        muharaf_md5=models["muharaf_md5"],
        blla_url=models["blla_url"],
        blla_md5=models["blla_md5"],
        recognition_url=models.get(
            "recognition_url",
            "https://zenodo.org/records/21788410/files/medium.safetensors?download=1",
        ),
        recognition_md5=models.get("recognition_md5", "e3411a453ce3b9e9efae3f8631d85762"),
        api_port=int(os.environ.get("API_PORT", "8000")),
        gradio_port=int(os.environ.get("GRADIO_PORT", "7860")),
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", "").strip(),
        openrouter_model=os.environ.get("OPENROUTER_MODEL", "stealth/space-bunny-alpha").strip()
        or "stealth/space-bunny-alpha",
    )
    for directory in (
        settings.models_dir,
        settings.ground_truth_dir,
        settings.documents_dir,
        settings.outputs_dir,
        settings.training_dir,
        settings.evaluation_dir,
        settings.catalog_db.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def reset_settings() -> None:
    get_settings.cache_clear()
