"""Fine-tune the Muharaf Kraken model on tahrir ground truth."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from app.config import Settings
from app.training.prepare import prepare_training


class TrainingNotReady(RuntimeError):
    pass


def experiment_config(settings: Settings, summary: dict) -> dict:
    train = settings.training
    return {
        "precision": settings.precision,
        "device": settings.device,
        "num_workers": int(train["num_workers"]),
        "num_threads": 1,
        "train": {
            "training_data": [summary["manifests"]["train"]],
            "evaluation_data": [summary["manifests"]["validation"]],
            "format_type": summary["format_type"],
            "linetype": "baselines" if summary["format_type"] == "page" else "bbox",
            "normalization": train["normalization"],
            "normalize_whitespace": bool(train["normalize_whitespace"]),
            "resize": train["resize"],
            "load": str(settings.recognition_model),
            "checkpoint_path": str(settings.training_dir / "checkpoints"),
            "weights_format": train["weights_format"],
            "quit": train["quit"],
            "epochs": int(train["epochs"]),
            "lag": int(train["lag"]),
            "min_epochs": int(train["min_epochs"]),
            "lrate": float(train["learning_rate"]),
            "batch_size": int(train["batch_size"]),
            "augment": bool(train["augment"]),
        },
    }


def write_experiment(settings: Settings, summary: dict) -> Path:
    path = settings.training_dir / "experiment.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(experiment_config(settings, summary), sort_keys=False), encoding="utf-8")
    return path


def ketos_executable() -> str:
    name = "ketos.exe" if sys.platform == "win32" else "ketos"
    sibling = Path(sys.executable).with_name(name)
    if sibling.is_file():
        return str(sibling)
    return name


def ketos_command(settings: Settings, experiment_path: Path) -> list[str]:
    return [
        ketos_executable(),
        "--device",
        settings.device,
        "--precision",
        settings.precision,
        "--config",
        str(experiment_path),
        "train",
    ]


def publish_best_weights(settings: Settings) -> Path | None:
    candidates = list(settings.training_dir.rglob("best_*.safetensors"))
    candidates += list(settings.training_dir.rglob("*.safetensors"))
    unique = [path for path in candidates if path.is_file() and path.resolve() != settings.finetuned_model.resolve()]
    if not unique:
        return None
    best = max(unique, key=lambda path: path.stat().st_mtime)
    settings.finetuned_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, settings.finetuned_model)
    return settings.finetuned_model


def run_finetune(settings: Settings, execute: bool) -> dict:
    summary = prepare_training(settings)
    accepted = summary["validation"]["accepted_lines"]
    train_lines = sum(1 for item in accepted if item["split"] == "train")
    val_lines = sum(1 for item in accepted if item["split"] == "validation")
    minimum = int(settings.training["min_reviewed_lines"])
    if train_lines == 0 or val_lines == 0:
        raise TrainingNotReady(
            "Eğitim için hem train hem validation sayfalarında doğrulanmış satır gerekli. "
            "Satırları rastgele bölmeyin; config.yaml içindeki sayfa listelerini doldurun."
        )
    if train_lines + val_lines < minimum:
        raise TrainingNotReady(
            f"Doğrulanmış satır sayısı yetersiz ({train_lines + val_lines} < {minimum})."
        )
    experiment_path = write_experiment(settings, summary)
    command = ketos_command(settings, experiment_path)
    result = {
        "summary": summary,
        "experiment": str(experiment_path),
        "command": command,
        "executed": False,
        "best_model": None,
    }
    if not execute:
        subprocess.run([ketos_executable(), "train", "--help"], check=True)
        return result
    subprocess.run(command, check=True)
    result["executed"] = True
    published = publish_best_weights(settings)
    result["best_model"] = str(published) if published else None
    return result
