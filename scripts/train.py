"""Validate ground truth and fine-tune from the Muharaf weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.utils import configure_stdio
from app.training.finetune import TrainingNotReady, run_finetune


def main() -> None:
    configure_stdio()
    parser = argparse.ArgumentParser(description="Tahrir ground truth ile Kraken ince ayarı")
    parser.add_argument(
        "--run",
        action="store_true",
        help="ketos train çalıştır. Verilmezse veri hazırlanır ve komut yazılır.",
    )
    args = parser.parse_args()
    settings = get_settings()
    try:
        result = run_finetune(settings, execute=args.run)
    except TrainingNotReady as exc:
        print(exc)
        raise SystemExit(0) if not args.run else SystemExit(2)
    printable = {
        "executed": result["executed"],
        "command": result["command"],
        "experiment": result["experiment"],
        "format_type": result["summary"]["format_type"],
        "counts": result["summary"]["counts"],
        "line_counts": result["summary"]["line_counts"],
        "warnings": result["summary"]["warnings"],
        "rejected": result["summary"]["validation"]["rejected"],
        "best_model": result["best_model"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
