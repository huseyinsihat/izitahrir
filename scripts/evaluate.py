"""CER/WER for the base model and, when present, the fine-tuned model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.evaluation.report import evaluate
from app.utils import configure_stdio


def main() -> None:
    configure_stdio()
    report = evaluate(get_settings())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
