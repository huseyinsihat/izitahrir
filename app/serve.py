"""Arayüz ve API'yi aynı adreste açar."""

from __future__ import annotations

import os
import subprocess
import sys

import gradio as gr
import uvicorn

from app.api.main import create_app
from app.config import get_settings
from app.ui.gradio_app import build_demo


def ensure_models() -> None:
    script = get_settings().root / "scripts" / "download_models.py"
    completed = subprocess.run([sys.executable, str(script)], check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def build_app():
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    settings = get_settings()
    ensure_models()
    app = create_app()
    demo = build_demo()
    return gr.mount_gradio_app(
        app,
        demo,
        path="/",
        server_name="0.0.0.0",
        server_port=settings.gradio_port,
        show_api=False,
        ssr_mode=False,
        allowed_paths=[
            str(settings.documents_dir),
            str(settings.outputs_dir),
            str(settings.ground_truth_dir),
        ],
    )


def main() -> None:
    settings = get_settings()
    uvicorn.run(build_app(), host="0.0.0.0", port=settings.gradio_port)


if __name__ == "__main__":
    main()
