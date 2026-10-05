"""Baseline segmentation with Kraken's BLLA model."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.config import Settings
from app.htr.models import load_segmenter


def segment_page(image: Image.Image, settings: Settings, imagename: str = "page"):
    from kraken.configs import SegmentationInferenceConfig

    model = load_segmenter(Path(settings.segmentation_model))
    config = SegmentationInferenceConfig(
        text_direction=settings.text_direction,
        batch_size=1,
        num_threads=settings.thread_count(),
        **settings.accelerator_kwargs(),
    )
    segmentation = model.predict(image, config)
    if getattr(segmentation, "imagename", None) in (None, ""):
        segmentation.imagename = imagename
    return segmentation
