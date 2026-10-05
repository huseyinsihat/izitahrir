"""Recognition with a Kraken model. Output is Unicode logical order."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.config import Settings
from app.htr.models import load_recognizer
from app.utils.unicode_text import clean_prediction


def recognition_config(settings: Settings, return_line_image: bool = False):
    from kraken.configs import RecognitionInferenceConfig

    return RecognitionInferenceConfig(
        batch_size=settings.batch_size,
        num_line_workers=settings.num_line_workers,
        return_line_image=return_line_image,
        bidi_reordering=True,
        num_threads=settings.thread_count(),
        **settings.accelerator_kwargs(),
    )


def single_line_segmentation(image: Image.Image, _recognizer=None, _text_direction: str = "horizontal-tb"):
    """Treat an already cropped line as one box, the same way `kraken --no-segmentation` does.

    A synthetic baseline on a short strip is extracted as a blank or warped patch,
    so the recognizer returns an empty string.
    """
    from kraken.containers import BBoxLine, Segmentation

    width, height = image.size
    line = BBoxLine(id="line_0", bbox=(0, 0, width, height))
    return Segmentation(
        type="bbox",
        imagename="line",
        text_direction="horizontal-tb",
        script_detection=False,
        lines=[line],
    )


def prediction_text(record) -> str:
    # predict() already reorders to logical order when bidi_reordering is on.
    return clean_prediction(getattr(record, "prediction", "") or "")


def recognize_segmentation(image: Image.Image, segmentation, settings: Settings, model_path: Path | None = None):
    recognizer = load_recognizer(Path(model_path or settings.recognition_model))
    config = recognition_config(settings, return_line_image=True)
    return list(recognizer.predict(im=image, segmentation=segmentation, config=config))


def recognize_line_image(image: Image.Image, settings: Settings, model_path: Path | None = None) -> str:
    recognizer = load_recognizer(Path(model_path or settings.recognition_model))
    segmentation = single_line_segmentation(image, recognizer, settings.text_direction)
    config = recognition_config(settings, return_line_image=False)
    parts = [
        prediction_text(record)
        for record in recognizer.predict(im=image, segmentation=segmentation, config=config)
    ]
    return clean_prediction(" ".join(part for part in parts if part))
