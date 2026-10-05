"""Point and crop helpers for baseline lines."""

from __future__ import annotations

from PIL import Image


def as_points(value) -> list[tuple[int, int]] | None:
    if not value:
        return None
    points: list[tuple[int, int]] = []
    for point in value:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append((int(point[0]), int(point[1])))
    return points or None


def crop_polygon(image: Image.Image, boundary, baseline) -> Image.Image:
    points = as_points(boundary) or as_points(baseline)
    if not points:
        return image.copy()
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    width, height = image.size
    left = max(0, min(xs))
    top = max(0, min(ys))
    right = min(width, max(xs) + 1)
    bottom = min(height, max(ys) + 1)
    if right <= left or bottom <= top:
        return image.copy()
    return image.crop((left, top, right, bottom))
