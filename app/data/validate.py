"""Ground-truth checks run before a line can enter training."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.utils.unicode_text import normalize_ota, unicode_issues

MIN_WIDTH = 32
MIN_HEIGHT = 8
MIN_ASPECT = 1.2


def validate_line(line: dict) -> list[str]:
    issues: list[str] = []
    image_path = Path(line.get("image_path") or "")
    if not image_path.is_file():
        issues.append("missing_image")
        return issues
    try:
        with Image.open(image_path) as image:
            image.load()
            width, height = image.size
            extrema = image.convert("L").getextrema()
    except Exception:
        issues.append("unreadable_image")
        return issues
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        issues.append("crop_too_small")
    if height <= 0 or (width / height) < MIN_ASPECT:
        issues.append("crop_not_line_like")
    if extrema and extrema[0] == extrema[1]:
        issues.append("blank_image")

    text = line.get("text_ota")
    if text is None or not str(text).strip():
        issues.append("empty_transcription")
    else:
        raw = str(text)
        issues.extend(unicode_issues(raw))
        if not normalize_ota(raw):
            issues.append("empty_transcription")
    if not line.get("script"):
        issues.append("missing_script")
    if line.get("duplicate_of"):
        issues.append("duplicate")
    return issues


def validation_report(lines: list[dict], catalog) -> dict:
    accepted = []
    rejected = []
    seen: dict[tuple[str, str], str] = {}
    for line in lines:
        text = normalize_ota(line.get("text_ota") or "")
        digest = line.get("image_sha256") or ""
        duplicate_id = None
        if digest and text:
            existing = catalog.find_duplicate(digest, text, exclude_id=line["id"])
            if existing:
                duplicate_id = existing["id"]
            elif (digest, text) in seen:
                duplicate_id = seen[(digest, text)]
            else:
                seen[(digest, text)] = line["id"]
        checked = dict(line)
        checked["duplicate_of"] = duplicate_id
        issues = validate_line(checked)
        entry = {
            "line_id": line["id"],
            "page_id": line["page_id"],
            "split": line["split"],
            "verification_status": line["verification_status"],
            "source": line["source"],
            "issues": issues,
        }
        if issues:
            rejected.append(entry)
        else:
            accepted.append(entry)
    return {
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accepted_lines": accepted,
        "rejected_lines": rejected,
    }
