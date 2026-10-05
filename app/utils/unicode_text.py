"""Unicode helpers that keep historical Ottoman orthography intact."""

from __future__ import annotations

import unicodedata

BIDI_CONTROLS = set(range(0x202A, 0x202F)) | set(range(0x2066, 0x206A))
REPLACEMENT = "\ufffd"


def normalize_ota(text: str) -> str:
    """NFC only. Compatibility folding would change historical letter forms."""
    return unicodedata.normalize("NFC", text).strip()


def unicode_issues(text: str) -> list[str]:
    issues: list[str] = []
    if text is None:
        return ["missing_text"]
    if REPLACEMENT in text:
        issues.append("replacement_character")
    if any(ord(char) in BIDI_CONTROLS for char in text):
        issues.append("bidi_override")
    if unicodedata.normalize("NFC", text) != text:
        issues.append("not_nfc")
    return issues


def clean_prediction(text: str) -> str:
    """Drop bidi controls emitted by a model, then NFC. Letter identity stays."""
    kept = "".join(char for char in text if ord(char) not in BIDI_CONTROLS and char != REPLACEMENT)
    return normalize_ota(kept)
