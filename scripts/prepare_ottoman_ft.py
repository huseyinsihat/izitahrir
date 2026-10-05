"""Hold out modeller/testveri and build a Kraken path-format fine-tune set.

Images come from the local naskh and riqa zips. A line is excluded when its
Ottoman text or image hash matches the checked test pairs.
"""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIPS = {
    "naskh": ROOT / "modeller" / "veri" / "ottoman-newdata-ocr" / "NaskhTestset.zip",
    "riqa": ROOT / "modeller" / "veri" / "ottoman-newdata-ocr" / "RiqaTestset.zip",
}
TEST_ROOT = ROOT / "modeller" / "testveri"
OUT = ROOT / "datasets" / "auxiliary" / "ottoman_ft"
LIST_DIR = ROOT / "outputs" / "training" / "script_ft"
FOLD = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ؤ": "و",
        "ئ": "ی",
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ە": "ه",
    }
)


def norm_script(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("M"))
    text = text.translate(FOLD).replace("\u0640", "")
    return re.sub(r"\s+", "", text)


def test_block() -> tuple[set[str], set[str]]:
    texts: set[str] = set()
    hashes: set[str] = set()
    for folder in ("osmanlica-nesih", "osmanlica-rika"):
        for path in (TEST_ROOT / folder).glob("*.txt"):
            texts.add(norm_script(path.read_text(encoding="utf-8")))
        for path in (TEST_ROOT / folder).glob("*.png"):
            hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    return texts, hashes


def ottoman_name(image_name: str) -> str:
    return image_name.replace("/images/", "/ottoman/").rsplit(".", 1)[0] + ".gt.txt"


def main() -> None:
    blocked_text, blocked_hash = test_block()
    OUT.mkdir(parents=True, exist_ok=True)
    kept: list[tuple[str, Path]] = []
    skipped = 0
    for script, zip_path in ZIPS.items():
        with zipfile.ZipFile(zip_path) as archive:
            images = [name for name in archive.namelist() if name.endswith(".png")]
            for image_name in images:
                gt_name = ottoman_name(image_name)
                if gt_name not in archive.namelist():
                    skipped += 1
                    continue
                raw = archive.read(image_name)
                text = archive.read(gt_name).decode("utf-8").strip()
                if not text:
                    skipped += 1
                    continue
                if hashlib.sha256(raw).hexdigest() in blocked_hash or norm_script(text) in blocked_text:
                    skipped += 1
                    continue
                stem = f"{script}_{Path(image_name).stem}"
                image_path = OUT / f"{stem}.png"
                gt_path = OUT / f"{stem}.gt.txt"
                if not image_path.exists() or image_path.stat().st_size != len(raw):
                    image_path.write_bytes(raw)
                gt_path.write_text(text + "\n", encoding="utf-8")
                kept.append((script, image_path))

    random.Random(7).shuffle(kept)
    val_target = max(40, int(len(kept) * 0.08))
    val: list[Path] = []
    train: list[Path] = []
    val_counts = {"naskh": 0, "riqa": 0}
    for script, path in kept:
        if len(val) < val_target and val_counts[script] < val_target:
            val.append(path)
            val_counts[script] += 1
        else:
            train.append(path)
    LIST_DIR.mkdir(parents=True, exist_ok=True)
    (LIST_DIR / "train.lst").write_text("".join(f"{path}\n" for path in train), encoding="utf-8")
    (LIST_DIR / "val.lst").write_text("".join(f"{path}\n" for path in val), encoding="utf-8")
    print(f"kept={len(kept)} train={len(train)} val={len(val)} skipped={skipped} val_counts={val_counts}")


if __name__ == "__main__":
    main()
