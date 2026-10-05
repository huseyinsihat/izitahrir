"""Build an Ottoman-to-Turkish word list from the local parallel line zips.

Test lines in modeller/testveri are left out. The hand-written lexicon in
turkish.py still wins when both know a word.
"""

from __future__ import annotations

import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

from app.utils.turkish import canon

ROOT = Path(__file__).resolve().parents[1]
ZIPS = [
    ROOT / "modeller" / "veri" / "ottoman-newdata-ocr" / "NaskhTestset.zip",
    ROOT / "modeller" / "veri" / "ottoman-newdata-ocr" / "RiqaTestset.zip",
]
TEST_ROOT = ROOT / "modeller" / "testveri"
OUT = ROOT / "app" / "utils" / "lexicon_ota.json"
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
    return re.sub(r"\s+", "", text.translate(FOLD).replace("\u0640", ""))


def modernize(text: str) -> str:
    text = text.translate(str.maketrans("ÂÎÛâîû", "AIUaiu"))
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[’‘ʼʿʾ`´'ˈ]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def tr_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def arabic_key(token: str) -> str:
    letters = "".join(ch for ch in token if "\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F")
    return canon(letters)


def blocked() -> set[str]:
    found = set()
    for folder in ("osmanlica-nesih", "osmanlica-rika"):
        for path in (TEST_ROOT / folder).glob("*.txt"):
            found.add(norm_script(path.read_text(encoding="utf-8")))
    return found


def pairs_from(archive: zipfile.ZipFile, blocked_text: set[str]) -> list[tuple[str, str]]:
    found = []
    for name in archive.namelist():
        if "/ottoman/" not in name or not name.endswith(".gt.txt"):
            continue
        ottoman = archive.read(name).decode("utf-8").strip()
        if norm_script(ottoman) in blocked_text:
            continue
        turkish_name = name.replace("/ottoman/", "/turkish/")
        if turkish_name not in archive.namelist():
            continue
        turkish = modernize(archive.read(turkish_name).decode("utf-8"))
        src = ottoman.split()
        dst = turkish.split()
        if len(src) != len(dst) or not src:
            continue
        found.extend(zip(src, dst))
    return found


def main() -> None:
    blocked_text = blocked()
    counts: dict[str, Counter] = {}
    for zip_path in ZIPS:
        with zipfile.ZipFile(zip_path) as archive:
            for src, dst in pairs_from(archive, blocked_text):
                key = arabic_key(src)
                value = tr_lower(dst.strip(".,;:!?()[]\"“”«»"))
                if len(key) < 2 or not value or any("\u0600" <= ch <= "\u06FF" for ch in value):
                    continue
                if not all(ch.isalpha() or ch in "-" for ch in value):
                    continue
                counts.setdefault(key, Counter())[value] += 1
    chosen = {}
    for key, counter in counts.items():
        word, votes = counter.most_common(1)[0]
        if votes < 2:
            continue
        if sum(counter.values()) >= 3 and votes / sum(counter.values()) < 0.6:
            continue
        chosen[key] = word
    OUT.write_text(json.dumps(chosen, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(f"words={len(chosen)}")


if __name__ == "__main__":
    main()
