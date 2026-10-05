"""Ottoman Arabic-script text to readable Turkish Latin.

The recognizer returns the spelling on the page. This layer turns that
spelling into Turkish letters for the screen and the text export. Latin
text is left as written.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from app.utils.unicode_text import normalize_ota

_MADDA = "ا\u0653"
_STRIP = dict.fromkeys(
    (
        0x0640,
        0x064B,
        0x064C,
        0x064D,
        0x064E,
        0x064F,
        0x0650,
        0x0651,
        0x0652,
        0x0653,
        0x0654,
        0x0655,
        0x0656,
        0x0657,
        0x0658,
        0x0670,
    ),
    None,
)
_FOLD = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "ؤ": "و",
        "ئ": "ی",
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
    }
)
_VOWEL_LETTERS = set("اآویهوە")
_CONSONANTS = {
    "ب": "b",
    "پ": "p",
    "ت": "t",
    "ث": "s",
    "ج": "c",
    "چ": "ç",
    "ح": "h",
    "خ": "h",
    "د": "d",
    "ذ": "z",
    "ر": "r",
    "ز": "z",
    "ژ": "j",
    "س": "s",
    "ش": "ş",
    "ص": "s",
    "ض": "z",
    "ط": "t",
    "ظ": "z",
    "غ": "ğ",
    "ف": "f",
    "ق": "k",
    "ک": "k",
    "گ": "g",
    "ڭ": "n",
    "ل": "l",
    "م": "m",
    "ن": "n",
}
_RAW_LEXICON = {
    "بر": "bir",
    "و": "ve",
    "بو": "bu",
    "شو": "şu",
    "که": "ki",
    "کی": "ki",
    "ایله": "ile",
    "ایچون": "için",
    "دخی": "dahi",
    "هر": "her",
    "نه": "ne",
    "یوق": "yok",
    "یوک": "yok",
    "وار": "var",
    "اولان": "olan",
    "اولوب": "olup",
    "اولدی": "oldu",
    "اولمش": "olmuş",
    "اولمیوب": "olmayıp",
    "اولمسنده": "olmasında",
    "ایدی": "idi",
    "ایکن": "iken",
    "ایسه": "ise",
    "کبی": "gibi",
    "قدر": "kadar",
    "چوق": "çok",
    "اما": "ama",
    "اگر": "eğer",
    "زیرا": "zira",
    "هم": "hem",
    "بله": "bile",
    "کتاب": "kitap",
    "باب": "bab",
    "حضرت": "hazret",
    "افندی": "efendi",
    "پاشا": "paşa",
    "دولت": "devlet",
    "سلطان": "sultan",
    "تاریخ": "tarih",
    "سنه": "sene",
    "اوزره": "üzere",
    "حقنده": "hakkında",
    "طرفندن": "tarafından",
    "بالفعل": "bilfiil",
    "وجود": "vücut",
    "اعراب": "irap",
    "لازم": "lazım",
    "معرب": "muareb",
    "کلمه": "kelime",
    "کلمە": "kelime",
    "کلمهنک": "kelimenin",
    "کلمەنک": "kelimenin",
    "عامل": "amil",
    "عاملی": "amili",
    "قهوه": "kahve",
    "اسکمله": "iskemle",
    "اسکملهلری": "iskemleleri",
    "اتیلمش": "atılmış",
    "آتیلمش": "atılmış",
    "طرف": "taraf",
    "طرفک": "tarafın",
    "مقابل": "mukabil",
    "مقابلنده": "mukabilinde",
    "مقابلندکی": "mukabilindeki",
}
# Longest suffix first. Front/back vowels pick the Turkish form.
_RAW_SUFFIXES = (
    ("لرنده", "larında", "lerinde"),
    ("لری", "ları", "leri"),
    ("لر", "lar", "ler"),
    ("نده", "ında", "inde"),
    ("ندن", "ından", "inden"),
    ("نک", "nın", "nin"),
    ("یه", "ya", "ye"),
    ("دن", "dan", "den"),
    ("ده", "da", "de"),
    ("دا", "da", "de"),
    ("ک", "ın", "in"),
    ("ی", "ı", "i"),
)
_RAW_PREFIXES = (("بر", "bir"), ("بو", "bu"), ("شو", "şu"))


def canon(word: str) -> str:
    word = word.replace(_MADDA, "آ")
    word = word.translate(_STRIP)
    word = unicodedata.normalize("NFC", word).translate(_FOLD)
    return word.translate(_STRIP)


def _lexicon() -> dict[str, str]:
    return {canon(src): dst for src, dst in _RAW_LEXICON.items()}


def _suffixes() -> tuple[tuple[str, str, str], ...]:
    return tuple((canon(src), back, front) for src, back, front in _RAW_SUFFIXES)


def _prefixes() -> tuple[tuple[str, str], ...]:
    return tuple((canon(src), dst) for src, dst in _RAW_PREFIXES)


def _learned_lexicon() -> dict[str, str]:
    path = Path(__file__).with_name("lexicon_ota.json")
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in data.items()}


LEXICON = _lexicon()
LEARNED = _learned_lexicon()
SUFFIXES = _suffixes()
PREFIXES = _prefixes()


def _has_arabic(text: str) -> bool:
    return any("\u0600" <= char <= "\u06FF" or "\u0750" <= char <= "\u077F" for char in text)


def _back_vowel(text: str) -> bool:
    for char in reversed(text):
        if char in "aıouâ":
            return True
        if char in "eiöüî":
            return False
    return False


def _known(word: str) -> str | None:
    if word in LEXICON:
        return LEXICON[word]
    return LEARNED.get(word)


def _lexical_stem(word: str) -> str | None:
    hit = _known(word)
    if hit:
        return hit
    for suffix, back, front in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2:
            stem = word[: -len(suffix)]
            base = _known(stem)
            if base:
                return base + (back if _back_vowel(base) else front)
    return None


def _letters(word: str) -> str:
    raw = canon(word)
    out: list[str] = []
    index = 0
    while index < len(raw):
        char = raw[index]
        nxt = raw[index + 1] if index + 1 < len(raw) else ""
        prev = raw[index - 1] if index else ""
        if char == "آ":
            out.append("a")
        elif char == "ا":
            if index == 0 and nxt == "و":
                out.append("o")
                index += 2
                continue
            if index == 0 and nxt == "ی":
                out.append("i")
                index += 2
                continue
            out.append("i" if index == 0 else "a")
        elif char == "و":
            out.append("v" if prev in _VOWEL_LETTERS or nxt in _VOWEL_LETTERS else "u")
        elif char in ("ه", "ە"):
            out.append("e" if index == len(raw) - 1 or nxt not in _VOWEL_LETTERS else "h")
        elif char == "ی":
            out.append("i" if index == len(raw) - 1 or nxt not in _VOWEL_LETTERS else "y")
        elif char == "ع":
            if index == 0:
                out.append("i")
        elif char in _CONSONANTS:
            out.append(_CONSONANTS[char])
        index += 1
    return "".join(out)


def _known_token(token: str, allow_prefix: bool = True) -> bool:
    if _lexical_stem(token):
        return True
    if not allow_prefix:
        return False
    for prefix, _turkish in PREFIXES:
        if token.startswith(prefix) and len(token) > len(prefix) + 1:
            rest = token[len(prefix) :]
            if _lexical_stem(rest):
                return True
    return False


def read_word(word: str, allow_prefix: bool = True) -> str:
    token = canon(word)
    if not token:
        return ""
    if _known_token(token, allow_prefix=allow_prefix):
        hit = _lexical_stem(token)
        if hit:
            return hit
        for prefix, turkish in PREFIXES:
            if token.startswith(prefix) and len(token) > len(prefix) + 1:
                rest = token[len(prefix) :]
                if _lexical_stem(rest):
                    return f"{turkish} {read_word(rest, allow_prefix=False)}"
    return _letters(token)


def _arabic_words(text: str) -> list[str]:
    words: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            words.append("".join(buffer))
            buffer.clear()

    for char in text:
        if "\u0600" <= char <= "\u06FF" or "\u0750" <= char <= "\u077F" or char == "\u0653":
            buffer.append(char)
        else:
            flush()
    flush()
    return words


def uncertain_reading(text: str) -> bool:
    """True when fewer than half of the Ottoman words are in the lexicon.

    Those lines are the ones the letter-by-letter reading is likely to garble.
    """
    cleaned = normalize_ota(text or "")
    words = [word for word in _arabic_words(cleaned) if canon(word)]
    if not words:
        return False
    known = sum(1 for word in words if _known_token(canon(word)))
    return known * 2 < len(words)


def _capitalize(text: str) -> str:
    if not text:
        return text
    first = text[0]
    if first == "i":
        first = "İ"
    elif first == "ı":
        first = "I"
    else:
        first = first.upper()
    return first + text[1:]


def to_turkish(text: str) -> str:
    """Return Turkish Latin. Text that is already Latin stays unchanged."""
    cleaned = normalize_ota(text or "")
    if not cleaned or not _has_arabic(cleaned):
        return cleaned
    pieces: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            pieces.append(read_word("".join(buffer)))
            buffer.clear()

    for char in cleaned:
        if "\u0600" <= char <= "\u06FF" or "\u0750" <= char <= "\u077F" or char == "\u0653":
            buffer.append(char)
        else:
            flush()
            pieces.append(char)
    flush()
    joined = re.sub(r"\s+", " ", "".join(pieces)).strip()
    return _capitalize(joined)
