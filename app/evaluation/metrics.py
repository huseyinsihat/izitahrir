"""Character and word error rates for Ottoman transcriptions."""

from __future__ import annotations


def levenshtein(left: list, right: list) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_item in enumerate(left, start=1):
        current = [i]
        for j, right_item in enumerate(right, start=1):
            cost = 0 if left_item == right_item else 1
            current.append(min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    if reference == "":
        return 0.0 if hypothesis == "" else 1.0
    return levenshtein(list(reference), list(hypothesis)) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein(ref_words, hyp_words) / len(ref_words)


def aggregate(pairs: list[tuple[str, str]]) -> dict:
    if not pairs:
        return {"cer": None, "wer": None, "lines": 0}
    ref_chars = 0
    hyp_distance = 0
    ref_words = 0
    word_distance = 0
    for reference, hypothesis in pairs:
        ref_chars += len(reference)
        hyp_distance += levenshtein(list(reference), list(hypothesis))
        words = reference.split()
        ref_words += len(words)
        word_distance += levenshtein(words, hypothesis.split())
    return {
        "cer": (hyp_distance / ref_chars) if ref_chars else None,
        "wer": (word_distance / ref_words) if ref_words else None,
        "lines": len(pairs),
    }
