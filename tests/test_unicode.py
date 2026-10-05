from app.utils.unicode_text import clean_prediction, normalize_ota, unicode_issues


def test_nfc_keeps_ottoman_letters():
    text = "قريه پ چ ژ گ ڭ"
    assert normalize_ota(text) == text


def test_nfc_does_not_apply_compatibility_folding():
    isolated = "\ufb50"
    assert normalize_ota(isolated) == isolated


def test_nfd_is_reported_and_composed():
    alef_hamza = "\u0623"
    decomposed = "\u0627\u0654"
    assert unicode_issues(decomposed) == ["not_nfc"]
    assert normalize_ota(decomposed) == alef_hamza


def test_bidi_override_and_replacement_are_rejected():
    assert "bidi_override" in unicode_issues("قرية\u202e")
    assert "replacement_character" in unicode_issues("قري\ufffd")


def test_prediction_cleanup_drops_controls_only():
    assert clean_prediction("پ\u202eچ") == "پچ"
