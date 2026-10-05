from PIL import Image

from app.htr.pipeline import is_line_image
from app.utils.turkish import to_turkish, uncertain_reading


def test_latin_text_stays():
    assert to_turkish("Kahve iskemleleri") == "Kahve iskemleleri"


def test_empty_stays_empty():
    assert to_turkish("  ") == ""


def test_nesih_sentence_is_turkish():
    script = "بركلمهنك معرب اولمسنده بالفعل وجود اعراب لازم اولميوب عاملی"
    text = to_turkish(script)
    assert "bir" in text.lower()
    assert "kelimenin" in text.lower()
    assert "olmasında" in text.lower()
    assert "olmayıp" in text.lower()
    assert not any("\u0600" <= char <= "\u06FF" for char in text)


def test_elyazisi_sentence_is_turkish():
    script = "قهوه اسکملهلری آتیلمش طرفك مقابلنده کی سدده"
    text = to_turkish(script).lower()
    assert text.startswith("kahve")
    assert "iskemleleri" in text
    assert "atılmış" in text
    assert "tarafın" in text
    assert "mukabilinde" in text


def test_learned_word_keeps_turkish_letters():
    assert to_turkish("اتش").replace("İ", "i").lower() == "ateş"
    assert to_turkish("اتمام").replace("İ", "i").lower() == "itmam"


def test_clean_nesih_is_not_uncertain():
    script = "بركلمهنك معرب اولمسنده بالفعل وجود اعراب لازم اولميوب عاملی"
    assert uncertain_reading(script) is False


def test_rika_misread_is_uncertain():
    script = "قطرنده بر طوپوز واردر . شمدیکی قوت تمرینلرنده"
    assert uncertain_reading(script) is True


def test_latin_and_empty_are_not_uncertain():
    assert uncertain_reading("") is False
    assert uncertain_reading("Kahve iskemleleri") is False


def test_line_strip_is_not_segmented_as_a_page():
    assert is_line_image(Image.new("RGB", (1014, 59), "white"))
    assert is_line_image(Image.new("RGB", (1111, 285), "white"))
    assert not is_line_image(Image.new("RGB", (1200, 800), "white"))
