"""Visual reading uses a fake HTTP response. It never calls OpenRouter."""

import json

from PIL import Image

from app.htr.visual_read import apply_proposals, line_frames, propose_readings, transcribe_line


class _Settings:
    def __init__(self, key="test-key", model="stealth/space-bunny-alpha"):
        self.openrouter_api_key = key
        self.openrouter_model = model


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def _line(script="قطرنده بر طوپوز", turkish="Ktrnde bir tupuz"):
    return {
        "row": 0,
        "image": Image.new("RGB", (48, 16), "white"),
        "script": script,
        "turkish": turkish,
    }


def test_thin_strip_is_split_into_readable_frames():
    frames = line_frames(Image.new("RGB", (1600, 40), "white"))
    assert len(frames) >= 2
    for frame in frames:
        assert frame.size[1] > 120
        assert frame.size[0] / frame.size[1] < 4


def test_missing_key_skips_visual_read():
    called = False

    def post(url, headers, body, timeout):
        nonlocal called
        called = True
        return _Response(200)

    proposals, note = propose_readings([_line()], _Settings(key=""), post=post)
    assert proposals == []
    assert note == ""
    assert called is False


def test_fake_response_is_read_then_translated():
    def post(url, headers, body, timeout):
        assert body["model"] == "stealth/space-bunny-alpha"
        assert body["messages"][0]["content"][1]["type"] == "image_url"
        payload = {"osmanlica": "بر كتاب"}
        return _Response(
            200,
            {"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]},
        )

    ottoman, turkish = transcribe_line(_line()["image"], _Settings(), post=post)
    assert ottoman == "بر كتاب"
    assert turkish == "Bir kitap"


def test_model_latin_reading_is_kept():
    def post(url, headers, body, timeout):
        prompt = body["messages"][0]["content"][0]["text"]
        assert "osmanlica" in prompt
        assert "turkce" in prompt
        payload = {"osmanlica": "مثلا زيد", "turkce": "Mesela Zeyd [...]"}
        return _Response(
            200,
            {"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]},
        )

    ottoman, turkish = transcribe_line(_line()["image"], _Settings(), post=post)
    assert ottoman == "مثلا زيد"
    assert turkish == "Mesela Zeyd [...]"


def test_reasoning_field_is_dropped_after_400():
    calls = []

    def post(url, headers, body, timeout):
        calls.append("reasoning_effort" in body)
        if len(calls) == 1:
            return _Response(400, {"error": {"message": "unsupported"}})
        payload = {"osmanlica": "كتاب"}
        return _Response(
            200,
            {"choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}]},
        )

    ottoman, turkish = transcribe_line(_line()["image"], _Settings(), post=post)
    assert calls == [True, False]
    assert ottoman == "كتاب"
    assert "kitap" in turkish.lower()


def test_failed_call_keeps_fast_text():
    def post(url, headers, body, timeout):
        return _Response(502, {"error": {"message": "upstream"}})

    proposals, note = propose_readings([_line()], _Settings(), post=post)
    assert proposals == []
    assert note == "Görsel okuma tamamlanamadı."


def test_update_skips_a_row_the_user_edited():
    rows = [[1, "Ktrnde", "قطرنده", "İncelendi"]]
    proposals = [
        {
            "row": 0,
            "ottoman": "كتاب",
            "turkish": "Kitap",
            "fast_ottoman": "قطرنده",
            "fast_turkish": "Ktrnde",
        }
    ]
    updated, applied, skipped = apply_proposals(rows, proposals)
    assert applied == 1
    assert skipped == 0
    assert updated[0][1] == "Kitap"
    assert updated[0][2] == "كتاب"

    edited = [[1, "elle düzeltme", "قطرنده", "İncelendi"]]
    updated, applied, skipped = apply_proposals(edited, proposals)
    assert applied == 0
    assert skipped == 1
    assert updated[0][1] == "elle düzeltme"
