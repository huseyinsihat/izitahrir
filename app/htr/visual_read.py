"""Visual line reading for a tahrir line the fast recognizer could not spell.

The fast recognizer stays in front. A line whose Ottoman spelling misses the
lexicon is sent as an image. The model returns Ottoman spelling and a Latin
reading. ``to_turkish`` is only the fallback when that Latin reading is absent.
"""

from __future__ import annotations

import base64
import io
import json
import re
import ssl
import time
from typing import Callable

import httpx
from PIL import Image

from app.utils.turkish import to_turkish, uncertain_reading
from app.utils.unicode_text import normalize_ota

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_PROMPT = (
    "Bu görüntü, bir tahrir defteri satırının bir karesidir. "
    "Yazı çoğu zaman siyakat ya da süratli divani el yazısıdır. "
    "Yalnız bu karede görünen mürekkebi oku. "
    "Ezber cümle, açıklama veya satırda olmayan kelime ekleme. "
    "Çerçevedeki boş kenar yazı değildir.\n"
    'Yalnız şu JSON nesnesini döndür: {"osmanlica":"...","turkce":"..."}\n'
    "osmanlica: görünen yazıyı Arap harfli Osmanlı imlasıyla, göründüğü gibi yaz. "
    "Parantez, rakam ve kısaltmayı yerinde bırak.\n"
    "turkce: aynı yazının Latin harfli okuması. Anlamı özetleme, bugünkü dile çevirme. "
    "Bilinen kelimeleri alışılmış Türkçe yazılışıyla yaz. Okunamayan yeri [...] ile işaretle."
)
_LATIN = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]")
_TARGET_HEIGHT = 150
_STRIP_WIDTH = 720
_OVERLAP = 36

Post = Callable[..., httpx.Response]


class VisualReadError(Exception):
    """The visual reading could not be used."""


def image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _frame(image: Image.Image, background: tuple) -> Image.Image:
    width, height = image.size
    frame_h = max(height + 80, int(width * 0.55))
    frame_w = width + 48
    canvas = Image.new("RGB", (frame_w, frame_h), background)
    canvas.paste(image, (24, (frame_h - height) // 2))
    return canvas


def line_frames(image: Image.Image) -> list[Image.Image]:
    """Scale a thin line and cut a wide one into frames the visual model accepts.

    Frames run from the right of the line to the left, which is the reading order.
    """
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 2 or height < 2:
        return [rgb]
    if height < _TARGET_HEIGHT:
        scale = _TARGET_HEIGHT / height
        rgb = rgb.resize((max(1, round(width * scale)), _TARGET_HEIGHT), Image.Resampling.LANCZOS)
        width, height = rgb.size
    background = rgb.getpixel((0, 0))
    if width <= _STRIP_WIDTH:
        return [_frame(rgb, background)]
    frames: list[Image.Image] = []
    right = width
    while right > 0:
        left = max(0, right - _STRIP_WIDTH)
        frames.append(_frame(rgb.crop((left, 0, right, height)), background))
        if left == 0:
            break
        right = left + _OVERLAP
    return frames


def _default_post(url: str, headers: dict, body: dict, timeout: float) -> httpx.Response:
    return httpx.post(
        url,
        headers=headers,
        json=body,
        timeout=httpx.Timeout(timeout, connect=20.0),
        verify=ssl.create_default_context(),
    )


def _keep_script(text: str) -> str:
    kept: list[str] = []
    for char in text:
        if (
            "\u0600" <= char <= "\u06FF"
            or "\u0750" <= char <= "\u077F"
            or char.isspace()
            or char.isdigit()
            or char in ".,،؛:!?؟-()[]/"
        ):
            kept.append(char)
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return ""


def _usable_turkish(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned or len(cleaned) > 800 or not _LATIN.search(cleaned):
        return ""
    return cleaned


def _parse_readings(content: str) -> tuple[str, str]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
    if isinstance(data, dict):
        ottoman = str(data.get("osmanlica") or data.get("ottoman") or "").strip()
        turkish = str(data.get("turkce") or data.get("turkish") or "").strip()
        if ottoman or turkish:
            return ottoman, turkish
    return text, ""


def _body(model: str, image: Image.Image, *, reasoning: bool, json_mode: bool = True) -> dict:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 2500,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_url(image)}},
                ],
            }
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if reasoning:
        payload["reasoning_effort"] = "low"
    return payload


def _failure_code(response: httpx.Response) -> int | None:
    if response.status_code >= 400:
        return response.status_code
    try:
        payload = response.json()
    except (TypeError, ValueError):
        return 502
    if not isinstance(payload, dict):
        return 502
    error = payload.get("error")
    if isinstance(error, dict) and error.get("code"):
        try:
            return int(error["code"])
        except (TypeError, ValueError):
            return 502
    if "choices" not in payload:
        return 502
    return None


def _readings_from_message(message: dict) -> tuple[str, str]:
    ottoman, turkish = _parse_readings(_message_text(message))
    return normalize_ota(_keep_script(ottoman)), _usable_turkish(turkish)


def _transcribe_frame(
    frame: Image.Image, settings, send: Post, headers: dict, model: str
) -> tuple[str, str]:
    def call(*, reasoning: bool, json_mode: bool) -> httpx.Response:
        return send(
            OPENROUTER_URL,
            headers,
            _body(model, frame, reasoning=reasoning, json_mode=json_mode),
            120,
        )

    # response_format leaves this model's content empty, so the JSON shape is
    # requested in the prompt and parsed from the text.
    response = call(reasoning=True, json_mode=False)
    code = _failure_code(response)
    if code == 400:
        response = call(reasoning=False, json_mode=False)
        code = _failure_code(response)
    elif code in (502, 503, 504):
        time.sleep(1)
        response = call(reasoning=True, json_mode=False)
        code = _failure_code(response)
        if code in (502, 503, 504):
            time.sleep(1.5)
            response = call(reasoning=True, json_mode=False)
            code = _failure_code(response)
    if code:
        raise VisualReadError("Görsel okuma tamamlanamadı")
    try:
        script, turkish = _readings_from_message(response.json()["choices"][0]["message"])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise VisualReadError("Görsel okuma tamamlanamadı") from exc
    if not script:
        raise VisualReadError("Görsel okuma boş döndü")
    return script, turkish


def transcribe_line(image: Image.Image, settings, post: Post | None = None) -> tuple[str, str]:
    """Return Ottoman spelling and the Latin reading of one line image."""
    api_key = getattr(settings, "openrouter_api_key", "") or ""
    if not api_key.strip():
        raise VisualReadError("API anahtarı yok")
    send = post or _default_post
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    model = getattr(settings, "openrouter_model", "") or "stealth/space-bunny-alpha"
    scripts: list[str] = []
    readings: list[str] = []
    for frame in line_frames(image):
        try:
            script, turkish = _transcribe_frame(frame, settings, send, headers, model)
        except (VisualReadError, httpx.HTTPError):
            continue
        if not script:
            continue
        scripts.append(script)
        readings.append(turkish or to_turkish(script))
    script = normalize_ota(" ".join(part for part in scripts if part))
    if not script:
        raise VisualReadError("Görsel okuma boş döndü")
    reading = re.sub(r"\s+", " ", " ".join(readings)).strip()
    if _usable_turkish(reading):
        return script, reading
    return script, to_turkish(script)


def proposal_for_line(line: dict, settings, post: Post | None = None) -> dict:
    """Read one uncertain line. Raises when the call cannot be used."""
    ottoman, turkish = transcribe_line(line["image"], settings, post=post)
    return {
        "row": int(line["row"]),
        "ottoman": ottoman,
        "turkish": turkish,
        "fast_ottoman": str(line.get("script") or ""),
        "fast_turkish": str(line.get("turkish") or ""),
    }


def outcome_note(failed: int, proposals: list) -> str:
    if failed and not proposals:
        return "Görsel okuma tamamlanamadı."
    if failed:
        return "Bazı satırların görsel okuması tamamlanamadı."
    return ""


def propose_readings(lines: list[dict], settings, post: Post | None = None) -> tuple[list[dict], str]:
    """Read uncertain lines. A missing key or a failed call leaves the fast text in place."""
    pending = [line for line in lines if uncertain_reading(str(line.get("script") or ""))]
    if not pending:
        return [], ""
    if not (getattr(settings, "openrouter_api_key", "") or "").strip():
        return [], ""
    proposals: list[dict] = []
    failed = 0
    for line in pending:
        try:
            proposals.append(proposal_for_line(line, settings, post=post))
        except (VisualReadError, httpx.HTTPError):
            failed += 1
    return proposals, outcome_note(failed, proposals)


def apply_proposals(rows: list, proposals: list[dict]) -> tuple[list[list], int, int]:
    """Write a proposal only when that row still shows the fast reading."""
    updated = [list(row) for row in rows]
    applied = 0
    skipped = 0
    for item in proposals:
        row_index = int(item["row"])
        if row_index < 0 or row_index >= len(updated) or len(updated[row_index]) < 3:
            continue
        row = updated[row_index]
        same_turkish = str(row[1]).strip() == str(item.get("fast_turkish") or "").strip()
        same_ottoman = str(row[2]).strip() == str(item.get("fast_ottoman") or "").strip()
        if not same_turkish or not same_ottoman:
            skipped += 1
            continue
        turkish = str(item.get("turkish") or "").strip()
        ottoman = str(item.get("ottoman") or "").strip()
        if not turkish and not ottoman:
            continue
        if turkish:
            row[1] = turkish
        if ottoman:
            row[2] = ottoman
        applied += 1
    return updated, applied, skipped
