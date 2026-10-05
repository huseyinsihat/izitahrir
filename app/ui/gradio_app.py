"""Belge yükle, oku, düzelt, kaydet."""

from __future__ import annotations

import os
import time
from pathlib import Path

import gradio as gr
import httpx
from PIL import Image

from app.config import get_settings
from app.data.catalog import CatalogError
from app.htr.models import ModelMissingError, load_recognizer
from app.htr.visual_read import VisualReadError, apply_proposals, outcome_note, proposal_for_line
from app.preprocessing.pages import IngestError
from app.service import HTRService
from app.utils.turkish import uncertain_reading

STATUS_TO_CODE = {
    "incelenmedi": "unverified",
    "incelendi": "reviewed",
    "uzman onaylı": "expert_verified",
    "uzman onayli": "expert_verified",
    "unverified": "unverified",
    "reviewed": "reviewed",
    "expert_verified": "expert_verified",
}

CSS = """
body, .gradio-container {
  background: #f3efe6 !important;
  color: #1c1915 !important;
}
.gradio-container {
  max-width: 920px !important;
  margin: 0 auto !important;
  padding: 32px 22px 56px !important;
}
footer { display: none !important; }

.mast {
  border-top: 3px solid #8a3d32;
  border-bottom: 1px solid #e4dccf;
  padding: 18px 0 16px;
  margin-bottom: 22px;
}
.kicker {
  margin: 0 0 6px;
  font-size: 0.72rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #8a3d32;
  font-weight: 650;
}
.mast h1 {
  margin: 0;
  font-size: 2rem;
  line-height: 1.1;
  font-weight: 600;
  letter-spacing: -0.03em;
  color: #1b1916;
}
.lede {
  margin: 8px 0 0;
  color: #5c564c;
  font-size: 1rem;
}
.steps {
  margin: 12px 0 0;
  color: #6d665c;
  font-size: 0.92rem;
}
.steps span { color: #c4b8a8; padding: 0 6px; }

.section {
  margin: 8px 0 8px;
  font-size: 0.72rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #6d665c;
  font-weight: 650;
}
.hint {
  margin: 8px 0 0;
  color: #6d665c;
  font-size: 0.88rem;
}

.block, .form {
  border-color: #e4dccf !important;
  background: #fffdf9 !important;
  box-shadow: none !important;
}
button.primary {
  background: #1e3a34 !important;
  border-color: #1e3a34 !important;
  color: #f6f1e8 !important;
}
button.primary:hover {
  background: #16302b !important;
  border-color: #16302b !important;
}
button.secondary {
  background: #fffdf9 !important;
  border-color: #d9d0c3 !important;
  color: #1c1915 !important;
}
.read-btn button, button.read-btn {
  min-height: 46px;
}

#okunan table { width: 100%; }
#okunan th {
  font-size: 0.75rem;
  font-weight: 650;
  color: #6d665c;
  background: #faf7f2;
}
#okunan td, #okunan textarea, #okunan input {
  font-size: 1rem;
  line-height: 1.45;
}
#okunan tbody td:nth-child(2),
#okunan tbody td:nth-child(2) textarea,
#okunan tbody td:nth-child(2) input {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
#okunan tbody td:nth-child(3),
#okunan tbody td:nth-child(3) textarea,
#okunan tbody td:nth-child(3) input {
  direction: rtl;
  text-align: right;
  font-size: 1.35rem;
  line-height: 1.8;
  font-family: "Traditional Arabic", "Scheherazade New", "Noto Naskh Arabic", "Segoe UI", serif;
}
.line-view img, .line-view button img {
  object-fit: contain !important;
  background: #f6f1e6 !important;
  max-height: 150px;
}
.flat, .flat.block {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
}
#okunan .add-row-button {
  display: none !important;
}
"""

HEAD = """
<script>
const phrases = {
  "Drop File Here": "Dosyayı bırakın",
  "- or -": "veya",
  "or": "veya",
  "Click to Upload": "Dosya seçin"
};
function translateNode(node) {
  if (!node) return;
  if (node.nodeType === Node.TEXT_NODE) {
    const next = phrases[(node.textContent || "").trim()];
    if (next) node.textContent = next;
    return;
  }
  if (node.nodeType === Node.ELEMENT_NODE) {
    node.childNodes.forEach(translateNode);
  }
}
function run() { translateNode(document.body); }
new MutationObserver(run).observe(document.documentElement, {childList: true, subtree: true, characterData: true});
document.addEventListener("DOMContentLoaded", run);
</script>
"""


def _theme() -> gr.Theme:
    return gr.themes.Base(
        primary_hue=gr.themes.colors.stone,
        secondary_hue=gr.themes.colors.stone,
        neutral_hue=gr.themes.colors.stone,
        font=["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"],
        font_mono=["Consolas", "Segoe UI", "monospace"],
    ).set(
        body_background_fill="#f3efe6",
        body_background_fill_dark="#f3efe6",
        background_fill_primary="#fffdf9",
        background_fill_primary_dark="#fffdf9",
        block_background_fill="#fffdf9",
        block_background_fill_dark="#fffdf9",
        block_border_color="#e4dccf",
        block_border_color_dark="#e4dccf",
        block_label_text_color="#5c564c",
        block_label_text_color_dark="#5c564c",
        body_text_color="#1c1915",
        body_text_color_dark="#1c1915",
        button_primary_background_fill="#1e3a34",
        button_primary_background_fill_dark="#1e3a34",
        button_primary_background_fill_hover="#16302b",
        button_primary_background_fill_hover_dark="#16302b",
        button_primary_text_color="#f6f1e8",
        button_primary_text_color_dark="#f6f1e8",
        button_secondary_background_fill="#fffdf9",
        button_secondary_background_fill_dark="#fffdf9",
        button_secondary_text_color="#1c1915",
        button_secondary_text_color_dark="#1c1915",
        button_secondary_border_color="#d9d0c3",
        button_secondary_border_color_dark="#d9d0c3",
        input_background_fill="#fffdf9",
        input_background_fill_dark="#fffdf9",
        shadow_drop="none",
        shadow_drop_lg="none",
        block_shadow="none",
        block_radius="10px",
        button_large_radius="10px",
    )


def _service() -> HTRService:
    return HTRService()


def _rows(data) -> list[tuple]:
    if data is None:
        return []
    if hasattr(data, "itertuples"):
        return [tuple(row) for row in data.itertuples(index=False)]
    return [tuple(row) for row in data]


def _seconds(elapsed: float) -> str:
    return f"{elapsed:.1f}".replace(".", ",")


def _status_code(label: str) -> str:
    text = str(label or "").strip()
    if text in ("unverified", "reviewed", "expert_verified"):
        return text
    # İ.casefold() is not "i", so the Turkish capital is folded by hand first.
    key = text.replace("İ", "i").replace("I", "ı").casefold()
    return STATUS_TO_CODE.get(key, "")


def _hidden():
    return gr.update(visible=False)


def _proposal_text(proposals: list[dict]) -> str:
    blocks = ["Ayrıntılı okuma hazır. Güncelle, elle değiştirilmemiş satırlara yazar."]
    for item in proposals:
        number = int(item["row"]) + 1
        blocks.append(
            f"\n\n**Satır {number}**\n\nOsmanlıca: {item['ottoman']}\n\nOkuma: {item['turkish']}"
        )
    return "".join(blocks)


def read_upload(file_obj):
    hidden = _hidden()
    if file_obj is None:
        yield None, "Önce bir JPG, PNG veya PDF seçin.", [], [], [], hidden, [], hidden
        return
    path = Path(file_obj if isinstance(file_obj, str) else file_obj.name)
    started = time.perf_counter()
    try:
        service = _service()
        payload = service.ingest(path.name, path.read_bytes())
        payload = service.read_document(payload["document"]["id"])
    except (IngestError, CatalogError, ModelMissingError) as exc:
        yield None, str(exc), [], [], [], hidden, [], hidden
        return
    except Exception as exc:
        yield None, f"Okuma başarısız: {exc}", [], [], [], hidden, [], hidden
        return
    images = []
    rows = []
    line_ids = []
    pending = []
    for index, line in enumerate(payload["lines"], start=1):
        with Image.open(line["image_path"]) as image:
            rgb = image.convert("RGB")
            images.append((rgb, f"Satır {index}"))
            script = line.get("text_pred") or ""
            turkish = line.get("text_ota") or ""
            if uncertain_reading(script):
                pending.append(
                    {
                        "row": index - 1,
                        "image": rgb.copy(),
                        "script": script,
                        "turkish": turkish,
                    }
                )
        line_ids.append(line["id"])
        rows.append(
            [
                index,
                line.get("text_ota") or "",
                line.get("text_pred") or "",
                "İncelendi",
            ]
        )
    elapsed = _seconds(time.perf_counter() - started)
    count = len(rows)
    if count == 0:
        message = "Bu görüntüde satır bulunamadı."
    else:
        message = f"{count} satır okundu · {elapsed} sn. Türkçe metni kontrol edin."
    document_id = payload["document"]["id"]
    settings = get_settings()
    if not pending or not (getattr(settings, "openrouter_api_key", "") or "").strip():
        yield document_id, message, images, rows, line_ids, hidden, [], hidden
        return
    proposals: list[dict] = []
    failed = 0
    total = len(pending)
    for index, line in enumerate(pending, start=1):
        yield (
            document_id,
            f"{message} Ayrıntılı okuma sürüyor · {index}/{total}.",
            images,
            rows,
            line_ids,
            hidden,
            [],
            hidden,
        )
        try:
            proposals.append(proposal_for_line(line, settings))
        except (VisualReadError, httpx.HTTPError):
            failed += 1
    note = outcome_note(failed, proposals)
    if note:
        message = f"{message} {note}"
    if not proposals:
        yield document_id, message, images, rows, line_ids, hidden, [], hidden
        return
    yield (
        document_id,
        f"{message} Ayrıntılı okuma hazır.",
        images,
        rows,
        line_ids,
        gr.update(value=_proposal_text(proposals), visible=True),
        proposals,
        gr.update(visible=True),
    )


def apply_update(table, proposals):
    rows = [list(row) for row in _rows(table)]
    updated, applied, skipped = apply_proposals(rows, list(proposals or []))
    if applied:
        message = f"{applied} satır güncellendi."
        if skipped:
            message += f" {skipped} satır elle değiştiği için bırakıldı."
        return updated, message, _hidden(), [], _hidden()
    if skipped:
        return rows, f"{skipped} satır elle değiştiği için bırakıldı.", gr.update(), proposals, gr.update()
    return rows, "Güncellenecek satır yok.", _hidden(), [], _hidden()


def save_corrections(document_id, table, line_ids):
    if not document_id:
        return "Önce belge okutun."
    service = _service()
    saved = 0
    errors = []
    ids = list(line_ids or [])
    for index, row in enumerate(_rows(table)):
        if len(row) < 2:
            continue
        if index >= len(ids):
            errors.append("Eklenen satırın görüntüsü yok, kaydedilmedi.")
            continue
        line_id = ids[index]
        text = str(row[1])
        status = _status_code(row[3] if len(row) > 3 else "İncelendi")
        if not status:
            errors.append("Durum: İncelenmedi, İncelendi veya Uzman onaylı olmalı.")
            continue
        try:
            service.save_ground_truth(line_id, text, status)
            saved += 1
        except CatalogError as exc:
            errors.append(str(exc))
    message = f"{saved} satır kaydedildi."
    if errors:
        message += " " + " ".join(errors)
    return message


def export_file(document_id, export_format):
    if not document_id:
        return gr.File(visible=False)
    filename, body, _media = _service().export(document_id, export_format)
    destination = get_settings().outputs_dir / "exports" / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(body)
    return gr.File(value=str(destination), visible=True)


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="İz-i Tahrir",
        theme=_theme(),
        css=CSS,
        head=HEAD,
        analytics_enabled=False,
    ) as demo:
        gr.HTML(
            """
            <header class="mast">
              <p class="kicker">TÜBİTAK 2204-B</p>
              <h1>İz-i Tahrir</h1>
              <p class="lede">Tahrir Defterlerinin Yapay Zekâ Destekli Çözümlenmesi</p>
              <p class="steps">Belge yükle <span>·</span> Oku <span>·</span> Kontrol et <span>·</span> Kaydet</p>
            </header>
            """
        )
        document_id = gr.State()
        line_ids = gr.State([])
        proposals = gr.State([])
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                upload = gr.File(
                    label="Belge",
                    file_types=[".jpg", ".jpeg", ".png", ".pdf"],
                    file_count="single",
                )
                read_button = gr.Button("Oku", variant="primary", elem_classes=["read-btn"])
                status = gr.Markdown("Henüz belge okunmadı.")
            with gr.Column(scale=6):
                gallery = gr.Gallery(
                    label="Satır görüntüsü",
                    columns=1,
                    height=220,
                    object_fit="contain",
                    allow_preview=True,
                    elem_classes=["line-view"],
                )
        table = gr.Dataframe(
            headers=["Satır", "Türkçe", "Osmanlıca", "Durum"],
            datatype=["number", "str", "str", "str"],
            col_count=(4, "fixed"),
            column_widths=["72px", "46%", "34%", "130px"],
            wrap=True,
            interactive=True,
            label="Okunan metin",
            elem_id="okunan",
        )
        gr.HTML(
            '<p class="hint">Türkçe sütunu çalışma metnidir; düzeltebilirsiniz. Osmanlıca, satırda okunan yazıdır. Ayrıntılı okuma bitince Güncelle, değiştirilmemiş satırlara yazar. Durum: İncelenmedi, İncelendi veya Uzman onaylı.</p>',
            elem_classes=["flat"],
        )
        proposal = gr.Markdown(visible=False)
        update_button = gr.Button("Güncelle", variant="primary", visible=False)
        with gr.Row():
            save_button = gr.Button("Kaydet", variant="primary", scale=2)
            txt_button = gr.Button("TXT", variant="secondary")
            json_button = gr.Button("JSON", variant="secondary")
            xml_button = gr.Button("PAGE-XML", variant="secondary")
        download = gr.File(label="İndirilen dosya", visible=False)

        read_button.click(
            read_upload,
            inputs=[upload],
            outputs=[document_id, status, gallery, table, line_ids, proposal, proposals, update_button],
        )
        update_button.click(
            apply_update,
            inputs=[table, proposals],
            outputs=[table, status, proposal, proposals, update_button],
        )
        save_button.click(
            save_corrections,
            inputs=[document_id, table, line_ids],
            outputs=[status],
        )
        txt_button.click(lambda doc: export_file(doc, "txt"), inputs=[document_id], outputs=[download])
        json_button.click(lambda doc: export_file(doc, "json"), inputs=[document_id], outputs=[download])
        xml_button.click(lambda doc: export_file(doc, "pagexml"), inputs=[document_id], outputs=[download])
    return demo


def main() -> None:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    settings = get_settings()
    try:
        load_recognizer(settings.recognition_model)
    except ModelMissingError:
        pass
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=settings.gradio_port,
        show_api=False,
        allowed_paths=[
            str(settings.documents_dir),
            str(settings.outputs_dir),
            str(settings.ground_truth_dir),
        ],
    )


if __name__ == "__main__":
    main()
