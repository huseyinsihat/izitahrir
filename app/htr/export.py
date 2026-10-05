"""TXT, JSON, and PAGE-XML export."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
ET.register_namespace("", PAGE_NS)


def _points(points: list | None) -> str:
    if not points:
        return ""
    return " ".join(f"{int(point[0])},{int(point[1])}" for point in points)


def lines_to_txt(lines: list[dict]) -> str:
    ordered = sorted(lines, key=lambda line: (line.get("page_index", 0), line["line_index"]))
    return "\n".join((line.get("text_ota") or line.get("text_pred") or "") for line in ordered) + "\n"


def document_to_json(document: dict, pages: list[dict], lines: list[dict]) -> str:
    by_page: dict[str, list[dict]] = {}
    for line in lines:
        by_page.setdefault(line["page_id"], []).append(line)
    payload = {
        "document_id": document["id"],
        "filename": document["filename"],
        "pages": [],
    }
    for page in pages:
        payload["pages"].append(
            {
                "page_id": page["id"],
                "page_index": page["page_index"],
                "image_path": page["image_path"],
                "split": page["split"],
                "width": page["width"],
                "height": page["height"],
                "lines": [
                    {
                        "line_id": line["id"],
                        "line_index": line["line_index"],
                        "image_path": line["image_path"],
                        "text_ota": line.get("text_ota") or "",
                        "text_pred": line.get("text_pred") or "",
                        "script": line["script"],
                        "source": line["source"],
                        "split": line["split"],
                        "verification_status": line["verification_status"],
                        "baseline": line.get("baseline"),
                        "boundary": line.get("boundary"),
                    }
                    for line in by_page.get(page["id"], [])
                ],
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def page_to_pagexml(page: dict, lines: list[dict], image_filename: str) -> str:
    page_el = ET.Element(f"{{{PAGE_NS}}}PcGts")
    page_node = ET.SubElement(
        page_el,
        f"{{{PAGE_NS}}}Page",
        {
            "imageFilename": image_filename,
            "imageWidth": str(page["width"]),
            "imageHeight": str(page["height"]),
            "readingDirection": "right-to-left",
        },
    )
    region = ET.SubElement(
        page_node,
        f"{{{PAGE_NS}}}TextRegion",
        {"id": f"region_{page['id']}", "type": "text"},
    )
    ET.SubElement(
        region,
        f"{{{PAGE_NS}}}Coords",
        {"points": f"0,0 {page['width']},0 {page['width']},{page['height']} 0,{page['height']} 0,0"},
    )
    for line in lines:
        text = line.get("text_ota") or line.get("text_pred") or ""
        node = ET.SubElement(
            region,
            f"{{{PAGE_NS}}}TextLine",
            {"id": line["id"], "readingDirection": "right-to-left"},
        )
        boundary = line.get("boundary") or [
            [0, 0],
            [page["width"], 0],
            [page["width"], page["height"]],
            [0, page["height"]],
            [0, 0],
        ]
        baseline = line.get("baseline") or _fallback_baseline(boundary)
        ET.SubElement(node, f"{{{PAGE_NS}}}Coords", {"points": _points(boundary)})
        ET.SubElement(node, f"{{{PAGE_NS}}}Baseline", {"points": _points(baseline)})
        equiv = ET.SubElement(node, f"{{{PAGE_NS}}}TextEquiv")
        unicode_el = ET.SubElement(equiv, f"{{{PAGE_NS}}}Unicode")
        unicode_el.text = text
    return ET.tostring(page_el, encoding="unicode", xml_declaration=False)


def _fallback_baseline(boundary: list) -> list:
    xs = [int(point[0]) for point in boundary]
    ys = [int(point[1]) for point in boundary]
    y = min(ys) + (3 * (max(ys) - min(ys)) // 4)
    return [[max(xs), y], [min(xs), y]]


def write_pagexml_file(path: Path, page: dict, lines: list[dict], image_filename: str) -> None:
    xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + page_to_pagexml(page, lines, image_filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")
