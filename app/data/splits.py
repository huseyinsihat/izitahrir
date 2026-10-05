"""Page-level split assignment. Lines never cross splits on their own."""

from __future__ import annotations

SPLIT_LISTS = {
    "train": "train_pages",
    "validation": "validation_pages",
    "test": "test_pages",
}


def find_conflicts(rules: dict) -> list[str]:
    seen: dict[str, str] = {}
    conflicts: list[str] = []
    for split, key in SPLIT_LISTS.items():
        for token in rules.get(key, []) or []:
            marker = str(token)
            previous = seen.get(marker)
            if previous and previous != split:
                conflicts.append(f"{marker} hem {previous} hem {split} listesinde")
            seen[marker] = split
    return conflicts


def page_tokens(page: dict, document: dict | None) -> set[str]:
    stem = (document or {}).get("stem", "")
    doc_id = page["document_id"]
    index = page["page_index"]
    tokens = {
        page["id"],
        f"{doc_id}:{index}",
    }
    if stem:
        tokens.add(f"{stem}:{index}")
        tokens.add(stem)
    return tokens


def resolve_split(page: dict, document: dict | None, rules: dict) -> str:
    tokens = page_tokens(page, document)
    matched: list[str] = []
    for split, key in SPLIT_LISTS.items():
        listed = {str(item) for item in (rules.get(key, []) or [])}
        if tokens & listed:
            matched.append(split)
    if len(matched) > 1:
        raise ValueError(f"Sayfa {page['id']} birden fazla bölüme düşüyor: {matched}")
    if matched:
        return matched[0]
    external = {str(item) for item in (rules.get("external_test_documents", []) or [])}
    doc_keys = {page["document_id"]}
    if document and document.get("stem"):
        doc_keys.add(document["stem"])
    if doc_keys & external:
        return "external_test"
    return "unassigned"


def apply_split_rules(catalog, rules: dict) -> dict[str, str]:
    conflicts = find_conflicts(rules)
    if conflicts:
        raise ValueError("; ".join(conflicts))
    assigned: dict[str, str] = {}
    for page in catalog.list_pages():
        document = catalog.get_document(page["document_id"])
        split = resolve_split(page, document, rules)
        catalog.set_page_split(page["id"], split)
        assigned[page["id"]] = split
    return assigned
