"""Import existing memories (a JSON file) into the local mem0 store, verbatim.

Accepts a list of strings, a list of {"memory"|"text": ..., "created_at"?,
"metadata"?} objects, or mem0's {"results": [...]}. Stored with extract=False
(no LLM pass, no paraphrase drift); embeddings are recomputed on insert and
any stored vectors are ignored. Re-running is safe: a content-hash idempotency
key skips memories already imported.
"""
from __future__ import annotations

import json
from pathlib import Path

import import_common as IC


def load(path) -> list:
    p = Path(path).expanduser()
    if not p.is_file():
        raise IC.ImportError_(f"Not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise IC.ImportError_(f"Not valid JSON: {e}") from None
    if isinstance(data, dict):
        data = data.get("results", data.get("memories"))
    if not isinstance(data, list):
        raise IC.ImportError_("Expected a list of memories (or {'results': [...]}).")
    out = []
    for item in data:
        if isinstance(item, str):
            text, meta = item, {}
        elif isinstance(item, dict):
            text = item.get("memory") or item.get("text") or ""
            meta = dict(item.get("metadata") or {})
            if item.get("created_at"):
                meta["original_created_at"] = item["created_at"]
        else:
            continue
        text = text.strip()
        if text:
            out.append((text, meta))
    if not out:
        raise IC.ImportError_("No memories found in that file.")
    return out


def run(path, add=None) -> dict:
    """`add(text, metadata, key)` is injectable for tests; default is mem0_core."""
    items = load(path)
    if add is None:
        import mem0_core as core

        def add(text, meta, key):
            return core.add(text, extract=False, metadata=meta, idempotency_key=key)
        digest = core.chunk_digest
    else:
        import hashlib
        digest = lambda t: hashlib.sha256(t.encode()).hexdigest()[:16]
    stored = skipped = 0
    for text, meta in items:
        meta.setdefault("source", "import")
        r = add(text, meta, "imp-" + digest(text))
        if str(r).startswith("skipped"):
            skipped += 1
        else:
            stored += 1
    return {"memories": len(items), "stored": stored, "skipped_existing": skipped}
