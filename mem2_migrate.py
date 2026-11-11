#!/usr/bin/env python3
"""One-shot: move full memory (mem2) off the legacy qdrant store into mem2.db.

The new db is rebuilt from the transcripts themselves (logs/ingested/*.log),
which are the source of truth. A qdrant vector is reused only when some point's
text matches a freshly parsed chunk exactly; anything else (files
re-chunked since, or ingested after the last qdrant index) is embedded anew.
Files outside the corpus are simply not carried over.

Reads a COPY of qdrant's storage.sqlite (qdrant holds a lock on the live dir
while mem2 runs) and needs no qdrant_client: the pickled points are decoded
with a stub unpickler. Writes to a temp file and renames it into place, and
refuses to overwrite an existing mem2.db. The legacy files are left untouched.

    python3 mem2_migrate.py                       # copy + migrate, default paths
    python3 mem2_migrate.py --qdrant X.sqlite     # use an existing copy
    python3 mem2_migrate.py --out /tmp/mem2.db    # write elsewhere (dry runs)
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import mem2_server as m2
from paths import QDRANT_MEM2_PATH

LEGACY_EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # the only model qdrant-mem2 ever held
STORAGE = QDRANT_MEM2_PATH / "collection" / "cchat-mem2" / "storage.sqlite"


class _Stub:
    """Stands in for any pickled qdrant/pydantic class; keeps its state dict."""

    def __init__(self, *args, **kwargs):
        self.__dict__["_args"] = args

    def __setstate__(self, state):
        if isinstance(state, tuple):  # (dict, slots)
            state = {k: v for part in state if isinstance(part, dict) for k, v in part.items()}
        inner = state.get("__dict__", state) if isinstance(state, dict) else {}
        self.__dict__.update(inner)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ("qdrant_client", "pydantic", "pydantic_core"):
            return _Stub
        return super().find_class(module, name)


def read_qdrant(storage: Path) -> dict[int, tuple[str, np.ndarray]]:
    """point_id -> (payload text, raw vector) from a qdrant local storage.sqlite.

    Point ids are not portable: they hash the source path, and the logs moved
    (~/.mem0/logs -> data/mem0/logs) after many points were written. Callers
    should match on text, which is all a vector depends on.
    """
    db = sqlite3.connect(f"file:{storage}?mode=ro", uri=True)
    out = {}
    try:
        for _id, blob in db.execute("SELECT id, point FROM points"):
            p = _Unpickler(__import__("io").BytesIO(blob)).load()
            vec = p.vector
            if isinstance(vec, dict):  # named vectors: mem2 only ever used the default
                vec = next(iter(vec.values()))
            out[int(p.id)] = ((p.payload or {}).get("text", ""),
                              np.asarray(vec, dtype=np.float32))
    finally:
        db.close()
    return out


def migrate(storage: Path, out: Path) -> dict:
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}; move it aside first")
    points = read_qdrant(storage)
    legacy = {text: vec for text, vec in points.values() if text}
    print(f"qdrant points read: {len(points)} ({len(legacy)} distinct texts)")

    tmp = out.with_name(out.name + ".migrating")
    for suffix in ("", "-wal", "-shm"):
        Path(str(tmp) + suffix).unlink(missing_ok=True)
    db = m2.open_db(tmp)
    with db:
        m2.set_meta(db, "embedder", LEGACY_EMBED_MODEL)
        m2.set_meta(db, "migrated_from", f"qdrant:{storage} at {time.strftime('%Y-%m-%dT%H:%M:%S')}")

    stats = {"files": 0, "chunks": 0, "reused": 0, "embedded": 0}
    for path in m2._log_paths():
        chunks = m2._parse_log(path)
        vectors, need = [], []
        for i, c in enumerate(chunks):
            vec = legacy.get(c["text"])
            if vec is not None:
                vectors.append(m2._unit(vec)[0])
            else:
                vectors.append(None)
                need.append(i)
        if need:
            if m2.EMBED_MODEL != LEGACY_EMBED_MODEL:
                raise SystemExit(
                    f"{path.name}: {len(need)} chunks need embedding but MEM2_EMBED_MODEL is "
                    f"{m2.EMBED_MODEL!r}; migrate with {LEGACY_EMBED_MODEL!r}, then --reembed")
            for k in range(0, len(need), m2._EMBED_BATCH):
                idx = need[k:k + m2._EMBED_BATCH]
                for i, v in zip(idx, m2._embed([chunks[i]["text"] for i in idx])):
                    vectors[i] = v
        m2.write_file_chunks(db, str(path), path.stat().st_mtime, chunks,
                             np.vstack(vectors) if vectors else
                             np.empty((0, m2.EMBED_DIMS), dtype=np.float32))
        stats["files"] += 1
        stats["chunks"] += len(chunks)
        stats["embedded"] += len(need)
        stats["reused"] += len(chunks) - len(need)
        if need:
            print(f"  {path.name}: {len(chunks)} chunks, {len(need)} embedded", flush=True)

    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    for suffix in ("-wal", "-shm"):
        Path(str(tmp) + suffix).unlink(missing_ok=True)
    tmp.rename(out)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qdrant", type=Path,
                    help="a copy of qdrant-mem2's storage.sqlite (default: copy the live one)")
    ap.add_argument("--out", type=Path, default=m2.DB_PATH)
    a = ap.parse_args()

    if a.qdrant:
        storage = a.qdrant
    else:
        if not STORAGE.exists():
            print(f"no legacy store at {STORAGE}; nothing to migrate "
                  f"(mem2 builds {m2.DB_PATH.name} from the logs on first use)")
            return 0
        tmpdir = Path(tempfile.mkdtemp(prefix="mem2-migrate-"))
        storage = tmpdir / "storage.sqlite"
        shutil.copy2(STORAGE, storage)
    stats = migrate(storage, a.out)
    print(f"done -> {a.out}: {stats}")
    if not a.qdrant:
        shutil.rmtree(storage.parent, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
