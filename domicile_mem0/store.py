# The single SQLite base. Facts, vectors, and Domicile's tracking payload
# (created_at/updated_at/hash, supersession chain with reason,
# use_count/salience) all live on the memory row — the chain is the history.
import hashlib
import json
import sqlite3
import threading
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    user_id TEXT,
    agent_id TEXT,
    run_id TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    embedding BLOB,
    embedding_int8 BLOB,
    embedding_scale REAL,
    dim INTEGER NOT NULL DEFAULT 0,
    hash TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    superseded_by TEXT,
    supersede_reason TEXT,
    use_count INTEGER NOT NULL DEFAULT 0,
    salience REAL NOT NULL DEFAULT 0.5
);
CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_mem_agent ON memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_mem_run ON memories(run_id);
-- Keyword side of hybrid search. External-content FTS5 kept in sync by
-- triggers, so it cannot drift from the facts (the old mirror could).
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text, content='memories', content_rowid='rowid');
-- The index holds ACTIVE facts only: a superseded fact left in it would still
-- count toward bm25's corpus statistics and shift every keyword ranking.
-- Because of that, never run FTS5 'rebuild' or 'integrity-check' here: both
-- compare against the whole content table, superseded rows included.
DROP TRIGGER IF EXISTS mem_fts_ai;
DROP TRIGGER IF EXISTS mem_fts_ad;
DROP TRIGGER IF EXISTS mem_fts_au;
CREATE TRIGGER IF NOT EXISTS mem_fts_ai2 AFTER INSERT ON memories
WHEN new.superseded_by IS NULL BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text); END;
CREATE TRIGGER IF NOT EXISTS mem_fts_ad2 AFTER DELETE ON memories
WHEN old.superseded_by IS NULL BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text); END;
CREATE TRIGGER IF NOT EXISTS mem_fts_au2 AFTER UPDATE OF text ON memories
WHEN old.superseded_by IS NULL AND new.superseded_by IS NULL BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text); END;
CREATE TRIGGER IF NOT EXISTS mem_fts_sup2 AFTER UPDATE OF superseded_by ON memories
WHEN old.superseded_by IS NULL AND new.superseded_by IS NOT NULL BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text); END;
CREATE TRIGGER IF NOT EXISTS mem_fts_uns2 AFTER UPDATE OF superseded_by ON memories
WHEN old.superseded_by IS NOT NULL AND new.superseded_by IS NULL BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text); END;
"""

COLUMNS = (
    "id, text, user_id, agent_id, run_id, metadata, embedding, "
    "embedding_int8, embedding_scale, dim, hash, created_at, updated_at, "
    "superseded_by, supersede_reason, use_count, salience"
)


def text_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class Store:
    def __init__(self, db_path):
        # Domicile's IPC runs every op via asyncio.to_thread: one connection,
        # shared across threads, serialized by this lock.
        self.lock = threading.RLock()
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.version = 0  # bumped on any vector-affecting write; drives the scan cache
        # WAL + busy timeout: several readers, one deliberate writer (phase-2 note).
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript(SCHEMA)
        self._migrate_fts_active_only()
        self.db.commit()

    def _migrate_fts_active_only(self):
        # user_version 2 = the keyword index holds active facts only. Older
        # databases indexed every row, so drop the superseded ones once.
        if self.db.execute("PRAGMA user_version").fetchone()[0] >= 2:
            return
        for r in self.db.execute(
                "SELECT rowid, text FROM memories WHERE superseded_by IS NOT NULL").fetchall():
            self.db.execute(
                "INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', ?, ?)",
                (r[0], r[1]))
        self.db.execute("PRAGMA user_version=2")

    def close(self):
        with self.lock:
            self.db.close()

    # -- embedder stamp -------------------------------------------------
    # Every vector in the store must come from one model. Dim cannot tell
    # (bge-small and MiniLM-L6 are both 384-d), so the model name is stamped
    # once per store and checked on every open. Re-embedding rewrites all
    # vectors, then restamps; a crash in between leaves the old stamp, so a
    # half-swapped store fails loudly instead of ranking garbage.
    def get_meta(self, key):
        with self.lock:
            self.db.execute("CREATE TABLE IF NOT EXISTS store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            r = self.db.execute("SELECT value FROM store_meta WHERE key=?", (key,)).fetchone()
            return r[0] if r else None

    def set_meta(self, key, value):
        with self.lock:
            self.db.execute("CREATE TABLE IF NOT EXISTS store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            self.db.execute("INSERT INTO store_meta(key, value) VALUES (?, ?) "
                            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
            self.db.commit()

    def check_embedder(self, name):
        have = self.get_meta("embedder")
        if have == name:
            return
        if have is None:
            n = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            if n == 0:
                self.set_meta("embedder", name)  # fresh store: first embedder wins
                return
            raise RuntimeError(
                f"store has {n} rows but no embedder stamp; refusing to guess. "
                f"Stamp it once: Store(path).set_meta('embedder', '<model that made the vectors>')")
        raise RuntimeError(
            f"embedder mismatch: store vectors were made by {have!r}, this process embeds with {name!r}. "
            f"Re-embed the whole store and restamp, or point at the right model.")

    def insert(self, text, embedding_blob, int8_blob, scale, dim, user_id=None,
               agent_id=None, run_id=None, metadata=None, salience=0.5,
               memory_id=None, created_at=None):
        with self.lock:
            now = created_at or time.time()
            mid = memory_id or str(uuid.uuid4())
            self.db.execute(
                f"INSERT INTO memories ({COLUMNS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (mid, text, user_id, agent_id, run_id,
                 json.dumps(metadata or {}), embedding_blob, int8_blob, scale,
                 dim, text_hash(text), now, now, None, None, 0, salience),
            )
            self.db.commit()
            self.version += 1
            return mid

    def get(self, memory_id):
        with self.lock:
            row = self.db.execute(
                f"SELECT {COLUMNS} FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
            return dict(row) if row else None

    def all_active(self):
        with self.lock:
            rows = self.db.execute(
                f"SELECT {COLUMNS} FROM memories WHERE superseded_by IS NULL"
            ).fetchall()
            return [dict(r) for r in rows]

    def list_rows(self, user_id=None, agent_id=None, run_id=None,
                  include_superseded=False, limit=100, offset=0):
        with self.lock:
            clauses, args = [], []
            if not include_superseded:
                clauses.append("superseded_by IS NULL")
            for col, val in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)):
                if val is not None:
                    clauses.append(f"{col}=?")
                    args.append(val)
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            rows = self.db.execute(
                f"SELECT {COLUMNS} FROM memories{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (*args, limit, offset),
            ).fetchall()
            return [dict(r) for r in rows]

    def update_text(self, memory_id, text, embedding_blob, int8_blob, scale, dim):
        with self.lock:
            self.db.execute(
                "UPDATE memories SET text=?, embedding=?, embedding_int8=?, "
                "embedding_scale=?, dim=?, hash=?, updated_at=? WHERE id=?",
                (text, embedding_blob, int8_blob, scale, dim, text_hash(text),
                 time.time(), memory_id),
            )
            self.db.commit()
            self.version += 1

    def update_metadata(self, memory_id, metadata):
        with self.lock:
            self.db.execute(
                "UPDATE memories SET metadata=?, updated_at=? WHERE id=?",
                (json.dumps(metadata), time.time(), memory_id),
            )
            self.db.commit()

    def mark_superseded(self, memory_id, superseded_by, reason):
        with self.lock:
            self.db.execute(
                "UPDATE memories SET superseded_by=?, supersede_reason=?, updated_at=? WHERE id=?",
                (superseded_by, reason, time.time(), memory_id),
            )
            self.db.commit()
            self.version += 1

    def delete(self, memory_id):
        with self.lock:
            cur = self.db.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            self.db.commit()
            self.version += 1
            return cur.rowcount > 0

    def increment_use(self, memory_ids):
        with self.lock:
            self.db.executemany(
                "UPDATE memories SET use_count=use_count+1 WHERE id=?",
                [(m,) for m in memory_ids],
            )
            self.db.commit()

    def count(self):
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def keyword(self, match, limit):
        """FTS5 bm25 ranking over ACTIVE facts; returns memory ids best-first."""
        with self.lock:
            rows = self.db.execute(
                "SELECT m.id FROM memories_fts f JOIN memories m ON m.rowid=f.rowid "
                "WHERE memories_fts MATCH ? AND m.superseded_by IS NULL "
                "ORDER BY bm25(memories_fts) LIMIT ?", (match, int(limit))).fetchall()
        return [r[0] for r in rows]

    def hashes(self, ids):
        with self.lock:
            q = ",".join("?" * len(ids))
            return {r[0] for r in self.db.execute(
                f"SELECT hash FROM memories WHERE id IN ({q})", list(ids))} if ids else set()
