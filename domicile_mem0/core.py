# MemoryCore: the ops _dispatch_to_core maps onto. Compatibility surface
# is these methods and their result shapes, not any server.
import json
import time
from collections import deque

import numpy as np

from . import extract as extract_mod
from .fts import _fts_match_query
from .retrieval import RRF_K
from .scoring import HINT_THRESHOLD, rank_scores
from .store import Store, text_hash
from . import vectors as V
from .vectors import Scanner, from_blob, normalize, quantize_int8, to_blob


class MemoryCore:
    def __init__(self, db_path, embedder, llm=None, custom_instructions=None):
        self.store = Store(db_path)
        self.embedder = embedder
        name = getattr(embedder, "name", None)  # test embedders carry no name: unchecked
        if name:
            self.store.check_embedder(name)
        self.llm = llm
        self.custom_instructions = custom_instructions
        self._cache_index = {}
        self._recent = {}  # upstream keeps last 10 msgs per scope for the process lifetime
        self._cache_cols = {}
        self._cache_version = -1
        self._cache_rows = []
        self._cache_scan = Scanner()
        self.rebuilds = 0  # full cache reloads (tests / diagnostics)

    def close(self):
        self.store.close()

    # -- internals -----------------------------------------------------

    def _embed_blob(self, text):
        vec = normalize(self.embedder.embed(text))
        codes, scale = quantize_int8(vec)
        return to_blob(vec), codes, scale, len(vec)

    _BLOBS = ("embedding", "embedding_int8")

    @staticmethod
    def _entry(row):
        """(float32 vec | None, int8 codes | None, scale) for the Scanner."""
        return (from_blob(row["embedding"]) if row.get("embedding") else None,
                np.frombuffer(row["embedding_int8"], dtype=np.int8)
                if row.get("embedding_int8") else None,
                row.get("embedding_scale") or 1.0)

    @staticmethod
    def _col_values(row):
        return {"user_id": row["user_id"], "agent_id": row["agent_id"],
                "run_id": row["run_id"], "salience": row["salience"],
                "created_at": row["created_at"] if row["created_at"] else np.nan,
                "use_count": row["use_count"]}

    _COL_DTYPES = {"user_id": object, "agent_id": object, "run_id": object,
                   "salience": np.float32, "created_at": np.float64,
                   "use_count": np.float32}

    def _lean(self, row):
        # The Scanner holds the vectors; don't pin ~2 KB of blobs per fact too.
        return {k: v for k, v in row.items() if k not in self._BLOBS}

    def _rebuild(self):
        tiered = self.store.count_active() >= V.EXACT_MAX
        rows = self.store.all_active(float_vectors=not tiered)
        self._cache_scan = Scanner.build([self._entry(r) for r in rows], tiered)
        self._cache_cols = {
            name: np.array([self._col_values(r)[name] for r in rows], dtype=dt)
            for name, dt in self._COL_DTYPES.items()}
        self._cache_rows = [self._lean(r) for r in rows]
        self._cache_index = {r["id"]: i for i, r in enumerate(self._cache_rows)}
        self.rebuilds += 1

    def _apply(self, memory_id):
        """Bring one fact's cache entry up to date. Copy-on-write throughout:
        a reader holding the previous snapshot keeps a consistent one."""
        row = self.store.get(memory_id)
        active = row is not None and row["superseded_by"] is None
        i = self._cache_index.get(memory_id)
        if i is None and not active:
            return
        rows, cols, scan = self._cache_rows, self._cache_cols, self._cache_scan
        if not active:                                   # deleted or superseded
            self._cache_rows = rows[:i] + rows[i + 1:]
            self._cache_cols = {n: np.delete(a, i) for n, a in cols.items()}
            self._cache_scan = scan.deleted(i)
            self._cache_index = {r["id"]: k for k, r in enumerate(self._cache_rows)}
            return
        vals = self._col_values(row)
        if i is None:                                    # new fact
            self._cache_rows = rows + [self._lean(row)]
            self._cache_cols = {n: np.append(a, np.array([vals[n]], dtype=a.dtype))
                                for n, a in cols.items()}
            self._cache_scan = scan.appended(self._entry(row))
            self._cache_index = {**self._cache_index, memory_id: len(rows)}
            return
        new_rows = list(rows)                            # text or metadata changed
        new_rows[i] = self._lean(row)
        new_cols = {}
        for n, a in cols.items():
            a = a.copy()
            a[i] = vals[n]
            new_cols[n] = a
        self._cache_rows, self._cache_cols = new_rows, new_cols
        self._cache_scan = scan.replaced(i, self._entry(row))

    def _snapshot(self):
        """(rows, index, scanner, cols) for the active facts.

        A write used to mean reloading every fact on the next read (250 ms at
        14k facts, and linear in store size). Now the store's journal says
        which ids changed and only those are applied; a full reload happens
        once at start-up, or if the journal has been outrun.
        """
        with self.store.lock:
            if self._cache_version != self.store.version:
                changed = self.store.changed_since(self._cache_version)
                if (changed is None or self._cache_version < 0
                        or len(changed) > self.MAX_INCREMENTAL):
                    self._rebuild()
                else:
                    for mid in dict.fromkeys(changed):
                        self._apply(mid)
                self._cache_version = self.store.version
            return self._cache_rows, self._cache_index, self._cache_scan, self._cache_cols

    MAX_INCREMENTAL = 64

    def _exact_vectors(self, rows):
        """Scanner callback: true float32 vectors for a few row indices."""
        def fetch(idx):
            ids = [rows[int(i)]["id"] for i in idx]
            blobs = self.store.float_vectors(ids)
            return np.vstack([from_blob(blobs[m]) for m in ids])
        return fetch

    def _ranked(self, query_text, user_id=None, agent_id=None, run_id=None,
                metadata_filter=None, pool=None, hybrid=True):
        # Filters are a mask applied AFTER the matmul. Vectorized:
        # one matmul, one boolean mask, argsort -- no per-row Python loop.
        # hybrid=True fuses FTS5 keyword hits by reciprocal rank (RRF_K=60,
        # Domicile's existing fusion) -- the keyword signal mem0's BM25 used
        # to carry. Returns [(row, cosine, score)], best first.
        query = normalize(self.embedder.embed(query_text))
        rows, index, scan, cols = self._snapshot()
        if not rows:
            return []
        mask = np.ones(len(rows), dtype=bool)
        for col, val in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)):
            if val is not None:
                mask &= cols[col] == val
        if metadata_filter:
            for i in np.flatnonzero(mask):
                meta = json.loads(rows[i]["metadata"])
                if any(meta.get(k) != v for k, v in metadata_filter.items()):
                    mask[i] = False
        pool = pool or 30
        sims = scan.sims(query, mask=mask, k=pool, exact=self._exact_vectors(rows))
        with np.errstate(invalid="ignore"):
            scores = rank_scores(sims, cols["salience"], cols["created_at"], cols["use_count"])
        scores = np.where(mask & np.isfinite(sims), scores, -np.inf)
        order = [int(i) for i in np.argsort(-scores) if np.isfinite(scores[i])]
        if not hybrid:
            return [(rows[i], float(sims[i]), float(scores[i])) for i in order]
        sem = order[:pool]
        match = _fts_match_query(query_text)
        kw = []
        if match:
            for mid in self.store.keyword(match, pool * 3):
                i = index.get(mid)
                if i is not None and mask[i]:
                    kw.append(i)
                if len(kw) >= pool:
                    break
        fused = {}
        for hits in (sem, kw):
            for rank, i in enumerate(hits):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank)
        best = sorted(fused, key=lambda i: -fused[i])
        # Tiered scan only: a keyword hit outside the candidate set has no
        # cosine yet. There are at most `pool` of them; score them exactly.
        blind = [i for i in best if not np.isfinite(sims[i])]
        if blind:
            sims = sims.copy()
            sims[blind] = self._exact_vectors(rows)(blind) @ query
            scores = scores.copy()
            scores[blind] = rank_scores(sims[blind], cols["salience"][blind],
                                        cols["created_at"][blind], cols["use_count"][blind])
        # score stays the calibrated semantic value; RRF only orders.
        return [(rows[i], float(sims[i]), float(scores[i])) for i in best]

    @staticmethod
    def _public(row, similarity=None, score=None):
        out = {
            "id": row["id"],
            "memory": row["text"],
            "user_id": row["user_id"],
            "agent_id": row["agent_id"],
            "run_id": row["run_id"],
            "metadata": json.loads(row["metadata"]),
            "hash": row["hash"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "use_count": row["use_count"],
            "salience": row["salience"],
        }
        if row["superseded_by"] is not None:
            out["superseded_by"] = row["superseded_by"]
            out["supersede_reason"] = row["supersede_reason"]
        if similarity is not None:
            out["similarity"] = similarity
        if score is not None:
            out["score"] = score
        return out

    # -- ops -----------------------------------------------------------

    def add_verbatim(self, text, user_id=None, agent_id=None, run_id=None,
                     metadata=None, salience=0.5):
        blob, codes, scale, dim = self._embed_blob(text)
        mid = self.store.insert(text, blob, codes, scale, dim, user_id,
                                agent_id, run_id, metadata, salience)
        return {"results": [{"id": mid, "memory": text, "event": "ADD"}]}

    def add(self, messages, user_id=None, agent_id=None, run_id=None,
            metadata=None, infer=True, agent_scoped=False):
        # Mirrors the INSTALLED mem0 2.0.19 add (main.py V3 pipeline): one
        # ADD-only extraction call that sees the top-25 existing memories under
        # integer ids, then md5 hash dedup. Extraction never updates or deletes.
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        if not infer:
            return self.add_verbatim(extract_mod.parse_messages(messages),
                                     user_id, agent_id, run_id, metadata)
        if self.llm is None:
            raise RuntimeError("add(infer=True) requires an llm callable")
        parsed = extract_mod.parse_messages(messages)
        existing = [row for row, _s, _c in self._ranked(
            parsed, user_id, agent_id, run_id, hybrid=False)[:25]]
        scope = f"{user_id}|{agent_id}|{run_id}"
        recent = self._recent.setdefault(scope, deque(maxlen=10))
        found = extract_mod.extract_additive(
            parsed, existing, self.llm, last_k_messages=list(recent),
            agent_scoped=agent_scoped or (bool(agent_id) and not user_id),
            custom_instructions=self.custom_instructions)
        recent.extend(m for m in messages if m.get("content") is not None)
        existing_hashes = {r["hash"] for r in existing if r.get("hash")}
        seen, results = set(), []
        for mem in found:
            text = (mem.get("text") or "").strip()
            if not text:
                continue
            h = text_hash(text)
            # The top-25 are ranked against the whole conversation, so an exact
            # restatement can sit outside them: check the store too.
            if (h in existing_hashes or h in seen
                    or self.store.has_active_hash(h, user_id, agent_id, run_id)):
                continue
            seen.add(h)
            meta = dict(metadata or {})
            if mem.get("attributed_to"):
                meta["attributed_to"] = mem["attributed_to"]
            blob, codes, scale, dim = self._embed_blob(text)
            mid = self.store.insert(text, blob, codes, scale, dim,
                                    user_id, agent_id, run_id, meta)
            results.append({"id": mid, "memory": text, "event": "ADD"})
        return {"results": results}

    def search(self, query, user_id=None, agent_id=None, run_id=None,
               metadata_filter=None, limit=10, threshold=None):
        ranked = self._ranked(query, user_id, agent_id, run_id, metadata_filter)
        limit = 10 if limit is None else max(0, int(limit))
        out = []
        if limit == 0:
            return {"results": out}
        for row, sim, score in ranked:
            if threshold is not None and score < threshold:
                continue
            out.append(self._public(row, similarity=sim, score=score))
            if len(out) >= limit:
                break
        if out:
            self.store.increment_use([r["id"] for r in out])
        return {"results": out}

    def hint(self, query, user_id=None, agent_id=None, run_id=None, limit=5):
        found = self.search(query, user_id, agent_id, run_id,
                            limit=limit, threshold=HINT_THRESHOLD)
        return {"results": [
            {"id": r["id"], "memory": r["memory"], "score": r["score"]}
            for r in found["results"]
        ]}

    def get(self, memory_id):
        row = self.store.get(memory_id)
        return self._public(row) if row else None

    def list_facts(self, user_id=None, agent_id=None, run_id=None,
                   include_superseded=False, limit=100, offset=0):
        rows = self.store.list_rows(user_id, agent_id, run_id,
                                    include_superseded, limit, offset)
        return {"results": [self._public(r) for r in rows]}

    list = list_facts

    def update(self, memory_id, text=None, metadata=None):
        row = self.store.get(memory_id)
        if row is None:
            return None
        if text is not None:
            blob, codes, scale, dim = self._embed_blob(text)
            self.store.update_text(memory_id, text, blob, codes, scale, dim)
        if metadata is not None:
            merged = {**json.loads(row["metadata"]), **metadata}
            self.store.update_metadata(memory_id, merged)
        return self.get(memory_id)

    def delete(self, memory_id):
        # Explicit op only. Chat mode never exposes this to the model (D1).
        return {"deleted": self.store.delete(memory_id), "id": memory_id}

    def mark_superseded(self, memory_id, superseded_by, reason):
        self.store.mark_superseded(memory_id, superseded_by, reason)
        return self.get(memory_id)
