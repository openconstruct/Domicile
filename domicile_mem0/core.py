# MemoryCore: the ops _dispatch_to_core maps onto. Compatibility surface
# is these methods and their result shapes, not any server.
import json
import time
from collections import deque

import numpy as np

from . import extract as extract_mod
from .fts import _fts_match_query
from .scoring import HINT_THRESHOLD, rank_score, rank_scores

RRF_K = 60  # Domicile's fusion damping (mem0_core.RRF_K)
from .store import Store, text_hash
from .vectors import cosine_scores, dequantize_int8, from_blob, normalize, quantize_int8, to_blob


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
        self._cache_matrix = None

    def close(self):
        self.store.close()

    # -- internals -----------------------------------------------------

    def _embed_blob(self, text):
        vec = normalize(self.embedder.embed(text))
        codes, scale = quantize_int8(vec)
        return to_blob(vec), codes, scale, len(vec)

    def _snapshot(self):
        # Rebuilt only when a write bumps the store version (read-mostly store).
        with self.store.lock:
            if self._cache_version != self.store.version:
                rows = self.store.all_active()
                self._cache_rows = rows
                self._cache_index = {r["id"]: i for i, r in enumerate(rows)}
                self._cache_matrix = np.vstack([
                    (from_blob(r["embedding"]) if r["embedding"] else
                     dequantize_int8(r["embedding_int8"], r["embedding_scale"]))
                    for r in rows]).astype(np.float32) if rows else None
                # Rook: float32 when stored -- int8 shifted cosines ~0.002 vs
                # legacy, enough to reorder near-ties and move hint thresholds.
                self._cache_cols = {
                    "user_id": np.array([r["user_id"] for r in rows], dtype=object),
                    "agent_id": np.array([r["agent_id"] for r in rows], dtype=object),
                    "run_id": np.array([r["run_id"] for r in rows], dtype=object),
                    "salience": np.array([r["salience"] for r in rows], dtype=np.float32),
                    "created_at": np.array([r["created_at"] if r["created_at"] else np.nan
                                            for r in rows], dtype=np.float64),
                    "use_count": np.array([r["use_count"] for r in rows], dtype=np.float32),
                }
                self._cache_version = self.store.version
            return self._cache_rows, self._cache_index, self._cache_matrix, self._cache_cols

    def _ranked(self, query_text, user_id=None, agent_id=None, run_id=None,
                metadata_filter=None, pool=None, hybrid=True):
        # Rook R2: filters are a mask applied AFTER the matmul. Vectorized:
        # one matmul, one boolean mask, argsort -- no per-row Python loop.
        # hybrid=True fuses FTS5 keyword hits by reciprocal rank (RRF_K=60,
        # Domicile's existing fusion) -- the keyword signal mem0's BM25 used
        # to carry. Returns [(row, cosine, score)], best first.
        query = normalize(self.embedder.embed(query_text))
        rows, index, matrix, cols = self._snapshot()
        if not rows:
            return []
        sims = cosine_scores(matrix, query)
        mask = np.ones(len(rows), dtype=bool)
        for col, val in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)):
            if val is not None:
                mask &= cols[col] == val
        if metadata_filter:
            for i in np.flatnonzero(mask):
                meta = json.loads(rows[i]["metadata"])
                if any(meta.get(k) != v for k, v in metadata_filter.items()):
                    mask[i] = False
        scores = rank_scores(sims, cols["salience"], cols["created_at"], cols["use_count"])
        scores = np.where(mask, scores, -np.inf)
        order = [int(i) for i in np.argsort(-scores) if mask[i]]
        if not hybrid:
            return [(rows[i], float(sims[i]), float(scores[i])) for i in order]
        pool = pool or 30
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
            if h in existing_hashes or h in seen:
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
        out = []
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
