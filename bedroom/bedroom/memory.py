"""Bedroom M2: memory + meditation episode layer.

Dollhouse implementations of Domicile's real memory surface, re-baselined
2026-10-19 against the release tree:

- mem0 (MCP server "memory"): memory_search / memory_add / memory_list /
  memory_delete. In the resident's tool list these arrive mcp__-prefixed
  (mcp__memory__memory_search etc.), so scenarios use the prefixed names
  as the expected tool -- that is the surface a student will actually see.
- mem2 (MCP server "mem2"): mem2_search / mem2_list_dates / mem2_get_chunk,
  likewise mcp__mem2__-prefixed.
- mem1: curated files with hard token caps (identity 4000, user 3500,
  practices 3000, goals 2500, open 3000, reflections 3000). Edited with
  the ordinary file tools; the cap is the postcondition that matters.

Dollhouse simplifications (documented, deliberate): search is token-overlap
ranking instead of embeddings; memory_add stores verbatim (the real
extractor path reconciles duplicates via an LLM); token counts use the
chars/4 approximation. One-shot episodes with seeded state, as in M1;
scenarios that need ids show a memory listing in the observation, the way
a resident who just ran memory_list would see it.
"""
import difflib, hashlib, json, math, re
from pathlib import Path

from .core import Sandbox, run_fs_tool, validate

MEMORY_TOOLS = ["mcp__memory__memory_search", "mcp__memory__memory_add",
                "mcp__memory__memory_list", "mcp__memory__memory_delete",
                "mcp__mem2__mem2_search", "mcp__mem2__mem2_list_dates",
                "mcp__mem2__mem2_get_chunk"]
MEM1_CAPS = {"identity.md": 4000, "user.md": 3500, "practices.md": 3000,
             "goals.md": 2500, "open.md": 3000, "reflections.md": 3000}


def approx_tokens(text):
    return math.ceil(len(text) / 4)


def _terms(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def _hex8(n):
    """Deterministic 8-char hex memory id, the shape memory_list prints."""
    return f"{0xAEEF0000 + n:08x}"


def _rank(query, texts, limit):
    q = _terms(query)
    scored = []
    for i, t in enumerate(texts):
        overlap = len(q & _terms(t))
        if overlap:
            scored.append((overlap, -i, i))
    scored.sort(reverse=True)
    return [i for _, _, i in scored[:limit]]


class FactStore:
    """Seeded mem0 stand-in: ordered facts with stable 8-char hex ids.

    Reply text mirrors the live server (captured 2026-10-22):
      search        [0.867] 2026-10-22 <text>
      list          <N> memories total, showing <K>:
                    - 2026-10-22 (aeef9cb6) <text>
      add extract   ADD: <stored fact>
      add verbatim  stored verbatim: <first 100 chars>
      delete        deleted: <first 80 chars>
                    no memory with id starting '<id>'
    """

    ID_BASE = 0xAEEF_0000

    def __init__(self, facts=None):
        self.facts = []
        for i, f in enumerate(facts or []):
            if isinstance(f, dict):
                self.facts.append({"id": _hex8(i), "date": f.get("date", ""),
                                   "text": f.get("text", "")})
            else:
                self.facts.append({"id": _hex8(i), "date": "", "text": str(f)})
        self._next = len(self.facts)

    def _line(self, f):
        return f"- {f['date']} ({f['id']}) {f['text']}" if f["date"] else \
            f"- ({f['id']}) {f['text']}"

    def search(self, query, limit=5):
        idx = _rank(query, [f["text"] for f in self.facts], int(limit))
        if not idx:
            return "no memories found"
        q = query.lower()
        out = []
        for i in idx:
            f = self.facts[i]
            sim = difflib.SequenceMatcher(None, q, f["text"].lower()).ratio()
            head = f"[{sim:.3f}] {f['date']} {f['text']}" if f["date"] else f"[{sim:.3f}] {f['text']}"
            out.append(head)
        return "\n".join(out)

    def list(self, limit=20):
        rows = self.facts[-int(limit):][::-1]
        if not rows:
            return "0 memories total"
        return (f"{len(self.facts)} memories total, showing {len(rows)}:\n"
                + "\n".join(self._line(f) for f in rows))

    def add(self, text, extract=True):
        fid = _hex8(self._next)
        self._next += 1
        self.facts.append({"id": fid, "date": "", "text": text})
        if extract:
            return f"ADD: {text}"
        return f"stored verbatim: {text[:100]}"

    def delete(self, memory_id):
        mid = str(memory_id or "")
        hits = [f for f in self.facts if f["id"] == mid or f["id"].startswith(mid)]
        if not hits:
            return f"no memory with id starting '{mid}'"
        if len(hits) > 1:
            raise ValueError(f"id prefix {mid!r} matches {len(hits)} memories; use more characters")
        self.facts.remove(hits[0])
        return f"deleted: {hits[0]['text'][:80]}"


class TranscriptStore:
    """Seeded mem2 stand-in: dated conversation chunks.

    Reply text mirrors the live server (captured 2026-10-22):
      search      [1] via=sem+kw score=0.0350 date=2026-09-16 chat=Main Chat ref=/path.log#19
                  <the chunk's log lines>
      list_dates  2026-10-21  (86,985 bytes)
      get_chunk   date: 2026-09-16
                  chat: Main Chat
                  ---
                  <text>
                  Chunk not found: /nonexistent.log#0
    """

    def __init__(self, chunks=None):
        self.chunks = []
        for c in chunks or []:
            c = dict(c)
            c.setdefault("chat", "Main Chat")
            self.chunks.append(c)

    def search(self, query, limit=5, date_from=None, date_to=None):
        pool = [c for c in self.chunks
                if (not date_from or c["date"] >= date_from)
                and (not date_to or c["date"] <= date_to)]
        idx = _rank(query, [c["text"] for c in pool], int(limit))
        if not idx:
            return "no conversations found"
        q = query.lower()
        out = []
        for n, i in enumerate(idx, 1):
            c = pool[i]
            sim = difflib.SequenceMatcher(None, q, c["text"].lower()).ratio()
            out.append(f"[{n}] via=sem+kw score={sim:.4f} date={c['date']} chat={c['chat']} "
                       f"ref={c['source_file']}#{c['chunk_index']}\n{c['text']}")
        return "\n\n".join(out)

    def list_dates(self):
        dates = sorted({c["date"] for c in self.chunks})
        if not dates:
            return "no logs"
        out = []
        for d in dates:
            size = sum(len(c["text"]) for c in self.chunks if c["date"] == d)
            out.append(f"{d}  ({size:,} bytes)")
        return "\n".join(out)

    def get_chunk(self, source_file, chunk_index):
        for c in self.chunks:
            if c["source_file"] == source_file and c["chunk_index"] == int(chunk_index):
                return f"date: {c['date']}\nchat: {c['chat']}\n---\n{c['text']}"
        return f"Chunk not found: {source_file}#{chunk_index}"


def run_memory_tool(store, tstore, sb, tool, args):
    name = tool.split("__")[-1]
    if name == "memory_search":
        return store.search(args["query"], args.get("limit", 5))
    if name == "memory_add":
        return store.add(args["text"], args.get("extract", True))
    if name == "memory_list":
        return store.list(args.get("limit", 20))
    if name == "memory_delete":
        return store.delete(args["memory_id"])
    if name == "mem2_search":
        return tstore.search(args["query"], args.get("limit", 5),
                             args.get("date_from"), args.get("date_to"))
    if name == "mem2_list_dates":
        return tstore.list_dates()
    if name == "mem2_get_chunk":
        return tstore.get_chunk(args["source_file"], args["chunk_index"])
    from .render import render_fs_call
    return render_fs_call(sb, tool, args)  # mem1 scenarios use the file tools


def check_mem_postconditions(store, sb, scenario):
    """Returns True when every declared postcondition holds."""
    for pc in scenario["expected"].get("postconditions", []):
        kind = pc["type"]
        if kind == "store_contains":
            if not any(pc["text"] in f["text"] for f in store.facts):
                return False
        elif kind == "store_not_contains":
            if any(pc["text"] in f["text"] for f in store.facts):
                return False
        elif kind == "store_count":
            if len(store.facts) != pc["equals"]:
                return False
        elif kind == "file_contains":
            p = sb.resolve(pc["path"])
            if not p.exists() or pc["text"] not in p.read_text():
                return False
        elif kind == "file_max_tokens":
            p = sb.resolve(pc["path"])
            if not p.exists() or approx_tokens(p.read_text()) > pc["max"]:
                return False
        else:
            raise ValueError(f"unknown postcondition {kind}")
    return True


def run_episode_mem(scenario, student_fn, trace_path=None):
    setup = scenario.get("setup", {})
    store = FactStore(setup.get("facts"))
    tstore = TranscriptStore(setup.get("chunks"))
    sb = Sandbox(setup.get("files"))
    # mem1 files are always in context in the house -- identity/user/practices/
    # goals/open/reflections are injected every session -- so the resident has
    # effectively read them already. Marking them read here satisfies the real
    # read-before-edit rule instead of training around it.
    from .render import _mark_read
    for rel in (setup.get("files") or {}):
        p = sb.resolve(rel)
        if p.exists():
            _mark_read(sb, p)
    obs = {"situation": scenario["situation"], "scenario_id": scenario["id"],
           "tools": MEMORY_TOOLS + ["Read", "Write", "Edit"],
           "files_root": "/"}
    if setup.get("show_listing"):
        obs["memory_listing"] = store.list()
    obs_hash = hashlib.sha256(json.dumps(obs, sort_keys=True).encode()).hexdigest()[:16]
    action, reward, reason = None, 0, "error"
    try:
        action = student_fn(obs)
        reward, reason = validate(scenario, action)
        if reward:
            try:
                run_memory_tool(store, tstore, sb, action["tool"], action.get("args", {}))
            except Exception as e:
                reward, reason = 0, f"tool error: {e}"
            if reward and not check_mem_postconditions(store, sb, scenario):
                reward, reason = 0, "postcondition failed"
    finally:
        trace = {"id": scenario["id"], "obs_hash": obs_hash, "action": action,
                 "reward": reward, "reason": reason}
        if trace_path:
            with open(trace_path, "a") as f:
                f.write(json.dumps(trace) + "\n")
        sb.cleanup()
    return trace


def load_mem_dir(d):
    from .core import load_scenario
    return [load_scenario(p) for p in sorted(Path(d).glob("*.json"))]


def run_eval_mem(scenarios, student_fn, trace_path=None):
    results = [run_episode_mem(s, student_fn, trace_path) for s in scenarios]
    by_family = {}
    for s, r in zip(scenarios, results):
        by_family.setdefault(s["family"], []).append(r["reward"])
    return {"overall": sum(r["reward"] for r in results) / max(1, len(results)),
            "by_family": {k: sum(v) / len(v) for k, v in by_family.items()},
            "n": len(results), "results": results}
