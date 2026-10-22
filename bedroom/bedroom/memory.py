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
import hashlib, json, math, re
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
    """Seeded mem0 stand-in: ordered facts with stable ids."""

    def __init__(self, facts=None):
        self.facts = [{"id": f"m-{i + 1:03d}", "text": t} for i, t in enumerate(facts or [])]
        self._next = len(self.facts) + 1

    def search(self, query, limit=5):
        idx = _rank(query, [f["text"] for f in self.facts], int(limit))
        if not idx:
            return "no memories found"
        return "\n".join(f'{self.facts[i]["id"]}: {self.facts[i]["text"]}' for i in idx)

    def list(self, limit=20):
        rows = self.facts[-int(limit):][::-1]
        return "\n".join(f'{f["id"]}: {f["text"]}' for f in rows) or "store is empty"

    def add(self, text):
        fid = f"m-{self._next:03d}"
        self._next += 1
        self.facts.append({"id": fid, "text": text})
        return f"stored {fid}"

    def delete(self, memory_id):
        hits = [f for f in self.facts if f["id"] == memory_id or f["id"].startswith(memory_id)]
        if len(hits) != 1:
            return f"error: {'ambiguous id' if hits else 'no such memory'}"
        self.facts.remove(hits[0])
        return f'deleted {hits[0]["id"]}'


class TranscriptStore:
    """Seeded mem2 stand-in: dated conversation chunks."""

    def __init__(self, chunks=None):
        self.chunks = chunks or []  # {source_file, chunk_index, date, text}

    def search(self, query, limit=5, date_from=None, date_to=None):
        pool = [c for c in self.chunks
                if (not date_from or c["date"] >= date_from)
                and (not date_to or c["date"] <= date_to)]
        idx = _rank(query, [c["text"] for c in pool], int(limit))
        if not idx:
            return "no conversations found"
        out = []
        for i in idx:
            c = pool[i]
            out.append(f'{c["source_file"]}#{c["chunk_index"]} ({c["date"]}): {c["text"][:80]}')
        return "\n".join(out)

    def list_dates(self):
        return "\n".join(sorted({c["date"] for c in self.chunks})) or "no logs"

    def get_chunk(self, source_file, chunk_index):
        for c in self.chunks:
            if c["source_file"] == source_file and c["chunk_index"] == int(chunk_index):
                return c["text"]
        return "error: no such chunk"


def run_memory_tool(store, tstore, sb, tool, args):
    name = tool.split("__")[-1]
    if name == "memory_search":
        return store.search(args["query"], args.get("limit", 5))
    if name == "memory_add":
        return store.add(args["text"])
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
