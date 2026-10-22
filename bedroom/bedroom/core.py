"""Bedroom M1 core: one-shot episode loop, scenario loader, strict binary validator, trace log.

Tree-agnostic per spec v0.1 sections 3, 5, 7. Tool names and arg keys are
Domicile's real registry names (local_tools._SCHEMAS), re-baselined
2026-10-18 against the release tree -- see bedroom-rebaseline-oct18.md in
the goal files for the mapping and the known fidelity deltas.
"""
import fnmatch, hashlib, json, os, tempfile
from pathlib import Path

def load_scenario(path):
    s = json.loads(Path(path).read_text())
    assert s["expected"]["tool"], s["id"]
    return s

def load_dir(d):
    return [load_scenario(p) for p in sorted(Path(d).glob("*.json"))]

class Sandbox:
    """Per-episode filesystem sandbox seeded from scenario setup.files."""
    def __init__(self, files=None):
        self.root = Path(tempfile.mkdtemp(prefix="bedroom-"))
        for i, (rel, content) in enumerate((files or {}).items()):
            p = (self.root / rel.lstrip("/")).resolve()
            assert str(p).startswith(str(self.root)), f"path escape: {rel}"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            # deterministic mtimes (newest-first Glob must not depend on fs tick): later files are newer
            t = 1_700_000_000 + i
            os.utime(p, (t, t))
    def resolve(self, rel):
        p = (self.root / str(rel).lstrip("/")).resolve()
        assert str(p).startswith(str(self.root)), f"path escape: {rel}"
        return p
    def cleanup(self):
        import shutil; shutil.rmtree(self.root, ignore_errors=True)

# Dollhouse tool implementations on the sandbox, mirroring Domicile's real
# local_tools registry. Fidelity deltas vs the real tools are listed in
# bedroom-rebaseline-oct18.md; the headline ones: Read returns raw text
# (the real tool returns numbered lines), Grep matches the pattern
# literally (real is ripgrep regex), Glob returns sorted paths (real is
# newest-first), Write does not enforce the real read-before-overwrite
# precondition in one-shot M1 episodes.
def run_fs_tool(sb, tool, args):
    if tool == "Read":
        return sb.resolve(args["path"]).read_text()
    if tool == "Glob":
        base = sb.resolve(args.get("path", "/"))
        pat = args.get("pattern", "*")
        return ["/" + str(p.relative_to(sb.root)) for p in sorted(base.rglob("*"))
                if p.is_file() and fnmatch.fnmatch(p.name, pat)]
    if tool == "Grep":
        needle = args["pattern"]
        base = sb.resolve(args.get("path", "/"))
        mode = args.get("output_mode", "content")
        ic = bool(args.get("ignore_case"))
        files, lines, count = [], [], 0
        for p in sorted(base.rglob("*")):
            if not p.is_file(): continue
            if args.get("glob") and not fnmatch.fnmatch(p.name, args["glob"]): continue
            text = p.read_text(errors="ignore")
            hay = text.lower() if ic else text
            if (needle.lower() if ic else needle) in hay:
                rel = "/" + str(p.relative_to(sb.root))
                files.append(rel)
                for ln in text.splitlines():
                    if (needle.lower() if ic else needle) in (ln.lower() if ic else ln):
                        lines.append(f"{rel}: {ln}"); count += 1
        if mode == "files_with_matches": return files
        if mode == "count": return count
        return lines
    if tool == "Write":
        p = sb.resolve(args["path"]); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"])
        return "ok"
    if tool == "Edit":
        p = sb.resolve(args["path"]); t = p.read_text()
        n = t.count(args["old"])
        assert n > 0, "edit target not found"
        assert n == 1 or args.get("replace_all"), "edit target not unique"
        p.write_text(t.replace(args["old"], args["new"], -1 if args.get("replace_all") else 1))
        return "ok"
    raise ValueError(f"unknown tool {tool}")

def validate(scenario, action):
    """Strict binary per spec section 7: exact tool + declared args_match, else 0."""
    if not isinstance(action, dict) or "tool" not in action:
        return 0, "not a single tool call"
    exp = scenario["expected"]
    if action["tool"] in exp.get("forbidden_tools", []):
        return 0, f"forbidden tool {action['tool']}"
    if action["tool"] != exp["tool"]:
        return 0, f"wrong tool: got {action['tool']}, want {exp['tool']}"
    args = action.get("args", {})
    for key, rule in exp.get("args_match", {}).items():
        val = str(args.get(key, ""))
        if "contains" in rule and not all(x in val for x in rule["contains"]):
            return 0, f"args.{key} missing {rule['contains']}"
        if "equals" in rule and val != rule["equals"]:
            return 0, f"args.{key} != {rule['equals']}"
        if "one_of" in rule and val not in rule["one_of"]:
            return 0, f"args.{key} not in {rule['one_of']}"
    # Post-condition check when declared: file must end in the expected state
    return 1, "ok"

def check_postcondition(sb, scenario):
    pc = scenario["expected"].get("postcondition")
    if not pc: return True
    p = sb.resolve(pc["path"])
    if not p.exists(): return False
    t = p.read_text()
    if "contains" in pc:
        wants = pc["contains"] if isinstance(pc["contains"], list) else [pc["contains"]]
        if not all(w in t for w in wants): return False
    if "not_contains" in pc and pc["not_contains"] in t: return False
    return True

def run_episode(scenario, student_fn, trace_path=None):
    """student_fn(observation) -> action dict. One shot, then done."""
    sb = Sandbox(scenario.get("setup", {}).get("files"))
    obs = {"situation": scenario["situation"], "scenario_id": scenario["id"],
           "tools": ["Read", "Write", "Edit", "Glob", "Grep"],
           "files_root": "/"}
    obs_hash = hashlib.sha256(json.dumps(obs, sort_keys=True).encode()).hexdigest()[:16]
    try:
        action = student_fn(obs)
        reward, reason = validate(scenario, action)
        if reward:
            try: run_fs_tool(sb, action["tool"], action.get("args", {}))
            except Exception as e: reward, reason = 0, f"tool error: {e}"
            if reward and not check_postcondition(sb, scenario):
                reward, reason = 0, "postcondition failed"
    finally:
        trace = {"id": scenario["id"], "obs_hash": obs_hash, "action": action if 'action' in dir() else None,
                 "reward": reward if 'reward' in dir() else 0, "reason": reason if 'reason' in dir() else "error"}
        if trace_path:
            with open(trace_path, "a") as f: f.write(json.dumps(trace) + "\n")
        sb.cleanup()
    return trace

def run_eval(scenarios, student_fn, trace_path=None):
    results = [run_episode(s, student_fn, trace_path) for s in scenarios]
    by_family = {}
    for s, r in zip(scenarios, results):
        by_family.setdefault(s["family"], []).append(r["reward"])
    return {"overall": sum(r["reward"] for r in results) / max(1, len(results)),
            "by_family": {k: sum(v)/len(v) for k, v in by_family.items()},
            "n": len(results), "results": results}
