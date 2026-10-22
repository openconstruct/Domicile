"""Freeze the EVAL pool: hash every eval scenario + the eval oracles
into bench/eval-freeze.json. A frozen bench is the Sec. 8 proof
instrument -- any later change to an eval file makes the hashes
mismatch, which is detectable, and a changed bench is by definition
a new bench version (new baseline required). Refuses to overwrite
an existing freeze; --verify re-checks a freeze instead.
"""
import hashlib, json, sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def collect():
    oracles = json.loads((BENCH_DIR / "oracles.json").read_text())["eval"]
    files = {}
    for p in sorted((BENCH_DIR / "eval").glob("*.json")):
        files[p.name] = _sha(p)
    blob = json.dumps({"files": files, "oracles": oracles}, sort_keys=True).encode()
    return {"version": "bedroom-bench-v1", "n_eval": len(files),
            "files": files, "eval_oracles_sha256": _sha_bytes(oracles),
            "aggregate_sha256": hashlib.sha256(blob).hexdigest()}


def _sha_bytes(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def main():
    out = BENCH_DIR / "eval-freeze.json"
    if "--verify" in sys.argv:
        frozen = json.loads(out.read_text())
        now = collect()
        ok = now["aggregate_sha256"] == frozen["aggregate_sha256"]
        print("FREEZE VERIFY", "OK" if ok else "MISMATCH -- bench has changed")
        return 0 if ok else 1
    if out.exists():
        print("refusing to overwrite an existing freeze; a changed bench is a new version")
        return 1
    frozen = collect()
    out.write_text(json.dumps(frozen, indent=1))
    print(f"froze {frozen['n_eval']} eval scenarios, aggregate {frozen['aggregate_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
