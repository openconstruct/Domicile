"""M3 smoke: oracle walking the graded web road per scenario must score
8/8; a lazy student that answers immediately, a search-only student that
answers from the top snippet, and a one-shot student must all score 0 --
every answer lives behind a fetch, a store, or a second source."""
import sys
sys.path.insert(0, __file__.rsplit("/tests/", 1)[0])
from bedroom.multi import load_multi_dir, run_eval_multi

base = __file__.rsplit("/tests/", 1)[0]
scenarios = load_multi_dir(base + "/scenarios/web")

WS = "WebSearch"
WF = "WebFetch"
M2S = "mcp__mem2__mem2_search"
MADD = "mcp__memory__memory_add"
MDEL = "mcp__memory__memory_delete"

ORACLE = {
    "web-0001": [
        {"tool": WS, "args": {"query": "Bramble default retention window"}},
        {"tool": WF, "args": {"url": "https://docs.bramble.example/config"}},
        {"text": "The default retention window is 30 days."},
    ],
    "web-0002": [
        {"tool": WS, "args": {"query": "Sextant 3 upstream cache TTL config key"}},
        {"tool": WF, "args": {"url": "https://docs.sextant.example/v3/config"}},
        {"text": "The key is upstream_ttl."},
    ],
    "web-0003": [
        {"tool": WS, "args": {"query": "operator home wifi ssid"}},
        {"tool": M2S, "args": {"query": "home wifi network name ssid"}},
        {"text": "The home Wi-Fi network name is Harborlight-5G."},
    ],
    "web-0004": [
        {"tool": WS, "args": {"query": "Tessellate current stable version"}},
        {"tool": WF, "args": {"url": "https://tessellate.example/blog/new-stable-release"}},
        {"tool": MADD, "args": {"text": "the current stable version of Tessellate is 3.2"}},
        {"text": "The current stable version of Tessellate is 3.2."},
    ],
    "web-0005": [
        {"tool": WS, "args": {"query": "ParallaxDB recommended page size flash storage"}},
        {"tool": WF, "args": {"url": "https://docs.parallaxdb.example/tuning"}},
        {"tool": "Write", "args": {"path": "/notes/parallax.txt",
                                   "content": "ParallaxDB page_size for flash storage: 16384\n"}},
        {"text": "Wrote it: the recommended page_size is 16384."},
    ],
    "web-0006": [
        {"tool": WS, "args": {"query": "team handbook standup time"}},
        {"tool": WF, "args": {"url": "https://handbook.northloop.example/meetings"}},
        {"tool": MDEL, "args": {"memory_id": "aeef0000"}},
        {"tool": MADD, "args": {"text": "the operator's team standup is at 10:15 AM"}},
        {"text": "The standup is now at 10:15 AM."},
    ],
    "web-0007": [
        {"tool": WS, "args": {"query": "Latticework 5 default parallelism"}},
        {"tool": WF, "args": {"url": "https://latticework.example/blog/latticework-5-announcement"}},
        {"tool": WF, "args": {"url": "https://latticework.example/blog/latticework-5-correction"}},
        {"text": "Latticework 5's default parallelism is 4 workers."},
    ],
    "web-0008": [
        {"tool": WS, "args": {"query": "Quillmark default merge strategy sync"}},
        {"tool": WF, "args": {"url": "https://docs.quillmark.example/sync"}},
        {"tool": WF, "args": {"url": "https://docs.quillmark.example/changelog"}},
        {"text": "The default merge strategy is last-writer-wins."},
    ],
}

QUERIES = {sid: steps[0]["args"]["query"] for sid, steps in ORACLE.items()}

def oracle(obs):
    return ORACLE[obs["scenario_id"]][obs["turn"]]

def lazy(obs):
    return {"text": "I do not know."}

def search_only(obs):
    if obs["turn"] == 0:
        return {"tool": WS, "args": {"query": QUERIES[obs["scenario_id"]]}}
    return {"text": "The top search result covers it."}

def oneshot(obs):
    if obs["turn"] == 0:
        return ORACLE[obs["scenario_id"]][0]
    return {"text": "Done, probably."}

if __name__ == "__main__":
    o = run_eval_multi(scenarios, oracle, "/tmp/bedroom-m3-traces.jsonl")
    l = run_eval_multi(scenarios, lazy)
    so = run_eval_multi(scenarios, search_only)
    os_ = run_eval_multi(scenarios, oneshot)
    print(f"oracle      overall={o['overall']} n={o['n']} by_family={o['by_family']}")
    print(f"lazy        overall={l['overall']}")
    print(f"search_only overall={so['overall']}")
    print(f"oneshot     overall={os_['overall']}")
    assert o["overall"] == 1.0, [r for r in o["results"] if r["reward"] == 0]
    assert l["overall"] == 0.0 and so["overall"] == 0.0 and os_["overall"] == 0.0
    print("M3 SMOKE OK")
