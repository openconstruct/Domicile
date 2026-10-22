"""Bedroom M3: web dollhouse.

Dollhouse implementations of Domicile's real web surface, per build spec
section 5: WebSearch / WebFetch (Domicile's registry names), DDGS-shaped in the real tool (a search
returns titled result blocks with URLs and snippets; a fetch returns the
page body). The backend here is a frozen, seeded corpus instead of the
network, cribbed from Enclosure's Corpus (harness/tools.py): an index of
{url, title, snippet, file} plus plain-text page bodies, ranked by term
overlap then difflib ratio, so the same query returns the same results
in the same order forever. A scenario may declare no_results_for
substrings; any query containing one comes back "(no results)" --
Enclosure's hubris instrument, here the "the web doesn't have that"
lever for fallback-routing scenarios.

Scenario setup shape (setup.web):
  corpus_id:      shared corpus to load ("dollhouse-web-v1"), or absent
  docs:           inline pages [{url, title, snippet, body}] merged over
                  the shared corpus (scenario-specific answer pages)
  no_results_for: [substring, ...]

Dollhouse simplifications (documented, deliberate): one shared corpus
plus per-scenario inline docs instead of the open web; ranking is term
overlap + difflib instead of a real index; fetch matches the URL exactly
(no redirects, no rendering); fetched bodies are capped at 20,000 chars.
An episode with no web setup gets an empty Dollhouse whose searches all
return "(no results)".
"""
import difflib, json, re
from pathlib import Path

WEB_TOOLS = ["WebSearch", "WebFetch"]
CORPUS_ROOT = Path(__file__).resolve().parent.parent / "scenarios" / "web" / "corpus"
FETCH_CAP = 12_000      # real WebFetch default max_chars (clamped 800..40000)


class Dollhouse:
    """Frozen web stand-in: shared corpus pages + per-scenario inline docs."""

    def __init__(self, web_setup=None):
        web_setup = web_setup or {}
        self.docs = []  # {url, title, snippet, body}
        self.blocked = [s.lower() for s in web_setup.get("no_results_for", [])]
        cid = web_setup.get("corpus_id")
        if cid:
            if cid != "dollhouse-web-v1":
                raise ValueError(f"unknown corpus {cid}")
            index = CORPUS_ROOT / "index.json"
            if index.is_file():
                for d in json.loads(index.read_text()):
                    f = CORPUS_ROOT / "pages" / d.get("file", "")
                    body = f.read_text(errors="replace") if f.is_file() else d.get("snippet", "")
                    self.docs.append({"url": d.get("url", ""), "title": d.get("title", ""),
                                      "snippet": d.get("snippet", ""), "body": body})
        for d in web_setup.get("docs", []):
            self.docs.append({"url": d.get("url", ""), "title": d.get("title", ""),
                              "snippet": d.get("snippet", ""), "body": d.get("body", "")})

    def _score(self, query, doc):
        hay = f"{doc['title']} {doc['snippet']} {doc['url']}".lower()
        q = query.lower()
        terms = [t for t in re.split(r"\W+", q) if t]
        overlap = sum(1 for t in terms if t in hay)
        return (overlap, difflib.SequenceMatcher(None, q, hay[:400]).ratio())

    def websearch(self, query, count=5):
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        q = query.strip()
        if any(b in q.lower() for b in self.blocked):
            return "(no results)"
        ranked = sorted(self.docs, key=lambda d: self._score(q, d), reverse=True)
        hits = [d for d in ranked if self._score(q, d)[0] > 0][: int(count or 5)]
        if not hits:
            return "(no results)"
        lines = []
        for d in hits:        # real WebSearch renders "- title\n  url\n  snippet[:420]"
            lines.append(f"- {d['title']}\n  {d['url']}\n  {d['snippet'][:420]}")
        return "\n\n".join(lines)

    def webfetch(self, url, max_chars=12000):
        u = str(url or "").strip()
        if not u:
            raise ValueError("url is empty")
        if not (u.startswith("http://") or u.startswith("https://")):
            raise ValueError("url must be http(s)")
        cap = max(800, min(int(max_chars or 12000), 40000))
        for d in self.docs:
            if d["url"] == u:
                text = d["body"]
                if len(text) > cap:
                    keep = cap // 2
                    text = text[:keep] + f"\n\n... [{len(text) - cap} chars trimmed] ...\n\n" + text[-keep:]
                return text
        raise ValueError(f"could not fetch: {u}")


def run_web_tool(dollhouse, tool, args):
    if tool == "WebSearch":
        return dollhouse.websearch(args["query"], args.get("count", 5))
    if tool == "WebFetch":
        return dollhouse.webfetch(args["url"], args.get("max_chars", 12000))
    raise ValueError(f"unknown tool {tool}")
