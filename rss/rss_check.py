#!/usr/bin/env python3
"""
Minimal RSS/Atom checker, stdlib only (no feedparser dependency).

Reads feeds.txt (one URL per line, # comments allowed), fetches each,
diffs against seen.json (per-feed set of seen entry ids), and writes
any new entries to latest.md -- overwritten each run, so it always
reflects "new since last check," not a growing log.

Usage:
    python3 rss_check.py            # check all feeds, update state
    python3 rss_check.py --dry-run  # show what's new, don't update seen.json

State lives in data/rss/. If feeds.txt is missing, the starter list from
templates/rss/feeds.txt is copied there on first run. Scheduling is
separate: the setup wizard installs an hourly timer that runs rss_notify.py.
"""
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import paths  # noqa: E402

FEEDS_FILE = paths.RSS_FEEDS
SEEN_FILE = paths.RSS_SEEN
FULL_FILE = paths.RSS_FULL
LATEST_FILE = paths.RSS_LATEST
STARTER_FEEDS = HERE.parent / "templates" / "rss" / "feeds.txt"

ATOM_NS = "{http://www.w3.org/2005/Atom}"
USER_AGENT = "domicile-rss/1.0 (+https://github.com/openconstruct/Domicile)"


def load_feeds():
    if not FEEDS_FILE.exists():
        if not STARTER_FEEDS.is_file():
            return []
        FEEDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        FEEDS_FILE.write_text(STARTER_FEEDS.read_text())
    urls = []
    for line in FEEDS_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            urls.append(line)
    return urls


def load_seen():
    if SEEN_FILE.exists():
        return json.loads(SEEN_FILE.read_text())
    return {}


def save_seen(seen):
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEEN_FILE.write_text(json.dumps(seen, indent=2, sort_keys=True))


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read()


def strip_html(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_entries(raw):
    """Return list of (entry_id, title, link, summary) for RSS 2.0 or Atom."""
    root = ET.fromstring(raw)
    entries = []

    # RSS 2.0: <rss><channel><item>...
    channel = root.find("channel")
    if channel is not None:
        for item in channel.findall("item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            guid = (item.findtext("guid") or link or title).strip()
            summary = strip_html(item.findtext("description") or "")
            entries.append((guid, title, link, summary))
        return entries

    # Atom: <feed><entry>...
    if root.tag == f"{ATOM_NS}feed" or root.tag == "feed":
        entry_list = root.findall(f"{ATOM_NS}entry")
        if not entry_list:
            entry_list = root.findall("entry")
        for entry in entry_list:
            title = (entry.findtext(f"{ATOM_NS}title") or entry.findtext("title") or "").strip()

            link = ""
            link_els = entry.findall(f"{ATOM_NS}link")
            if not link_els:
                link_els = entry.findall("link")
            # Prefer rel="alternate" (or no rel at all); fall back to first link.
            for le in link_els:
                if le.get("rel") in (None, "alternate"):
                    link = le.get("href", "").strip()
                    break
            if not link and link_els:
                link = link_els[0].get("href", "").strip()

            entry_id = (entry.findtext(f"{ATOM_NS}id") or entry.findtext("id") or "").strip() or link or title
            summary = strip_html(
                entry.findtext(f"{ATOM_NS}summary")
                or entry.findtext("summary")
                or entry.findtext(f"{ATOM_NS}content")
                or entry.findtext("content")
                or ""
            )
            entries.append((entry_id, title, link, summary))
        return entries

    return entries


def main():
    dry_run = "--dry-run" in sys.argv
    feeds = load_feeds()
    seen = load_seen()
    new_by_feed = []

    for url in feeds:
        seen_ids_list = seen.get(url, [])
        seen_ids = set(seen_ids_list)
        try:
            raw = fetch(url)
            entries = parse_entries(raw)
        except Exception as e:
            new_by_feed.append((url, None, str(e), 0))
            continue

        fresh = [e for e in entries if e[0] not in seen_ids]
        # arXiv feeds re-announce revised papers ("Announce Type: replace...").
        # They are marked seen below but are not news, so don't report them.
        n_before = len(fresh)
        fresh = [e for e in fresh if "Announce Type: replace" not in e[3][:120]]
        skipped_replace = n_before - len(fresh)
        if fresh or skipped_replace:
            new_by_feed.append((url, fresh, None, skipped_replace))

        if not dry_run:
            all_ids = [e[0] for e in entries]
            # Order-preserving: keep existing order, append newly-seen ids in
            # feed order, trim oldest-by-position if over cap. Old code used
            # set union + slice, which has no stable order (Python randomizes
            # str hash seed per process) -- that turned "cap exceeded" into
            # random eviction, so an id seen yesterday could vanish today and
            # get reported as "new" again tomorrow. CAP raised from 500 since
            # cs.AI alone runs ~600 items per fetch (arXiv's rolling window).
            CAP = 1500
            combined = seen_ids_list + [i for i in all_ids if i not in seen_ids]
            seen[url] = combined[-CAP:]

    if not dry_run:
        save_seen(seen)

    # Write latest.md
    stamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    lines = [f"# RSS check — {stamp}", ""]
    full = [f"# RSS check (full list) — {stamp}", ""]
    any_new = False
    # Show at most this many items per feed. A feed that dumps its whole window
    # at once (arXiv does) would otherwise put ~1,400 abstracts into the turn
    # that reads this. The complete list always goes to FULL_FILE.
    MAX_SHOWN = 15
    for url, fresh, err, n_replace in new_by_feed:
        if err:
            lines.append(f"## {url}")
            lines.append(f"(fetch/parse error: {err})")
            lines.append("")
            continue
        if fresh:
            any_new = True
            head = f"## {url} — {len(fresh)} new"
            if n_replace:
                head += f" ({n_replace} revisions of older papers not shown)"
            lines.append(head)
            full.append(head)
            for i, (entry_id, title, link, summary) in enumerate(fresh):
                block = [f"- **{title}**"]
                if link:
                    block.append(f"  {link}")
                if summary:
                    snippet = summary[:280] + ("..." if len(summary) > 280 else "")
                    block.append(f"  {snippet}")
                full.extend(block)
                if i < MAX_SHOWN:
                    lines.extend(block)
            if len(fresh) > MAX_SHOWN:
                lines.append(f"- ... and {len(fresh) - MAX_SHOWN} more (full list: {FULL_FILE})")
            lines.append("")
            full.append("")

    if not any_new:
        lines.append("(nothing new since last check)")

    LATEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    LATEST_FILE.write_text("\n".join(lines))
    FULL_FILE.write_text("\n".join(full))
    print("\n".join(lines))
    return 0 if any_new else 3


if __name__ == "__main__":
    raise SystemExit(main())
