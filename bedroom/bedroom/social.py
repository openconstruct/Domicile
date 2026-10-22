"""Bedroom M4: simulated Google + Discord dollhouses.

Dollhouse implementations of Domicile's Google/Discord surface, per build
spec section 6: seeded mail / calendar / Blogger state behind gmail_*,
calendar_* and blogger_post, and seeded channels behind discord_read /
discord_post. State changes only via tools -- a send, draft, post or
calendar create mutates the SocialState and nothing else does -- so the
multi runner can grade end-state postconditions against it.

Scenario setup shape (setup.social):
  inbox:    [{id, from, subject, body}] seeded mail
  calendar: [{id, title, start, end}] seeded events (ISO strings)
  discord:  {channel: [{author, text}]} seeded channel history

Search ranks by term overlap then difflib ratio over from+subject+body,
the web dollhouse's _score pattern, so the same query returns the same
results in the same order forever. Reads return titled text blocks the
way the real tools do. Bad message ids and unknown channels raise
ValueError; the runner feeds errors back as observations.

Dollhouse simplifications (documented, deliberate): channel names are
stored without the leading '#'; calendar_create does not reject overlaps
(double-booking is a graded failure, not an environment error); sends do
not thread (a reply is a fresh message whose subject carries the Re:).
An episode with no social setup gets an empty state whose searches all
return "(no results)".
"""
import difflib, re

SOCIAL_TOOLS = ["gmail_search", "gmail_read", "gmail_send", "gmail_draft",
                "calendar_list", "calendar_create", "blogger_post",
                "discord_read", "discord_post"]


class SocialState:
    """Seeded Google/Discord stand-in: inbox, calendar, channels, outboxes."""

    def __init__(self, social_setup=None):
        social_setup = social_setup or {}
        self.inbox = [dict(m) for m in social_setup.get("inbox", [])]
        self.calendar = [dict(e) for e in social_setup.get("calendar", [])]
        self.discord = {ch.lstrip("#"): [dict(m) for m in msgs]
                        for ch, msgs in social_setup.get("discord", {}).items()}
        self.sent = []
        self.drafts = []
        self.blog = []
        self._n = {"sent": 0, "draft": 0, "evt": 0, "post": 0}

    def _next_id(self, kind, prefix):
        self._n[kind] += 1
        return f"{prefix}-{self._n[kind]:03d}"

    def _score(self, query, msg):
        hay = f"{msg.get('from', '')} {msg.get('subject', '')} {msg.get('body', '')}".lower()
        q = query.lower()
        terms = [t for t in re.split(r"\W+", q) if t]
        overlap = sum(1 for t in terms if t in hay)
        return (overlap, difflib.SequenceMatcher(None, q, hay[:400]).ratio())

    def gmail_search(self, query, count=5):
        if not query or not query.strip():
            raise ValueError("query cannot be empty")
        q = query.strip()
        ranked = sorted(self.inbox, key=lambda m: self._score(q, m), reverse=True)
        hits = [m for m in ranked if self._score(q, m)[0] > 0][: int(count or 5)]
        if not hits:
            return "(no results)"
        lines = []
        for i, m in enumerate(hits, 1):
            lines.append(f"[{i}] ID: {m['id']}\nFrom: {m.get('from', '')}\n"
                         f"Subject: {m.get('subject', '')}\n{m.get('body', '')[:120]}")
        return "\n\n".join(lines)

    def gmail_read(self, message_id):
        for m in self.inbox:
            if m["id"] == message_id:
                return (f"From: {m.get('from', '')}\nSubject: {m.get('subject', '')}\n\n"
                        f"{m.get('body', '')}")
        raise ValueError(f"no such message: {message_id}")

    def gmail_send(self, to, subject, body):
        if not to or not to.strip():
            raise ValueError("recipient cannot be empty")
        mid = self._next_id("sent", "sent")
        self.sent.append({"id": mid, "to": to, "subject": subject or "", "body": body or ""})
        return f"sent {mid}"

    def gmail_draft(self, to, subject, body):
        if not to or not to.strip():
            raise ValueError("recipient cannot be empty")
        did = self._next_id("draft", "draft")
        self.drafts.append({"id": did, "to": to, "subject": subject or "", "body": body or ""})
        return f"draft saved {did}"

    def calendar_list(self):
        if not self.calendar:
            return "(no events)"
        rows = sorted(self.calendar, key=lambda e: (e.get("start", ""), e.get("title", "")))
        return "\n".join(f"{e['id']} | {e.get('title', '')} | "
                         f"{e.get('start', '')} -> {e.get('end', '')}" for e in rows)

    def calendar_create(self, title, start, end):
        eid = self._next_id("evt", "evt-new")
        self.calendar.append({"id": eid, "title": title, "start": start, "end": end})
        return f"created {eid}"

    def blogger_post(self, title, body):
        pid = self._next_id("post", "post")
        self.blog.append({"id": pid, "title": title or "", "body": body or ""})
        return f"posted {pid}"

    def _channel(self, channel):
        name = (channel or "").lstrip("#")
        if name not in self.discord:
            raise ValueError(f"no such channel: {channel}")
        return name

    def discord_read(self, channel, limit=20):
        name = self._channel(channel)
        msgs = self.discord[name][-int(limit or 20):]
        if not msgs:
            return "(no messages)"
        return "\n".join(f"{m.get('author', '')}: {m.get('text', '')}" for m in msgs)

    def discord_post(self, channel, text):
        name = self._channel(channel)
        self.discord[name].append({"author": "resident", "text": text or ""})
        return f"posted to #{name}"


def run_social_tool(state, tool, args):
    if tool == "gmail_search":
        return state.gmail_search(args["query"], args.get("count", 5))
    if tool == "gmail_read":
        return state.gmail_read(args["id"])
    if tool == "gmail_send":
        return state.gmail_send(args["to"], args.get("subject", ""), args.get("body", ""))
    if tool == "gmail_draft":
        return state.gmail_draft(args["to"], args.get("subject", ""), args.get("body", ""))
    if tool == "calendar_list":
        return state.calendar_list()
    if tool == "calendar_create":
        return state.calendar_create(args["title"], args["start"], args["end"])
    if tool == "blogger_post":
        return state.blogger_post(args.get("title", ""), args["body"])
    if tool == "discord_read":
        return state.discord_read(args["channel"], args.get("limit", 20))
    if tool == "discord_post":
        return state.discord_post(args["channel"], args["text"])
    raise ValueError(f"unknown tool {tool}")
