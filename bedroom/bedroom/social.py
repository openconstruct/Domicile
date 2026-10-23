"""Bedroom M4: Google (and Google Chat) dollhouse -- Domicile's real surface.

Re-shaped 2026-10-22 against domicile/google_tools.py. The old invented tool
family (gmail_*, calendar_*, blogger_post, discord_read/discord_post) is gone:
none of those names exists in Domicile, so a policy trained on them learned to
call tools the house does not have. This module now implements exactly the nine
model-callable Google tools, with the argument names, required-argument errors
and result text of google_tools.py:

  GmailSearch  query, max_results         -> "<id>  <date>  <from>  <subject>\\n    <snippet>"
  GmailRead    id                         -> "From: ..\\nTo: ..\\nSubject: ..\\nDate: ..\\n\\n<body>"
  GmailSend    to, subject, body          -> "tool=GmailSend; to=..; id=.."
  GmailDraft   to, subject, body          -> "tool=GmailDraft; to=..; id=.."
  CalendarList max_results, calendar_id   -> "<start>  <title>[  [<location>]]"
  CalendarCreate summary, start, end      -> "tool=CalendarCreate; id=..; start=.."
  BloggerPost  title, content_html, draft -> "tool=BloggerPost; id=..; status=draft|live"
  ChatRead     space, page_size           -> "<time>  <sender> (<HUMAN|BOT>): <text>"
  ChatSend     space, text                -> "tool=ChatSend; space=..; name=.."

Calendar times are RFC3339, as the real schema requires. Discord has no tool and
will not get one: the bridge delivers a reply, so answering is posting. Scenarios
that needed a "read the channel, answer the decision" shape use ChatRead instead.

Scenario setup shape (setup.social):
  inbox:    [{id, from, to, subject, date, body}]   `date`/`to` optional
  calendar: [{id, title|summary, start, end, location}]  RFC3339 start/end
  spaces:   {space: [{sender, sender_type, text, createTime}]}

Dollhouse simplifications (deliberate, documented): search ranks by term overlap
then difflib ratio over from+subject+body instead of Gmail's query language;
CalendarList returns the seeded events sorted by start rather than filtering to a
live "now"; sends do not thread (a reply is a fresh message); ids are synthetic
but stable; message dates are whatever the setup says, never the wall clock.

State changes only via tools -- a send/draft/post/create mutates the state and
nothing else does -- so the multi runner grades end-state postconditions.
"""
import difflib, re

SOCIAL_TOOLS = ["GmailSearch", "GmailRead", "GmailSend", "GmailDraft",
                "CalendarList", "CalendarCreate", "BloggerPost",
                "ChatRead", "ChatSend"]


def _s(args, key, required=True):
    """Mirror google_tools._s: blank or missing required arg is a ValueError."""
    v = args.get(key)
    if v is None or str(v).strip() == "":
        if required:
            raise ValueError(f"{key} is required")
        return ""
    return str(v)


def _snippet(body, n=120):
    return re.sub(r"\s+", " ", str(body or "")).strip()[:n]


def _hex_id(base, n, width=16):
    return f"{base + n:0{width}x}"


class SocialState:
    """Seeded Google stand-in: inbox, calendar, chat spaces, outboxes."""

    MAIL_ID_BASE = 0x18A0_0000_0000_0000
    EVT_ID_BASE = 0x6A1C_0000_0000_0000_0000_0000
    POST_ID_BASE = 7_300_000_000_000_000_000
    CHAT_ID_BASE = 0x1A2B_0000_0000_0000

    def __init__(self, social_setup=None):
        social_setup = social_setup or {}
        self.inbox = [dict(m) for m in social_setup.get("inbox", [])]
        for m in self.inbox:
            m.setdefault("date", "Thu, 1 Jan 2026 09:00:00 -0500")
            m.setdefault("to", "resident")
        self.calendar = []
        for n, e in enumerate(social_setup.get("calendar", [])):
            e = dict(e)
            e.setdefault("id", _hex_id(self.EVT_ID_BASE, n, 26))
            if "title" not in e:
                e["title"] = e.get("summary", "")
            self.calendar.append(e)
        # Chat spaces. The old `discord` setup key is accepted so older scenario
        # files still load, but the tool surface is ChatRead/ChatSend only.
        spaces = social_setup.get("spaces", social_setup.get("discord", {}))
        self.spaces = {}
        for name, msgs in (spaces or {}).items():
            rows = []
            for n, m in enumerate(msgs):
                m = dict(m)
                m.setdefault("sender", m.get("author", "teammate"))
                m.setdefault("sender_type", "HUMAN")
                m.setdefault("createTime", f"2026-01-01T09:{n % 60:02d}:00Z")
                rows.append(m)
            self.spaces[str(name)] = rows
        self.sent = []      # {id, to, subject, body}
        self.drafts = []    # {id, to, subject, body}
        self.blog = []      # {id, title, body, status}
        self._n = {"mail": len(self.inbox), "draft": 0, "evt": len(self.calendar),
                   "post": 0, "chat": 0}

    def _next_id(self, kind, base, width=16):
        n = self._n[kind]
        self._n[kind] += 1
        return _hex_id(base, n, width)

    def _score(self, query, msg):
        hay = f"{msg.get('from', '')} {msg.get('subject', '')} {msg.get('body', '')}".lower()
        q = query.lower()
        terms = [t for t in re.split(r"\W+", q) if t]
        overlap = sum(1 for t in terms if t in hay)
        return (overlap, difflib.SequenceMatcher(None, q, hay[:400]).ratio())

    # ---- Gmail

    def _find(self, message_id):
        for m in self.inbox:
            if str(m.get("id")) == str(message_id):
                return m
        raise ValueError(f"no message with id {message_id}")

    def gmail_search(self, query, max_results=20):
        q = _s({"query": query}, "query")
        limit = max(1, min(int(max_results or 20), 100))
        ranked = sorted(self.inbox, key=lambda m: self._score(q, m), reverse=True)
        hits = [m for m in ranked if self._score(q, m)[0] > 0][:limit]
        if not hits:
            return "(no matches)"
        return "\n".join(
            f"{m['id']}  {m.get('date', '')}  {m.get('from', '')}  {m.get('subject', '')}\n"
            f"    {_snippet(m.get('body'))}" for m in hits)

    def gmail_read(self, message_id):
        m = self._find(_s({"id": message_id}, "id"))
        return (f"From: {m.get('from', '')}\nTo: {m.get('to', '')}\n"
                f"Subject: {m.get('subject', '')}\nDate: {m.get('date', '')}\n\n"
                f"{m.get('body') or '(no plain-text body)'}")

    def gmail_send(self, to, subject, body):
        to, subject, body = _s({"to": to}, "to"), _s({"subject": subject}, "subject"), \
            _s({"body": body}, "body")
        mid = self._next_id("mail", self.MAIL_ID_BASE)
        self.sent.append({"id": mid, "to": to, "subject": subject, "body": body})
        return f"tool=GmailSend; to={to}; id={mid}"

    def gmail_draft(self, to, subject, body):
        to, subject, body = _s({"to": to}, "to"), _s({"subject": subject}, "subject"), \
            _s({"body": body}, "body")
        did = self._next_id("draft", self.MAIL_ID_BASE + 0x1000)
        self.drafts.append({"id": did, "to": to, "subject": subject, "body": body})
        return f"tool=GmailDraft; to={to}; id={did}"

    # ---- Calendar

    def calendar_list(self, max_results=10):
        limit = max(1, min(int(max_results or 10), 100))
        rows = sorted(self.calendar, key=lambda e: (e.get("start", ""), e.get("title", "")))[:limit]
        if not rows:
            return "(no upcoming events)"
        out = []
        for e in rows:
            line = f"{e.get('start', '')}  {e.get('title', '')}"
            if e.get("location"):
                line += f"  [{e['location']}]"
            out.append(line)
        return "\n".join(out)

    def calendar_create(self, summary, start, end):
        summary = _s({"summary": summary}, "summary")
        start, end = _s({"start": start}, "start"), _s({"end": end}, "end")
        eid = self._next_id("evt", self.EVT_ID_BASE, 26)
        self.calendar.append({"id": eid, "title": summary, "start": start, "end": end})
        return f"tool=CalendarCreate; id={eid}; start={start}"

    # ---- Blogger

    def blogger_post(self, title, content_html, draft=True):
        title = _s({"title": title}, "title")
        html = _s({"content_html": content_html}, "content_html")
        is_draft = True if draft is None else bool(draft)
        n = self._n["post"]
        self._n["post"] += 1
        pid = str(self.POST_ID_BASE + n)
        self.blog.append({"id": pid, "title": title, "body": html,
                          "status": "draft" if is_draft else "live"})
        return f"tool=BloggerPost; id={pid}; status={'draft' if is_draft else 'live'}"

    # ---- Google Chat

    def _space(self, space):
        name = _s({"space": space}, "space")
        key = name
        if key not in self.spaces:
            raise ValueError(f"space not found: {name}")
        return key

    def chat_read(self, space, page_size=20):
        key = self._space(space)
        limit = max(1, min(int(page_size or 20), 100))
        msgs = self.spaces[key][-limit:][::-1]      # newest first, as the real tool returns
        if not msgs:
            return "(no messages)"
        return "\n".join(
            f"{m.get('createTime', '')}  {m.get('sender', '')} "
            f"({m.get('sender_type', 'HUMAN')}): {m.get('text', '')}" for m in msgs)

    def chat_send(self, space, text):
        key = self._space(space)
        body = _s({"text": text}, "text")
        name = f"{key}/messages/{_hex_id(self.CHAT_ID_BASE, self._n['chat'])}"
        self._n["chat"] += 1
        self.spaces[key].append({"sender": "resident", "sender_type": "BOT",
                                 "text": body, "createTime": "2026-01-01T12:00:00Z"})
        return f"tool=ChatSend; space={key}; name={name}"


def run_social_tool(state, tool, args):
    a = args or {}
    if tool == "GmailSearch":
        return state.gmail_search(a.get("query"), a.get("max_results", 20))
    if tool == "GmailRead":
        return state.gmail_read(a.get("id"))
    if tool == "GmailSend":
        return state.gmail_send(a.get("to"), a.get("subject"), a.get("body"))
    if tool == "GmailDraft":
        return state.gmail_draft(a.get("to"), a.get("subject"), a.get("body"))
    if tool == "CalendarList":
        return state.calendar_list(a.get("max_results", 10))
    if tool == "CalendarCreate":
        return state.calendar_create(a.get("summary", a.get("title")), a.get("start"), a.get("end"))
    if tool == "BloggerPost":
        return state.blogger_post(a.get("title"), a.get("content_html", a.get("body")),
                                  a.get("draft"))
    if tool == "ChatRead":
        return state.chat_read(a.get("space", a.get("channel")), a.get("page_size", 20))
    if tool == "ChatSend":
        return state.chat_send(a.get("space", a.get("channel")), a.get("text"))
    raise ValueError(f"unknown tool {tool}")
