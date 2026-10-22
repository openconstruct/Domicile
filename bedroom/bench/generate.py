"""Bedroom bench generator: seeded archetypes -> disjoint pools.

Every scenario in the bench/train system is generated from an archetype
(the parameterized form of an M1-M5 hand-built scenario) by a seeded
RNG. Ground truth is computed from the planted content -- the
generator counts the lines it wrote, picks the slot it left free,
knows the fact it seeded -- so a generated scenario cannot be
miskeyed. Pools (train/dev/eval) are disjoint by construction:

- seed = sha256("bedroom-bench-v1|<pool>|<archetype>|<index>"), so
  pools cannot produce the same instance;
- each pool draws surface content (people, projects, orgs, streets)
  from its own vocabulary lists, so a template-identical pair still
  shares no surface content.

Archetypes flagged eval_only build the generalization slice: variants
that never appear in train/dev, so the eval report can separate
"learned the skill" from "learned the distribution".

Output (under bench/): {train,dev,eval}/*.json scenario files,
oracles.json (per pool, per scenario: the action list that must score
1), manifest.json (counts, archetypes, knobs). The audit that proves
oracle=1 / degenerate=0 / no leaks lives in audit.py.
"""
import hashlib, json, random
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
POOLS = ("train", "dev", "eval")
COUNTS = {"train": 70, "dev": 4, "eval": 12}
GEN_EVAL_COUNT = 8

# ---------------------------------------------------------------- vocab
# Disjoint surface vocabularies per pool. Numbers and dates come from
# the seeded RNG; identity content comes only from these lists.
VOCAB = {
    "train": {
        "people": ["Priya", "Tomas", "Dana", "Marta", "Jonas", "Silvie", "Henrik",
                   "Aiko", "Ruth", "Omar", "Lena", "Felix", "Ines", "Marco"],
        "projects": ["Bramble", "Fernway", "Tessellate", "Quillmark", "Parallax",
                     "Northloop", "Sextant", "Lattice", "Meridian", "Cobalt"],
        "orgs": ["Northloop", "Fernway Labs", "Helios Works", "Cartwheel Co",
                 "Basalt Group", "Umbral Systems"],
        "streets": ["Harbor Lane", "Old Mill Road", "Cinder Street", "Alder Avenue",
                    "Foundry Row", "Saltbox Lane"],
        "domains": ["example.com", "mail.example", "corp.example"],
    },
    "dev": {
        "people": ["Greta", "Yusuf", "Nadia", "Pavel", "Rosa", "Emil", "Tara",
                   "Hugo", "Mina", "Sven", "Alba", "Kurt", "Iris", "Dario"],
        "projects": ["Kelvin", "Ostara", "Briarwood", "Copperline", "Dunmore",
                     "Aldergrove", "Fable", "Granton", "Heliotrope", "Juniper"],
        "orgs": ["Copperline Ltd", "Dunmore Group", "Fable Works", "Granton Labs",
                 "Juniper Systems", "Ostara Co"],
        "streets": ["Copper Lane", "Dunmore Street", "Fable Road", "Granton Avenue",
                    "Juniper Row", "Ostara Lane"],
        "domains": ["sample.dev", "inbox.sample", "hq.sample"],
    },
    "eval": {
        "people": ["Vera", "Anton", "Selin", "Rurik", "Maren", "Dmitri", "Anya",
                   "Lars", "Inna", "Boris", "Tilda", "Oleg", "Sana", "Yuri"],
        "projects": ["Vessel", "Antonia", "Selkie", "RurikDB", "Marengo", "Dmitra",
                     "Anywhere", "Larsen", "Innate", "Borislav"],
        "orgs": ["Vessel Inc", "Selkie Labs", "Marengo Group", "Dmitra Works",
                 "Innate Systems", "Larsen Co"],
        "streets": ["Vessel Street", "Selkie Lane", "Marengo Road", "Innate Avenue",
                    "Larsen Row", "Antonia Lane"],
        "domains": ["bench.test", "post.bench.test", "office.bench.test"],
    },
}

FIRST_FACTS = ["prefers tea in the morning", "takes the 7:40 train",
               "keeps the server room key in the blue drawer",
               "answers email after 18:00", "uses a standing desk"]


def _rng(pool, archetype, index):
    h = hashlib.sha256(f"bedroom-bench-v1|{pool}|{archetype}|{index}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


def _pick(rng, seq):
    return seq[rng.randrange(len(seq))]


def _sample(rng, seq, n):
    return rng.sample(list(seq), min(n, len(seq)))


def _scenario(pool, archetype, index, family, difficulty, situation, setup,
              expected, notes):
    sid = f"{pool}-{archetype}-{index:03d}"
    sc = {"id": sid, "family": family, "difficulty": difficulty,
          "situation": situation, "setup": setup, "expected": expected,
          "max_turns": expected.pop("max_turns", 6),
          "notes": notes,
          "reward_rule": "sequence followed + postconditions hold + final answer correct = 1 else 0",
          "pool": pool, "archetype": archetype}
    return sc


def _final_contains(*vals):
    return {"final_answer_contains": [str(v) for v in vals]}


# ---------------------------------------------------------------- fs
def gen_fs_read_fact(rng, v, pool, idx):
    proj = _pick(rng, v["projects"])
    other = _pick(rng, [p for p in v["projects"] if p != proj])
    port = rng.randrange(2000, 9000)
    other_port = port + rng.choice([17, 100, 250])
    key = rng.choice(["deploy port", "metrics port", "admin port"])
    files = {
        f"/notes/{proj.lower()}-ops.txt":
            f"{proj} operations.\nThe {key} for {proj} is {port}.\nNightly jobs run at 02:00.\n",
        f"/notes/{other.lower()}-ops.txt":
            f"{other} operations.\nThe {key} for {other} is {other_port}.\nNightly jobs run at 03:00.\n"}
    sc = _scenario(pool, "fs-read-fact", idx, "fs", 1,
                   f"What is the {key} for {proj}? The answer is in the operations notes. "
                   f"Read the right file, then answer with the port number.",
                   {"files": files},
                   {"required_sequence": ["Read"], **_final_contains(port)},
                   "Read the right ops file among a sibling project's; values differ.")
    oracle = [{"tool": "Read", "args": {"path": f"/notes/{proj.lower()}-ops.txt"}},
              {"text": f"The {key} for {proj} is {port}."}]
    return sc, oracle


def gen_fs_write_note(rng, v, pool, idx):
    person = _pick(rng, v["people"])
    proj = _pick(rng, v["projects"])
    code = f"{rng.randrange(100, 999)}-{rng.randrange(100, 999)}"
    content = f"Access code for the {proj} lab, issued to {person}: {code}\n"
    path = f"/notes/{proj.lower()}-access.txt"
    sc = _scenario(pool, "fs-write-note", idx, "fs", 1,
                   f"Write a new file at {path} containing exactly this line and nothing else: "
                   f"'Access code for the {proj} lab, issued to {person}: {code}' followed by a newline.",
                   {"files": {}},
                   {"required_sequence": ["Write"],
                    "postconditions": [{"type": "file_equals", "path": path, "text": content}]},
                   "Verbatim write; the graded content is exact, punctuation included.")
    oracle = [{"tool": "Write", "args": {"path": path, "content": content}},
              {"text": "Written."}]
    return sc, oracle


def gen_fs_edit_unique(rng, v, pool, idx):
    proj = _pick(rng, v["projects"])
    old_v = f"{rng.randrange(1, 9)}.{rng.randrange(0, 9)}"
    new_v = f"{rng.randrange(1, 9)}.{rng.randrange(0, 9)}"
    while new_v == old_v:
        new_v = f"{rng.randrange(1, 9)}.{rng.randrange(0, 9)}"
    before = (f"{proj} release status\ncurrent version: {old_v}\n"
              f"channel: stable\nowner sign-off: pending\n")
    after = before.replace(f"current version: {old_v}", f"current version: {new_v}")
    path = f"/notes/{proj.lower()}-status.txt"
    sc = _scenario(pool, "fs-edit-unique", idx, "fs", 1,
                   f"In {path}, change the current version from {old_v} to {new_v}. "
                   f"Change nothing else in the file.",
                   {"files": {path: before}},
                   {"required_sequence": ["Edit"],
                    "postconditions": [{"type": "file_equals", "path": path, "text": after}]},
                   "Surgical edit; a full rewrite that alters any other line fails file_equals.")
    # The house Edit refuses a file you have not Read this session, so the
    # oracle reads first (required_sequence ["Edit"] is a subsequence check).
    oracle = [{"tool": "Read", "args": {"path": path}},
              {"tool": "Edit", "args": {"path": path,
                                        "old": f"current version: {old_v}",
                                        "new": f"current version: {new_v}"}},
              {"text": "Version updated."}]
    return sc, oracle


def gen_fs_grep_count(rng, v, pool, idx):
    proj = _pick(rng, v["projects"])
    token = rng.choice(["DEPLOY", "BACKUP", "PRUNE", "SYNC"])
    lines, hits = [], 0
    filler = ["INFO heartbeat ok", "INFO queue drained", "WARN disk at 61 percent",
              "INFO checkpoint saved", "INFO worker restarted"]
    for i in range(rng.randrange(9, 14)):
        day = f"{rng.randrange(1, 28):02d}"
        if rng.random() < 0.4:
            lines.append(f"2026-10-{day} {token} {proj.lower()} job finished rc=0")
            hits += 1
        else:
            lines.append(f"2026-10-{day} {rng.choice(filler)}")
    body = "\n".join(lines) + "\n"
    sc = _scenario(pool, "fs-grep-count", idx, "fs", 2,
                   f"How many lines in /logs/{proj.lower()}.log contain the word {token}? "
                   f"Use search, then answer with just the number.",
                   {"files": {f"/logs/{proj.lower()}.log": body}},
                   {"required_sequence": ["Grep"],
                    "final_answer_exact_lines": [str(hits)]},
                   "Grep count mode vs eyeballing the file; the count is computed from the planted lines.")
    oracle = [{"tool": "Grep", "args": {"pattern": token, "path": f"/logs/{proj.lower()}.log",
                                        "output_mode": "count"}},
              {"text": str(hits)}]
    return sc, oracle


def gen_fs_glob_list(rng, v, pool, idx):
    ext = rng.choice(["md", "csv", "log"])
    keep = sorted(f"report-{i}.{ext}" for i in _sample(rng, range(1, 30), rng.randrange(3, 6)))
    others = [f"draft-{i}.txt" for i in _sample(rng, range(1, 30), 3)] + \
             [f"notes-{i}.{ext}.bak" for i in _sample(rng, range(1, 30), 2)]
    files = {f"/data/{name}": f"contents of {name}\n" for name in keep + others}
    want = [f"/data/{name}" for name in keep]
    sc = _scenario(pool, "fs-glob-list", idx, "fs", 2,
                   f"List every file in /data whose name ends in .{ext} — full paths, "
                   f"one per line, nothing else in your answer.",
                   {"files": files},
                   {"required_sequence": ["Glob"], "final_answer_exact_lines": want},
                   "Glob pattern discipline; .bak siblings and other extensions are bait.")
    oracle = [{"tool": "Glob", "args": {"pattern": f"*.{ext}", "path": "/data"}},
              {"text": "\n".join(want)}]
    return sc, oracle


# ---------------------------------------------------------------- memory
MS = "mcp__memory__memory_search"
MA = "mcp__memory__memory_add"
MD = "mcp__memory__memory_delete"
M2S = "mcp__mem2__mem2_search"


def gen_mem_search(rng, v, pool, idx):
    person = _pick(rng, v["people"])
    city_a, city_b = _sample(rng, ["Alder", "Bexley", "Corvin", "Dunmore", "Eastvale",
                                   "Falkirk", "Grantham", "Holloway"], 2)
    facts = [f"{person} lives in {city_a}",
             f"{person} works in {city_b}",
             f"{person}'s sister lives in {city_b}",
             _pick(rng, FIRST_FACTS).replace("the operator", person)]
    rng.shuffle(facts)
    sc = _scenario(pool, "mem-search", idx, "memory", 1,
                   f"Which city does {person} live in? Search long-term memory, then answer.",
                   {"facts": facts},
                   {"required_sequence": [MS], **_final_contains(city_a)},
                   "Near distractors share the person and the city tokens; the graded fact is residence.")
    oracle = [{"tool": MS, "args": {"query": f"{person} lives in"}},
              {"text": f"{person} lives in {city_a}."}]
    return sc, oracle


def gen_mem_add(rng, v, pool, idx):
    person = _pick(rng, v["people"])
    proj = _pick(rng, v["projects"])
    day = _pick(rng, ["Tuesday", "Thursday", "Friday"])
    fact = f"{person} reviews {proj} deploys every {day}"
    sc = _scenario(pool, "mem-add", idx, "memory", 1,
                   f"Remember this for later: {fact}. Store it in long-term memory.",
                   {"facts": [_pick(rng, FIRST_FACTS)]},
                   {"required_sequence": [MA],
                    "postconditions": [{"type": "store_contains", "text": fact}]},
                   "The fact must land in the store verbatim enough to match; narrating it earns 0.")
    oracle = [{"tool": MA, "args": {"text": fact}}, {"text": "Stored."}]
    return sc, oracle


def gen_mem_delete(rng, v, pool, idx):
    person = _pick(rng, v["people"])
    old_num = rng.randrange(1000, 9999)
    facts = [f"{person}'s old badge number is {old_num}",
             f"{person} works on the night shift",
             _pick(rng, FIRST_FACTS).replace("the operator", person)]
    rng.shuffle(facts)
    # ids are assigned in list order by FactStore
    pos = [i for i, f in enumerate(facts) if "badge number" in f][0]
    target_id = f"m-{pos + 1:03d}"
    sc = _scenario(pool, "mem-delete", idx, "memory", 2,
                   f"{person}'s badge number is no longer valid and the stored fact about it "
                   f"should be removed from long-term memory. Delete exactly that one fact; "
                   f"keep everything else.",
                   {"facts": facts, "show_listing": True},
                   {"required_sequence": [MD],
                    "postconditions": [{"type": "store_not_contains", "text": "badge number"},
                                       {"type": "store_count", "equals": len(facts) - 1}]},
                   "Delete exactly one targeted fact among keepers; the listing shows the ids.")
    oracle = [{"tool": MD, "args": {"memory_id": target_id}}, {"text": "Deleted the stale badge fact."}]
    return sc, oracle


def gen_mem_route(rng, v, pool, idx):
    street_old, street_new = _sample(rng, v["streets"], 2)
    num_old, num_new = rng.randrange(10, 99), rng.randrange(100, 999)
    old_addr, new_addr = f"{num_old} {street_old}", f"{num_new} {street_new}"
    facts = [f"the operator lives at {old_addr}", _pick(rng, FIRST_FACTS)]
    chunk = {"source_file": "2026-09-02-chat.jsonl", "chunk_index": 0, "date": "2026-09-02",
             "text": f"We moved last week. The new address is {new_addr}; the {street_old} "
                     f"house is sold, so anything mailed there is lost."}
    sc = _scenario(pool, "mem-route", idx, "memory", 3,
                   "What is the operator's current home address? Make sure long-term memory "
                   "holds the current address, not a stale one, then answer.",
                   {"facts": facts, "chunks": [chunk], "show_listing": True},
                   {"required_sequence": [MS, M2S, MD, MA],
                    "postconditions": [{"type": "store_contains", "text": new_addr},
                                       {"type": "store_not_contains", "text": old_addr}],
                    **_final_contains(new_addr)},
                   "Escalation by construction: mem0 is stale, the correction lives only in mem2; "
                   "routing is paid by the required sequence.")
    stale_id = "m-001"
    oracle = [{"tool": MS, "args": {"query": "operator home address"}},
              {"tool": M2S, "args": {"query": "moved new address"}},
              {"tool": MD, "args": {"memory_id": stale_id}},
              {"tool": MA, "args": {"text": f"the operator lives at {new_addr}"}},
              {"text": f"The current address is {new_addr}."}]
    return sc, oracle


# ---------------------------------------------------------------- web
def gen_web_fact(rng, v, pool, idx):
    proj = _pick(rng, v["projects"])
    setting = rng.choice(["retention window", "sync interval", "upload limit"])
    unit = {"retention window": "days", "sync interval": "minutes", "upload limit": "MB"}[setting]
    wrong = rng.choice([7, 14, 15, 100, 250])
    right = wrong + rng.choice([8, 16, 23, 240, 500])
    slug = proj.lower()
    docs = [
        {"url": f"https://forum.{slug}.example/t/{setting.replace(' ', '-')}",
         "title": f"What is the {setting}? - {proj} forum",
         "snippet": f"Mine looks like about {wrong} {unit}. Is that the default? Nobody in the thread got a straight answer.",
         "body": f"What is the {setting}?\n\nOP: Mine seems to be around {wrong} {unit}. Is that the default?\n\n"
                 f"Reply 1: Same here, roughly that. I never found a setting for it.\n\n"
                 f"Reply 2: I mailed support twice and got nothing. {wrong} {unit} is my guess too, only a guess.\n\n"
                 f"Reply 3: Check the configuration reference before you trust any of us. Nobody here has."},
        {"url": f"https://docs.{slug}.example/config",
         "title": f"{proj} configuration reference",
         "snippet": f"Reference for {slug}.toml: schedules, limits, and defaults.",
         "body": f"{proj} configuration reference.\n\nThe default {setting} is {right} {unit} "
                 f"when the setting is not configured.\n\nRaising it is the usual fix when the "
                 f"default feels too small; the reference is the only authoritative source."},
    ]
    sc = _scenario(pool, "web-fact", idx, "web", 2,
                   f"What is the default {setting} for {proj}, in {unit}? It is not in memory "
                   f"or in your files; look it up on the web and open the right page before answering.",
                   {"web": {"docs": docs}},
                   {"required_sequence": ["WebSearch", "WebFetch"],
                    **_final_contains(f"{right} {unit}")},
                   "Snippet bait: the forum thread guesses wrong in its snippet; the graded value "
                   "exists only in the reference page body.")
    oracle = [{"tool": "WebSearch", "args": {"query": f"{proj} default {setting}"}},
              {"tool": "WebFetch", "args": {"url": f"https://docs.{slug}.example/config"}},
              {"text": f"The default {setting} is {right} {unit}."}]
    return sc, oracle


# ---------------------------------------------------------------- social
GS = "gmail_search"
GR = "gmail_read"
GE = "gmail_send"
GD = "gmail_draft"
CL = "calendar_list"
CC = "calendar_create"
BP = "blogger_post"
DR = "discord_read"
DP = "discord_post"


def _mail_pair(rng, v, proj_a, proj_b):
    """Target sender + similar distractor sender asking about sibling projects."""
    a = _pick(rng, v["people"])
    dom = _pick(rng, v["domains"])
    return a, dom


def gen_mail_reply_fact(rng, v, pool, idx):
    proj, other = _sample(rng, v["projects"], 2)
    sender, dom = _pick(rng, v["people"]), _pick(rng, v["domains"])
    window = rng.choice([14, 21, 30, 45])
    other_window = window + rng.choice([7, 9, 16])
    note = (f"{proj} ops notes.\nThe shipped default retention window for {proj} "
            f"snapshots is {window} days when retention_days is not set.\n")
    inbox = [
        {"id": "msg-101", "from": f"{sender.lower()}@{dom}",
         "subject": f"{proj} retention for the runbook",
         "body": f"Hi -- I'm writing the runbook and need the shipped default retention window "
                 f"for {proj} snapshots. What is it, in days?"},
        {"id": "msg-102", "from": f"{sender.lower()}@{dom}",
         "subject": f"{other} retention for the runbook",
         "body": f"Separate thread: same question for {other} -- the shipped default retention "
                 f"window, in days? (Answer on the {other} thread, not this one.)"},
    ]
    files = {f"/notes/{proj.lower()}-ops.txt": note,
             f"/notes/{other.lower()}-ops.txt":
                 f"{other} ops notes.\nThe shipped default retention window for {other} "
                 f"snapshots is {other_window} days.\n"}
    sc = _scenario(pool, "mail-reply-fact", idx, "social", 2,
                   f"{sender} emailed asking for the shipped default retention window for "
                   f"{proj} snapshots for the runbook. Find the email, check the ops notes, "
                   f"and reply to {sender} with the window. Reply only to that email.",
                   {"social": {"inbox": inbox}, "files": files},
                   {"required_sequence": [GS, GR, "Read", GE],
                    "postconditions": [
                        {"type": "mail_sent", "to_contains": f"{sender.lower()}@{dom}",
                         "subject_contains": proj, "body_contains": f"{window} days"},
                        {"type": "mail_sent_count", "equals": 1}]},
                   "Distractor: a same-sender thread about the sibling project with a different "
                   "window; exactly-once grading on the reply.")
    oracle = [{"tool": GS, "args": {"query": f"{sender} {proj} retention runbook"}},
              {"tool": GR, "args": {"id": "msg-101"}},
              {"tool": "Read", "args": {"path": f"/notes/{proj.lower()}-ops.txt"}},
              {"tool": GE, "args": {"to": f"{sender.lower()}@{dom}",
                                    "subject": f"Re: {proj} retention for the runbook",
                                    "body": f"Hi {sender} -- the shipped default retention window "
                                            f"for {proj} snapshots is {window} days."}},
              {"text": f"Replied to {sender} with the {proj} window."}]
    return sc, oracle


def gen_mail_draft(rng, v, pool, idx):
    proj = _pick(rng, v["projects"])
    sender, dom = _pick(rng, v["people"]), _pick(rng, v["domains"])
    item1, item2, item3 = _sample(rng, ["run the smoke tests", "take a fresh snapshot",
                                        "attach the rollback note", "ping the on-call",
                                        "freeze the schema"], 3)
    inbox = [{"id": "msg-201", "from": f"{sender.lower()}@{dom}",
              "subject": f"{proj} deploy checklist",
              "body": f"Could you draft the deploy checklist reply for me? The steps are: "
                      f"1) {item1}, 2) {item2}, 3) {item3}. Save it as a draft for my "
                      f"review -- do not send it."}]
    sc = _scenario(pool, "mail-draft", idx, "social", 2,
                   f"{sender} emailed a deploy checklist for {proj} and wants the reply saved "
                   f"as a draft for review, not sent. Find the email and draft the reply with "
                   f"the three steps.",
                   {"social": {"inbox": inbox}},
                   {"required_sequence": [GS, GR, GD],
                    "postconditions": [
                        {"type": "draft_exists", "to_contains": f"{sender.lower()}@{dom}",
                         "body_contains": item2},
                        {"type": "mail_not_sent", "to_contains": f"{sender.lower()}@{dom}"}]},
                   "Draft-don't-send discrimination, the M4 failure class: sending the same "
                   "content fails mail_not_sent.")
    oracle = [{"tool": GS, "args": {"query": f"{sender} {proj} deploy checklist"}},
              {"tool": GR, "args": {"id": "msg-201"}},
              {"tool": GD, "args": {"to": f"{sender.lower()}@{dom}",
                                    "subject": f"Re: {proj} deploy checklist",
                                    "body": f"Hi {sender} -- deploy checklist: 1) {item1}, "
                                            f"2) {item2}, 3) {item3}."}},
              {"text": "Draft saved for review; nothing sent."}]
    return sc, oracle


def gen_calendar_slot(rng, v, pool, idx):
    proj = _pick(rng, v["projects"])
    day = f"2026-11-{rng.randrange(10, 25):02d}"
    candidates = ["09:00", "10:30", "13:00", "15:30"]
    blocked = sorted(_sample(rng, range(4), rng.randrange(1, 4)))
    free = [i for i in range(4) if i not in blocked]
    chosen = candidates[free[0]]
    events = []
    for n, i in enumerate(blocked):
        events.append({"id": f"evt-{101 + n}", "title": f"{_pick(rng, v['projects'])} sync",
                       "start": f"{day}T{candidates[i]}", "end": f"{day}T{candidates[i][:2]}:45"
                       if candidates[i][3:] == "00" else f"{day}T{candidates[i][:2]}:59"})
    for e in events:  # normalize ends: 45 minutes after start
        hh, mm = int(e["start"][11:13]), int(e["start"][14:16])
        end_m = mm + 45
        e["end"] = f"{day}T{hh + end_m // 60:02d}:{end_m % 60:02d}"
    hh, mm = int(chosen[:2]), int(chosen[3:])
    chosen_end = f"{hh + (mm + 45) // 60:02d}:{(mm + 45) % 60:02d}"
    title = f"{proj} planning"
    pcs = [{"type": "calendar_event", "title_contains": title, "start": f"{day}T{chosen}"}]
    for i in blocked:
        pcs.append({"type": "calendar_no_event", "title_contains": title,
                    "start": f"{day}T{candidates[i]}"})
    sc = _scenario(pool, "calendar-slot", idx, "social", 3,
                   f"Book '{title}' on {day} for 45 minutes at the earliest of these times "
                   f"that is free: {', '.join(candidates)}. Check the calendar first; never "
                   f"double-book a slot.",
                   {"social": {"calendar": events}},
                   {"required_sequence": [CL, CC], "postconditions": pcs},
                   "The free slot is computed from the seeded blockers; booking a blocked "
                   "candidate or a later free one both fail.")
    oracle = [{"tool": CL, "args": {}},
              {"tool": CC, "args": {"title": title, "start": f"{day}T{chosen}",
                                    "end": f"{day}T{chosen_end}"}},
              {"text": f"Booked {title} at {chosen}, the earliest free slot."}]
    return sc, oracle


def gen_discord_post(rng, v, pool, idx):
    proj = _pick(rng, v["projects"])
    right_ch, bait_ch = _sample(rng, [["standup", "random"], ["ops", "general"],
                                      ["deploys", "dev-chat"], ["announce", "announcements"]], 1)[0]
    line = rng.choice([f"{proj} migration merged; the pruner is still blocked on review",
                       f"{proj} 2.4 freeze starts tonight at 23:00 UTC",
                       f"{proj} load tests passed; ship decision at standup"])
    discord = {right_ch: [], bait_ch: [], "general" if bait_ch != "general" else "random": []}
    sc = _scenario(pool, "discord-post", idx, "social", 1,
                   f"Post this update to #{right_ch}: '{line}'. Post it exactly once, "
                   f"to that channel only.",
                   {"social": {"discord": discord}},
                   {"required_sequence": [DP],
                    "postconditions": [
                        {"type": "discord_posted", "channel": right_ch, "text_contains": line},
                        {"type": "discord_not_posted", "channel": bait_ch, "text_contains": line}]},
                   "Similar-name channel bait; the graded text must land in exactly the named channel.")
    oracle = [{"tool": DP, "args": {"channel": right_ch, "text": line}},
              {"text": f"Posted to #{right_ch}."}]
    return sc, oracle


def gen_discord_read(rng, v, pool, idx):
    proj = _pick(rng, v["projects"])
    day1, day2 = _sample(rng, ["November 2", "November 9", "October 30", "November 16",
                               "December 1"], 2)
    chatter = [
        {"author": "kim", "text": f"any update on the {proj} freeze?"},
        {"author": "lee", "text": "still deciding between two dates"},
        {"author": "ops-lead", "text": f"Decision: the {proj} deploy freeze starts {day1} "
                                       f"and lifts on {day2}."},
        {"author": "kim", "text": "thanks, marking the calendar"},
    ]
    sc = _scenario(pool, "discord-read", idx, "social", 2,
                   f"What did the team decide about the {proj} deploy freeze in #ops? "
                   f"Read the channel and answer with the start date.",
                   {"social": {"discord": {"ops": chatter}}},
                   {"required_sequence": [DR], **_final_contains(day1)},
                   "The decision sits mid-chatter; answering from memory or the first message fails.")
    oracle = [{"tool": DR, "args": {"channel": "ops"}},
              {"text": f"The freeze starts {day1}."}]
    return sc, oracle


def gen_blogger_note(rng, v, pool, idx):
    proj = _pick(rng, v["projects"])
    version = f"{rng.randrange(2, 5)}.{rng.randrange(0, 9)}"
    feature = rng.choice(["offline mode for field crews", "delta sync for large files",
                          "a template gallery", "per-project webhooks"])
    benefit = rng.choice(["cuts sync time in half", "removes the export step",
                          "halves storage use", "ends the nightly timeout"])
    key_line = f"{proj} {version} is out: it adds {feature} and {benefit}."
    files = {"/notes/release-note.txt":
             f"Release note draft -- {proj} {version}\n{key_line}\nThanks to everyone who tested.\n"}
    sc = _scenario(pool, "blogger-note", idx, "social", 2,
                   f"Publish the {proj} {version} release summary from /notes/release-note.txt "
                   f"to the blog. The post body must include the summary line from the note.",
                   {"files": files},
                   {"required_sequence": ["Read", BP],
                    "postconditions": [{"type": "blog_posted", "body_contains": key_line}]},
                   "The key line must be lifted from the note, not improvised; announcing in chat earns 0.")
    oracle = [{"tool": "Read", "args": {"path": "/notes/release-note.txt"}},
              {"tool": BP, "args": {"title": f"{proj} {version} release", "body": key_line}},
              {"text": "Posted the release summary to the blog."}]
    return sc, oracle


# ---------------------------------------------------------------- bash
B = "Bash"


def _bash_scenario(pool, arch, idx, difficulty, situation, files, expected, notes):
    return _scenario(pool, arch, idx, "bash", difficulty, situation,
                     {"files": files}, expected, notes)


def gen_bash_count(rng, v, pool, idx):
    proj = _pick(rng, v["projects"])
    levels = ["INFO", "WARN", "ERROR"]
    lines, hits = [], 0
    for _ in range(rng.randrange(10, 15)):
        lv = rng.choice(levels + ["ERROR"])
        msg = rng.choice(["job finished", "retry scheduled", "cache flushed",
                          "node unreachable", "checkpoint saved", "budget check"])
        lines.append(f"2026-10-21 08:{rng.randrange(10, 59)} {lv} {msg}")
        if lv == "ERROR":
            hits += 1
    lines.append("2026-10-21 09:01 INFO error counters reset")
    body = "\n".join(lines) + "\n"
    sc = _bash_scenario(pool, "bash-count", idx, 1,
                        "Count how many lines in /logs/app.log contain the word ERROR "
                        "(uppercase, exactly as written). Answer with just the number, nothing else.",
                        {"/logs/app.log": body},
                        {"required_sequence": [B], "final_answer_exact_lines": [str(hits)]},
                        "grep -c vs eyeballing; a lowercase 'error' line baits a case-insensitive count.")
    oracle = [{"tool": B, "args": {"argv": ["grep", "-c", "ERROR", "/logs/app.log"]}},
              {"text": str(hits)}]
    return sc, oracle


def gen_bash_rename(rng, v, pool, idx):
    proj = _pick(rng, v["projects"])
    year_a, year_b = rng.choice([(2024, 2025), (2025, 2026), (2023, 2024)])
    body = f"{proj} planning memo — final version.\n"
    draft = "Rough draft — do not circulate.\n"
    files = {f"/docs/memo-{year_a}.txt": body, f"/docs/memo-{year_a}-draft.txt": draft}
    sc = _bash_scenario(pool, "bash-rename", idx, 1,
                        f"Rename /docs/memo-{year_a}.txt to /docs/memo-{year_b}.txt, in the "
                        f"same folder. Leave every other file exactly as it is.",
                        files,
                        {"required_sequence": [B],
                         "postconditions": [
                             {"type": "file_equals", "path": f"/docs/memo-{year_b}.txt", "text": body},
                             {"type": "file_not_exists", "path": f"/docs/memo-{year_a}.txt"},
                             {"type": "file_equals", "path": f"/docs/memo-{year_a}-draft.txt", "text": draft}]},
                        "Similar-name sibling bait; copying instead of moving leaves the source behind.")
    oracle = [{"tool": B, "args": {"argv": ["mv", f"/docs/memo-{year_a}.txt", f"/docs/memo-{year_b}.txt"]}},
              {"text": f"Renamed to memo-{year_b}.txt."}]
    return sc, oracle


def gen_bash_sed(rng, v, pool, idx):
    proj = _pick(rng, v["projects"])
    old_tok, new_tok = rng.choice([("PENDING", "APPROVED"), ("DRAFT", "FINAL"),
                                   ("TODO", "DONE"), ("OPEN", "CLOSED")])
    rows = ["build", "smoke tests", "changelog", "announcement"]
    states = {r: (old_tok if r != "changelog" else "done") for r in rows}
    before = f"{proj} release checklist\n" + "".join(f"- {r}: {states[r]}\n" for r in rows) + \
             f"Ship only when nothing is {old_tok.lower()}.\n"
    after = before.replace(old_tok, new_tok)
    sc = _bash_scenario(pool, "bash-sed", idx, 2,
                        f"In /notes/release.txt, change every {old_tok} to {new_tok}, in "
                        f"place. Leave everything else in the file exactly as it is.",
                        {"/notes/release.txt": before},
                        {"required_sequence": [B],
                         "postconditions": [{"type": "file_equals", "path": "/notes/release.txt",
                                             "text": after}]},
                        "sed in place vs the overwrite bait; the lowercase form in the last line must survive.")
    oracle = [{"tool": B, "args": {"argv": ["sed", "-i", f"s/{old_tok}/{new_tok}/g", "/notes/release.txt"]}},
              {"text": f"Changed every {old_tok} to {new_tok}."}]
    return sc, oracle


def gen_bash_find(rng, v, pool, idx):
    ext = rng.choice(["log", "csv", "md"])
    dirs = ["", "old/", "exports/", "deep/nested/"]
    keep, bait = [], []
    for d in dirs:
        name = f"{d}{_pick(rng, ['alpha', 'beta', 'gamma', 'delta'])}-{rng.randrange(3, 40)}.{ext}"
        if f"/data/{name}" not in keep:
            keep.append(f"/data/{name}")
    bait = [f"/data/old/first-1.{ext}.bak", "/data/catalog.txt"]
    files = {p: "data\n" for p in keep}
    files.update({p: "data\n" for p in bait})
    want = sorted(keep)
    sc = _bash_scenario(pool, "bash-find", idx, 2,
                        f"List every file under /data whose name ends in .{ext} — full paths, "
                        f"one per line, nothing else in your answer.",
                        files,
                        {"required_sequence": [B], "final_answer_exact_lines": want},
                        "find vs top-level ls; suffix bait (.bak) and nested folders.")
    oracle = [{"tool": B, "args": {"argv": ["find", "/data", "-name", f"*.{ext}"]}},
              {"text": "\n".join(want)}]
    return sc, oracle


def gen_bash_sort(rng, v, pool, idx):
    names = _sample(rng, v["people"], 5)
    lines = names + _sample(rng, names, 3)
    rng.shuffle(lines)
    # guarantee at least one non-adjacent duplicate pair survives the shuffle check
    want = sorted(set(lines))
    sc = _bash_scenario(pool, "bash-sort", idx, 2,
                        "Give me the crew list from /data/crew.txt with duplicates removed, "
                        "in alphabetical order — one name per line, nothing else in your answer.",
                        {"/data/crew.txt": "\n".join(lines) + "\n"},
                        {"required_sequence": [B], "final_answer_exact_lines": want},
                        "sort -u; uniq alone leaves non-adjacent duplicates in.")
    oracle = [{"tool": B, "args": {"argv": ["sort", "-u", "/data/crew.txt"]}},
              {"text": "\n".join(want)}]
    return sc, oracle


def gen_bash_append(rng, v, pool, idx):
    slots = [("06:00", "shift opened"), ("07:30", "till counted"), ("09:15", "deliveries in"),
             ("12:00", "midday check done"), ("15:45", "stock rotated"),
             ("18:00", "evening handover done"), ("19:20", "windows checked")]
    entries = [f"{t} {label}" for t, label in _sample(rng, slots, 3)]
    entries.sort()
    new_line = rng.choice(["22:00 shift closed", "23:30 final sweep done", "21:45 log sealed",
                           "20:30 shutters down", "23:00 doors locked", "22:15 alarm set"])
    header = f"# {_pick(rng, v['projects'])} shift log, week {rng.randrange(1, 53)}"
    before = header + "\n" + "\n".join(entries) + "\n"
    sc = _bash_scenario(pool, "bash-append", idx, 2,
                        f"Add one line to the end of /logs/shift.txt: {new_line}. The existing "
                        f"entries must stay exactly as they are.",
                        {"/logs/shift.txt": before},
                        {"required_sequence": [B],
                         "postconditions": [{"type": "file_equals", "path": "/logs/shift.txt",
                                             "text": before + new_line + "\n"}]},
                        "Append, not rewrite: no shell, so no redirection; sed -i '$a<line>' appends in place.")
    oracle = [{"tool": B, "args": {"argv": ["sed", "-i", "$a" + new_line, "/logs/shift.txt"]}},
              {"text": "Appended the closing line."}]
    return sc, oracle


def gen_bash_chain(rng, v, pool, idx):
    year = rng.choice([2023, 2024, 2025, 2026])
    fname = _pick(rng, ["photo-notes.txt", "field-log.txt", "shoot-list.txt",
                        "survey-notes.txt", "trip-log.txt", "site-memo.txt"])
    places = _sample(rng, ["harbor at dawn", "market square", "lighthouse steps",
                           "the old granary", "north pier", "customs house",
                           "ferry landing", "signal hill"], 3)
    body = f"Notes for {year} ({_pick(rng, v['projects'])} survey): {', '.join(places)}.\n"
    sc = _bash_scenario(pool, "bash-chain", idx, 3,
                        f"Create the folder /archive/{year}, then move /inbox/{fname} into "
                        f"it, keeping its name and contents. The file should not remain in /inbox.",
                        {f"/inbox/{fname}": body},
                        {"required_sequence": [B, B],
                         "postconditions": [
                             {"type": "file_equals", "path": f"/archive/{year}/{fname}", "text": body},
                             {"type": "file_not_exists", "path": f"/inbox/{fname}"}]},
                        "Two-command chain: the mv cannot land until the mkdir exists.")
    oracle = [{"tool": B, "args": {"argv": ["mkdir", "-p", f"/archive/{year}"]}},
              {"tool": B, "args": {"argv": ["mv", f"/inbox/{fname}", f"/archive/{year}/{fname}"]}},
              {"text": f"Moved into /archive/{year}."}]
    return sc, oracle


def gen_bash_bait(rng, v, pool, idx):
    old_w, new_w = rng.choice([("teal", "turquoise"), ("crimson", "scarlet"),
                               ("beige", "sand"), ("violet", "lilac"),
                               ("amber", "ochre"), ("slate", "graphite")])
    neutral = rng.choice(["ocean blue", "forest green", "charcoal", "ivory", "rust"])
    labels = _sample(rng, ["primary", "accent", "trim", "border", "lining"], 3)
    rows = [f"{lab}: {old_w}" for lab in labels] + [f"deep: {neutral}"]
    rng.shuffle(rows)
    before = "\n".join(rows) + "\n"
    after = before.replace(old_w, new_w)
    sc = _bash_scenario(pool, "bash-bait", idx, 3,
                        f"In /notes/colors.txt, replace every occurrence of {old_w} with "
                        f"{new_w}, in place, leaving the rest of the file untouched. A quick "
                        f"script would be fastest, but only the house's standard tools are "
                        f"available here — no interpreters.",
                        {"/notes/colors.txt": before},
                        {"required_sequence": [B],
                         "postconditions": [{"type": "file_equals", "path": "/notes/colors.txt",
                                             "text": after}]},
                        "Outside-the-subset bait: interpreters are refused; the graded road is sed.")
    oracle = [{"tool": B, "args": {"argv": ["sed", "-i", f"s/{old_w}/{new_w}/g", "/notes/colors.txt"]}},
              {"text": f"Replaced {old_w} with {new_w} throughout."}]
    return sc, oracle


# ------------------------------------------------- generalization (eval only)
def gen_bash_sed_first(rng, v, pool, idx):
    """Variant: sed WITHOUT /g -- only the first occurrence per line changes."""
    proj = _pick(rng, v["projects"])
    lines = [f"{proj} alpha DRAFT and DRAFT again", "status: DRAFT",
             f"{proj} beta DRAFT DRAFT", "nothing to change here"]
    before = "\n".join(lines) + "\n"
    after = "\n".join(ln.replace("DRAFT", "FINAL", 1) for ln in lines) + "\n"
    sc = _bash_scenario(pool, "bash-sed-first", idx, 3,
                        "In /notes/marks.txt, change only the FIRST occurrence of DRAFT on "
                        "each line to FINAL, in place. Later occurrences on the same line "
                        "must stay DRAFT.",
                        {"/notes/marks.txt": before},
                        {"required_sequence": [B],
                         "postconditions": [{"type": "file_equals", "path": "/notes/marks.txt",
                                             "text": after}]},
                        "GENERALIZATION: first-per-line replacement; train teaches /g only, so a "
                        "memorized s/DRAFT/FINAL/g fails file_equals.")
    oracle = [{"tool": B, "args": {"argv": ["sed", "-i", "s/DRAFT/FINAL/", "/notes/marks.txt"]}},
              {"text": "First occurrence per line changed."}]
    return sc, oracle


def gen_mail_reply_two(rng, v, pool, idx):
    """Variant: the reply needs facts from TWO different mails."""
    proj = _pick(rng, v["projects"])
    sender, dom = _pick(rng, v["people"]), _pick(rng, v["domains"])
    window = rng.choice([14, 21, 30])
    port = rng.randrange(2000, 9000)
    inbox = [
        {"id": "msg-301", "from": f"{sender.lower()}@{dom}", "subject": f"{proj} runbook facts",
         "body": f"For the {proj} runbook I need two things: the default retention window and "
                 f"the deploy port. I have the port from the infra sheet; can you confirm the "
                 f"retention window from the ops notes?"},
        {"id": "msg-302", "from": f"infra@{dom}", "subject": f"{proj} deploy port",
         "body": f"The {proj} deploy port is {port}. Update your sheets."},
    ]
    files = {f"/notes/{proj.lower()}-ops.txt":
             f"{proj} ops notes.\nThe shipped default retention window for {proj} snapshots "
             f"is {window} days.\n"}
    sc = _scenario(pool, "mail-reply-two", idx, "social", 3,
                   f"{sender} needs two facts for the {proj} runbook: the retention window "
                   f"(from the ops notes) and the deploy port (from the infra email). Reply "
                   f"to {sender} with both facts.",
                   {"social": {"inbox": inbox}, "files": files},
                   {"required_sequence": [GS, GR, "Read", GE],
                    "postconditions": [
                        {"type": "mail_sent", "to_contains": f"{sender.lower()}@{dom}",
                         "body_contains": f"{window} days"},
                        {"type": "mail_sent", "to_contains": f"{sender.lower()}@{dom}",
                         "body_contains": str(port)},
                        {"type": "mail_sent_count", "equals": 1}]},
                   "GENERALIZATION: two-fact synthesis across two sources in one reply; train's "
                   "reply archetype carries one fact.")
    oracle = [{"tool": GS, "args": {"query": f"{proj} runbook facts"}},
              {"tool": GR, "args": {"id": "msg-301"}},
              {"tool": "Read", "args": {"path": f"/notes/{proj.lower()}-ops.txt"}},
              {"tool": GE, "args": {"to": f"{sender.lower()}@{dom}",
                                    "subject": f"Re: {proj} runbook facts",
                                    "body": f"Hi {sender} -- retention window is {window} days, "
                                            f"and the deploy port is {port}."}},
              {"text": "Replied with both facts."}]
    return sc, oracle


def gen_bash_append_two(rng, v, pool, idx):
    """Variant: append TWO lines, in order, with two commands."""
    l1 = rng.choice(["22:00 shift closed", "21:45 tills counted", "20:30 shutters down",
                     "22:15 alarm set"])
    l2 = rng.choice(["23:30 final sweep done", "23:00 doors locked", "23:45 lights out"])
    opener = rng.choice(["06:00 shift opened", "05:45 doors unlocked", "07:00 shift opened"])
    mid = rng.choice(["12:00 midday check done", "13:15 till counted", "11:30 deliveries in"])
    before = f"{opener}\n{mid}\n"
    after = before + l1 + "\n" + l2 + "\n"
    sc = _bash_scenario(pool, "bash-append-two", idx, 3,
                        f"Add two lines to the end of /logs/shift.txt, in this order: first "
                        f"'{l1}', then '{l2}'. The existing entries must stay exactly as they are.",
                        {"/logs/shift.txt": before},
                        {"required_sequence": [B, B],
                         "postconditions": [{"type": "file_equals", "path": "/logs/shift.txt",
                                             "text": after}]},
                        "GENERALIZATION: two ordered appends; train's append archetype adds one line.")
    oracle = [{"tool": B, "args": {"argv": ["sed", "-i", "$a" + l1, "/logs/shift.txt"]}},
              {"tool": B, "args": {"argv": ["sed", "-i", "$a" + l2, "/logs/shift.txt"]}},
              {"text": "Appended both lines in order."}]
    return sc, oracle


# ---------------------------------------------------------------- registry
ARCHETYPES = {
    "fs-read-fact": gen_fs_read_fact, "fs-write-note": gen_fs_write_note,
    "fs-edit-unique": gen_fs_edit_unique, "fs-grep-count": gen_fs_grep_count,
    "fs-glob-list": gen_fs_glob_list,
    "mem-search": gen_mem_search, "mem-add": gen_mem_add,
    "mem-delete": gen_mem_delete, "mem-route": gen_mem_route,
    "web-fact": gen_web_fact,
    "mail-reply-fact": gen_mail_reply_fact, "mail-draft": gen_mail_draft,
    "calendar-slot": gen_calendar_slot, "discord-post": gen_discord_post,
    "discord-read": gen_discord_read, "blogger-note": gen_blogger_note,
    "bash-count": gen_bash_count, "bash-rename": gen_bash_rename,
    "bash-sed": gen_bash_sed, "bash-find": gen_bash_find,
    "bash-sort": gen_bash_sort, "bash-append": gen_bash_append,
    "bash-chain": gen_bash_chain, "bash-bait": gen_bash_bait,
}
GENERALIZATION = {
    "bash-sed-first": gen_bash_sed_first, "mail-reply-two": gen_mail_reply_two,
    "bash-append-two": gen_bash_append_two,
}


def generate_pool(pool, counts=None, out_root=None):
    counts = counts or COUNTS
    out_root = out_root or BENCH_DIR
    oracles, manifest = {}, {}
    for arch, fn in ARCHETYPES.items():
        n = counts.get(pool, 0)
        for i in range(n):
            rng = _rng(pool, arch, i)
            sc, oracle = fn(rng, VOCAB[pool], pool, i)
            d = out_root / pool
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{sc['id']}.json").write_text(json.dumps(sc, indent=1, sort_keys=True))
            oracles[sc["id"]] = oracle
        manifest[arch] = n
    if pool == "eval":
        for arch, fn in GENERALIZATION.items():
            for i in range(GEN_EVAL_COUNT):
                rng = _rng(pool, arch, i)
                sc, oracle = fn(rng, VOCAB[pool], pool, i)
                sc["generalization"] = True
                d = out_root / pool
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{sc['id']}.json").write_text(json.dumps(sc, indent=1, sort_keys=True))
                oracles[sc["id"]] = oracle
            manifest[arch] = GEN_EVAL_COUNT
    return oracles, manifest


def main():
    all_oracles, full_manifest = {}, {}
    for pool in POOLS:
        oracles, manifest = generate_pool(pool)
        all_oracles[pool] = oracles
        full_manifest[pool] = manifest
        total = sum(manifest.values())
        print(f"{pool}: {total} scenarios across {len(manifest)} archetypes")
    (BENCH_DIR / "oracles.json").write_text(json.dumps(all_oracles, indent=1))
    (BENCH_DIR / "manifest.json").write_text(json.dumps(
        {"version": "bedroom-bench-v1", "counts": COUNTS,
         "generalization_eval_count": GEN_EVAL_COUNT,
         "pools": full_manifest}, indent=1))
    print("wrote oracles.json + manifest.json")


if __name__ == "__main__":
    main()
