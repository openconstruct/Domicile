# Domicile tool contract (what the model sees)

A native local model in Domicile calls tools through `local_tools.dispatch()` and
receives one **string** back. That string is what Bedroom trains on, so this file
records each tool's call shape, its exact result text, and its failure text.

**Status tags**
- **VERIFIED**: the same calls were run through the real `local_tools` and through
  Bedroom's dollhouse, and the output matched (`tests/conformance_domicile.py`,
  30 cases, 0 deltas on 2026-10-22). The examples below are real output.
- **SOURCE-READ**: signatures and docstrings read from the source; output not run.
- **IMPLEMENTED**: exists in Domicile and has unit tests, but has not been run live.

Two kinds of failure text, for every local tool:
- `DENIED: <reason>`: a guard refused (policy). Not a bug; the model should adapt.
- `ERROR: <ExceptionType>: <message>`: the tool ran and raised.

Results longer than 12,000 chars are cut to the first and last 6,000 with
`... [N chars trimmed] ...` between them.

---

## 1. File tools (VERIFIED)

Paths are absolute. `file_path`, `old_string`, `new_string` are accepted as aliases
for `path`, `old`, `new` on Read/Write/Edit/Glob/Grep.

### Read
`{"path": str, "offset": int (1-based), "limit": int (default 2000)}`

```
Read {"path": "/d/a.txt"}
     1	alpha
     2	beta
     3	gamma

Read {"path": "/d/a.txt", "offset": 2, "limit": 1}
     2	beta
[showing lines 2-2 of 3; call Read again with offset=3 to continue]

Read {"path": "/d/nope.txt"}
ERROR: FileNotFoundError: /d/nope.txt
```
Empty file: `(empty file)`. Offset past the end: `(file has only N lines)`.
Reading marks the file as read for Write/Edit below.

### Write
`{"path": str, "content": str}`
```
Write {"path": "/d/new.txt", "content": "hi\n"}
tool=Write; path=/d/new.txt; bytes=3; created=True

Write on an existing file that was not Read this session:
DENIED: refused: /d/a.txt exists but has not been read this session. Read it first -- overwriting a file blind is how work gets lost.
```
After a Read, an overwrite returns `...; created=False`. If the file changed on disk
since the Read: `DENIED: refused: <path> changed on disk since you last read it. Read it again before modifying it.`

### Edit
`{"path": str, "old": str, "new": str, "replace_all": bool}`
```
Edit (after Read)      tool=Edit; path=/d/a.txt; replaced=1
Edit without Read      DENIED: refused: read /d/a.txt before editing it.
old not present        ERROR: ValueError: old string not found (whitespace must match exactly)
old appears 4 times    ERROR: ValueError: old string appears 4 times; pass replace_all or include more surrounding context to make it unique
```
Also: `old must not be empty`, `old and new are identical; nothing to change`.

### Glob
`{"pattern": "**/*.md", "path": "/d"}`. Matches files only, newest first, max 200;
`{a,b}` braces expand.
```
Glob {"pattern": "*.txt", "path": "/d"}
/d/a.txt
/d/nonl.txt
/d/b.txt
```
No match: `(no matches)`. Over 200: last line `... N more not shown; narrow the pattern`.

### Grep (ripgrep regex)
`{"pattern": str, "path": str, "glob": str, "output_mode": "content"|"files_with_matches"|"count",
"ignore_case": bool, "line_numbers": bool, "after"|"before"|"context": int, "multiline": bool,
"type": str, "head_limit": int (250), "offset": int}`
```
content (default)       /d/a.txt:1:alpha
                        /d/b.txt:3:alpha again
files_with_matches      /d/a.txt
                        /d/b.txt
count                   /d/a.txt:1
                        /d/b.txt:1
no match                (no matches)
over head_limit         ...last line: [N more lines; raise head_limit or use offset]
```

---

## 2. Bash (VERIFIED for the commands Bedroom implements)

`{"argv": [str, ...], "cwd": str, "timeout_s": number}`. **argv is a list. There is no
shell**: pipes, redirects (`>`, `>>`), globs, `&&` and `$()` are not interpreted; they
reach the program as literal arguments. Chain steps with separate calls.

Result rendering: stdout, then `[stderr]\n<text>` if any, then `[timed out after Ns]`
if it timed out, then `[exit N]` if the code is nonzero. No output at all:
`(no output)`.

```
["ls","/d"]                      a.txt
                                 b.txt
                                 nonl.txt
                                 sub
["wc","-l","/d/a.txt"]           3 /d/a.txt
["grep","-c","alpha","/d/b.txt"] 1
["grep","zzz","/d/b.txt"]        [exit 1]
["ls","/d/*.txt"]                [stderr]
                                 ls: cannot access '/d/*.txt': No such file or directory
                                 [exit 2]
["echo","x",">>","/d/a.txt"]     x >> /d/a.txt        (">>" is just an argument)
["sed","-i","$aNEW","/d/a.txt"]  (no output)          (appends a line "NEW")
["bash","-c","ls /d"]            DENIED: 'bash' is not available: commands run as an argv list with no shell, so shells and privilege tools are blocked. Run the program directly instead of via a shell.
["python3","-c","...os.remove..."]  DENIED: refused: inline code (-c) that looks destructive. Write it to a file and run that, so it can be read before it executes.
[]                               DENIED: empty command
["notacommand"]                  ERROR: ValueError: command not found: notacommand
```

Guards (from `local_tools.guard_argv`): blocked executables are `bash sh zsh fish dash
ksh csh tcsh cmd powershell pwsh sudo doas su pkexec`. For interpreters (`python*
perl ruby node deno bun php lua awk sed env xargs find git vim ssh make docker nohup
setsid timeout ...`) an inline-code flag (`-c -e -E --eval --exec -s -p ...`) is
refused only if the line looks destructive (`rm mv cp dd tee chmod --force -f --delete
...`). Destructive git (`reset --hard`, `clean -f`, `push --force`, `branch -D`,
`checkout --`, `restore`, ...) is refused.

**Bedroom's simulation limit:** Bash runs only `ls cat grep wc sort uniq head tail mv cp
mkdir echo sed find`. Anything else returns
`[stderr]\n<cmd>: not available in this simulated environment\n[exit 127]`
where the real tool would run it. The real tool has no allowlist.

### ToolWake (SOURCE-READ, not simulated)
`{"argv": [...], "cwd": str, "label": str}`. Starts a background job and returns a task id
at once; Domicile starts a NEW turn carrying the exit code and output tail when the job
ends. Same argv rules as Bash.

---

## 3. Web (WebSearch/WebFetch: SOURCE-READ and result-shaped to source)

### WebSearch `{"query": str, "count": int (default 5, max 10)}`
```
- <title>
  <url>
  <snippet, max 420 chars>

- <title>
  ...
```
No results: `(no results)`. Empty query: `ERROR: ValueError: query is empty`.
Missing dependency: `ERROR: RuntimeError: ddgs not installed (pip install ddgs)`.

### WebFetch `{"url": str, "max_chars": int (default 12000, clamped 800-40000)}`
Returns cleaned page text (scripts, styles, nav, header, footer removed). Longer than
the cap: the first and last halves with `... [N chars trimmed] ...` between. Non-text
content type: `[non-text content-type '...']` plus the first 800 chars. Errors:
`url is empty`, `url must be http(s)`, HTTP failures as `ERROR: HTTPStatusError: ...`.

Tool names are CamelCase: `WebSearch`, `WebFetch`. (Bedroom's old lowercase
`websearch`/`webfetch` did not exist in Domicile.)

---

## 4. Memory (names VERIFIED; argument lists SOURCE-READ; result text seen live)

Memory tools are MCP tools, so their model-facing names are `mcp__<server>__<tool>`.
Each returns a plain string.

### mcp__memory__memory_search `{"query": str, "limit": int (5)}`
Semantic search. One line per hit, best first:
```
[0.867] 2026-10-22 Jerry decided to house Bedroom under `domicile/bedroom` ...
[0.857] 2026-09-16 On 2026-09-16, Jerry decided to ship Bedroom ...
```
(`[similarity] date text`)

### mcp__memory__memory_add `{"text": str, "extract": bool (true), "metadata": object, "idempotency_key": str, "guard": bool (false)}`
`extract=true` sends the text to an extractor that splits it into facts and reconciles
with stored ones; `extract=false` stores it verbatim. Live replies (VERIFIED, 2026-10-22):
```
extract=true    ADD: Test probe ZZTEST-... uses 'a purple walrus' as the sentinel for tool reply capture.
extract=false   stored verbatim: <first 100 chars of the text>
```
The extractor rewrites the text, and the reply shows the stored fact, prefixed with the
action (`ADD:`; reconciliation can presumably also report update or delete actions;
only `ADD:` was seen).

### mcp__memory__memory_list `{"limit": int (20)}`
```
8347 memories total, showing 2:
- 2026-10-22 (aeef9cb6) 2026-10-22: Rook made Bedroom's dollhouse tools conform ...
- 2026-10-22 (01378448) On 2026-10-20, Jerry evaluated Lili samples ...
```
(`- date (8-char id) text`; the id is what `memory_delete` takes.)

### mcp__memory__memory_delete `{"memory_id": str}`
Id or unique id prefix from `memory_list`. Live replies (VERIFIED):
```
deleted: <first 80 chars of the removed memory>
no memory with id starting 'zzzzzzzz'
```

### mcp__mem2__mem2_search `{"query": str, "limit": int (5), "date_from": "YYYY-MM-DD", "date_to": "YYYY-MM-DD"}`
Hybrid semantic + keyword search of past conversations:
```
[1] via=sem+kw score=0.0350 date=2026-09-16 chat=Main Chat ref=/home/.../2026-09-16.log#19
[2026-09-16 06:16] Jerry: ...
[2026-09-16 06:17] Rook: ...
```
`ref=<source_file>#<chunk_index>` feeds `mem2_get_chunk`.

### mcp__mem2__mem2_list_dates `{}`
One line per log: `2026-10-21  (86,985 bytes)`.

### mcp__mem2__mem2_get_chunk `{"source_file": str, "chunk_index": int}`
Full text of one chunk (`source_file` = the path before `#` in a `ref=`). Live replies (VERIFIED):
```
date: 2026-09-16
chat: Main Chat
---
[2026-09-16 06:16] Jerry: ...
[2026-09-16 06:17] Rook: ...

Chunk not found: /nonexistent.log#0
```

**Bedroom gap:** its memory dollhouse returns simplified strings and one-line tool
descriptions. Until it is re-shaped to the formats above and fed the real docstrings,
memory scenarios are not contract-faithful.

---

## 5. Google tools (IMPLEMENTED in Domicile 2026-10-22; unit-tested with mocked clients, not live)

`domicile/google_tools.py` wraps the existing `google_auth/*_client.py` functions and
registers them in `local_tools`. Code and tests: `google_tools.py`, `_google_tools_test.py`
(17 tests). **Not run against a live Google account.**

**Opt-in.** The names are in no default tier (`readonly`/`write`/`sandbox`/`wakeup`/`full`).
A role gets them only if its tool spec lists them, because reading someone's mail is a
per-role privacy decision.

**Approval gate.** Write tools call `google_tools.set_approver(fn)`'s callback
(`fn(tool, summary, args) -> bool`) before any network call. With no approver set they
are refused, so they fail closed until the approve popup is wired in:
`DENIED: GmailSend needs your approval and no approver is configured, so it was not run.`
Declined: `DENIED: GmailSend was not approved.` Blogger posts default to a draft;
`draft=false` is labelled `PUBLISH LIVE` in the approval summary.

| Tool | Arguments (* required) | Success text | Gate |
|---|---|---|---|
| `GmailSearch` | `query`, `max_results` (20, max 100) | one block per message: `<id>  <date>  <from>  <subject>` then `    <snippet>`; or `(no matches)` | none |
| `GmailRead` | `id`* | `From: ..\nTo: ..\nSubject: ..\nDate: ..\n\n<body>` | none |
| `GmailSend` | `to`* `subject`* `body`* `thread_id` | `tool=GmailSend; to=<addr>; id=<message id>` | approve |
| `CalendarList` | `max_results` (10), `calendar_id` (primary) | `<start>  <title>  [<location>]` per event, soonest first; or `(no upcoming events)` | none |
| `CalendarCreate` | `summary`* `start`* `end`* (RFC3339) `description` `calendar_id` | `tool=CalendarCreate; id=<event id>; start=<start>` | approve |
| `BloggerPost` | `title`* `content_html`* `draft` (true) | `tool=BloggerPost; id=<post id>; status=draft\|live` | approve |
| `ChatRead` | `space`*, `page_size` (20) | `<time>  <sender> (<HUMAN\|BOT>): <text>` per message, newest first; or `(no messages)` | none |
| `ChatSend` | `space`* `text`* `thread` | `tool=ChatSend; space=<space>; name=<message resource name>` | approve |

Failures use the standard forms: a missing required argument is
`ERROR: ValueError: <arg> is required`; a client failure (expired token, API error) is
`ERROR: <ExceptionType>: <message>`. Results are cut at 12,000 chars like every tool.

**Discord** has no tool and will not get one: the bridge delivers a turn's reply to the
user's DM or channel, so answering *is* posting. Bedroom's `discord_read`/`discord_post`
and the lowercase `gmail_*`, `calendar_*`, `blogger_post` names should be removed or
renamed to the above; the Bedroom social dollhouse still uses the old invented names
and is **not contract-faithful** until it is re-shaped to this table.

## 6. Command-line reference (Bedroom)

All run from the `bedroom/` directory. Bench scripts take no options.

| Command | Purpose | Expected result |
|---|---|---|
| `python3 tests/conformance_domicile.py [--domicile DIR] [-v]` | Compare the dollhouse to the real `local_tools` | `0 deltas; 30 cases compared, 2 listed sim-limits skipped`; exit 1 on any difference |
| `python3 -m bedroom.sync_domicile_schemas [--domicile DIR]` | Re-export the real tool schemas | `wrote .../bedroom/domicile_tools.json (8 tools)` |
| `python3 bench/generate.py` | Regenerate train/dev/eval scenarios and oracles (deterministic) | `train: 1680 ...  dev: 96 ...  eval: 312 ...` |
| `python3 bench/audit.py` | Run every oracle through the dollhouse | `AUDIT OK` (oracle scores 1, lazy student 0, no answer leaks) |
| `python3 bench/freeze.py` / `--verify` | Hash the eval pool; refuses to overwrite a freeze | `froze 312 eval scenarios, aggregate ...` / `FREEZE VERIFY OK` |
| `python3 tests/smoke_test*.py` | Mock-student smoke tests | each ends `... SMOKE OK` |
| `python3 -m bedroom.train.eval_hf MODEL [--pool train\|dev\|eval] [--limit N] [--max-new-tokens 512] [--batch 32] [--out FILE]` | Score a model (HF weights) on a pool | JSON report with overall and by-family/difficulty rewards |
| `python3 -m bedroom.train.train_hf MODEL --out DIR [--rounds 10] [--scenarios-per-round 8] [--group-size 8] [--lr 2e-5] [--lora-r 16] [--temperature 1.0] [--max-new-tokens 512] [--weights fam=w,...] [--seed 0] [--difficulty-schedule "0:1,12:2,22:3"]` | LoRA GRPO | per-round log: `round N: reward=... loss=... dead_groups=a/b by_family={...}`; adapter checkpoints in DIR |

Notes: `--difficulty-schedule` is `round:max_difficulty` pairs (step function).
`dead_groups` counts scenarios where all samples got the same reward (no learning signal).
After any change to Domicile's tools: run `sync_domicile_schemas`, then the conformance
test, then (if scenarios or oracles changed) `generate`, `audit`, and a new `freeze`; a
changed eval pool needs a fresh baseline.

---

## 7. Not yet covered

ReadImage, ExpressToPane, SetWakeup, CompactNow, and the full Bash (everything outside
the 14 simulated commands). Bedroom's social
dollhouse and memory dollhouse (re-shape to sections 4-5).
