# AGENTS.md — Domicile, for agents living or working in this tree

Domicile is a home for one continuous agent: one memory, one history, running on
your own machine, reachable from a terminal UI, Discord, Google Chat or a web
page. This file is the map: what exists, where it lives, how it is built, and
the rules for changing it. If you are the resident, it is also how you make the
place yours.

Paths below are relative to the install directory (wherever this folder lives).
Never hardcode it: `paths.py` finds it from its own location.

---

## 1. Rules for anyone editing this tree

**Do not destroy the user's data.** Never delete, overwrite, truncate, `git reset
--hard`, `git clean`, force-push or rebase anything you did not create in this
session. Treat these as read-only unless the task names the file:

- `data/memory/*.md` (curated memory), `data/chats/`, `data/sessions/`,
  `data/plans/` (history), `data/mem0/` (memory store, logs, cursors),
  `data/backup/`, `data/backup-state.json`
- `.env` (all credentials) and `data/google/token.json`
- any `*.sqlite`, `storage.sqlite`, qdrant `meta.json`

**Editable:** the `*.py` files, `providers/`, `google_auth/`, `rss/`,
`templates/`, `webui/`, and config *schema* when a task calls for it. Touch only
what the task needs; no drive-by refactors.

**Concurrency.** The mem0 store is single-writer and the running app holds its
lock. Don't run `mem0_core`, `mem0_server`, `mem0_nightly` or `mem0_audit`
while the app may be running, don't run the nightly job against the live store
to "test" it, and never start a second instance. A running process does not
pick up edits: say whether a restart is needed, and mark a fix UNVERIFIED if
you could not test it against a fresh start.

**Git.** Work on a branch. Stage the files you changed, not `git add -A`. Never
rewrite published history. Describe what and why in the message.

**Tests.** Run the code path you changed. Clean up scratch files in the same
session. `./run_tests.sh --quick` runs everything; one file:
`python3 _test_runner.py _name_test.py`.

**Stop and leave a note** when a task needs a protected path, spans several of
{app code, mem0, mem2, backup} and you are unsure the paths line up, fails tests
you can't explain, needs history rewritten, or is growing past ~5 files
without converging.

---

## 2. Features

- **Chat harness** (`cchat.py`): Textual terminal UI, one continuous chat, modes
  (chat / build, F5), per-turn model, provider and thinking-level switching
  (F3/F4/F6), side panel, `/compact`.
- **Providers**: Claude (via the Claude Code CLI/SDK), Codex (ChatGPT login via
  the `codex` binary), any OpenAI-compatible endpoint, and Responses-style
  endpoints. Automatic fallback down a provider list on caps, rate limits and
  expired auth.
- **Memory in three layers** (section 5): curated files always in context, a
  searchable fact store, and search over raw transcripts.
- **Tools** (section 6): files, shell, web, images, background jobs, wakeups,
  display pane. Permissions are profiles, enforced by a write sandbox.
- **Bridges**: Discord, Google Chat (polling), and an OpenAI-compatible HTTP API
  with a bundled web UI (`api_server.py`, optional, needs aiohttp).
- **Wakeups**: the agent can wake itself on a schedule (quiet hours supported)
  and spend the time on its own interests; results go to the user.
- **Google**: Gmail (unread digest to the agent), Calendar, Blogger, Chat.
- **RSS**: an hourly feed check; new items reach the agent as a background task.
- **Backups**: a snapshot every 4 days (run from inside the app) to a private
  GitHub repo, with a visibility check on every push and a restore path
  (`backup.py`). Without a token it only commits a local snapshot under
  `data/backup/`.
- **Setup wizard** (`wizard.py`, `install.sh`): provider, context window, memory
  import (ChatGPT, Claude, SQL, text, mem0), persona, Discord, Google, systemd
  units, backup, smoke test. `python3 wizard.py --edit` reopens the editors.
- **Health**: `selftest.py`.

---

## 3. Layout

| Area | Files |
|---|---|
| App | `cchat.py` (UI + turn loop), `config_loader.py`, `paths.py`, `credentials.py`, `file_cache.py` |
| Providers | `providers/` (`base`, `openai_compatible`, `openai_responses`, `codex_cli`, `mcp_client`), `provider_rollover.py` |
| Context | `context_diet.py` (compaction, tool-result budgets), `tool_spill.py` + `fetch_result.py` (trimmed output stays recoverable), `midturn.py` (messages sent mid-turn) |
| Memory | `mem1.py`; `mem0_core.py`, `mem0_server.py`, `mem0_ipc.py`, `mem0_hourly.py`, `mem0_nightly.py`, `mem0_rescan.py`, `mem0_guard.py`, `mem0_audit.py`, `mem0_*_llm.py`; `mem2_server.py`; `meditate_dryrun.py` + `meditate_apply.py` |
| Tools | `local_tools.py`, `sandbox.py`, `allow.py`, `bgrun.py`, `bgtask_runner.py`, `codex_tools_server.py`, `expression.py`, `pdf_read_hook.py` |
| Bridges | `discord_bridge.py`, `chat_bridge.py`, `api_server.py` + `webui/`, `inbound_files.py` |
| Integrations | `google_auth/`, `rss/` |
| Backup | `backup.py`, `sync_guard.py` |
| Setup | `wizard.py` (UI only), `wizard_*.py` (logic), `import_*.py`, `install.sh`, `templates/` |
| Ops | ``selftest.py`, `inputlog.py`, `run_tests.sh`, `_test_runner.py` |

Runtime state is under `data/` (git-ignored; only `data/.gitkeep` ships).

---

## 4. Configuration and data

- **`paths.py`** is the only place that knows where things are. `ROOT` is the
  folder it sits in; data is `ROOT/data`, or `$DOMICILE_DATA` if set (tests and
  the systemd units use that). Import constants from it; never build a path from
  `~` or a literal folder name.
- **`.env`** holds every secret, read only through `credentials.py`
  (`get`, `read_key`, `save`; `save` is atomic and keeps mode 600). Config files
  refer to a key by name (`key_env`), never by value.
- **`data/config.json`**: `active_provider`, `provider_order`, `providers`
  (per-provider `adapter`, `models` per role, `thinking`, `vision`),
  `context`, `backup`, `hourly_ingest`, `google.blog_url`. Read with
  `config_loader.load_config()`; write with `save_config()` (deep-merge, atomic).
- **`data/roles.json`**: what each role may do (`profile`, `write_roots`,
  `max_tool_rounds`). Which model a role uses lives in `config.json`. Edit
  permissions with `python3 allow.py`, not by hand.
- **`data/identity.json`**: operator and agent names, memory scope key. Read via
  `config_loader.load_identity()`; code never assumes a name.

---

## 5. Memory

1. **mem1**, `data/memory/*.md`: always in the system prompt, small and curated.
   `identity.md`, `user.md`, `practices.md`, `goals.md`, `open.md`,
   `reflections.md`, plus one job-description file per mode (`chat.md`,
   `code-building.md`, `compact.md`, created from `templates/memory/`).
   `mem1.py` enforces per-file token caps; going over truncates the tail, so
   keep files lean and move detail to mem0.
2. **mem0**, MCP server `memory`: extracted facts in a Qdrant store, searched on
   demand. Filled by hourly and nightly ingestion of transcripts (`mem0_hourly`,
   `mem0_nightly`) and by the agent via `memory_add`. Tools: `memory_search`,
   `memory_add`, `memory_list`, `memory_delete`.
3. **mem2**, MCP server `mem2`: semantic search over raw transcript chunks.
   Tools: `mem2_search`, `mem2_get_chunk`, `mem2_list_dates`.

Search mem0 before saying you don't remember something. You may fix bugs in the
code behind these layers; don't change the routing between them or edit memory
content without a specific instruction.

---

## 6. Tools and permissions

**Tool set** (`local_tools.py`, one implementation shared by every provider):
`Read`, `Write`, `Edit`, `Glob`, `Grep`, `Bash`, `ReadImage`, `WebSearch`,
`WebFetch`, `ToolWake`, `SetWakeup`, `CompactNow`, `ExpressToPane`. Each has an
OpenAI-style schema in `local_tools._SCHEMAS`; that schema is the single source
the other paths derive from.

**How each provider gets them**
- OpenAI-compatible: schemas sent directly, executed in `providers/openai_compatible.py`.
- Claude: served in-process as an MCP server named `cchat`; the SDK's built-in
  tools are off unless `claude_builtin_tools` is set in `config.json`.
- Codex: `codex_tools_server.py` is a stdio MCP server (`domicile_tools`)
  generated from the same schemas. Codex's own native tools are disabled; the
  adapter refuses to run if that lockdown can't be applied (fail closed).
- Memory tools come from the `memory` and `mem2` MCP servers; `providers/mcp_client.py`
  keeps long-lived sessions to them.

**Profiles** (set per role in `roles.json`, changed with `allow.py`):
`safe` (default: write only inside `write_roots`, Bash in bwrap with no network,
no ToolWake), `wakeup` (safe plus ToolWake; jobs run in bwrap with network),
`full` (unrestricted, must be named explicitly), `none` (no tools; compaction).
Protected files (`.env`, `config.json`, `roles.json`) can never be inside a
writable root. The agent cannot change its own profile.

**Long output.** Tool results over a limit are trimmed in context but saved whole
by `tool_spill.py`; the marker says how to fetch the rest with `fetch_result.py`.

**Background jobs.** `ToolWake` starts a detached job (`bgrun.py`); when it
finishes, a `data/bgtasks/` entry opens a new turn with the output. Gmail and RSS
checks use the same mechanism.

**Adding a tool:** add its schema to `_SCHEMAS`, its implementation to
`local_tools.py`, its phase tier in `cchat.py`'s tool lists, and a test. The
Codex server registers it automatically from the schema.

---

## 7. Conventions

- **Logic apart from UI.** Every wizard step is a UI-free `wizard_*.py` module
  (testable, reusable by a web wizard) plus a thin screen in `wizard.py`.
- **Writes are create-only or atomic.** Anything that could replace a user's file
  refuses instead; config and credentials go through temp-file-and-rename.
- **Fail closed.** Unknown, unparseable or unverifiable means deny (sandbox
  roots, codex lockdown, private-repo check), not allow.
- **Errors are written for the person running it** ("the Gmail API isn't switched
  on, open this link"), never a bare traceback; setup modules raise a
  `*SetupError` for that.
- **Optional dependencies import lazily** (discord, Google, aiohttp, mem0) so a
  missing feature never breaks start-up.
- **No hardcoded paths, names or secrets.** Use `paths.py`, `load_identity()` and
  `credentials.py`. `_identity_neutral_test.py` fails the build if a real name,
  machine or family detail appears in a shipped file.
- **Tests** are `_name_test.py`, run through `_test_runner.py`, which points `HOME`
  and the data directory at a disposable sandbox and refuses to run tests that
  mutate rollover state anywhere else. Tests use fakes, not the network.
- **Comments** are labels, with a short *why* only where the code can't say it.
  No dates, no names, no history: version control keeps the history.

---

## 8. Making it yours

- **Persona:** edit `data/memory/identity.md`, `user.md`, `practices.md`
  (`python3 wizard.py --edit` shows the live token budget). `goals.md`, `open.md`
  and `reflections.md` are the agent's own notes. `templates/memory/` holds the
  per-mode job descriptions copied in at setup.
- **Name and user:** `data/identity.json`; the web UI and API model id follow it.
- **Feeds:** `data/rss/feeds.txt`, one URL per line.
- **Providers and models:** `data/config.json`, or rerun the provider step.
- **Wakeups:** the agent's `SetWakeup` tool, or `data/wakeup.json`.
- **A new channel:** follow `discord_bridge.py` (push) or `chat_bridge.py`
  (poll): own polling and dedup, accept a handler, know nothing about the model,
  and start from `cchat.py` guarded so its absence is silent.
- **A new provider:** add a block under `providers` in `config.json`. Only
  genuinely new wire formats need an adapter in `providers/`.
