# Domicile

A home for one continuous agent: one memory, one history, running on your own
machine. A Textual terminal app, with optional Discord, Google (Gmail, Calendar,
Google Chat), RSS and HTTP bridges. `AGENTS.md` is the map of the codebase.

  Domicile is a harness for one agent only. It has one chat session with permanence, a thorough 3 layer memory system, and an active context manager for keeping costs low.  Domicile has a great selection of built in tools, and it's simple for a decent agent to make others. Domicile is @20k python LoC so it's easy for agents to understand and extend.
  
  Domicile uses a shared-state UI with a CLI and a web viewpoint. They share the same featureset, and the webUI can be accesseed from the local network, so you can chat in bed. The UI has system information, agent settings ( provider, model, thinking level, role). The resident agent has 3 roles, chat, build, and compaction. Compaction is both used for compacting context and extraction to flash memory. For on the go conversations we have google and discord connectors. Google supports more than chat, Domicile agents can be set up with their own google account and use email, chat, calendar, and blogger.
  
  Backups of all memory layers are automatic and designed to land in a private github repo. If it is not private YOUR FULL CHAT LOGS ARE PUBLIC.

  Provider tools are turned off by default, so the agent has one set of tools, regardless of provider.

  Domicile is designed to let the same agent run on multiple backends. We support claude code, codex, and OpenAI compatible API. If a limit is hit, Domicile will automatically roll over to the next provider. Local models are simple to use, just point to the inference API.

  Domicile's memory system has 3 layers: 

  1. agent edited injected memory, this includes persona, user, and system practices
  2. flash memory, similar to mem0 this extracts facts from conversations and indexes them for searching
  3. full memory, the complete and full chat logs are always saved and indexed so they can be searched in full, the lossless layer.

  Domicile can import full chat histories from Anthropic or ChatGPT, import chatlogs or textbased memory systems, and import from mem0.

 



## Quickstart (Debian/Ubuntu)

    sudo apt install python3 python3-venv tmux
    git clone https://github.com/openconstruct/Domicile && cd Domicile
    ./install.sh

The folder can be called anything and live anywhere; Domicile finds itself.
`install.sh` makes a venv at `~/.cchat-venv`, installs dependencies, and runs the
setup wizard: provider, context window, optional memory import, persona,
optional Discord and Google, the background service, optional private GitHub
backups, and an offline smoke test. Your data lives in `data/` inside the
install folder.

Using the Anthropic provider? Install the Claude Code CLI and run `claude` once
to log in before your first conversation.

## Your agent's memory

Three persona files are loaded on every turn, in `data/memory/`:

- `identity.md`: who the agent is, its values and voice
- `user.md`: what the agent should know about you
- `practices.md`: rules and habits for working together

**Grow** (default) starts them near-empty; the agent picks its own name in your
first conversation and fills the files in through use. **Define** lets you name it
now and import an existing persona. Files have token caps; the wizard refuses an
oversized file instead of truncating it. Raise mem1 room in the Context step, or
trim the file.

## Later changes

    ./install.sh --edit

Opens an editor for the three memory files (with a live token counter) and
`config.json`. Restart cchat after editing config.

## Running

With the service enabled: `tmux -L cchat attach -t main` (detach with Ctrl-b d).
By hand: `~/.cchat-venv/bin/python cchat.py`.

Credentials live in `.env`, which is gitignored. Never commit it.
