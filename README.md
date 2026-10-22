# Domicile

A home for one continuous agent: one memory, one history, running on your own
machine. A Textual terminal app, with optional Discord, Google (Gmail, Calendar,
Google Chat), RSS and HTTP bridges. `AGENTS.md` is the map of the codebase.

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
