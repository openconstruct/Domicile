# Bedroom

An optional training and evaluation environment for Domicile's tool use.
**Domicile is the agent. Bedroom is where you measure and improve how well
a small local model drives Domicile's tools.** Nothing in Domicile imports
Bedroom, and Domicile's install does not need anything listed here.

Designed to work with any local model that fits in VRAM. The results we
publish are from LFM2.5-2.6B only; other models have not been tested, and a
different chat template or tool-call format may need a new adapter in
`students/`. No model weights are shipped in this repo.

## What's in it

- `bedroom/` — the harness: sandboxed dollhouses for Domicile's tool
  families (filesystem, mem0/mem2/mem1 memory, web, Gmail/calendar/
  Blogger/Discord, bash) and a multi-round runner with strict binary
  postconditions.
- `scenarios/` — 60 hand-built scenarios across those families.
- `bench/` — a generated bench of 2,088 scenarios in disjoint pools:
  1,680 train, 96 dev, 312 eval. The eval pool is hash-frozen
  (`bench/eval-freeze.json`); `bench/audit.py` checks soundness, leaks
  and duplicates.
- `students/` — adapters that let a llama-server model play the scenarios.
- `bedroom/train/` — a GRPO training loop (needs torch).
- `tests/` — smoke tests.

## Run it

Everything except training is standard-library Python 3.11+. From this
directory:

    python3 tests/smoke_test.py            # filesystem
    python3 tests/smoke_test_m2.py         # memory
    python3 tests/smoke_test_m3.py         # web
    python3 tests/smoke_test_m4.py         # social
    python3 tests/smoke_test_m5.py         # bash + runner
    python3 tests/smoke_test_grpo.py       # GRPO math, no torch
    python3 bench/audit.py                 # bench audit
    python3 bench/freeze.py --verify       # eval set unchanged

For the training campaign (baseline eval, GRPO on train, final eval on the
frozen set) install the extras with `pip install -r requirements.txt` and
follow `INSTRUCTIONS.md`.

These tests are separate from Domicile's `run_tests.sh`, which only picks up
`_*_test.py` files in the top directory, so Bedroom never joins the quick run.

## Fidelity to Domicile's tools

A policy learns from the strings it sees, so Bedroom reproduces Domicile's
model-facing contract rather than a lookalike:

- `bedroom/domicile_tools.json` -- the exact tool schemas Domicile shows a native
  local model (Read, Write, Edit, Glob, Grep, Bash, WebSearch, WebFetch). Regenerate
  with `python3 -m bedroom.sync_domicile_schemas` when `local_tools._SCHEMAS` changes.
- `bedroom/render.py` -- tool results in `local_tools.dispatch()` shape: numbered
  `Read` lines, regex `Grep` as `path:line:text`, newest-first `Glob`, the
  read-before-write/edit rule, `DENIED:` / `ERROR: <Type>:` prefixes, Bash
  `[stderr]` / `[exit N]` rendering, and the real guard messages.
- `tests/conformance_domicile.py` -- runs the same calls through the real
  `local_tools` and fails on any difference in output, vendored constants or
  schemas. Needs a full Domicile checkout; skips otherwise. Run it whenever
  Domicile's tools change.

Known simulation limits: Bash runs only the commands in `bash.ALLOWED_COMMANDS`
(others report `not available in this simulated environment`), `grep`/`find` in Bash
match literally, Grep ignores `type`/`multiline`. The memory (`mcp__memory__*`,
`mcp__mem2__*`) and social tool schemas and results are NOT verified against the
house and may differ.
