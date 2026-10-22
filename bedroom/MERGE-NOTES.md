# Bedroom → Domicile merge notes (Nova, 2026-10-21)

For Jerry's merge into the Domicile codebase, gated on the
Chromebook clean-install test. Everything below lives under
`bedroom/` in this goal's files tree and is stdlib-only except
where noted.

## What merges cleanly (harness core, no new deps)

- `bedroom/core.py` — episode loop, validator, fs sandbox
  (Read/Write/Edit/Glob/Grep), traces, eval.
- `bedroom/memory.py` — mem0/mem2/mem1 dollhouse.
- `bedroom/web.py` — web dollhouse (search/fetch over a fake index).
- `bedroom/social.py` — gmail/calendar/blogger/discord dollhouse.
- `bedroom/bash.py` — Bash dollhouse on the house contract (argv
  list, no shell; ALLOWED_COMMANDS is one frozenset constant —
  swap in Rook's enumeration of the real subset when merging).
- `bedroom/multi.py` — multi-round runner, postconditions
  (exactly-once counts, file_equals, final_answer_exact_lines),
  run_curriculum.
- `scenarios/` — the 60 hand-built M1–M5 scenarios.
- `students/` — llama-server adapters (oracle + LFM variants);
  eval-side only, merge optional.

## Bench (additive, self-contained under `bench/`)

- `bench/generate.py` — 24 archetypes + 3 eval-only generalization
  variants; pools train 1,680 / dev 96 / eval 312, disjoint seed
  spaces + per-pool vocabularies, oracle-by-construction.
- `bench/audit.py` — soundness/leak/duplicate audit.
  **Status 2026-10-21: AUDIT OK on all 2,088** (audit-report.json).
- `bench/freeze.py` — EVAL hash freeze. **Not yet executed:**
  waiting on the DEV calibration verdict (below).

## Training regimen (`bedroom/train/`, needs torch)

- `bedroom/train/grpo_loop.py` — GRPO per Jerry's 2026-10-21 call:
  episode-group rollouts against the runner, binary reward,
  group-relative advantages, loss masked to policy tokens,
  KL-to-reference penalty, curriculum sampling over TRAIN,
  per-round checkpoints + rounds.jsonl.
- Pure-math smoke passes; the tiny-model end-to-end smoke runs
  once the torch venv on Nova's VM finishes installing.

## Status update (2026-10-21 evening)

- **DEV calibration DONE: 0.479 overall — inside the 40–50% band**
  (bash the weak family at 0.125). bench/dev-calibration-oct21.json.
- **EVAL FROZEN**: 312 scenarios, aggregate SHA-256
  6931e5f634afe294…, bench/eval-freeze.json, verify OK.
- **GRPO loop smoke-verified end to end** (tiny model, CPU):
  rollouts, masked loss, optimizer step, checkpoint — SMOKE OK.
- Remaining is the campaign itself (baseline EVAL → GRPO on TRAIN
  → final EVAL → delta), on Jerry's hardware per the ladder.
  (Longer design notes were not included in this bundle.)
