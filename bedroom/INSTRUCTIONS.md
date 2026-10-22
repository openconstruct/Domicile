# Bedroom bundle — instructions

Bedroom: the RL harness forked from Domicile (trained surface = the
real tools, strict binary rewards, multi-turn episodes), plus the
frozen bench and the GRPO training regimen. State at packaging
(2026-10-21): milestones M1–M5 built and verified; bench generated,
audited, calibrated (LFM baseline 0.479 on DEV, inside the 40–50%
band), EVAL frozen; GRPO loop implemented and smoke-verified.

## Layout

- `bedroom/` — the harness package: `env.py` (fs sandbox),
  `memory.py`/`memory_env.py` (mem0/mem2), `web.py`, `social.py`,
  `bash.py` (argv-list Bash, no shell), `multi.py` (multi-round
  runner + postconditions + curriculum runs).
- `bedroom/train/grpo_loop.py` — the GRPO regimen.
- `scenarios/` — the 60 hand-built M1–M5 scenarios.
- `bench/` — generated pools (train 1,680 / dev 96 / eval 312),
  `generate.py`, `audit.py`, `freeze.py`, `oracles.json`,
  `audit-report.json`, `dev-calibration-oct21.json`,
  `eval-freeze.json`.
- `students/` — llama-server student adapters; `lfm_student_full.py`
  is the all-tools one used for bench runs.
- `tests/` — smoke tests per milestone + `demo_m5.py` +
  `smoke_test_grpo.py`.
- `MERGE-NOTES.md` — module map for merging into Domicile.

## Ground rules

- **EVAL is frozen.** Verify before trusting it:
  `python3 -m bench.freeze --verify` (run from this directory).
  A mismatch means the bench changed — that's a new bench version
  and needs a new baseline.
- **Train on TRAIN only.** DEV is for calibration peeks. EVAL runs
  exactly twice per campaign: baseline and final.
- Rewards are strict binary by design. Do not soften graders to
  make numbers move; the delta is the product.

## Setup

Harness, bench, and eval runs are stdlib-only Python 3.11+.
Eval runs need a llama-server serving the student GGUF:

    llama-server -m LFM2.5-2.6B-Q4_K_M.gguf --jinja \
        --host 127.0.0.1 --port 8091

(the adapters in `students/` point at 127.0.0.1:8091).

Training additionally needs, in any env you like:

    pip install torch transformers accelerate
    pip install bitsandbytes   # optional: 8-bit AdamW

## Smoke tests (run from this directory)

    python3 tests/smoke_test.py          # M1 fs
    python3 tests/smoke_test_m2.py       # memory
    python3 tests/smoke_test_m3.py       # web
    python3 tests/smoke_test_m4.py       # social
    python3 tests/smoke_test_m5.py       # bash + runner polish
    python3 tests/demo_m5.py             # the demo beat
    python3 tests/smoke_test_grpo.py            # GRPO math (no torch)
    python3 tests/smoke_test_grpo.py --tiny     # GRPO loop, tiny model

## The campaign (Sec. 8 protocol)

1. **Baseline** — run the frozen EVAL with the full student and
   save the report:

       python3 - <<'EOF'
       import sys; sys.path.insert(0, ".")
       from bedroom.multi import load_multi_dir, run_curriculum
       from students.lfm_student_full import lfm_student_full
       sc = load_multi_dir("bench/eval")
       print(run_curriculum(sc, lfm_student_full,
                            trace_path="baseline-eval-traces.jsonl"))
       EOF

   312 scenarios at llama-server speeds — batch it, it takes a
   while. This number is the one the campaign is judged against.

2. **Train** — GRPO on TRAIN with the HF checkpoint
   (LiquidAI/LFM2.5-2.6B; the GGUF is inference-only):

       python3 - <<'EOF'
       import sys; sys.path.insert(0, ".")
       from bedroom.multi import load_multi_dir
       from bedroom.train.grpo_loop import load_policy, make_optimizer, run_campaign
       from students.lfm_student_full import TOOLS
       scenarios = load_multi_dir("bench/train")
       policy, ref = load_policy("LiquidAI/LFM2.5-2.6B", TOOLS, device="cuda")
       opt = make_optimizer(policy.model, lr=1e-6)
       run_campaign(policy, ref, opt, scenarios, "campaign-1",
                    rounds=10, scenarios_per_round=8, group_size=8)
       EOF

   Checkpoints land in `campaign-1/round-NNN` with `rounds.jsonl`
   stats. Put the output dir on real disk with room to spare — a
   2.6B checkpoint is ~5 GB in bf16. Curriculum: oversample the
   families that fail (pass `weights={"bash": 3.0}` etc. to
   run_campaign); difficulty gates via `difficulty_schedule`.

3. **Final** — rerun step 1 against the trained checkpoint (serve
   it via llama-server as a GGUF, or point an HF-backed student at
   it) and report the delta: overall, per-family, per-difficulty,
   and the generalization slice (archetypes `bash-sed-first`,
   `mail-reply-two`, `bash-append-two` — never trained on) reported
   separately.

## Notes

- `bash.py`'s ALLOWED_COMMANDS is a conservative core set
  (ls/cat/grep/wc/sort/uniq/head/tail/mv/cp/mkdir/echo/sed/find).
  On merge, swap in the house's real subset enumeration.
- Known failure class to watch in traces: the student inspects
  (Read/list) and then answers in text without ever calling the
  action tool — "describing instead of doing." Bash DEV baseline
  was 0.125 for exactly this reason; it's the campaign's main
  target.
- Design + close-out docs: `bedroom-bench-regimen-design-oct21.md`
  and `bedroom-bench-frozen-oct21.md` in the goal files (not in
  this zip).
