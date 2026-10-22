"""GRPO campaign launcher (LoRA by default).

    python3 -m bedroom.train.train_hf MODEL --out campaign-1 --rounds 40 \
        --scenarios-per-round 8 --group-size 8 --lr 2e-5 --lora-r 16 \
        --weights bash=3

Checkpoints are adapter dirs (round-NNN/); score one with
`python3 -m bedroom.train.eval_hf campaign-1/round-039 --pool eval`.
"""
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--scenarios-per-round", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--weights", default="", help="family=weight,family=weight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty-schedule", default="",
                    help="round:max_difficulty,... e.g. 0:1,10:2,20:3 (default: all difficulties)")
    a = ap.parse_args()
    import torch
    from bedroom.multi import load_multi_dir
    from bedroom.train.grpo_loop import load_policy, make_optimizer, run_campaign
    from students.lfm_student_full import TOOLS
    sched = {int(k): int(v) for k, v in (x.split(":") for x in a.difficulty_schedule.split(",") if x)}
    weights = {k: float(v) for k, v in (w.split("=") for w in a.weights.split(",") if w)}
    train = load_multi_dir(str(Path(__file__).resolve().parent.parent.parent / "bench" / "train"))
    policy, ref = load_policy(a.model, TOOLS, device="cuda", lora_r=a.lora_r)
    policy.max_new_tokens = a.max_new_tokens
    opt = make_optimizer(policy.model, lr=a.lr)
    t0 = time.time()

    def log(*x):
        print(f"[{time.time() - t0:7.0f}s peak {torch.cuda.max_memory_allocated() / 1e9:.1f}GB]", *x, flush=True)

    run_campaign(policy, ref, opt, train, a.out, rounds=a.rounds,
                 scenarios_per_round=a.scenarios_per_round, group_size=a.group_size,
                 weights=weights, difficulty_schedule=sched or None, temperature=a.temperature, seed=a.seed, log=log)


if __name__ == "__main__":
    main()
