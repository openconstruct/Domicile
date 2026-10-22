"""Run a Bedroom pool against an HF model (or a trained checkpoint dir).

    python3 -m bedroom.train.eval_hf MODEL_OR_CKPT --pool dev [--limit N] [--out report.json]

Greedy decoding, same render/parse path the GRPO loop trains with, so the
baseline and final numbers measure exactly what training optimizes.
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--pool", default="dev", choices=["train", "dev", "eval"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import random, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from bedroom.multi import load_multi_dir, run_curriculum
    from bedroom.train.grpo_loop import Batcher, HFPolicy
    from students.lfm_student_full import TOOLS
    adapter = (Path(a.model) / "adapter_config.json").exists()
    base_id = json.loads((Path(a.model) / "adapter_config.json").read_text())["base_model_name_or_path"] if adapter else a.model
    tok = AutoTokenizer.from_pretrained(a.model if adapter else base_id)
    model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16).to("cuda")
    if adapter:                       # LoRA checkpoint: attach and merge for speed
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.model).merge_and_unload()
    model = model.eval()
    pol = HFPolicy(model, tok, TOOLS, max_new_tokens=a.max_new_tokens)
    pol.batcher = Batcher(pol, max_batch=a.batch)
    bench = Path(__file__).resolve().parent.parent.parent / "bench" / a.pool
    sc = load_multi_dir(str(bench))
    if a.limit:
        random.Random(0).shuffle(sc)
        sc = sc[:a.limit]
    t0 = time.time()
    rep = run_curriculum(sc, lambda obs: pol.act(obs, temperature=0.0), workers=a.batch)
    rep["secs"] = round(time.time() - t0, 1)
    rep["n_scenarios"] = len(sc)
    rep["model"] = a.model
    rep["pool"] = a.pool
    print(json.dumps({k: v for k, v in rep.items() if k != "results"}, indent=1, default=str))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
