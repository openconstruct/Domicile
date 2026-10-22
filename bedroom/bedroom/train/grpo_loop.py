"""Bedroom training regimen: GRPO over multi-turn tool episodes.

Decided by Jerry 2026-10-21 ("Straight to GRPO"). The regimen trains
ONLY on the bench's TRAIN pool; EVAL never enters this process (the
Sec. 8 protocol runs it separately, before and after a campaign).

Shape of the algorithm, per episode group:

- For one scenario, sample G episodes from the current policy at
  temperature. An episode is a full multi-round run against
  bedroom.multi.run_episode_multi; the runner's strict binary verdict
  is the reward. No critic, no partial credit.
- Advantage is group-relative: A_i = (r_i - mean(r)) / (std(r)+eps).
  An all-same-reward group contributes no gradient (and is counted,
  since a rising share of all-1 groups is the learning signal).
- The training sequence for an episode is the transcript: the
  rendered prompt, then alternating policy generations and tool
  observations. The loss touches ONLY policy-generated tokens
  (mask 1); observation and prompt tokens are mask 0.
- Loss per episode: -A_i * mean_logprob(action tokens)
  + beta * mean(logp_ref - logp_policy) over action tokens
  (KL-to-reference penalty, k1 estimator).

Policy rendering uses the model's own chat template with the full
tool array, the same presentation the llama-server adapters use at
eval time; generations are parsed back into Bedroom actions (JSON
tool-call extraction, {"text": ...} fallback). Fidelity note: during
training the running transcript is extended with plain tool-result
segments rather than re-rendered through the template each turn;
the graded surface (tool names, arg shapes, observation text) is
identical, the wrapping is the eval adapters' job to match at
baseline/final time.

Hardware posture per the Oct 19 sizing: written for a ~2.6B policy
(the campaign student is the HF checkpoint, LiquidAI/LFM2.5-2.6B;
the GGUF on Nova's VM is inference-only). bf16 + gradient
checkpointing on CUDA, 8-bit AdamW when bitsandbytes is present,
plain AdamW otherwise. CPU works for smoke tests with tiny models
and nothing else.
"""
import json, math, random, re
from pathlib import Path

from ..multi import run_episode_multi

# ------------------------------------------------------------ advantages

def group_advantages(rewards):
    """GRPO group-relative advantages for one scenario's episode group."""
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
    if std < 1e-8:
        return [0.0] * n
    return [(r - mean) / std for r in rewards]


# ------------------------------------------------------------ action parsing

_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)
# Native tool-call formats of the models we train. LFM2.5: Pythonic list
# between special markers. Qwen3.5: XML-ish <function=..><parameter=..>.
_LFM_CALL = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.DOTALL)
_QWEN_CALL = re.compile(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", re.DOTALL)
_QWEN_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def _parse_native(text):
    """Return an action dict from a model-native tool call, or None."""
    m = _LFM_CALL.search(text)
    if m:
        import ast
        try:
            node = ast.parse(m.group(1).strip(), mode="eval").body
            if isinstance(node, ast.List) and node.elts:
                node = node.elts[0]          # one action per turn: take the first
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                return {"tool": node.func.id,
                        "args": {k.arg: ast.literal_eval(k.value) for k in node.keywords if k.arg}}
        except Exception:
            return None
        return None
    m = _QWEN_CALL.search(text)
    if m:
        args = {}
        for k, v in _QWEN_PARAM.findall(m.group(2)):
            try:
                args[k] = json.loads(v)
            except Exception:
                args[k] = v
        return {"tool": m.group(1), "args": args}
    return None


def parse_action(text):
    """Model generation -> Bedroom action dict.

    Accepts a JSON object with a "tool" key ({"tool": ..., "args": {...}}
    or the OpenAI-ish {"name": ..., "arguments": {...}}), else treats the
    text as the student's final answer.
    """
    text = (text or "").strip()
    if "</think>" in text:               # reasoning is not part of the action or answer
        text = text.split("</think>")[-1].strip()
    text = text.replace("<think>", "").strip()
    native = _parse_native(text)
    if native is not None:
        return native
    text = _LFM_CALL.sub("", text).strip() if "<|tool_call_start|>" in text else text
    m = _JSON_OBJ.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            obj = None
        if isinstance(obj, dict):
            if "tool" in obj:
                return {"tool": obj["tool"], "args": obj.get("args", {}) or {}}
            if "name" in obj and ("arguments" in obj or "args" in obj):
                args = obj.get("arguments", obj.get("args", {}))
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                return {"tool": obj["name"], "args": args or {}}
    return {"text": text[:500]}


# ------------------------------------------------------------ HF policy

class HFPolicy:
    """The policy under training: renders obs, generates one turn."""

    def __init__(self, model, tokenizer, tools_spec, max_new_tokens=300):
        self.model = model
        self.tok = tokenizer
        self.tools_spec = tools_spec
        self.max_new_tokens = max_new_tokens

    def render(self, obs):
        content = obs["situation"]
        if "memory_listing" in obs:
            content += "\n\nCurrent memory listing:\n" + obs["memory_listing"]
        messages = [{"role": "user", "content": content}]
        for h in obs.get("history", []):
            messages.append({"role": "assistant", "content": None,
                             "tool_calls": [{"id": "call_0", "type": "function",
                                             "function": {"name": h["tool"],
                                                          "arguments": json.dumps(h.get("args", {}))}}]})
            messages.append({"role": "tool", "tool_call_id": "call_0",
                             "content": str(h["result"])})
        try:
            return self.tok.apply_chat_template(
                messages, tools=self.tools_spec, tokenize=False,
                add_generation_prompt=True)
        except Exception:
            return content

    def generate_batch(self, prompts, temperature=1.0):
        """Generate one completion per prompt in a single left-padded batch.

        Tokenizes with add_special_tokens=False (the chat template already
        carries BOS), the same way episode_tokens does for training.
        """
        import torch
        tok = self.tok
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **enc, max_new_tokens=self.max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5) if temperature > 0 else None,
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=tok.pad_token_id)
        n = enc["input_ids"].shape[1]
        return [tok.decode(o[n:], skip_special_tokens=True) for o in out]

    def gen_text(self, prompt, temperature=1.0):
        """One completion; routed through the batcher when one is attached."""
        b = getattr(self, "batcher", None)
        if b is not None:
            return b.submit(prompt, temperature)
        return self.generate_batch([prompt], temperature)[0]

    def act(self, obs, temperature=1.0):
        return parse_action(self.gen_text(self.render(obs), temperature))


class Batcher:
    """Merges generate calls from concurrent episode threads into batches.

    Episodes run in threads (they block on the model between tool calls);
    each submit() waits while a worker thread collects requests for a short
    window and runs them as one left-padded batch.
    """

    def __init__(self, policy, max_batch=32, window=0.3):
        import queue, threading
        self.policy, self.max_batch, self.window = policy, max_batch, window
        self.q = queue.Queue()
        self._queue_mod = queue
        self._threading = threading
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, prompt, temperature):
        item = {"prompt": prompt, "temp": temperature, "ev": self._threading.Event()}
        self.q.put(item)
        item["ev"].wait()
        if "err" in item:
            raise item["err"]
        return item["out"]

    def _loop(self):
        import time
        while True:
            batch = [self.q.get()]
            deadline = time.time() + self.window
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.q.get(timeout=max(0.0, deadline - time.time())))
                except self._queue_mod.Empty:
                    break
            for temp in sorted({b["temp"] for b in batch}):
                grp = [b for b in batch if b["temp"] == temp]
                try:
                    outs = self.policy.generate_batch([g["prompt"] for g in grp], temp)
                    for g, o in zip(grp, outs):
                        g["out"] = o
                except Exception as e:  # surface in the waiting threads
                    for g in grp:
                        g["err"] = e
                for g in grp:
                    g["ev"].set()


# ------------------------------------------------------------ episode tokens

def episode_tokens(policy, obs_record):
    """Rebuild (input_ids, action_mask) from a recorded rollout.

    obs_record: {"prompt": str, "turns": [{"gen_text": str,
    "obs_text": str}, ...]} -- segments the rollout captured. Prompt
    and observation segments mask 0; generation segments mask 1.
    """
    tok = policy.tok
    ids, mask = [], []
    p = tok(obs_record["prompt"], add_special_tokens=False)["input_ids"]
    ids += p
    mask += [0] * len(p)
    for turn in obs_record["turns"]:
        g = tok(turn["gen_text"], add_special_tokens=False)["input_ids"]
        ids += g
        mask += [1] * len(g)
        if turn.get("obs_text"):
            o = tok(turn["obs_text"], add_special_tokens=False)["input_ids"]
            ids += o
            mask += [0] * len(o)
    return ids, mask


class RecordingPolicy:
    """Wraps a policy so a rollout leaves a tokenizable transcript."""

    def __init__(self, policy):
        self.policy = policy
        self.record = None

    def start(self, obs):
        self.record = {"prompt": self.policy.render(obs), "turns": []}

    def act(self, obs, temperature=1.0):
        text = self.policy.gen_text(self.policy.render(obs), temperature)
        if self.record is not None:
            prev = self.record["turns"][-1] if self.record["turns"] else None
            hist = obs.get("history", [])
            if prev is not None and hist:
                prev["obs_text"] = f"\nTool result: {hist[-1]['result']}\n"
            self.record["turns"].append({"gen_text": text, "obs_text": ""})
        return parse_action(text)


def rollout_group(policy, scenario, group_size, temperature=1.0):
    """Sample G episodes of one scenario (concurrently); return records + rewards."""
    from concurrent.futures import ThreadPoolExecutor

    def one(_):
        rec = RecordingPolicy(policy)
        state = {"started": False}

        def student(obs):
            if not state["started"]:
                rec.start(obs)
                state["started"] = True
            return rec.act(obs, temperature=temperature)

        trace = run_episode_multi(scenario, student)
        return {"record": rec.record, "reward": trace["reward"],
                "reason": trace["reason"], "trace": trace}

    with ThreadPoolExecutor(max_workers=group_size) as ex:
        return list(ex.map(one, range(group_size)))


# ------------------------------------------------------------ training step

def sequence_logprobs(model, tok, ids, mask, device):
    """Per-token logprobs at action (mask=1) positions, teacher-forced.

    Only the action rows are upcast and softmaxed, so a 4k-token episode
    does not materialize a 4k x vocab float32 matrix.
    """
    import torch
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(input_ids=x).logits[0, :-1]
    targets = x[0, 1:]
    sel = torch.tensor(mask[1:], dtype=torch.bool, device=device)
    lg = logits[sel].float()
    logp = torch.log_softmax(lg, dim=-1)
    tok_lp = logp.gather(1, targets[sel].unsqueeze(1)).squeeze(1)
    m = torch.ones_like(tok_lp)
    return tok_lp.sum() / m.sum().clamp(min=1.0), tok_lp, m


def grpo_step(policy, ref_model, batch, optimizer, beta=0.04, max_len=4096):
    """One GRPO update over a batch of episode dicts with advantages.

    batch: [{"ids": [...], "mask": [...], "advantage": float}, ...]
    Returns the mean loss as a float.
    """
    import torch
    device = policy.model.device
    policy.model.train()
    total = 0.0
    optimizer.zero_grad()
    for ep in batch:
        ids, mask = ep["ids"][:max_len], ep["mask"][:max_len]
        if sum(mask) == 0:
            continue
        lp, tok_lp, m = sequence_logprobs(policy.model, policy.tok, ids, mask, device)
        with torch.no_grad():
            if ref_model is None:        # LoRA: the reference is the base with adapters off
                with policy.model.disable_adapter():
                    _, ref_tok_lp, _ = sequence_logprobs(policy.model, policy.tok, ids, mask, device)
            else:
                _, ref_tok_lp, _ = sequence_logprobs(ref_model, policy.tok, ids, mask, device)
        kl = ((ref_tok_lp - tok_lp) * m).sum() / m.sum().clamp(min=1.0)
        loss = -ep["advantage"] * lp + beta * kl
        (loss / len(batch)).backward()
        total += float(loss.detach())
    torch.nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    return total / max(1, len(batch))


# ------------------------------------------------------------ campaign

def sample_scenarios(scenarios, rng, n, weights=None, max_difficulty=3):
    """Curriculum-weighted draw: family weights + difficulty gate."""
    weights = weights or {}
    pool = [s for s in scenarios if s.get("difficulty", 1) <= max_difficulty]
    w = [max(0.05, float(weights.get(s["family"], 1.0))) for s in pool]
    return rng.choices(pool, weights=w, k=n)


def run_campaign(policy, ref_model, optimizer, scenarios, out_dir,
                 rounds=10, scenarios_per_round=8, group_size=8,
                 weights=None, difficulty_schedule=None, beta=0.04,
                 temperature=1.0, seed=0, log=print):
    """GRPO over the TRAIN pool. One round = sample scenarios, roll out
    groups, build the masked batch, one optimizer step, log, checkpoint.

    difficulty_schedule: {round_index: max_difficulty}; default all 3.
    Returns the round stats list; also written to out_dir/rounds.jsonl.
    """
    import torch
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    stats_all = []
    for rnd in range(rounds):
        # step schedule: the entry with the largest round key <= rnd applies
        _ks = [k for k in (difficulty_schedule or {}) if k <= rnd]
        max_d = difficulty_schedule[max(_ks)] if _ks else 3
        batch_scenarios = sample_scenarios(scenarios, rng, scenarios_per_round,
                                           weights, max_d)
        batch, fam_rewards, dead_groups = [], {}, 0
        from concurrent.futures import ThreadPoolExecutor
        policy.model.eval()
        with ThreadPoolExecutor(max_workers=len(batch_scenarios)) as ex:
            all_eps = list(ex.map(
                lambda sc: rollout_group(policy, sc, group_size, temperature),
                batch_scenarios))
        for sc, episodes in zip(batch_scenarios, all_eps):
            rewards = [e["reward"] for e in episodes]
            advs = group_advantages(rewards)
            if all(a == 0.0 for a in advs):
                dead_groups += 1
            fam_rewards.setdefault(sc["family"], []).extend(rewards)
            for ep, adv in zip(episodes, advs):
                ids, mask = episode_tokens(policy, ep["record"])
                batch.append({"ids": ids, "mask": mask, "advantage": adv})
        loss = grpo_step(policy, ref_model, batch, optimizer, beta=beta)
        stats = {"round": rnd, "loss": loss, "dead_groups": dead_groups,
                 "groups": len(batch_scenarios),
                 "mean_reward": sum(sum(v) for v in fam_rewards.values()) /
                                max(1, sum(len(v) for v in fam_rewards.values())),
                 "by_family": {k: sum(v) / len(v) for k, v in sorted(fam_rewards.items())}}
        stats_all.append(stats)
        with open(out / "rounds.jsonl", "a") as f:
            f.write(json.dumps(stats) + "\n")
        policy.model.save_pretrained(out / f"round-{rnd:03d}")
        policy.tok.save_pretrained(out / f"round-{rnd:03d}")
        log(f"round {rnd}: reward={stats['mean_reward']:.3f} loss={loss:.4f} "
            f"dead_groups={dead_groups}/{len(batch_scenarios)} by_family={stats['by_family']}")
    return stats_all


def load_policy(model_id, tools_spec, device="cpu", dtype=None, lora_r=0, lora_alpha=None):
    """Load a policy + reference for a campaign.

    lora_r > 0: bf16 base + fp32 LoRA adapters; returns (policy, None) and
    the reference is the same model with adapters disabled. Full-parameter
    bf16 at small learning rates rounds the updates away, so LoRA is the
    default for real runs. lora_r == 0 keeps the full-parameter path.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
    if lora_r:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha or 2 * lora_r, lora_dropout=0.0,
                         target_modules="all-linear", task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        ref = None
    else:
        ref = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    pol = HFPolicy(model, tok, tools_spec)
    pol.batcher = Batcher(pol)
    return pol, ref


def make_optimizer(model, lr=1e-6):
    import torch
    params = [p for p in model.parameters() if p.requires_grad]
    if any(p.dtype == torch.float32 for p in params) and \
            not any(p.dtype != torch.float32 for p in params):
        return torch.optim.AdamW(params, lr=lr)       # LoRA adapters: small, fp32
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr)
    except Exception:
        return torch.optim.AdamW(params, lr=lr)
