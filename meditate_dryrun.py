#!/usr/bin/env python3
"""Generate memory consolidation proposals without changing facts.

COST NOTE: meditation is expensive by construction. Only already-superseded
facts leave the pool; every other fact is re-judged on every run, because a
fact left alone yesterday may pair with something new today. A run is
~(live facts / 40) judge calls on the ACTIVE provider at thinking high, and
each call carries identity.md + user.md as background (--no-context to drop it).

--store rebuild reads the disposable store; --store live reads through IPC.
Proposals are written to data/mem0/<tag>.json for review before applying.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from credentials import read_key

from openai import OpenAI

import mem0_ipc
import paths

REBUILD = str(paths.MEM0_DIR / 'qdrant-rebuild')
COL = 'ccchat'
CFG = json.load(open(paths.CONFIG_PATH))
BATCH = 40
MAX_TOKENS = 32768

SYSTEM = """You are the judgment pass over a long-term memory store for a
personal AI assistant. Each fact has an id, date, kind, attribution, and
text. Your job: find (a) redundant near-duplicate facts where one clearly
supersedes another, (b) small clusters of related facts that deserve ONE
synthesized meta-fact.

Rules:
- Conservative. Propose only when confidence is high. Two facts being on
  the same topic is NOT redundancy; one must be strictly less useful than
  the other (e.g., superseded by a later resolution of the same event).
- Do NOT merge distinct states: "used Vulkan for inference" and "abandoned
  ROCm" are different facts. Preferences, decisions, and lessons beat
  play-by-play events when they overlap.
- Supersede proposals name keep_id (the better fact) and drop_ids (the
  weaker ones). Never propose deleting both.
- Meta proposals name member_ids (2-5 facts) and a synthesis text under 60
  words that preserves every load-bearing detail (dates, names, numbers).
- Output ONLY a JSON object: {"proposals":[{"action":"supersede"|"meta",
  "keep_id":"...","drop_ids":[],"member_ids":[],"synthesis":"","rationale":
  "...","confidence":0.0-1.0}]}
- If a batch has no confident proposals, return {"proposals":[]}.
"""

def background_context() -> str:
    """identity.md + user.md, so the judge knows who the operator and agent are."""
    d = paths.MEM1_DIR
    parts = []
    for name in ('identity.md', 'user.md'):
        try:
            parts.append(f'### {name}\n' + (d / name).read_text().strip())
        except OSError:
            pass
    if not parts:
        return ''
    return ('\n\nBACKGROUND (who the people and projects in the facts are; '
            'context only -- never a source of proposals, and facts that '
            'merely repeat this background are not automatically redundant):\n\n'
            + '\n\n'.join(parts))


PROMPT_TMPL = """Facts (batch {i}/{n}):
{facts}

Identify supersede and meta proposals per the system rules."""


def load_facts(store: str) -> list[dict]:
    if store == 'rebuild':
        from qdrant_client import QdrantClient
        c = QdrantClient(path=REBUILD)
        pts, _ = c.scroll(collection_name=COL, limit=2000, with_payload=True,
                          with_vectors=False)
        facts = []
        for p in pts:
            pl = p.payload or {}
            facts.append({
                'id': str(p.id),
                'date': pl.get('date', '?'),
                'kind': pl.get('kind', '?'),
                'attr': pl.get('attributed_to') or '?',
                'text': (pl.get('data') or '')[:500],
            })
        return facts
    if store == 'live':
        return mem0_ipc.call('list_facts', {})
    raise ValueError(f'unknown --store {store!r}')


USAGE = {'pt': 0, 'ct': 0}


def make_judge(provider, effort):
    """Return (name, fn(system, user) -> str) for the CURRENT provider.

    Background jobs follow the active provider and use its `chat` model with
    thinking high. --provider exists for debugging, not as policy.
    """
    import config_loader as CL
    prov = provider or CL.active_provider()
    models = CL.models_for(prov, 'chat')
    if not models:
        raise RuntimeError(f'provider {prov!r} has no chat model configured')
    model = models[0]
    eff = effort if effort in CL.thinking_levels(prov) else None
    adapter = CL.adapter_for(prov)
    name = f'{prov}/{model}/{eff or "default"}'

    if adapter == 'claude_cli':
        import shutil
        import subprocess
        binary = os.environ.get('CCHAT_CLAUDE_BIN') or shutil.which('claude')

        def fn(system, user):
            env = dict(os.environ)
            env['CCHAT_MEM0_NESTED'] = '1'
            argv = [binary, '-p', '--setting-sources', '',
                    '--strict-mcp-config', '--max-turns', '1',
                    '--disallowed-tools',
                    'Bash,Read,Write,Edit,Glob,Grep,WebSearch,WebFetch,Task',
                    '--model', model, '--system-prompt', system]
            if eff:
                argv += ['--effort', eff]
            proc = subprocess.run(argv, input=user, capture_output=True,
                                  text=True, timeout=900, env=env, cwd='/tmp')
            if proc.returncode != 0:
                raise RuntimeError(f'claude exit {proc.returncode}: '
                                   f'{(proc.stderr or proc.stdout)[:300]}')
            return proc.stdout
        return name, fn

    if adapter == 'openai_compatible':
        blk = CL.provider_block(prov)
        client = OpenAI(
            base_url=blk['base_url'],
            api_key=read_key(blk.get('key_file'), blk.get('key_env')),
            default_headers={'x-opencode-session': 'cchat-meditation',
                             'x-opencode-client': 'cchat-mem0'})
        kind = (CL.thinking_block(prov) or {}).get('kind')

        def fn(system, user):
            kw = {}
            if eff and kind == 'reasoning_effort':
                kw['reasoning_effort'] = eff
            r = client.chat.completions.create(
                model=model, max_tokens=MAX_TOKENS, **kw,
                messages=[{'role': 'system', 'content': system},
                          {'role': 'user', 'content': user}])
            u = getattr(r, 'usage', None)
            if u:
                USAGE['pt'] += u.prompt_tokens or 0
                USAGE['ct'] += u.completion_tokens or 0
            return r.choices[0].message.content or ''
        return name, fn

    if adapter == 'codex_cli':
        import asyncio
        from providers import get_provider
        inst = get_provider(force=prov)

        def fn(system, user):
            return asyncio.run(inst.asummarize(
                user, MAX_TOKENS, system=system, model=model, effort=eff))
        return name, fn

    raise NotImplementedError(
        f'meditation judge not wired for adapter {adapter!r} ({prov}); '
        'wire it before switching active_provider to it')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--store', choices=('rebuild', 'live'), default='rebuild',
                     help='which store to read facts from (default: rebuild)')
    ap.add_argument('--tag', default=None,
                     help='run tag; default: meditation-<UTC timestamp>-run1 '
                          '(fresh each run, never clobbers a prior log)')
    ap.add_argument('--provider', default=None,
                     help='debug override; default is the ACTIVE provider')
    ap.add_argument('--effort', default='high', help='thinking level')
    ap.add_argument('--no-context', action='store_true',
                     help='omit identity.md/user.md background')
    ap.add_argument('--batches', default=None,
                     help='comma-separated 1-based batch numbers to run')
    args = ap.parse_args()

    run_tag = args.tag or (
        'meditation-' + datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')
        + '-run1')
    out = paths.MEM0_DIR / f'{run_tag}.json'

    facts = load_facts(args.store)
    facts.sort(key=lambda f: (f['date'], f['kind']))
    batches = [facts[i:i + BATCH] for i in range(0, len(facts), BATCH)]
    print(f'{len(facts)} facts -> {len(batches)} batches of <= {BATCH}',
          flush=True)

    judge_name, judge = make_judge(args.provider, args.effort)
    print(f'judge: {judge_name}', flush=True)
    system_prompt = SYSTEM + ('' if args.no_context else background_context())
    log_ctx = not args.no_context

    log = {'run': run_tag, 'store': args.store, 'judge': judge_name,
           'context': log_ctx,
           'started': datetime.now(timezone.utc).isoformat(),
           'batches': [], 'proposals': []}
    only = None
    if os.environ.get('MEMO_RETRY_BATCHES'):
        only = {int(x) for x in os.environ['MEMO_RETRY_BATCHES'].split(',')}
        if out.exists():
            log = json.loads(out.read_text())
            log['started'] = datetime.now(timezone.utc).isoformat()
            keep = [e for e in log['batches'] if e['batch'] not in only]
            log['batches'] = keep
            log['proposals'] = [p for e in keep for p in e['proposals']]
            log.setdefault('retries', []).append(
                {'at': log['started'], 'batches': sorted(only)})

    if args.batches:
        only = {int(x) for x in args.batches.split(',')}

    for i, b in enumerate(batches, 1):
        if only is not None and i not in only:
            continue
        fact_lines = '\n'.join(
            f"- id={f['id']} date={f['date']} kind={f['kind']} attr={f['attr']}\n  {f['text']}"
            for f in b)
        t0 = time.time()
        u0 = dict(USAGE)
        try:
            user = PROMPT_TMPL.format(i=i, n=len(batches), facts=fact_lines)
            raw = judge(system_prompt, user)
            if not raw.strip():
                raise ValueError('empty judge response')
            parsed = json.loads(extract_json_shim(raw))
            props = parsed.get('proposals', [])
        except Exception as e:
            print(f'[{i}/{len(batches)}] ERROR {type(e).__name__}: {e}',
                  flush=True)
            props = []
            raw = ''
        entry = {'batch': i, 'facts': len(b),
                 'usage': ({k: USAGE[k] - u0[k] for k in USAGE}
                           if USAGE['pt'] or USAGE['ct'] else None),
                 'secs': round(time.time() - t0, 1),
                 'proposals': props}
        log['batches'].append(entry)
        log['proposals'].extend(props)
        out.write_text(json.dumps(log, indent=1))
        print(f'[{i}/{len(batches)}] {len(b)} facts, {len(props)} proposals, '
              f'{entry["secs"]}s', flush=True)

    log['finished'] = datetime.now(timezone.utc).isoformat()
    if USAGE['pt'] or USAGE['ct']:
        log['usage_total'] = dict(USAGE)
        print(f'USAGE: {USAGE["pt"]} in / {USAGE["ct"]} out tokens', flush=True)
    out.write_text(json.dumps(log, indent=1))
    print(f'DONE: {len(log["proposals"])} total proposals -> {out}',
          flush=True)


def extract_json_shim(text: str) -> str:
    t = (text or '').strip()
    if t.startswith('```'):
        t = t.strip('`')
        if t.startswith('json'):
            t = t[4:]
    start = t.find('{')
    end = t.rfind('}')
    return t[start:end + 1] if start >= 0 and end > start else t


if __name__ == '__main__':
    sys.exit(main())
