#!/usr/bin/env python3
"""Sample training rollouts from the head at the gate's settings and grade them with the gate's grader.

  python3 miner/kit/generate.py --head ./head --url http://127.0.0.1:8000 --k 8 --out rollouts.jsonl
  python3 miner/kit/generate.py --head ./head --url http://127.0.0.1:8000 --pool local --tasks 0-7495 --k 8 --out rollouts.jsonl

The default live pool includes every active task; --tasks selects exact string task ids and/or zero-based positions/ranges in
that complete active list. Local operation is explicit: --pool local preserves the original numeric pool behavior, optionally
reading a JSON array or JSONL rows from --local-pool PATH.

Rollouts are sampled from the TRAINER's rendering of the batch prompt (one user message = MANDATE + "\\n\\n" + problem), so the
tokens you upload are tokens the head actually produced for that exact context. That matters: the v8.3.3l step replays the
head's top_k 20 -> top_p 0.95 support at temperature 1 and drops any sampled token outside it from the policy-gradient term
(support renorm), so hand-written or other-model text mostly trains nothing but KL. A live task's supplied batch prompt is used
verbatim. Each line: {task, prompt, answer, k, rollouts: [{output, tokens, stop, kind, how, reencode_ok}]}; kind is the gate
grader's verdict (positive / negative / neutral = capped or no single final boxed / indeterminate). output = decode(tokens
without the final EOS); reencode_ok says whether the trainer's re-tokenization of output gives back exactly the sampled tokens.
"""
import argparse, asyncio, json, secrets, sys
from pathlib import Path
KIT = Path(__file__).resolve().parent
REPO = KIT.parents[1]
for directory in (KIT, REPO / 'v2', REPO / 'v2' / 'miner'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
import prompts, sampler, grade  # noqa: E402
import live_pool  # noqa: E402


def parse_tasks(spec, n):
    """Legacy local selection: zero-based inclusive numeric ranges."""
    out = []
    for part in spec.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return [t for t in out if 0 <= t < n]


def _position(part, n):
    try:
        position = int(part)
    except ValueError:
        raise ValueError(f'{part!r} is not an active task id; use a nonnegative zero-based position for numeric selection') from None
    if not 0 <= position < n:
        raise ValueError(f'task position {position} is outside the active list (0-{n - 1})')
    return position


def select_tasks(tasks, spec=None):
    """Select rows by exact id or position/range; omitted or empty spec means the complete pool."""
    if spec is None or not spec.strip():
        return list(enumerate(tasks))
    by_id = {task['task_id']: position for position, task in enumerate(tasks)}
    if len(by_id) != len(tasks):
        raise ValueError('the active pool contains duplicate task ids')
    selected = []
    for item in spec.split(','):
        item = item.strip()
        if not item:
            raise ValueError('--tasks contains an empty selection')
        if item in by_id:
            selected.append((by_id[item], tasks[by_id[item]]))
            continue
        first, separator, last = item.partition('-')
        if separator:
            if not first or not last or '-' in last:
                raise ValueError(f'invalid task selection {item!r}; use ID, INDEX or START-END')
            start, stop = _position(first, len(tasks)), _position(last, len(tasks))
            if stop < start:
                raise ValueError(f'task range {item!r} has its end before its start')
            selected.extend(enumerate(tasks[start:stop + 1], start))
        else:
            selected.append((position := _position(item, len(tasks)), tasks[position]))
    return selected


def local_tasks(path=None):
    """Load the legacy bundled pool, or explicit JSON-array/JSONL task rows."""
    if path is None:
        return [dict(task_id=position, problem=row['prompt'], answer=row['answer']) for position, row in enumerate(prompts.pool())]
    raw = Path(path).read_text()
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        raise ValueError('--local-pool must contain a JSON array or JSONL task rows')
    tasks = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'local task row {position} is not an object')
        task_id = row.get('task_id', row.get('id', position))
        problem = row.get('problem', row.get('prompt'))
        prompt = row.get('prompt') if 'problem' in row and 'prompt' in row else None
        tasks.append(dict(task_id=task_id, problem=problem, answer=row['answer'], prompt=prompt))
    return tasks


def select_pool(args):
    """Return ``(position, task, batch_prompt)`` rows for the requested pool."""
    if args.pool == 'live':
        tasks = list(live_pool.active_tasks(public_url=args.public_url, cache=args.cache))
        selected = select_tasks(tasks, args.tasks)
    else:
        tasks = local_tasks(args.local_pool)
        positions = parse_tasks(args.tasks, len(tasks)) if args.tasks is not None else list(range(len(tasks)))
        selected = [(position, tasks[position]) for position in positions]
    rows = []
    for position, task in selected:
        if not isinstance(task.get('task_id'), (str, int)) or not isinstance(task.get('problem'), str) or not isinstance(task.get('answer'), str):
            raise ValueError(f'task at position {position} needs task_id, problem and answer')
        # The live helper supplies the exact official batch prompt. A bare local problem retains the legacy renderer.
        prompt = task.get('prompt') if args.pool == 'live' and task.get('prompt') is not None else prompts.batch_prompt(task['problem'])
        rows.append((position, task, prompt))
    return rows


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--head', required=True, help='head checkpoint dir (tokenizer is read from here)')
    ap.add_argument('--url', action='append', required=True, help='vLLM server base URL (repeat for several)')
    ap.add_argument('--model', default='m', help='--served-model-name of the vLLM server')
    ap.add_argument('--pool', choices=('live', 'local'), default='live', help='live public active pool (default) or explicit local operation')
    ap.add_argument('--tasks', default=None, help="live: comma-separated exact string task IDs and/or zero-based positions/inclusive ranges in the complete active list, e.g. gen:42,math:9,0-9,20; default: ALL active tasks. local: legacy numeric positions/ranges, default 0-7495")
    ap.add_argument('--local-pool', type=Path, default=None, help='optional local task JSON array or JSONL file (only with --pool local; default: bundled legacy pool)')
    ap.add_argument('--public-url', default='https://affine.io', help='public service base URL for the live pool')
    ap.add_argument('--cache', default=None, help='optional live-pool cache directory')
    ap.add_argument('--k', type=int, default=8, help='samples per task (the trainer keeps up to 8 per prompt)')
    ap.add_argument('--seed', default=None, help='plan seed for request seeds (default: random)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--concurrency', type=int, default=512)
    return ap


def parse_args(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.pool == 'local' and args.tasks is None:
        args.tasks = '0-7495'
    if args.pool != 'local' and args.local_pool is not None:
        ap.error('--local-pool requires --pool local')
    if args.k < 1:
        ap.error('--k must be positive')
    return args


def run(args):
    seed = args.seed or secrets.token_hex(16)
    grade.init(args.head); tok = grade._G['tok']
    rows = select_pool(args)
    jobs, meta = [], {}
    for key, (position, task, prompt) in enumerate(rows):
        ids = prompts.batch_ids(tok, prompt); meta[key] = (task, prompt)
        jobs += [((key, j), ids, sampler.request_seed(seed, 'miner', position, j)) for j in range(args.k)]
    res = asyncio.run(sampler.sample_all(args.url, jobs, model=args.model, concurrency=args.concurrency))
    with open(args.out, 'w') as f:
        for key in range(len(rows)):
            task, prompt = meta[key]
            rs = []
            for j in range(args.k):
                out, stop = res[(key, j)]
                kind, how = grade.grade(out, stop, task['answer'], task['problem'])
                body = out[:-1] if (stop == 'eos' and out and out[-1] in sampler.EOS) else out
                text = tok.decode(body, skip_special_tokens=False)
                ok = tok(text, add_special_tokens=False)['input_ids'] == body
                rs.append(dict(output=text, tokens=len(out), stop=stop, kind=kind, how=how, reencode_ok=ok))
            f.write(json.dumps(dict(task=task['task_id'], prompt=prompt, answer=task['answer'], k=args.k, seed=seed, rollouts=rs)) + '\n')
    print(json.dumps(dict(out=args.out, tasks=len(rows), samples=len(jobs), seed=seed)))


def main(argv=None):
    run(parse_args(argv))


if __name__ == '__main__':
    main()
