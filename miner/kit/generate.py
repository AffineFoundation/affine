#!/usr/bin/env python3
"""Sample training rollouts from the head at the gate's settings and grade them with the gate's grader.

  python3 generate.py --head ./head --url http://127.0.0.1:8000 --tasks 0-7495 --k 8 --out rollouts.jsonl

Rollouts are sampled from the TRAINER's rendering of the batch prompt (one user message = MANDATE + "\\n\\n" + problem), so the
tokens you upload are tokens the head actually produced for that exact context. That matters: the v8.3.3j step replays the
head's top_k 20 -> top_p 0.95 support at temperature 1 and drops any sampled token outside it from the policy-gradient term
(support renorm), so hand-written or other-model text mostly trains nothing but KL.
Each line: {task, prompt, answer, k, rollouts: [{output, tokens, stop, kind, how, reencode_ok}]}; kind is the gate grader's verdict
(positive / negative / neutral = capped or no single final boxed / indeterminate). output = decode(tokens without the final EOS);
reencode_ok says whether the trainer's re-tokenization of output gives back exactly the sampled tokens.
"""
import argparse, asyncio, json, os, secrets, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts, sampler, grade  # noqa: E402


def parse_tasks(spec, n):
    out = []
    for part in spec.split(','):
        a, _, b = part.partition('-')
        out += list(range(int(a), int(b or a) + 1))
    return [t for t in out if 0 <= t < n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--head', required=True, help='head checkpoint dir (tokenizer is read from here)')
    ap.add_argument('--url', action='append', required=True, help='vLLM server base URL (repeat for several)')
    ap.add_argument('--model', default='m', help='--served-model-name of the vLLM server')
    ap.add_argument('--tasks', default='0-7495', help='task ids, e.g. 0-999,2000-2100')
    ap.add_argument('--k', type=int, default=8, help='samples per task (the trainer keeps up to 8 per prompt)')
    ap.add_argument('--seed', default=None, help='plan seed for request seeds (default: random)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--concurrency', type=int, default=512)
    a = ap.parse_args()
    seed = a.seed or secrets.token_hex(16)
    grade.init(a.head); tok = grade._G['tok']
    pool = prompts.pool(); tasks = parse_tasks(a.tasks, len(pool))
    jobs, meta = [], {}
    for t in tasks:
        p = prompts.batch_prompt(pool[t]['prompt']); ids = prompts.batch_ids(tok, p); meta[t] = (p, ids)
        jobs += [((t, j), ids, sampler.request_seed(seed, 'miner', t, j)) for j in range(a.k)]
    res = asyncio.run(sampler.sample_all(a.url, jobs, model=a.model, concurrency=a.concurrency))
    with open(a.out, 'w') as f:
        for t in tasks:
            rs = []
            for j in range(a.k):
                out, stop = res[(t, j)]
                kind, how = grade.grade(out, stop, pool[t]['answer'], pool[t]['prompt'])
                body = out[:-1] if (stop == 'eos' and out and out[-1] in sampler.EOS) else out
                text = tok.decode(body, skip_special_tokens=False)
                ok = tok(text, add_special_tokens=False)['input_ids'] == body
                rs.append(dict(output=text, tokens=len(out), stop=stop, kind=kind, how=how, reencode_ok=ok))
            f.write(json.dumps(dict(task=t, prompt=meta[t][0], answer=pool[t]['answer'], k=a.k, seed=seed, rollouts=rs)) + '\n')
    print(json.dumps(dict(out=a.out, tasks=len(tasks), samples=len(jobs), seed=seed)))


if __name__ == '__main__':
    main()
