#!/usr/bin/env python3
"""Turn rollouts.jsonl (from generate.py) into an upload-ready batch file and check it with the validator's own checker.

  python3 make_batch.py --head ./head --hotkey 5F... --rollouts rollouts.jsonl --out batch.json \\
      [--select correct1|mixed|all] [--max-prompts 125] [--per-prompt 8] [--lr 5e-7 --kl 0.02] [--max-tokens 10096]

--select correct1 (default): one correct rollout per prompt, reward 1.0, only prompts the head got right at least once but not
          always (0 < c < k), hardest first (lowest c/k). This is the shape of every GREEN so far (125 x 1, v1 defaults).
--select mixed:  prompts with 0 < c < k, up to --per-prompt rollouts each, reward 1.0 correct / 0.0 otherwise.
--select all:    every prompt, up to --per-prompt rollouts each, reward 1.0 / 0.0.
--max-tokens:    skip rollouts longer than this (default 12288 = the cap; the trainer subtracts 0.5*clip((len-10096)/2192, 0, 1)
                 from longer ones, len incl. EOS; trainer v8.3.3m from epoch 590, before: (len-6000)/2192 at an 8192 cap).
--lr/--kl:       write affine-batch-v2 with a train block (lr 1.25e-7..1e-6, kl 0.005..0.1); omit both for v1 defaults.
Observations, not rules: see miner/README.md "How to win".
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kitpath  # noqa: E402,F401
import ff_format  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--head', required=True); ap.add_argument('--hotkey', required=True)
    ap.add_argument('--rollouts', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--select', choices=('correct1', 'mixed', 'all'), default='correct1')
    ap.add_argument('--max-prompts', type=int, default=125); ap.add_argument('--per-prompt', type=int, default=8)
    ap.add_argument('--max-tokens', type=int, default=12288)
    ap.add_argument('--lr', type=float); ap.add_argument('--kl', type=float)
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.rollouts) if l.strip()]
    groups = []
    for r in recs:
        rs = [x for x in r['rollouts'] if x['tokens'] <= a.max_tokens and x.get('reencode_ok', True)]
        k = len(r['rollouts']); c = sum(x['kind'] == 'positive' for x in r['rollouts'])
        if a.select in ('correct1', 'mixed') and not 0 < c < k:
            continue
        if a.select == 'correct1':
            good = sorted((x for x in rs if x['kind'] == 'positive'), key=lambda x: x['tokens'])
            if not good:
                continue
            groups.append((c / k, dict(prompt=r['prompt'], rollouts=[dict(output=good[0]['output'], reward=1.0)])))
        else:
            sel = rs[:a.per_prompt]
            if sel:
                groups.append((c / k, dict(prompt=r['prompt'], rollouts=[dict(output=x['output'], reward=1.0 if x['kind'] == 'positive' else 0.0) for x in sel])))
    groups.sort(key=lambda g: g[0])
    out, n_roll = [], 0
    for _, g in groups:
        if len(out) >= a.max_prompts or n_roll + len(g['rollouts']) > 1000:
            break
        out.append(g); n_roll += len(g['rollouts'])
    doc = dict(version='affine-batch-v1', hotkey=a.hotkey, groups=out)
    if a.lr is not None or a.kl is not None:
        train = {k: v for k, v in (('lr', a.lr), ('kl', a.kl)) if v is not None}
        doc = dict(version='affine-batch-v2', hotkey=a.hotkey, train=train, groups=out)
    raw = json.dumps(doc, ensure_ascii=False).encode()
    tok = ff_format.tokenizer(a.head)
    ok, summary, _ = ff_format.check(raw, a.hotkey, tok, 125, 1000, v2=True)
    if not ok:
        sys.exit('INVALID: %s' % summary)
    Path(a.out).write_bytes(raw)
    print(json.dumps(dict(out=a.out, **summary)))


if __name__ == '__main__':
    main()
