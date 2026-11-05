"""hard-v1 difficulty pass analysis (host). Pulls RUN/grades.jsonl from the pass pod (pp.sh), aggregates per task over every pass on
head 78fff7a0 and applies the keep rule. Per-task numbers stay in private/ (0600); stdout prints aggregates only.
usage: analyze.py stageB   -> private/stageB_tasks.jsonl (tasks worth more rollouts after pass 0)
       analyze.py final    -> private/passrates.private.jsonl, private/slice.jsonl, ../manifest parts (private/pass_summary.json)"""
import collections, json, os, subprocess, sys, hashlib
D = "/home/const/subnet120-rewrite/state/task-slices/hard-v1"; P = D + "/private"; RAD = "/home/const/subnet120-rewrite/state/root-audits/20261105-hard-slice-scope"
HEAD = "8389b90be4367d8eccf444f92508e38d87fe3736ae23c195e609f027991985b6"
LO, HI, CAPMAX = 0.05, 0.6, 0.33
def w(path, text):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as f: f.write(text)
def pull():
    if os.environ.get("GRADES"): return [json.loads(l) for l in open(os.environ["GRADES"]) if l.strip()]
    raw = subprocess.run([RAD + "/pod/pp.sh", 'cat /opt/prior/runs/hardv1/grades.jsonl'], capture_output=True, timeout=600).stdout
    w(P + '/grades.private.jsonl', raw.decode()); return [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
def agg(G):
    A = collections.defaultdict(lambda: dict(n=0, c=0, w=0, u=0, b=0, x=0, tok=0, passes=0))
    for g in G:
        if g.get('head') != HEAD or g.get('err'): continue
        a = A[g['task_id']]
        for k in ('n', 'c', 'w', 'u', 'b', 'x', 'tok'): a[k] += int(g.get(k) or 0)
        a['passes'] += 1
    return A
cands = {json.loads(l)['id']: json.loads(l) for l in open(P + '/candidates.jsonl')}
mode = sys.argv[1]; G = pull(); A = agg(G)
errs = sum(1 for g in G if g.get('err'))
if mode == 'stageB':
    keep = [t for t, a in A.items() if a['n'] and not (a['c'] == a['n']) and not (a['b'] >= 0.75 * a['n'])]
    drop_all = sum(1 for a in A.values() if a['n'] and a['c'] == a['n']); drop_cap = sum(1 for a in A.values() if a['n'] and a['c'] != a['n'] and a['b'] >= 0.75 * a['n'])
    w(P + '/stageB_tasks.jsonl', ''.join(json.dumps(cands[t], sort_keys=True, ensure_ascii=False) + '\n' for t in sorted(keep, key=lambda t: int(t.split(':')[1]))))
    print(json.dumps(dict(stageA_tasks=len(A), prompt_too_long=errs, all_solved_dropped=drop_all, cap_bound_dropped=drop_cap, to_stageB=len(keep),
                          stageB_sha256=hashlib.sha256(open(P + '/stageB_tasks.jsonl', 'rb').read()).hexdigest())))
    sys.exit()
# final
rows = []; hist = collections.Counter(); kept = []; reasons = collections.Counter(); by_src = collections.Counter(); cap_hist = collections.Counter()
for t, a in A.items():
    p = a['c'] / a['n'] if a['n'] else 0.0; cr = a['b'] / a['n'] if a['n'] else 1.0
    rows.append(dict(id=t, src=cands[t]['src'], n=a['n'], c=a['c'], b=a['b'], u=a['u'], p=round(p, 4), cap=round(cr, 4), tok=a['tok']))
    hist[min(int(p * 10), 9)] += 1; cap_hist[min(int(cr * 10), 9)] += 1
    if cr >= CAPMAX: reasons['cap_hit>=0.33'] += 1
    elif p < LO: reasons['p<0.05'] += 1
    elif p > HI: reasons['p>0.6'] += 1
    else: kept.append(t); by_src[cands[t]['src']] += 1
w(P + '/passrates.private.jsonl', ''.join(json.dumps(r, sort_keys=True) + '\n' for r in sorted(rows, key=lambda r: int(r['id'].split(':')[1]))))
K = [A[t] for t in kept]
kp = collections.Counter(); 
for t in kept:
    a = A[t]; kp['%.1f-%.1f' % (int(a['c'] / a['n'] * 10) / 10, int(a['c'] / a['n'] * 10) / 10 + 0.1)] += 1
nd = collections.Counter(a['n'] for a in A.values())
summ = dict(head=HEAD, rollouts_per_task_distribution={str(k): v for k, v in sorted(nd.items())}, never_scored_grader_timeout=len(cands) - len(A), rule=dict(pass_rate=[LO, HI], cap_hit_max_exclusive=CAPMAX, rollouts='adaptive: stage A 4 rollouts on every candidate; stage B +4 for tasks not 4/4 solved and not >=3/4 capped; stage C +4 for tasks with 1-6 of 8 solved and < 4/8 capped (12 total). The same keep rule is applied to every task on all its rollouts.'),
            tasks_sampled=len(A), prompt_too_long=errs, drop_reasons=dict(reasons), kept=len(kept), kept_by_src=dict(by_src),
            all_tasks_p_hist_deciles={'%.1f-%.1f' % (k / 10, k / 10 + 0.1): hist[k] for k in range(10)},
            all_tasks_cap_hist_deciles={'%.1f-%.1f' % (k / 10, k / 10 + 0.1): cap_hist[k] for k in range(10)},
            kept_p_hist=dict(sorted(kp.items())),
            kept_mean_p=round(sum(a['c'] for a in K) / max(1, sum(a['n'] for a in K)), 4),
            kept_pooled_cap_hit=round(sum(a['b'] for a in K) / max(1, sum(a['n'] for a in K)), 4),
            kept_max_task_cap_hit=round(max((a['b'] / a['n'] for a in K), default=0), 4),
            kept_mean_tokens_per_rollout=round(sum(a['tok'] for a in K) / max(1, sum(a['n'] for a in K))),
            all_pooled_cap_hit=round(sum(a['b'] for a in A.values()) / max(1, sum(a['n'] for a in A.values())), 4),
            rollouts_total=sum(a['n'] for a in A.values()))
slice_rows = []
for j, t in enumerate(sorted(kept, key=lambda t: int(t.split(':')[1]))):
    c = cands[t]; slice_rows.append(dict(slice_id='hard-v1:%d' % j, cand_id=t, src=c['src'], src_id=c['src_id'], uid=c['uid'], license=c['license'],
                                         problem=c['problem'], answer=c['answer']))
w(P + '/slice.jsonl', ''.join(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n' for r in slice_rows))
summ['slice_sha256'] = hashlib.sha256(open(P + '/slice.jsonl', 'rb').read()).hexdigest()
summ['kept_by_license'] = dict(collections.Counter(r['license'] for r in slice_rows))
w(P + '/pass_summary.json', json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))
