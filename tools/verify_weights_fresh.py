#!/usr/bin/env python3
"""READ-ONLY SN120 weight freshness check (no signing, no writes, no chain transactions).

Prints UID 0 ("const", owner validator) last_update block + CT age, the on-chain weight
vector hash (sha256(json.dumps(Weights[120][0]))[:10], same convention as RESUME 17), whether it changed
vs ~1h and ~2h ago, pending timelocked commits, whether the on-chain vector matches the
writer's last submission, the newest hourly flip assessment's assessment_stale / evidence cutoff age, and the flip
live-log freeze check (digest unchanged while score_sha256 changes and epochs advance, or scores.json self-pinned).
Exit 0 = OK; 1 = stale weights (>75 min since last_update) or stale assessment; 2 = check error.
"""
import datetime, glob, hashlib, json, os, sys, time
import bittensor as bt
from bittensor._generated.storage import SubtensorModule as S

NETUID, UID = 120, 0
STATE = '/home/const/subnet120-rewrite/state/live-math-launch-preparation-v1/reward-state-v1'
ADIR = STATE + '/current-assessment-flip-live-1'   # 2026-10-23: live weight path is flip-incentive (v9 retired at 2 PM CR cutover)
FLIP_VERSION = 'sn120-flip-live-weights-v1-nonzero-floor'
import importlib.util as _u
_s = _u.spec_from_file_location('_flip_freeze', '/home/const/subnet120-rewrite/state/root-audits/20261023-flip-freeze-fix/flip_freeze.py')
flip_freeze = _u.module_from_spec(_s); _s.loader.exec_module(flip_freeze)
MAX_AGE_MIN = 75
CT = datetime.timezone(datetime.timedelta(hours=-6), 'CR')  # America/Costa_Rica (UTC-6, no DST); keys keep the old _ct names

def h(w): return hashlib.sha256(json.dumps([list(x) for x in w]).encode()).hexdigest()[:10]
def ct(ts): return datetime.datetime.fromtimestamp(ts, CT).strftime('%m-%d %H:%M CR')

def main():
    out, problems = {}, []
    with bt.Subtensor('finney') as c:
        blk = c.block() if callable(c.block) else c.block
        now = c.at(blk).timestamp().timestamp()
        lu = int(c.query(S.LastUpdate, [NETUID])[UID])
        lu_ts = c.at(lu).timestamp().timestamp()
        w = c.query(S.Weights, [NETUID, UID])
        w1 = c.query(S.Weights, [NETUID, UID], block=blk - 300)
        w2 = c.query(S.Weights, [NETUID, UID], block=blk - 600)
        pending = []
        for key, entries in c.query_map(S.TimelockedWeightCommits, [NETUID]):
            for e in entries:
                pending.append(dict(epoch_key=key, commit_block=int(e[1])))
    age_min = (now - lu_ts) / 60
    out.update(block=blk, last_update_block=lu, last_update_ct=ct(lu_ts), last_update_age_min=round(age_min, 1),
               onchain_hash=h(w), onchain_nonzero=sum(1 for _, v in w if v),
               hash_1h_ago=h(w1), hash_2h_ago=h(w2),
               changed_vs_1h=h(w) != h(w1), changed_vs_2h=h(w) != h(w2), pending_commits=pending)
    try:
        # Remember each submitted vector (own cache file only) so a revealed vector can be matched to its window.
        hist_path = os.path.expanduser('~/.cache/sn120_verify_submissions.json')
        try: hist = json.load(open(hist_path))
        except Exception: hist = {}
        sub = json.load(open(STATE + '/weights.json'))['latest']
        m = max(sub['weights']) or 1
        hist[str(sub['window_end'])] = dict(block=sub['block'], u16=sorted([u, round(x / m * 65535)] for u, x in zip(sub['uids'], sub['weights']) if round(x / m * 65535)))
        hist = dict(sorted(hist.items())[-12:])
        os.makedirs(os.path.dirname(hist_path), exist_ok=True)
        tmp = hist_path + '.tmp'; json.dump(hist, open(tmp, 'w')); os.replace(tmp, hist_path)
        got = {u: v for u, v in w if v}
        def same(exp): return len(exp) == len(got) and all(abs(got.get(u, -999) - v) <= 1 for u, v in exp)
        matches = [int(k) for k, v in hist.items() if same(v['u16'])]
        out.update(last_submission_window_end_ct=ct(sub['window_end']), last_submission_block=sub['block'],
                   onchain_matches_last_submission=same(hist[str(sub['window_end'])]['u16']),
                   onchain_matches_submission_window_ct=[ct(k) for k in matches])
    except Exception as e:
        out['submission_compare_error'] = repr(e)[:200]
    files = sorted(glob.glob(ADIR + '/assessment-*.json'), key=lambda f: int(f.rsplit('-', 1)[1][:-5]))
    a = json.load(open(files[-1]))['payload']
    ev = a.get('evidence_cutoff') or 0
    out.update(assessment_file=os.path.basename(files[-1]), assessment_stale=a.get('assessment_stale'),
               evidence_error=a.get('evidence_error'), evidence_cutoff_ct=ct(ev) if ev else None,
               evidence_cutoff_age_min=round((time.time() - ev) / 60, 1) if ev else None)
    if age_min > MAX_AGE_MIN: problems.append(f'last_update {age_min:.0f} min old (> {MAX_AGE_MIN})')
    if a.get('assessment_stale') is not False: problems.append(f"assessment stale (error {a.get('evidence_error')}, cutoff {out['evidence_cutoff_ct']})")
    if a.get('version') != FLIP_VERSION: problems.append(f"assessment version {a.get('version')} is not the flip live rule")
    fp, fw, finfo = flip_freeze.check()
    out.update(flip_windows=finfo['flip_windows'], flip_scores=finfo['scores'], flip_warnings=fw)
    problems += fp
    if not out['changed_vs_2h']: out['warning'] = 'on-chain vector unchanged for >= 2h'
    out['status'] = 'FAIL' if problems else 'OK'; out['problems'] = problems
    print(json.dumps(out, indent=1))
    return 1 if problems else 0

if __name__ == '__main__':
    try: sys.exit(main())
    except Exception as e:
        print(json.dumps(dict(status='ERROR', error=repr(e)[:500]))); sys.exit(2)
