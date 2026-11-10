#!/usr/bin/env python3
"""SN120 promotion-gate verdict ledger (Jacob t690u, 2026-10-26).

Append-only public ledger of every gate verdict -> state/LEDGER.public.jsonl and https://affine.io/gate.json,
plus per-epoch gate timing (pre-probe on H, post-probe on C) for the site's epoch timing chart.
Private: state/REJECTED.private.jsonl keeps every rejected candidate id + batch manifest refs.
Never publishes seeds or the private sample task ids.
"""
import hashlib, json, os, time
from pathlib import Path

VERSION = 'sn120-promotion-gate-ledger-v1'
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
V = RA / '20261026-promotion-gate'
ST = V / 'state'; EP = ST / 'epochs'
LIFE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1')
PROBE_RESULTS = RA / '20261022-env-probe/state/results'
LEDGER = ST / 'LEDGER.public.jsonl'
REJECTED = ST / 'REJECTED.private.jsonl'
SITE_OUT = Path('/home/const/subnet120/affine/website/gate.json')
FORBIDDEN = ('seed', 'indices', 'per_task', 'sample_tasks')



SCORE_ALL_FLAG = Path('/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/SCORE-ALL-EPOCHS.json')


def SCORE_RULE():
    """Jacob t737u 2026-10-27: scores update on pass and fail (flag staged/SCORE-ALL-EPOCHS.json)."""
    try: f = json.loads(SCORE_ALL_FLAG.read_text())
    except Exception: f = {}
    if f.get('enabled') is True:
        return ('Scores update on pass and fail: from E%s a RED epoch is scored like a GREEN one (head probe vs the discarded '
                'candidate probed on that epoch\'s own batch tasks; zero flip credit for RED epochs before E%s).' % (f.get('from_epoch', 169), f.get('from_epoch', 169)))
    return 'Zero flip credit for that epoch.'

def rj(p, default=None):
    try:
        return json.loads(Path(p).read_bytes())
    except Exception:
        return default


def sha_file(p):
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception:
        return None


def mtime(p):
    try:
        return round(Path(p).stat().st_mtime, 1)
    except Exception:
        return None


def cr(t):
    return time.strftime('%Y-%m-%d %H:%M:%S CR (UTC-6)', time.gmtime(t - 6 * 3600)) if t else None


def manifest_window(e):
    plan = rj(LIFE / 'epochs' / str(e) / 'PLAN.json') or {}
    m = plan.get('manifest') or {}
    m = m.get('payload', m) if isinstance(m, dict) else {}
    return m.get('start'), m.get('deadline')


def pre_probe(head, gather_start):
    r = rj(PROBE_RESULTS / ('%s.private.json' % head)) or {}
    if not r:
        return None
    return dict(checkpoint=head, started_at=r.get('started_at'), finished_at=r.get('finished_at'),
                tasks=r.get('samples'), partial=r.get('partial'),
                reused=bool(gather_start and r.get('finished_at') and r['finished_at'] < gather_start))


def _head_chain_has(candidate):
    """True only after PE HEAD-CHAIN append (PIT cutover / pe_apply.record_applied)."""
    if not isinstance(candidate, str) or len(candidate) != 64:
        return False
    pe_state = os.environ.get('PE_STATE') or os.environ.get('PE_ST')
    roots = []
    if pe_state:
        roots.append(Path(pe_state))
    roots.append(LIFE.parent / 'pe' if hasattr(LIFE, 'parent') else None)
    roots.append(ST / 'pe')
    seen = set()
    for root in roots:
        if root is None:
            continue
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        chain = rj(root / 'HEAD-CHAIN.json') or {}
        for row in chain.get('applied') or []:
            if not isinstance(row, dict):
                continue
            if row.get('candidate') == candidate or row.get('new_head') == candidate:
                return True
    return False


def entry(e):
    d = EP / str(e); dec = rj(d / 'DECISION.json')
    if not dec or dec.get('decision') not in ('green', 'red'):
        return None
    life = LIFE / 'epochs' / str(e)
    applied = rj(life / 'PROMOTION-GATE.json') or {}
    tl = rj(d / 'TIMELINE.json') or {}
    ev = dec.get('evaluated') or {}
    start, deadline = manifest_window(e)
    if dec['decision'] == 'green':
        if _head_chain_has(dec.get('candidate')):
            verdict = 'promoted'
        else:
            pe_block = None
            pe_state = os.environ.get('PE_STATE') or os.environ.get('PE_ST')
            if pe_state:
                for row in (rj(Path(pe_state) / 'GREENS.json') or []):
                    if isinstance(row, dict) and row.get('candidate') == dec.get('candidate'):
                        pe_block = row
                        break
            if pe_block and (
                pe_block.get('state') == 'blocked_incomplete_master'
                or pe_block.get('apply_blocked') == 'incomplete-fp32-master'
            ):
                verdict = 'blocked'
            else:
                verdict = 'not-promoted'
    else:
        verdict = 'rejected'
    return dict(
        epoch=e, head=dec['head'], candidate=dec['candidate'], decision=dec['decision'], verdict=verdict,
        applied=applied.get('decision'), mode=dec.get('mode'),
        z=ev.get('z'), candidate_cap_rate=ev.get('c_cap_rate'), n=ev.get('n'),
        head_pass=ev.get('h_pass'), candidate_pass=ev.get('c_pass'),
        batch_tasks_dropped=dec.get('batch_overlap'), reasons=dec.get('reasons'),
        reason=(
            'blocked_incomplete_master (HEAD never moved)' if verdict == 'blocked'
            else ('green DECISION without HEAD-CHAIN append' if verdict == 'not-promoted'
                  else ('; '.join(dec.get('reasons') or []) or (
                      ('z %s %.1f and cap-hit < %.2f' % (
                          '>=' if dec.get('z_inclusive') else '>',
                          dec.get('z_min', -1.0), dec.get('cap_max', 0.20),
                      )) if verdict == 'promoted' else None)))
        ),
        thresholds=dict(z_min=dec.get('z_min', -1.0), z_inclusive=bool(dec.get('z_inclusive')), cap_max=dec.get('cap_max', 0.20),
                        min_tasks=dec.get('min_tasks', 200)), red_credit=dec.get('red_credit'),
        commitment=dec.get('commitment'),
        timestamps=dict(gather_start=start, upload_deadline=deadline,
                        training_result=tl.get('result_mtime') or mtime(life / 'TRAINING-RESULT.json'),
                        pre_probe=pre_probe(dec['head'], start),
                        post_probe=dict(start=tl.get('fetch_start') or tl.get('h_ready'), fetch_start=tl.get('fetch_start'),
                                        fetched=tl.get('fetched'), serve_start=tl.get('serve_start'), served=tl.get('served'),
                                        run_start=tl.get('run_start'), end=tl.get('run_done')),
                        decided_at=dec.get('decided_at'), finalized=mtime(life / 'FINALIZED.json')),
        decided_at_cr=dec.get('decided_at_cr'), timing_s=dec.get('timing'))


def private_reject(e, ent):
    life = LIFE / 'epochs' / str(e)
    tr = rj(life / 'TRAINING-RESULT.json') or {}
    desc = ((((tr.get('payload') or {}).get('report') or {}).get('publication_receipt') or {}).get('payload') or {}).get('descriptor') or {}
    return dict(epoch=e, head=ent['head'], candidate=ent['candidate'], z=ent['z'], cap=ent['candidate_cap_rate'], n=ent['n'],
                training_result_sha256=sha_file(life / 'TRAINING-RESULT.json'),
                dispatch_sha256=sha_file(life / 'coordinator' / 'DISPATCH.json'),
                k1_dispatch_sha256=sha_file(life / 'TRAIN-DISPATCH-K1.json') if (life / 'TRAIN-DISPATCH-K1.json').exists() else None,
                training_job_sha256=sha_file(life / 'TRAINING-JOB.json'),
                finalized_sha256=sha_file(life / 'FINALIZED.json'),
                checkpoint_descriptor=desc,
                r2_prefix='mimo/affine-mimo9b-grpo-20261013/checkpoints/%s/' % ent['candidate'],
                retention='checkpoint files kept in R2 and on the probe pod (gate ckpt cache, newest 4 + served) while disk allows; hash recorded regardless',
                recorded_at=time.time())


def load_jsonl(p):
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                try: out.append(json.loads(line))
                except Exception: pass
    return out


def append(p, obj, mode=0o644):
    new = not p.exists()
    with open(p, 'a') as f:
        f.write(json.dumps(obj, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())
    if new: os.chmod(p, mode)


def current_status():
    s = rj(ST / 'status.json') or {}
    e = s.get('epoch'); out = dict(epoch=e, head=s.get('head'), mode=s.get('mode'), stage=(s.get('state') or {}).get(str(e)))
    if e is not None:
        tl = rj(EP / str(e) / 'TIMELINE.json') or {}
        start, deadline = manifest_window(e)
        out.update(gather_start=start, upload_deadline=deadline, pre_probe=pre_probe(s.get('head'), start) if s.get('head') else None,
                   post_probe=dict(start=tl.get('fetch_start') or tl.get('h_ready'), end=tl.get('run_done')) if tl else None,
                   candidate=tl.get('candidate'))
        prev = s.get('state') or {}
        out['stages'] = {k: v for k, v in prev.items() if k.isdigit()}
    return out


HELDOUT_RULE = ('Promotion gate (held-out, live from E271; text updated 2026-10-30): the candidate is promoted (GREEN) only if its '
                'weighted per-task gain over the head is positive and weighted_z > 1.00 (strict), the estimated cap-hit rate stays under 0.20 '
                'and at least 100 tasks are kept. Head and candidate are scored on the same tasks of a private held-out probe set (one round '
                'per head; set size and candidate samples are fixed per round: 1000 tasks and 4 candidate samples per task before, 2000 tasks and 8 candidate samples per task in rounds opened after 2026-11-01 14:19 CR, the first 4 with the same request seeds as head block 1; the head gets 4 samples on every task + 12 per kept task); gain = inverse-variance-weighted mean of the per-task '
                'pass-rate differences (weights from the round baseline), weighted_z = gain / SE, SE = sqrt(se_task^2 + sigma_run^2 (1 + 1/3)): se_task is the per-task standard error of the gain and '
                'sigma_run the head\'s run-to-run noise measured on the round\'s 3 replicate head runs (E271-E353: sigma_run floored '
                'at 0.022 = 2.2 points; from E354 no fixed floor, the measured value is used; rounds opened after 2026-11-01 14:19 CR: the pooled value over the last 20 finished rounds, frozen when the round opens, published as sigma_run in current.json). From E296 (free-file '
                'cutover) the turn batch is not excluded: every kept task counts. The round commitment is public '
                '(https://affine.io/gate/heldout/current.json); a round is revealed only if a GREEN closed it, and then only the GREEN turn '
                '(seed, task ids, head baseline, the GREEN candidate per-task grades: https://affine.io/gate/heldout/<round>.json, '
                'https://affine.io/gate/reveal/<epoch>.json); RED turns and their probe results are never published. RED = candidate '
                'discarded, head kept. In each entry, paired_z holds the weighted_z.')


def _prior_rule():
    # prior-v1 baseline (RA/20261101-baseline-switch): rule text when PROMOTION-GATE.json heldout.baseline == 'prior-v1'
    try:
        f = json.loads(Path('/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json').read_text())
        if (f.get('heldout') or {}).get('baseline') != 'prior-v1': return None
        import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
        import prior_gate as _PG; return 'Promotion gate (held-out, ' + _PG.RULE + ') Earlier rule: ' + HELDOUT_RULE
    except Exception: return None


def K1_RULE_SUFFIX():
    # t712u (2026-10-27) K=1 per-UID: from K1.json from_epoch the epoch batch is ONE UID's turn batch and credit is per UID
    f = rj(Path('/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/K1.json')) or {}
    if f.get('enabled') is not True: return ''
    return (' K=1 from E%s: the epoch batch is one UID turn (see https://affine.io/k1.json); head and candidate are '
            'scored on the same private held-out tasks (see the rule above); GREEN = promoted '
            'and the turn UID joins the last-10 distinct GREEN payees (1/N each); RED = head kept, the turn earns nothing; '
            'no fixed-reference guard.' % f.get('from_epoch'))


def tick():
    ledger = load_jsonl(LEDGER); seen = {(r['epoch'], r['candidate']) for r in ledger}
    rej = load_jsonl(REJECTED); rseen = {(r['epoch'], r['candidate']) for r in rej}
    for d in sorted((p for p in EP.glob('*') if p.name.isdigit()), key=lambda p: int(p.name)):
        e = int(d.name); ent = entry(e)
        if not ent: continue
        if (e, ent['candidate']) not in seen:
            if ent['timestamps']['finalized'] is None and ent['applied'] is None:
                continue  # wait until the lifecycle applied it so `applied` is recorded
            ent['ledgered_at'] = time.time(); append(LEDGER, ent); ledger.append(ent); seen.add((e, ent['candidate']))
        if ent['decision'] == 'red' and (e, ent['candidate']) not in rseen:
            append(REJECTED, private_reject(e, ent), 0o600); rseen.add((e, ent['candidate']))
    doc = dict(version=VERSION, updated_at=time.time(), updated_at_cr=cr(time.time()),
               rule=(_prior_rule() or HELDOUT_RULE) + ' History: E152-E153 paired z > -1, E154-E167 paired z >= -1.5, E168-E270 paired z > 0 on a 1000-task '
                    'private probe sample minus the epoch batch (cap-hit < 0.20); held-out gate from E271. ' + SCORE_RULE() + ' Append-only.'
                    + K1_RULE_SUFFIX(),
               counts=dict(total=len(ledger),
                           promoted=sum(r.get('verdict') == 'promoted' for r in ledger),
                           rejected=sum(r.get('verdict') == 'rejected' for r in ledger),
                           blocked=sum(r.get('verdict') == 'blocked' for r in ledger),
                           not_promoted=sum(r.get('verdict') == 'not-promoted' for r in ledger)),
               current=current_status(), entries=ledger)
    try:   # z bar: fixed at z > 1.00 (RA/20261102-zbar-flat); gate.json z_bar + /gate/zbar.json
        import prior_gate as _PG; zb = _PG.z_bar_public()
        if zb:
            doc['z_bar'] = zb
            zp = SITE_OUT.with_name('gate'); zp.mkdir(exist_ok=True); zp = zp / 'zbar.json'; zt_ = zp.with_name('.zbar.json.tmp')
            zt_.write_text(json.dumps(zb, indent=1, sort_keys=True) + '\n'); os.chmod(zt_, 0o644); os.replace(zt_, zp)
    except Exception as ex:
        doc['z_bar'] = None; print('z_bar publish error: %s' % ex, flush=True)
    s = json.dumps(doc, sort_keys=True)
    for bad in ('"seed"', '"indices"', '"per_task"', 'revealed_'):
        if bad in s: raise ValueError('refusing to publish %s' % bad)
    tmp = SITE_OUT.with_name(SITE_OUT.name + '.tmp'); tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    os.chmod(tmp, 0o644); os.replace(tmp, SITE_OUT)
    return doc['counts']


if __name__ == '__main__':
    import sys
    if '--watch' not in sys.argv:
        print(tick()); raise SystemExit
    last = None
    while True:
        try:
            tick(); last = None
        except Exception as ex:
            if str(ex) != last: print(json.dumps(dict(event='error', error=str(ex)[:300], at=time.time())), flush=True)
            last = str(ex)
        time.sleep(20)
