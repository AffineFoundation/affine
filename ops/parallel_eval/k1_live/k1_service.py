#!/usr/bin/env python3
"""SN120 K=1 per-UID service (affinevalidator, systemd --user affine-k1.service). Jacob t712u 2026-10-27 11:52 CR.
Every TICK seconds:
  1. chain snapshot (read-only Finney query: Keys + BlockAtRegistration for netuid 120) -> state/K1-CHAIN.json
     (the v8.2.9l lifecycle reads it at the seal to pick the turn; no chain call inside the lifecycle).
  2. queue (v8.2.9q, Jacob t758u 2026-10-28 12:39 CR, teutonic.ai-style "register to submit"): a registration = (uid, hotkey,
     BlockAtRegistration); each registration gets ONE turn ever. Queue = unserved registrations (UID 0 excluded) by
     (registration block, uid). Published line (next QUEUE_N) = unserved registrations that had a ready batch (>= min_tasks
     distinct admitted tasks) at the LAST seal, in that order (a predictor; the seal decides). staged/K1.json
     "queue_rule": "fewest-turns" -> the v8.2.9o order. -> state/K1-QUEUE.public.json (read by training_params).
  3. credit: every K1 turn (epochs/N/K1-TURN.json) joined with the promotion-gate verdict (gate state/epochs/N/DECISION.json).
     GREEN = earned. Payees = the last up to 10 DISTINCT GREEN UIDs (most recent GREEN first; an 11th pushes out the oldest),
     each 1/N (N = payees whose earning hotkey is still registered on that UID). RED / skipped / no verdict earn nothing.
     No payee yet -> hard nonzero floor = the frozen last-good pre-K1 scores (K1-FLOOR-SCORES.json), never all-zero.
     Writes website/scores.json (read by the unchanged hourly mech-0 writer weights_flip.py) and website/k1.json (public log).
Never writes chain, TRAIN-CONTROL, the lifecycle or the gate. Bans are not applied (Jacob t712u "remove all the bans")."""
import hashlib, json, os, sys, time, traceback
from pathlib import Path

SQUAT_DIR = Path('/home/const/subnet120-rewrite/state/root-audits/20261105-k1-squat-trim/release-signed')
if str(SQUAT_DIR) not in sys.path:
    sys.path.insert(0, str(SQUAT_DIR))

NETUID = 120
V = Path('/home/const/subnet120-rewrite/state/root-audits/20261027-k1-per-uid')
ST = V / 'state'
LIFE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1')
GATE = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/epochs')
FLAG = Path('/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/K1.json')
WEB = Path('/home/const/subnet120/affine/website')
SCORES = WEB / 'scores.json'
K1PUB = WEB / 'k1.json'
FLOOR = ST / 'K1-FLOOR-SCORES.json'
TURNS = ST / 'K1-TURNS.jsonl'
TICK, CHAIN_EVERY, QUEUE_N, PAYEES = 60, 300, 8, 10   # free-file (pt898u): public preview of the next 8
FF_FLAG = FLAG.parent / 'FF.json'
FF_READY = Path('/home/const/subnet120-rewrite/state/root-audits/20261030-free-file/state/FF-READY.json')
GRANTS = Path('/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/K1-GRANTS.json')


def open_grants(chain, ts):
    """K1-GRANTS (Jacob 2026-10-30 19:30 CR), same rule as ff_lifecycle.open_grants/grant_rows: one fresh free-file turn per granted
    hotkey, ahead of the queue, oldest RED epoch first; used only by a real turn (not by a skip); hotkey must be registered now."""
    g = rj(GRANTS) or {}
    if g.get('enabled') is not True: return []
    by_hk = {r['hotkey']: (int(u), r) for u, r in chain['uids'].items() if int(u) != 0 and isinstance(r, dict) and type(r.get('miner_hex')) is str}
    out = []
    for r in sorted(g.get('grants') or [], key=lambda r: int(r.get('red_epoch', 0))):
        hk = r.get('hotkey')
        if hk not in by_hk or any(t.get('hotkey') == hk and int(t.get('epoch_index', -1)) > int(r['red_epoch']) for t in ts): continue
        out.append((by_hk[hk][0], by_hk[hk][1], dict(red_epoch=int(r['red_epoch']), weighted_z=r.get('weighted_z'))))
    return out


def ff_state():
    """free-file (Jacob pt898u): (enabled, old format still open, hotkeys whose current file passed the format check)."""
    f = rj(FF_FLAG) or {}
    if f.get('enabled') is not True:
        return False, True, set()
    r = rj(FF_READY) or {}
    valid = {hk for hk, v in (r.get('miners') or {}).items() if v.get('valid') is True} if time.time() - float(r.get('checked_at', 0)) < 600 else set()
    return True, time.time() < int(f.get('legacy_cutoff_unix', 0)), valid
B58 = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'


def cr(ts=None):
    ts = time.time() if ts is None else ts
    return time.strftime('%Y-%m-%d %H:%M:%S CR (UTC-6)', time.gmtime(ts - 6 * 3600))


def log(**kw):
    kw['at'] = round(time.time(), 1); kw['at_cr'] = cr(kw['at']); print(json.dumps(kw, default=str), flush=True)


def wj(path, obj, mode=0o600):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    t = path.with_name('.' + path.name + '.tmp'); fd = os.open(t, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, 'w') as f: f.write(json.dumps(obj, indent=1, sort_keys=True) + '\n')
    os.chmod(t, mode); os.replace(t, path)


def rj(path, default=None):
    try: return json.loads(Path(path).read_text())
    except Exception: return default


def ss58_pub(addr):
    n = 0
    for ch in addr: n = n * 58 + B58.index(ch)
    raw = n.to_bytes(35, 'big')   # 1 prefix byte (formats < 64) + 32 key + 2 checksum
    pre = raw[:1]; key = raw[1:33]; chk = raw[33:]
    if hashlib.blake2b(b'SS58PRE' + pre + key, digest_size=64).digest()[:2] != chk:
        raise ValueError('ss58 checksum')
    return key.hex()


def chain_snapshot():
    import bittensor as bt
    from bittensor._generated.storage import SubtensorModule as S
    with bt.Subtensor('finney') as c:
        blk = c.block() if callable(c.block) else c.block
        keys = dict(c.query_map(S.Keys, [NETUID], block=blk))
        reg = dict(c.query_map(S.BlockAtRegistration, [NETUID], block=blk))
        n = int(c.query(S.SubnetworkN, [NETUID], block=blk))
    if not keys or len(keys) != n:
        raise RuntimeError('incomplete metagraph keys %d of %d' % (len(keys), n))
    uids = {}
    for u, hk in keys.items():
        uids[str(int(u))] = dict(hotkey=str(hk), miner_hex=ss58_pub(str(hk)), reg_block=int(reg.get(u, 0)))
    return dict(version='sn120-k1-chain-v1', netuid=NETUID, block=int(blk), taken_at=time.time(), taken_at_cr=cr(), n=n, uids=uids)


def turns():
    out = []
    try:
        for line in TURNS.read_text().splitlines():
            try: out.append(json.loads(line))
            except Exception: pass
    except FileNotFoundError: pass
    return out


RULES = ('hotkey-registration-once', 'hotkey-once', 'registration-once', 'fewest-turns')   # v8.2.9s adds hotkey-registration-once
_READY = {}


def served(ts):
    """registrations that already had a turn: (set of (uid, hotkey, reg_block), legacy set of (uid, hotkey))."""
    full, legacy = set(), set()
    for t in ts:
        try: u, hk = int(t['uid']), t['hotkey']
        except Exception: continue
        rb = t.get('reg_block')
        if type(rb) is not int:
            rb = (rj(LIFE / 'epochs' / str(t.get('epoch_index')) / 'K1-TURN.json') or {}).get('reg_block')
        if type(rb) is int and rb > 0: full.add((u, hk, rb))
        else: legacy.add((u, hk))
    return full, legacy


def last_seal_tasks():
    """-> (epoch, {miner_hex: distinct admitted tasks}) from the newest sealed dispatch (same rows the K=1 seal uses)."""
    eps = sorted((int(p.name) for p in LIFE.glob('epochs/*') if p.name.isdigit() and (p / 'coordinator' / 'DISPATCH.json').is_file()))
    if not eps: return None, {}
    e = eps[-1]; path = LIFE / 'epochs' / str(e) / 'coordinator' / 'DISPATCH.json'
    key = (e, path.stat().st_mtime)
    if _READY.get('key') == key: return e, _READY['val']
    d = rj(path) or {}; d = d.get('payload', d)
    rows = list(d.get('selected') or []) + [x for x in (d.get('excluded') or []) if x.get('reason') in ('training-task-budget', 'cross-miner-task-collision')]
    tasks = {}
    for r in rows:
        if isinstance(r, dict) and type(r.get('miner')) is str:
            tasks.setdefault(r['miner'], set()).add((r.get('env_id'), r.get('index')))
    val = {m: len(v) for m, v in tasks.items()}
    _READY.update(key=key, val=val)
    return e, val


def queue(chain, ts, rule='hotkey-once', min_tasks=8):
    queue._last_squat_meta = None
    if rule != 'fewest-turns':
        full, legacy = served(ts)
        hks = {t.get('hotkey') for t in ts}
        regs = {(u, rb) for (u, _h, rb) in full}   # v8.2.9s served registrations
        _, ready = last_seal_tasks()
        ff_on, legacy_ok, ff_valid = ff_state()
        rows = []
        for u, r in chain['uids'].items():
            u = int(u)
            if u == 0: continue
            if rule in ('hotkey-once', 'hotkey-registration-once') and r['hotkey'] in hks: continue
            if rule == 'hotkey-registration-once' and int(r.get('reg_block') or 0) > 0 and (u, int(r['reg_block'])) in regs: continue
            if rule == 'registration-once' and ((u, r['hotkey'], r['reg_block']) in full or (u, r['hotkey']) in legacy): continue
            rows.append((r['reg_block'], u, r['hotkey'], ready.get(r.get('miner_hex'), 0) if legacy_ok else 0))
        rows.sort()
        detail = {}
        if rule == 'hotkey-registration-once' and ff_on:
            import k1_squat_trim as squat
            base = [(b, u, hk) for (b, u, hk, n) in rows]
            ordered, squat_meta = squat.demote(
                base,
                chain_block=int(chain.get('block') or 0),
                chain_taken_at=float(chain.get('taken_at') or time.time()),
                turns=ts,
                valid=ff_valid,
            )
            by = {(b, u, hk): (b, u, hk, n) for (b, u, hk, n) in rows}
            rows = [by[(b, u, hk)] for (b, u, hk) in ordered]
            detail = {(d['uid'], d['hotkey']): d for d in squat_meta['details']}
            queue._last_squat_meta = squat_meta
        if ff_on:   # ready = a valid batch file now, or (until the cutoff) a ready old-format batch at the last seal
            gr = open_grants(chain, ts) if rule == 'hotkey-registration-once' else []
            out = [dict(position=i - len(gr), uid=u, hotkey=r['hotkey'], registration_block=int(r.get('reg_block') or 0), turns=1,
                         file_valid=r['hotkey'] in ff_valid, old_format_tasks_last_seal=0, ready_last_seal=r['hotkey'] in ff_valid, grant=g)
                    for i, (u, r, g) in enumerate(gr)]
            for i, (b, u, hk, n) in enumerate(rows):
                rec = dict(position=i, uid=u, hotkey=hk, registration_block=b, turns=0, file_valid=hk in ff_valid,
                           old_format_tasks_last_seal=n, ready_last_seal=hk in ff_valid or n >= min_tasks)
                d = detail.get((u, hk))
                if d:
                    rec.update(expected_wait_seconds=d['expected_wait_seconds'], expected_wait_days=d['expected_wait_days'],
                               deadline_unix=d['deadline_unix'], deadline_cr=d['deadline_cr'],
                               expired_back_of_queue=bool(d['expired_back_of_queue']),
                               registration_time_unix=d['registration_time_unix'])
                out.append(rec)
            return out
        return [dict(position=i, uid=u, hotkey=hk, registration_block=b, turns=0, last_seal_tasks=n, ready_last_seal=n >= min_tasks)
                for i, (b, u, hk, n) in enumerate(rows)]
    counts = {}
    for t in ts: counts[(int(t['uid']), t['hotkey'])] = counts.get((int(t['uid']), t['hotkey']), 0) + 1
    rows = []
    for u, r in chain['uids'].items():
        u = int(u)
        if u == 0: continue
        rows.append((counts.get((u, r['hotkey']), 0), r['reg_block'], u, r['hotkey']))
    rows.sort()
    return [dict(position=i, uid=u, hotkey=hk, turns=n, registration_block=b) for i, (n, b, u, hk) in enumerate(rows)]



HELDOUT_PUB = WEB / 'gate' / 'heldout'


def pass_rates(e, dec, ev):
    """Per-turn aggregate pass rates for the 'Pass rate & cap-hit per turn' chart (Jacob pt904u 2026-10-31).
    Old gate (to E270): DECISION.json evaluated.{h_pass,c_pass,n} on the ~1,000-task private gate sample (turn batch removed).
    Held-out gate (from E272): DECISION.json redacts n and the pass rates while the held-out round is open ('published with the
    round reveal'). Once the round's public reveal /gate/heldout/<round>.json says status=closed, publish the gate's own
    aggregates for that turn (EVALUATED.private.json evaluated.{h_pass,c_pass,n}: the numbers the decision used, on the round's
    kept tasks). Only these three aggregates; no task ids or per-task cells. Open round: hidden, labelled. Fail-closed: labelled."""
    if ev.get('h_pass') is not None or ev.get('c_pass') is not None:
        return dict(pass_basis='gate-sample')
    if not dec.get('decision'):
        return dict(pass_basis=None, pass_note='gate decision pending')
    ho = dec.get('heldout') or {}
    rid = ho.get('round_id')
    if not rid:
        return dict(pass_basis=None, pass_note='no pass rates in the gate record')
    rr = rj(HELDOUT_PUB / (str(rid) + '.json')) or next((r for r in (rj(HELDOUT_PUB / 'index.json') or {}).get('rounds') or []
                                                       if r.get('round_id') == rid), {})
    if rr.get('status') != 'closed':
        return dict(pass_basis='heldout', heldout_round=rid, pass_note='held-out round open: pass rates are published when the round closes')
    pv = (rj(GATE / str(e) / 'EVALUATED.private.json') or {}).get('evaluated') or {}
    if pv.get('h_pass') is None and pv.get('c_pass') is None:
        why = (dec.get('reasons') or ['no evaluation'])[0] if (ho.get('fail_closed') or dec.get('no_decision') or (dec.get('heldout') or {}).get('no_decision')) else 'no evaluation recorded'
        return dict(pass_basis='heldout', heldout_round=rid, pass_note=str(why)[:200])
    f = lambda x: float(x) if isinstance(x, (int, float)) else None
    return dict(pass_basis='heldout', heldout_round=rid, h_pass=f(pv.get('h_pass')), c_pass=f(pv.get('c_pass')),
                n=pv.get('n') if type(pv.get('n')) is int else None)


def verdicts(chain):
    """-> list of per-turn records (epoch order) with verdict."""
    out = []
    f = rj(FLAG) or {}
    start = int(f.get('from_epoch', 10 ** 9))
    for d in sorted((p for p in LIFE.glob('epochs/*') if p.name.isdigit() and int(p.name) >= start), key=lambda p: int(p.name)):
        t = rj(d / 'K1-TURN.json')
        if not t: continue
        e = int(d.name)
        rec = dict(epoch_index=e, status=t.get('status'), uid=t.get('uid'), hotkey=t.get('hotkey'), reg_block=t.get('reg_block'),
                   queue_rule=t.get('queue_rule', 'fewest-turns'), tasks=t.get('tasks'),
                   rollouts=t.get('rollouts'), queue_position=t.get('queue_position'), reason=t.get('reason'),
                   skipped_not_ready=t.get('skipped_not_ready_count'), head=t.get('head'), turn_at=t.get('at'))
        if t.get('status') == 'turn':
            dec = rj(GATE / str(e) / 'DECISION.json') or {}
            pg = rj(d / 'PROMOTION-GATE.json') or {}
            # PIT terminal blocked (no candidate / incomplete master): never last-10
            # GREEN credit; chart shows blocked; settlement stays pending_operator.
            blocked_dec = dec.get('decision') in (
                'blocked_no_candidate', 'blocked_incomplete_master',
            ) or dec.get('settlement') == 'pending_operator'
            if blocked_dec:
                ev = dec.get('evaluated') or {}
                rec.update(
                    verdict='blocked',
                    candidate=dec.get('candidate'),
                    z=ev.get('z'),
                    n=ev.get('n'),
                    cap=ev.get('c_cap_rate'),
                    h_pass=ev.get('h_pass'),
                    c_pass=ev.get('c_pass'),
                    decided_at_cr=dec.get('decided_at_cr'),
                    chart_status=dec.get('chart_status') or dec.get('decision') or 'blocked',
                    settlement=dec.get('settlement') or 'pending_operator',
                )
                rec.update(pass_rates(e, dec, ev))
                kd = rj(d / 'TRAIN-DISPATCH-K1.json') or {}
                rec['task_indices'] = sorted({int(x['index']) for x in ((kd.get('payload') or kd).get('selected') or []) if 'index' in x}) or None
                if (WEB / 'k1' / 'turns' / str(e) / 'turn.json').exists():
                    rec['artifacts'] = '/k1/turns/%d/turn.json' % e
                if (WEB / 'gate' / 'reveal' / ('%d.json' % e)).exists():
                    rec['gate_reveal'] = '/gate/reveal/%d.json' % e
                out.append(rec)
                continue
            v = dec.get('decision') if dec.get('decision') in ('green', 'red') else None
            if v is None and pg.get('decision') in ('green', 'red'): v = pg['decision']
            ev = dec.get('evaluated') or {}
            rg = rj(GATE / str(e) / 'REGATE.json') or {}   # 20261029 t773u: operator clean re-gate supersedes the verdict (pay only)
            if rg.get('decision') in ('green', 'red') and rg.get('candidate') == dec.get('candidate'):
                rec['signed_verdict'] = v; rec['regate'] = rg.get('note', 'operator re-gate'); v = rg['decision']; ev = rg.get('evaluated') or ev
            # 20261105-e757-rejudge: REJUDGE.json (full corrected numbers) supersedes for pay+chart; keep original z in audit
            rjg = rj(GATE / str(e) / 'REJUDGE.json') or {}
            if rjg.get('decision') in ('green', 'red') and rjg.get('candidate') == dec.get('candidate'):
                rec['signed_verdict'] = rec.get('signed_verdict') or v
                rec['audit'] = dict(rjg.get('audit') or {}, original_z=(rjg.get('audit') or {}).get('original_z', (dec.get('evaluated') or {}).get('z')))
                v = rjg['decision']; ev = rjg.get('evaluated') or ev
                rec['rejudge'] = rjg.get('note') or 'operator re-judge'
            held = rj(GATE / str(e) / 'HELD.json') or {}
            chart_status = None
            if dec.get('no_decision') or (dec.get('heldout') or {}).get('no_decision') or dec.get('chart_status') == 'no_decision':
                chart_status = 'no_decision'
                if v is None:
                    v = 'red'  # lifecycle-compatible; chart shows no_decision
            if held.get('active') and v not in ('green', 'red'):
                chart_status = 'held'
            if (dec.get('audit') or {}).get('probe_invalid') or (rjg.get('audit') or {}).get('probe_invalid') or (rg.get('why') and 'probe' in str(rg.get('why')).lower() and 'error' in str(rg.get('why')).lower()):
                # judged on an errored probe (pre-correction): chart must not show as GREEN
                if not rjg.get('decision'):
                    chart_status = chart_status or 'invalid'
            # e757 contaminated green until REJUDGE lands: head_pass==0 with huge z
            if v == 'green' and ev.get('h_pass') is not None and float(ev.get('h_pass')) == 0.0 and float(ev.get('z') or 0) > 10 and not rjg.get('decision'):
                chart_status = 'invalid'
                rec['audit'] = dict(rec.get('audit') or {}, original_z=ev.get('z'), probe_invalid=True,
                                    why='head probe error-rate catastrophic; awaiting/apply re-judge')
            rec.update(verdict=v or ('timeout-promote' if pg.get('decision') == 'timeout-promote' else 'pending'),
                       candidate=dec.get('candidate'), z=ev.get('z'), n=ev.get('n'), cap=ev.get('c_cap_rate'),
                       h_pass=ev.get('h_pass'), c_pass=ev.get('c_pass'), decided_at_cr=dec.get('decided_at_cr'),
                       chart_status=chart_status)
            rec.update(pass_rates(e, dec, ev))   # 20261031 pt904u: pass rates for the held-out gate format too
            # t764u transparency: the trained task indices (filtered signed dispatch) + post-decision public artifacts
            kd = rj(d / 'TRAIN-DISPATCH-K1.json') or {}
            rec['task_indices'] = sorted({int(x['index']) for x in ((kd.get('payload') or kd).get('selected') or []) if 'index' in x}) or None
            if (WEB / 'k1' / 'turns' / str(e) / 'turn.json').exists():
                rec['artifacts'] = '/k1/turns/%d/turn.json' % e
            if (WEB / 'gate' / 'reveal' / ('%d.json' % e)).exists():
                rec['gate_reveal'] = '/gate/reveal/%d.json' % e
        out.append(rec)
    return out


def payees(recs, chain):
    """last up to PAYEES distinct GREEN UIDs (most recent first), earning hotkey still registered on that uid."""
    seen, out = set(), []
    for r in sorted((r for r in recs if r.get('verdict') == 'green'), key=lambda r: -r['epoch_index']):
        u = int(r['uid'])
        if u in seen: continue
        seen.add(u)
        out.append(r)
        if len(out) >= PAYEES: break
    valid = [r for r in out if (chain['uids'].get(str(r['uid'])) or {}).get('hotkey') == r['hotkey']]
    return out, valid


def scores_doc(recs, chain, q):
    allp, valid = payees(recs, chain)
    now = time.time()
    miners, floor_info = [], dict(floor_applied=False, reason='k1_payees', n=len(valid))
    if valid:
        s = 1.0 / len(valid)
        for r in valid:
            c = chain['uids'][str(r['uid'])]
            miners.append(dict(miner_hex=c['miner_hex'], uid=int(r['uid']), hotkey=r['hotkey'], share_now=s, flip_score=s,
                               k1_green_epoch=r['epoch_index'], k1_z=r.get('z'), banned=False, floor_share=False))
    else:
        fl = rj(FLOOR) or {}
        for m in fl.get('miners') or []:
            try: sh = float(m.get('share_now') or 0)
            except Exception: sh = 0
            if sh > 0 and isinstance(m.get('miner_hex'), str):
                miners.append(dict(miner_hex=m['miner_hex'], uid=m.get('uid'), hotkey=m.get('hotkey'), share_now=sh, flip_score=sh,
                                   banned=False, floor_share=True))
        floor_info = dict(floor_applied=True, reason='no_k1_green_yet_last_good_pre_k1_scores', n=len(miners),
                          source=str(FLOOR), source_score_sha256=fl.get('score_sha256'))
    body = dict(version='sn120-k1-per-uid-live-v1', mode='live', live_weights=True, enabled=True,
                generated_at=now, generated_at_cr=cr(now), netuid=NETUID,
                formula=dict(version='k1-last10-green-equal-v1',
                             text=('K=1 per-UID turns (Jacob t712u 2026-10-27). Each epoch trains one UID\'s batch. A turn whose candidate '
                                   'passes the promotion gate (GREEN: held-out gate (live from E271): GREEN only if the weighted per-task gain over the head is positive and weighted_z > 1.00 (strict), estimated cap-hit < 0.20 and >= 100 kept tasks; head and candidate scored on the same tasks of a private held-out probe set (4 seeded samples per task, shared seeds); from E296 the turn batch is not excluded; baseline revealed only after a GREEN or when the round closes (https://affine.io/gate.json rule)) earns. Weight = 1/N to each of '
                                   'the last up to 10 distinct GREEN UIDs (an 11th pushes out the oldest). RED, skipped and pending turns earn '
                                   'nothing; unearned mass burns to UID 0. No bans. Nonzero floor: until the first GREEN K=1 turn, the last good '
                                   'pre-K=1 scores are kept.'),
                             payees_max=PAYEES, nonzero_weight_floor=floor_info, bans='none (all SN120 bans removed, Jacob t712u)'),
                miners=sorted(miners, key=lambda m: -m['share_now']),
                k1=dict(payees=[dict(uid=r['uid'], hotkey=r['hotkey'], green_epoch=r['epoch_index'], z=r.get('z'),
                                     still_registered=r in valid) for r in allp],
                        chain_block=chain.get('block')))
    body['score_sha256'] = hashlib.sha256(json.dumps(dict(m=[(m['miner_hex'], m['share_now']) for m in body['miners']]),
                                                     sort_keys=True).encode()).hexdigest()
    return body


def tick(state):
    now = time.time()
    chain = rj(ST / 'K1-CHAIN.json')
    if chain is None or now - float(chain.get('taken_at', 0)) > CHAIN_EVERY:
        try:
            chain = chain_snapshot(); wj(ST / 'K1-CHAIN.json', chain, 0o600)
            log(event='chain', block=chain['block'], n=chain['n'])
        except Exception as e:
            log(event='chain_error', error=repr(e)[:300])
            if chain is None: return
    ts = turns()
    f = rj(FLAG) or {}
    rule = f.get('queue_rule') if f.get('queue_rule') in RULES else 'hotkey-once'
    q = queue(chain, ts, rule, int(f.get('min_tasks', 8)))
    caps = dict(max_rollouts=f.get('max_rollouts', 1000), min_tasks=f.get('min_tasks', 8),
                rollouts_per_task=[f.get('min_rollouts_per_task', 2), f.get('max_rollouts_per_task', 64)])
    recs = verdicts(chain)
    if rule == 'fewest-turns':
        qpub = dict(version='sn120-k1-queue-v1', generated_at=now, generated_at_cr=cr(now), chain_block=chain['block'],
                    enabled=f.get('enabled') is True, from_epoch=f.get('from_epoch'), next=q[:QUEUE_N], queue_length=len(q),
                    queue_rule=rule,
                    order_rule='(turns taken by this uid+hotkey, registration block, uid) ascending; UID 0 excluded; a turn counts when '
                               'its batch is dispatched (pass or fail); a UID without a ready batch at the seal is skipped and not charged; '
                               're-registering with a new hotkey resets the count',
                    caps=caps)
    else:
        se, _ = last_seal_tasks()
        line = [dict(r, line=i) for i, r in enumerate(x for x in q if x['ready_last_seal'])]
        fs, lg = served(ts)
        shk = sorted({t.get('hotkey') for t in ts if t.get('hotkey')})
        hot = rule in ('hotkey-once', 'hotkey-registration-once')
        qpub = dict(version='sn120-k1-queue-v2', generated_at=now, generated_at_cr=cr(now), chain_block=chain['block'],
                    enabled=f.get('enabled') is True, from_epoch=f.get('from_epoch'), queue_rule=rule,
                    queue_rule_from_epoch=f.get('queue_rule_from_epoch'),
                    next=line[:QUEUE_N], queue_length=len(line), readiness_epoch=se,
                    registrations_unserved=len([r for r in q if not r.get('grant')]), registrations_served=len(fs) + len(lg),
                    grants=[r for r in q if r.get('grant')],
                    grants_rule='Jacob 2026-10-30 19:30 CR: each hotkey whose turn since the held-out gate went live (e275) was RED with 0 < weighted_z <= 1.00 '
                                'gets ONE fresh free-file turn, ahead of the queue (oldest RED epoch first). It is used only by a real turn: a missing or '
                                'invalid file is a skip and the grant stays open. The hotkey must still be registered. next lists granted hotkeys only '
                                'while their file is valid.',
                    hotkeys_unserved=len({r['hotkey'] for r in q if not r.get('grant')}), hotkeys_served=len(shk), served_hotkeys=shk,
                    served=[dict(uid=t.get('uid'), hotkey=t.get('hotkey'), reg_block=t.get('reg_block'), epoch_index=t.get('epoch_index')) for t in ts],
                    **({'served_registrations': sorted([u, rb] for (u, _h, rb) in fs)} if rule == 'hotkey-registration-once' else {}),
                    order_rule=((('hotkey-once (Jacob t761u 2026-10-28, teutonic.ai "each hotkey gets exactly ONE submission, permanently"): each '
                                 'HOTKEY gets exactly one K=1 turn, permanently (pass or fail), whatever UID or registration block it is on later. '
                                 'Order = registration block, then uid (UID 0 excluded); hotkeys that already had a turn (served_hotkeys) are out of the queue. '
                                 'At the seal the first unserved hotkey with a ready batch (>= min_tasks distinct admitted tasks) takes the turn; others are '
                                 'skipped and keep their place. To try again, register a NEW hotkey. Do not deregister a paying hotkey: payees are the last 10 '
                                 'distinct GREEN UIDs while the earning hotkey stays on that UID. ' + ('v8.2.9s (Jacob 2026-10-29): a REGISTRATION (uid, registration block) that has had a turn also never gets another, whatever hotkey is on it now: swap_hotkey keeps the uid and registration block, so a swapped-in hotkey on a served registration stays out of the queue. A genuine re-registration (new registration block) with a never-served hotkey is eligible. ' if rule == 'hotkey-registration-once' else '')) if hot else
                                ('registration-once (Jacob t758u 2026-10-28): a registration = (uid, hotkey, registration block) on the metagraph. Each '
                                 'registration gets ONE turn, ever (pass or fail). Order = registration block, then uid (UID 0 excluded); registrations that '
                                 'already had a turn are out of the queue. At the seal the first unserved registration with a ready batch (>= min_tasks distinct '
                                 'admitted tasks) takes the turn; others are skipped and keep their place. To get another turn, register again. ')) +
                               'next/queue_length = unserved entries that had a ready batch at the last seal (readiness_epoch), in turn order: a forecast; the seal decides.')
                    if not ff_state()[0] else
                    (__import__('k1_squat_trim', fromlist=['ORDER_RULE_PLAIN']).ORDER_RULE_PLAIN.replace(
                     'next = the next unserved members with a valid file now',
                     'next = the next %d unserved members with a valid file now' % QUEUE_N)),
                    caps=caps)
    meta = getattr(queue, '_last_squat_meta', None)
    if isinstance(meta, dict) and rule == 'hotkey-registration-once':
        expired = [d for d in meta.get('details') or [] if d.get('expired_back_of_queue')]
        qpub.update(
            turn_rate_per_day=meta.get('turn_rate_per_day'),
            expired_back_of_queue_count=meta.get('expired_back_of_queue_count'),
            expired_back_of_queue=[dict(uid=d['uid'], hotkey=d['hotkey'], registration_block=d['registration_block'],
                                       expected_wait_seconds=d['expected_wait_seconds'], expected_wait_days=d['expected_wait_days'],
                                       deadline_unix=d['deadline_unix'], deadline_cr=d['deadline_cr'],
                                       registration_time_unix=d['registration_time_unix'], expired_back_of_queue=True)
                                  for d in expired],
            squat_trim=dict(enabled=True,
                            rule='age > 2 x max(1 day, N_ahead/turn_rate) and no valid file -> back of queue; valid file never demoted; recomputed each turn',
                            checked_at=meta.get('checked_at'), checked_at_cr=meta.get('checked_at_cr'),
                            active_count=meta.get('active_count'), unserved=meta.get('unserved')),
        )
        if meta.get('order_rule'):
            qpub['order_rule'] = meta['order_rule'].replace(
                'next = the next unserved members with a valid file now',
                'next = the next %d unserved members with a valid file now' % QUEUE_N)
    wj(ST / 'K1-QUEUE.public.json', qpub, 0o644)
    if f.get('enabled') is True:
        doc = scores_doc(recs, chain, q)
        wj(SCORES, doc, 0o644)
        allp, valid = payees(recs, chain)
        wj(K1PUB, dict(version='sn120-k1-public-v1', generated_at=now, generated_at_cr=cr(now), queue=qpub,
                       payees=doc['k1']['payees'], turns=recs[-200:], floor=doc['formula']['nonzero_weight_floor'],
                       score_sha256=doc['score_sha256']), 0o644)
        key = (doc['score_sha256'], len(recs))
        if state.get('last') != key:
            state['last'] = key
            log(event='scores', payees=[m['uid'] for m in doc['miners']][:12], n=len(doc['miners']),
                floor=doc['formula']['nonzero_weight_floor'].get('floor_applied'), turns=len(recs))


def main():
    once = '--once' in sys.argv
    state = {}
    while True:
        try: tick(state)
        except Exception: log(event='tick_error', error=traceback.format_exc()[-800:])
        if once: return
        time.sleep(TICK)


if __name__ == '__main__':
    main()
