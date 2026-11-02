#!/usr/bin/env python3
"""affine.io /k1/tables.json - the extra public fields the Kings / Queue / History tables need (Jacob pt1100u, 2026-11-02).

Everything here is either already public on affine.io or public on chain:
  - per decided turn: gain, SE, decided_at (unix), from the published k1/turns/<e>/turn.json decision block;
    the batch file link and the effective train params ONLY for GREEN turns (turn.json publishes them only then);
  - coldkey per hotkey (chain SubtensorModule.Owner, public), for hotkeys shown in the tables;
  - per queued hotkey: file check time, groups, rollouts and format from ff/status.json (already public);
  - epoch timing: median epoch length (last 20 finalized epochs, gate.json timestamps) and when the current epoch opened;
  - train param defaults (mining.json batch_file.train.fields) so the page can show only non-default knobs.
Never reads miner files, RED/pending train params, priors, selection probabilities, IRT params or holdout data.
"""
import json, os, statistics, tempfile, time
from pathlib import Path

WEB = Path('/home/const/subnet120/affine/website')
OUT = WEB / 'k1' / 'tables.json'
STATE = Path('/home/const/subnet120/affine/site-tools/state')
TURN_CACHE = STATE / 'k1-tables-turns.json'
OWNER_CACHE = STATE / 'k1-tables-coldkeys.json'
OWNER_TTL = 6 * 3600        # re-check a hotkey's owner every 6 h (owner swaps are rare)
OWNER_BUDGET = 400          # max chain lookups per run
NETUID = 120


def rj(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def wj(p, obj, mode=0o644):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix='.' + p.name + '.')
    with os.fdopen(fd, 'w') as f:
        json.dump(obj, f, separators=(',', ':'), sort_keys=True)
    os.chmod(tmp, mode); os.replace(tmp, p)


def num(x):
    try:
        return None if x is None else float(x)
    except Exception:
        return None


def turn_fields(e, verdict):
    d = rj(WEB / 'k1' / 'turns' / str(e) / 'turn.json')
    if not isinstance(d, dict):
        return None
    dec = d.get('decision') or {}
    ev = dec.get('evaluated') or {}
    out = dict(gain=num(ev.get('gain', ev.get('delta'))), se=num(ev.get('se')), decided_at=num(dec.get('decided_at')),
               decision=dec.get('decision'))
    if verdict == 'green' and dec.get('decision') == 'green':
        b = d.get('batch') or {}
        if b.get('published') and b.get('url') and (WEB / b['url'].lstrip('/')).exists():
            out['file'] = b['url']; out['file_bytes'] = b.get('bytes')
        t = d.get('turn') or {}
        if isinstance(t.get('train'), dict):
            out['train'] = {k: v for k, v in t['train'].items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
            out['given'] = [k for k in (t.get('train_given') or []) if isinstance(k, str)]
    return out


def owners(hotkeys):
    cache = rj(OWNER_CACHE, {}) or {}
    now = time.time()
    todo = [h for h in hotkeys if h and (h not in cache or now - cache[h].get('at', 0) > OWNER_TTL)][:OWNER_BUDGET]
    if todo:
        try:
            import bittensor as bt
            from bittensor._generated.storage import SubtensorModule as S
            with bt.Subtensor('finney') as c:
                for h in todo:
                    try:
                        ck = str(c.query(S.Owner, [h]))
                        if ck and ck.startswith('5') and len(ck) >= 46:
                            cache[h] = dict(ck=ck, at=now)
                    except Exception:
                        pass
        except Exception as ex:
            print('owner lookup failed:', ex)
        wj(OWNER_CACHE, cache, 0o600)
    return {h: cache[h]['ck'] for h in hotkeys if h in cache}


def main():
    k1 = rj(WEB / 'k1.json', {}) or {}
    gate = rj(WEB / 'gate.json', {}) or {}
    mining = rj(WEB / 'mining.json', {}) or {}
    ff = rj(WEB / 'ff' / 'status.json', {}) or {}
    turns = [t for t in (k1.get('turns') or []) if t.get('status') == 'turn']

    # per decided turn (immutable once decided -> cached)
    cache = rj(TURN_CACHE, {}) or {}
    per = {}
    for t in turns:
        e = t.get('epoch_index'); v = t.get('verdict')
        if e is None or v not in ('green', 'red'):
            continue
        key = str(e); c = cache.get(key)
        if not c or c.get('decision') != v:
            c = turn_fields(e, v)
            if c is None:
                continue
            cache[key] = c
        # privacy guard: train params / file only on GREEN
        if v != 'green':
            c = {k: c[k] for k in ('gain', 'se', 'decided_at', 'decision') if k in c}
        per[key] = c
    wj(TURN_CACHE, cache, 0o600)

    # epoch timing from the gate ledger
    ents = sorted((x for x in (gate.get('entries') or []) if isinstance(x.get('epoch'), int)), key=lambda x: x['epoch'])
    fin = [(x['epoch'], num((x.get('timestamps') or {}).get('finalized'))) for x in ents]
    fin = [(e, f) for e, f in fin if f]
    durs = [b[1] - a[1] for a, b in zip(fin, fin[1:]) if b[0] == a[0] + 1 and 0 < b[1] - a[1] < 6 * 3600][-20:]
    cur_epoch = (mining.get('training') or {}).get('epoch_index')
    opened = None
    if fin and cur_epoch is not None and fin[-1][0] == int(cur_epoch) - 1:
        opened = fin[-1][1]

    # queue files (already public in ff/status.json)
    q = k1.get('queue') or {}
    fm = ff.get('miners') or {}
    qf = {}
    for r in q.get('next') or []:
        m = fm.get(r.get('hotkey')) or {}
        if m:
            qf[r['hotkey']] = dict(checked_at=num(m.get('checked_at')), groups=m.get('groups'), rollouts=m.get('rollouts'),
                                   format=m.get('format'), valid=m.get('valid'))

    hks = []
    for r in q.get('next') or []: hks.append(r.get('hotkey'))
    for p in k1.get('payees') or []: hks.append(p.get('hotkey'))
    for t in reversed(turns): hks.append(t.get('hotkey'))
    seen = set(); hks = [h for h in hks if h and not (h in seen or seen.add(h))]

    fields = ((mining.get('batch_file') or {}).get('train') or {}).get('fields') or {}
    doc = dict(
        version='sn120-k1-tables-v1',
        generated_at=round(time.time(), 1),
        note='Extra public fields for the affine.io Kings / Queue / History tables. Train params and batch file links are '
             'published for GREEN turns only; RED and pending stay private. Coldkeys are the on-chain owners of the hotkeys.',
        epoch=dict(current=cur_epoch, opened_at=opened, median_s=(statistics.median(durs) if durs else None), sample=len(durs)),
        train_defaults={k: v.get('default') for k, v in fields.items() if isinstance(v, dict) and 'default' in v},
        turns=per,
        queue_files=qf,
        coldkeys=owners(hks),
        chain_block=q.get('chain_block'),
    )
    wj(OUT, doc)
    print('wrote', OUT, len(per), 'turns', len(doc['coldkeys']), 'coldkeys', 'median', doc['epoch']['median_s'])


if __name__ == '__main__':
    main()
