"""SN120 v9 PUBLIC per-epoch scores artifact (read-only projection; never affects weights).
Called by hourly_entry_v9.py after every writer run (errors are swallowed there) and usable by hand:
    python3 public_scores_v9.py --config LIVE-CONFIG-v9.ACTIVE.signed.json --authority <hex> [--out DIR]
Writes atomically (mode 0644) into the affine.io static site:
    <site>/scores.json          latest  (https://affine.io/scores.json)
    <site>/scores/e<E>.json     one per newest score epoch E (https://affine.io/scores/e<E>.json), overwritten while E is newest
    <site>/scores/index.json    list of published epochs
Contents: weight-formula version + exact parameters, per-UID inputs per epoch (verified groups, 3-group units/batches,
hits, per-unit credit, window credit, v9 gate diagnostics, zero-groups floor, final share and on-chain u16 weight when the
submitted vector is known), audit/placebo status, per-task claim counts (task index -> groups / UIDs; no rollout text),
and epoch/scoring status with ETA.  Aggregates only: UID, hotkey ss58 (public), counts and scores."""
import argparse, collections, gzip, hashlib, json, math, os, statistics, time
from pathlib import Path
VERSION = 'sn120-v9-public-scores-v1'
N = Path('/home/const/subnet120-rewrite/state/root-audits/20261020-measured-impact-v9')
SITE = Path('/home/const/subnet120/affine/website')
ASSESS = Path('/home/const/subnet120-rewrite/state/live-math-launch-preparation-v1/reward-state-v1/current-assessment-v9-live-1')
CACHE = N / 'public-cache'
B58 = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
def ss58(pub_hex, prefix=42):
    raw = bytes([prefix]) + bytes.fromhex(pub_hex); b = raw + hashlib.blake2b(b'SS58PRE' + raw, digest_size=64).digest()[:2]
    n = int.from_bytes(b, 'big'); s = ''
    while n: n, r = divmod(n, 58); s = B58[r] + s
    return '1' * (len(b) - len(b.lstrip(b'\0'))) + s
def _atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True); t = path.with_name('.' + path.name + '.tmp%d' % os.getpid())
    data = json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    fd = os.open(t, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    with os.fdopen(fd, 'wb') as f: f.write(data)
    os.chmod(t, 0o644); os.replace(t, path)
def cr(ts): return time.strftime('%Y-%m-%d %H:%M CR (UTC-6)', time.gmtime(ts - 6 * 3600)) if ts else None
def claims(E, budget_ok):
    """Per-task claim counts for epoch E from the sealed read-only document fetch (cached; task index + counts only)."""
    cf = CACHE / f'claims-e{E}.json'
    if cf.exists(): return json.loads(cf.read_text())
    src = N / 'work' / f'docs-e{E}.jsonl.gz'
    if not src.exists() or not budget_ok(): return None
    g, u, tr, per_uid = collections.Counter(), collections.defaultdict(set), collections.Counter(), collections.defaultdict(lambda: [0, 0])
    for line in gzip.open(src, 'rt'):
        r = json.loads(line); t = int(r['index']); g[t] += 1; u[t].add(r.get('miner_full') or r.get('miner')); tr[t] += bool(r.get('trained'))
        k = r.get('miner_full') or r.get('miner'); per_uid[k][0] += 1; per_uid[k][1] += bool(r.get('trained'))
    out = dict(epoch=E, groups_submitted=sum(g.values()), tasks_claimed=len(g),
               per_task={str(t): dict(groups=g[t], miners=len(u[t]), trained=tr[t]) for t in sorted(g)},
               per_miner={k: dict(groups_submitted=v[0], groups_trained=v[1]) for k, v in per_uid.items()})
    CACHE.mkdir(exist_ok=True, mode=0o700); t = cf.with_name(cf.name + '.tmp'); t.write_text(json.dumps(out)); os.replace(t, cf)
    return out
def build(w, c, deadline_s=20.0):
    t0 = time.time(); ok = lambda: time.time() - t0 < deadline_s
    sh, burn, info = w.v9_vector(c)
    E = info.get('score_epoch'); doc = dict(version=VERSION, generated_at=time.time(), generated_at_cr=cr(time.time()))
    if E is None:
        doc.update(status='no_valid_recent_score', reasons=info.get('reasons'), weights='100% burn to UID 0 until a valid score is installed'); return None, doc
    p = w._score(c, E)[1]; q = c.get('window_payout')
    rule = info.get('rule') or 'hard-gate'
    formula = dict(weights_module=w.VERSION, config_version=c['version'], config_sequence=c['sequence'], rule=rule, miner_share=c['miner_share'],
                   max_uid_share=c['max_uid_share'], max_score_age_epochs=c['max_score_age_epochs'])
    if q:
        formula.update(window_epochs=q['window_epochs'], min_window_groups=q['min_window_groups'], min_newest_groups=q['min_newest_groups'], text=(
            'credit(unit) = groups_in_unit * (1 - p_unit); units are fixed 3-group batches by slot (slots 0-2, 3-5, 6-8; a partial unit is tested '
            'against the same 3-group null); p_unit = P(max group score of an honest 3-group unit >= this unit max) from the epoch out-of-sample null. '
            'credit_i = sum over the miner\'s verified units in the last %d sealed epochs ending at the newest scored epoch (missing scores skipped). '
            'share_i = credit_i / sum_j credit_j over miners with >= %d verified group(s) in the window AND >= %d in the newest scored epoch '
            '(zero-groups floor: otherwise 0). Weight = share (mech 0, 100%% to miners); if the newest valid score is older than %d sealed epochs '
            'or nobody qualifies, 100%% burns to UID 0. Group score = leave-task-out gradient dot with a reference drawn from the public '
            'post-commit block-hash seed.') % (q['window_epochs'], q['min_window_groups'], q['min_newest_groups'], c['max_score_age_epochs']))
        _, rows, wd = w.window_credit(c, E, p)
    else:
        rows, wd = {}, None
    # last submitted vector (log written by the writer's recipients step) and its assessment
    sub = None; logs = sorted((Path(c['log_dir'])).glob('weights-*.json'))
    if logs:
        lg = json.loads(logs[-1].read_text()); sub = dict(cutoff=lg.get('cutoff'), cutoff_cr=cr(lg.get('cutoff')), miners_paid=lg.get('miners_paid'),
                                                           burn_share=lg.get('burn_share'), digest=lg.get('digest'), score_epoch=(lg.get('v9') or {}).get('score_epoch'),
                                                           rule=(lg.get('v9') or {}).get('rule'))
    pts = {}
    if sub and (ASSESS / f"assessment-{sub['cutoff']}.json").exists():
        pts = json.loads((ASSESS / f"assessment-{sub['cutoff']}.json").read_text())['payload']['points']
    gate = p['window_gate']; win = list(range(E - (q['window_epochs'] if q else 4) + 1, E + 1))
    cl = {e: claims(e, ok) for e in win}
    miners = []
    for m in sorted(set(rows) | set(gate) | set(sh)):
        a = rows.get(m, {}); g = gate.get(m, {})
        cm = {str(e): (cl[e] or {}).get('per_miner', {}).get(m) for e in win if cl.get(e)}
        miners.append(dict(uid=a.get('uid', g.get('uid')), hotkey=ss58(m) if len(m) == 64 else m,
                           window=dict(groups=a.get('groups', 0), units=a.get('units', 0), hits=a.get('hits', 0), credit=a.get('credit', 0.0)),
                           newest_epoch_groups=a.get('newest_groups', 0), zero_groups_floor_ok=a.get('floor_ok', False),
                           by_epoch=a.get('by_epoch', {}), submitted_and_trained_by_epoch={k: v for k, v in cm.items() if v},
                           v9_gate_diagnostic=dict(B=g.get('B'), hits=g.get('hits'), stouffer_z=g.get('stouffer_z'), binom_p=g.get('binom_p'), gated=g.get('gated')),
                           share_now=sh.get(m, 0.0), share_last_submitted=pts.get(m, 0.0)))
    miners.sort(key=lambda r: (-r['share_now'], r['uid'] if r['uid'] is not None else 1 << 30))
    st = json.loads((N / 'status.json').read_text()) if (N / 'status.json').exists() else {}
    lags = [st['installed'][k]['at'] - json.loads((N / 'work' / f'seed-e{k}.json').read_text())['deadline']
            for k in sorted(st.get('installed', {}), key=int)[-6:] if (N / 'work' / f'seed-e{k}.json').exists()]
    lag = statistics.median(lags) if lags else None
    scoring = {}
    for k in sorted(set(st.get('installed', {})) | set(st.get('started', {})) | set(st.get('errors', {})), key=int)[-12:]:
        s = 'scored' if k in st.get('installed', {}) else ('failed_retrying' if k in st.get('errors', {}) else 'scoring')
        sd = (N / 'work' / f'seed-e{k}.json'); dl = json.loads(sd.read_text())['deadline'] if sd.exists() else None
        scoring[k] = dict(state=s, deadline_cr=cr(dl), installed_at_cr=cr(st.get('installed', {}).get(k, {}).get('at')),
                          error=(st.get('errors', {}).get(k) or {}).get('error'), eta_cr=cr(dl + lag) if (s != 'scored' and dl and lag) else None)
    mj = {}
    try:
        m0 = json.loads((SITE / 'mining.json').read_text()); mj = dict(epoch_id=m0.get('epoch_id'), accepting_submissions=m0.get('accepting_submissions'),
                                                                      deadline=m0.get('deadline'), deadline_cr=cr(m0.get('deadline')), notice=m0.get('notice'))
    except Exception: pass
    fp = p['false_positive_checks']
    doc.update(
        score_epoch=E, score_sha256=info.get('score_sha256'), latest_sealed_epoch=info.get('latest_sealed'), formula=formula, window=wd,
        last_submitted_vector=sub,
        seed=dict((k, p['seed'].get(k)) for k in ('rule', 'epoch_id', 'deadline', 'anchor_block', 'margin_blocks', 'seed_block', 'seed_block_hash', 'seed')),
        reference=p.get('reference'), null=p.get('null'), base_checkpoint=p.get('base_checkpoint'),
        epoch_totals=dict(groups_verified=p.get('valid_groups'), docs_on_base=p.get('docs_on_base'), miners=p.get('miners')),
        audit=dict(v9_placebo_label_shuffle=fp.get('label_shuffle'), v9_uid_shuffle=fp.get('uid_shuffle'), unit_hit_rate=fp.get('unit_hit_rate'),
                   expected_unit_hit_rate=fp.get('expected_unit_hit_rate'), window_summary=p.get('window_summary'),
                   diagnostic_alarms=info.get('diagnostic_alarms'),
                   continuous_label_audit=('not on the v9 weight path: v9 scores every verified group with the gradient test above; the old continuous '
                                           're-audit counters (accepted/rejected) on the dashboard are no longer produced, so they read 0/0')),
        per_task_claims={str(e): (dict(groups_submitted=v['groups_submitted'], tasks_claimed=v['tasks_claimed'], per_task=v['per_task']) if v else None) for e, v in cl.items()},
        epoch_status=dict(current=mj, scoring=scoring, median_score_lag_hours=round(lag / 3600, 2) if lag else None,
                          note='weights update at the first hourly run after a new epoch score is installed'),
        miners=miners)
    return E, doc
def publish(w, c, site=SITE, deadline_s=20.0):
    E, doc = build(w, c, deadline_s)
    _atomic(site / 'scores.json', doc)
    if E is not None:
        _atomic(site / 'scores' / f'e{E}.json', doc)
        idx = sorted({int(f.stem[1:]) for f in (site / 'scores').glob('e*.json') if f.stem[1:].isdigit()})
        _atomic(site / 'scores' / 'index.json', dict(version=VERSION, epochs=idx, latest='https://affine.io/scores.json'))
    return E
if __name__ == '__main__':
    import importlib.util
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default=str(N / 'LIVE-CONFIG-v9.ACTIVE.signed.json')); ap.add_argument('--authority', required=True)
    ap.add_argument('--out', default=str(SITE)); a = ap.parse_args()
    spec = importlib.util.spec_from_file_location('_w9pub', str(N / 'weights_v9.py')); w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
    print(json.dumps(dict(published_epoch=publish(w, w.load_config(a.config, a.authority), Path(a.out), 120.0))))
