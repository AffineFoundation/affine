"""Signed host composition v9-live = the exact signed v7b writer (locks, single-writer gate, submission journal, rate limits,
registration snapshot, mech-0 SetWeights) with the weight RULE replaced by v9 (weights_v9.py): 100% v9 gated-miner shares,
remainder (and every fallback) burned to UID 0.  The continuous-audit evidence step is not called.
  --check : authenticate + build, no chain calls.   --plan : full run with execute=False against a COPY of chain state
  (no broadcast; nothing shared is written).   --execute : the live hourly run (systemd drop-in)."""
import argparse, hashlib, importlib.util, json, shutil, signal, sys, tempfile, types
from pathlib import Path
VERSION = 'hourly-host-composition-v9-live-weights'
HERE = Path(__file__).resolve().parent

def prepare(document, authority, plan_dir=None):
    import base64
    from nacl.signing import VerifyKey
    canonical = lambda v: json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    def need(c, m):
        if not c: raise ValueError(m)
    need(type(document) is dict and set(document) == {'payload', 'signer', 'signature'} and document['signer'] == authority, 'explicit operator authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']), base64.b64decode(document['signature'], validate=True))
    p = document['payload']
    need(set(p) == {'version', 'execute_enabled', 'entry', 'base_entry', 'base_composition', 'weights_module', 'config_path',
                    'maximum_seconds', 'evidence_timeout_seconds'} and p['version'] == VERSION and type(p['execute_enabled']) is bool, 'exact v9 composition')
    def load(name, ref):
        need(type(ref) is dict and set(ref) == {'path', 'sha256'} and hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() == ref['sha256'], 'pinned ' + name)
        spec = importlib.util.spec_from_file_location(name, ref['path']); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    need(Path(p['entry']['path']) == Path(__file__).resolve() and hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == p['entry']['sha256'], 'exact host entry')
    base = load('_v7b_host_composition', p['base_entry'])
    base_p, parent, context = base.prepare(json.loads(base.pinned(p['base_composition'])), authority)   # exact v7b writer + budgets
    need(base_p['maximum_seconds'] == p['maximum_seconds'] and base_p['evidence_timeout_seconds'] == p['evidence_timeout_seconds'], 'same budgets as v7b')
    wv9 = load('_sn120_v9_live_weights', p['weights_module']); wv9.load_config(p['config_path'], authority)
    writer = context['writer']; run = writer.run_once; g = run.__globals__
    rep = wv9.make(p['config_path'], authority, g['OWNER'], g['ChainAdapter'])
    loader = rep.pop('evidence_loader'); Adapter = RealAdapter = rep['ChainAdapter']
    if plan_dir is not None:
        rep['ASSESSMENT_DIRECTORY'] = 'current-assessment-v9-live-PLAN'
        def factory(state_dir, **kw):
            d = Path(plan_dir) / 'chain-state'; d.mkdir(parents=True, exist_ok=True)
            for n in ('weights.json',):
                if (Path(state_dir) / n).exists() and not (d / n).exists():
                    st = json.loads((Path(state_dir) / n).read_text()); st.pop('last_submitted_window', None)   # plan the current hour even if already sent
                    (d / n).write_text(json.dumps(st))
            return RealAdapter(str(d), **kw)
        Adapter = factory
    new_run = types.FunctionType(run.__code__, dict(g, **rep), run.__name__, run.__defaults__, run.__closure__)
    new_run.__kwdefaults__ = dict(run.__kwdefaults__ or {}, adapter_factory=Adapter)
    context = dict(context, writer=types.SimpleNamespace(**dict(vars(writer), run_once=new_run)), evidence_loader=loader)
    return p, parent, context, wv9

PUBLIC_SCORES_SHA256 = 'f3a46bc9dcfcfd41dc30007d68fda215b96d9a561d6bd11fba354793ef748e65'   # seq7: public_scores_v9.py is pinned through this (signed) entry file
def _publish_public_scores(wv9, p, authority):
    """Best effort, after the weight run: refresh https://affine.io/scores.json. Never raises, never touches weights."""
    try:
        f = HERE / 'public_scores_v9.py'; b = f.read_bytes()
        if hashlib.sha256(b).hexdigest() != PUBLIC_SCORES_SHA256: raise ValueError('public_scores_v9.py not the pinned file')
        spec = importlib.util.spec_from_file_location('_sn120_v9_public_scores', str(f)); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        m.publish(wv9, wv9.load_config(p['config_path'], authority))
    except Exception as e:
        print(json.dumps({'public_scores_error': '%s: %s' % (type(e).__name__, str(e)[:200])}))

def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--policy', required=True); ap.add_argument('--authority', required=True)
    m = ap.add_mutually_exclusive_group(required=True); m.add_argument('--check', action='store_true'); m.add_argument('--plan', action='store_true'); m.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    def expired(*_): raise TimeoutError('bounded existing hourly writer')
    signal.signal(signal.SIGALRM, expired); signal.alarm(1600)
    try:
        doc = json.loads(Path(a.policy).absolute().read_bytes()); tmp = tempfile.mkdtemp(prefix='v9plan-') if a.plan else None
        p, parent, context, wv9 = prepare(doc, a.authority, tmp)
        c = wv9.load_config(p['config_path'], a.authority); sh, burn, info = wv9.v9_vector(c)
        if a.check:
            print(json.dumps({'checked': True, 'chain_calls': 0, 'version': VERSION, 'config_sequence': c['sequence'], 'miners_paid': len(sh),
                              'burn_share': burn, 'v9': info})); return
        if a.plan:   # no global lock, no broadcast, chain-state COPY; shows the exact vector the live run would submit now
            import time
            w = context['writer']; g = w.run_once.__globals__; kw = w.run_once.__kwdefaults__
            pp = g['validate_policy'](context['parent_policy'], a.authority, context['cutover'], context['anchor'])
            cutoff = int(time.time()) // 3600 * 3600
            ev = context['evidence_loader'](None, cutoff=cutoff); asmt = g['calculate'](ev['snapshots'], {}, cutoff); g['validate_assessment'](asmt, pp, cutoff)
            ad = kw['adapter_factory'](g['authenticate_cutover'](context['cutover'], context['anchor'], a.authority)[0]['chain_state'], netuid=120, expected_owner=g['OWNER'])
            try:
                regs = ad.registrations(); pts, sel, exc = g['recipients'](asmt, regs)
                r = ad.submit_hour(pts, sel, cutoff, execute=False, zero_total_policy=pp['zero_total_policy'], registration_change_policy=pp['registration_change_policy'])
                pl = None
                if r.get('uids'):   # also build (never sign/send) the exact journaled timelocked intent the live run uses
                    import sys as _s
                    wst = _s.modules['_sn120_v9_weight_submission_transaction']; wsr = _s.modules['_sn120_v9_weight_submission_reconciliation']
                    from pathlib import Path as _P
                    wallet = ad.bt.Wallet(name='default', hotkey='default', path=str(_P.home() / '.bittensor/wallets'))
                    blk = int(ad.chain.block)
                    it = wst.make_recorded_intent(ad.bt, netuid=120, uids=r['uids'], weights=r['weights'], version_key=int(ad.query('WeightsVersionKey', [120], blk) or 0))
                    try: pl = ad.chain.plan(it, wallet)
                    except Exception as e:
                        pl = None; r['journal_plan_ok'] = False; r['journal_plan_detail'] = '%s: %s' % (type(e).__name__, str(e)[:200])
                if r.get('uids') and pl is not None:
                    r['journal_plan_ok'] = bool(pl.ok); r['journal_vector'] = getattr(it, 'journal_vector', None); r['journal_owner_uid'] = getattr(it, 'journal_owner_uid', None)
                    if pl.ok:
                        wsr.make_intent(owner=ad.owner, owner_uid=it.journal_owner_uid, netuid=120, window_end=cutoff, expected_weights=it.journal_vector,
                                        assessment_sha256='0' * 64, policy_sha256='0' * 64, registrations_sha256='0' * 64, attempt_start_block=blk)
                        r['journal_intent_valid'] = True
                    else: r['journal_plan_detail'] = str(pl)[:300]
            finally:
                ch = getattr(ad, 'chain', None)
                if ch is not None and hasattr(ch, 'close'): ch.close()
            out = dict(plan=True, cutoff=cutoff, status=r.get('status'), uids=r.get('uids'), weights=r.get('weights'), remaining_blocks=r.get('remaining_blocks'),
                       excluded_unregistered=len(exc), v9=asmt['v9'], plan_dir=tmp,
                       **{k: r.get(k) for k in ('journal_plan_ok', 'journal_vector', 'journal_owner_uid', 'journal_intent_valid', 'journal_plan_detail', 'detail')})
            Path(tmp, 'plan.json').write_text(json.dumps(out, default=str)); print(json.dumps({k: v for k, v in out.items() if k not in ('uids', 'weights', 'journal_vector')}, default=str))
            print(json.dumps(dict(n_uids=len(r.get('uids') or []), uid0_weight=dict(zip(r.get('uids') or [], r.get('weights') or [])).get(0)))); return
        if not p['execute_enabled']: raise ValueError('explicit signed host enablement')
        try:
            result = parent.run(context, execute=True)
            print(json.dumps({k: result[k] for k in ('status', 'window_end', 'remaining_blocks') if k in result}))
        finally:
            _publish_public_scores(wv9, p, a.authority)
    finally: signal.alarm(0)

if __name__ == '__main__': main()
