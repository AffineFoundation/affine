"""SN120 v9 LIVE weight rule (100% v9; replaces the audit-evidence current rule on the mech-0 weight path).

  share(miner) = miner_share * V_i / sum_j V_j   over miners passing the v9 gate in the newest signed v9 score
                 (V = max(0, sum z), Stouffer + binomial gate at alpha/N_UID over the 4-epoch window), optionally capped
                 at max_uid_share per UID;   share(UID 0 = owner hotkey) = 1 - sum(shares)   (burn)
  Fallback = 100% burn to UID 0 when: no miner passes the gate; the newest v9 score is missing or more than
  max_score_age_epochs sealed epochs old; the false-positive alarm trips (label-shuffle placebo gated a miner, or the
  uncorrected Stouffer flag rate > max_flag_rate); the score / config signature fails.  Never pays an unregistered key.
The continuous-audit evidence step is NOT called: evidence_loader below only reads the signed v9 score (milliseconds).
Mech-1 is untouched (this path submits the netuid-120 mech-0 vector exactly as before, via the same journal/adapter).

seq5 (Jacob 2026-10-20 16:30 CR "someone always passes the bar"): when the signed LIVE-CONFIG carries
  payout = {rule: 'always-pay-softmax-topk-v1', temperature T, top_k K, min_units U, fallback_top_n F}
the hard gate is DIAGNOSTIC ONLY (still computed by the scorer and recorded in info) and mech-0 is paid as:
  eligible = miners in the newest signed v9 score with B (scored units in the window) >= U and finite pooled Stouffer z
  pool     = top K eligible by z among z > 0;  if none has z > 0: top F eligible by z (at least one miner is always paid)
  share_i  = miner_share * softmax(z_i / T) over pool  (optional max_uid_share cap; only capped excess / miner_share<1 burns)
  ties broken by miner public key (deterministic).  Unregistered pool members are dropped and the rest renormalised.
  Placebo/flag-rate alarms are recorded as diagnostics and no longer force a burn; a missing/stale/badly signed score still
  burns 100% to UID 0 (and so does an empty eligible set).  A config WITHOUT payout keeps the old hard-gate rule exactly.

seq6 (Jacob 2026-10-21 18:56 CR "a bit more of a moving average to the scoring so that over epochs miners stabilize"):
when the signed LIVE-CONFIG ALSO carries
  smoothing = {rule: 'share-ema-v1', half_life_epochs h, lookback_epochs N, min_share m}
the always-pay share vector s_E (above) is computed for the newest signed score E AND for each older signed score E-k,
k < N, from the same scores_dir (each verified against the scorer key + epoch binding; a missing/invalid older score is
skipped, never guessed), and
  ema_i = sum_k a(1-a)^k s_{E-k}(i) / sum_k a(1-a)^k     (a = 1 - 2^(-1/h); a miner absent from epoch E-k contributes 0)
  paid  = miners with ema_i >= m * miner_share that are ELIGIBLE in the newest score E (B >= min_units, finite z),
          renormalised to miner_share   (so a miner who stops submitting decays toward 0 and is cut entirely once it
          leaves the newest eligible set or falls under m; nothing older than N epochs ever counts).
Keyed by miner public key only (no coldkey grouping).  Stateless: a pure function of the signed score files.
If the smoothed set is empty the unsmoothed s_E is used.  A config WITHOUT smoothing keeps the seq5 rule exactly."""
import base64, hashlib, inspect, json, math, os, stat, textwrap, time
from pathlib import Path
from nacl.signing import VerifyKey
VERSION = 'sn120-v9-live-weights-v1'; CONFIG_VERSION = 'sn120-v9-live-weights-config-v1'
SCORE_VERSION = 'sn120-measured-impact-score-v9-shadow'
EPOCHS = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs')
UNITS = 1_000_000_000; BURN = '__burn_uid0_owner__'
CONFIG_FIELDS = {'version', 'sequence', 'issued_at', 'mode', 'miner_share', 'max_uid_share', 'scorer_pubkey', 'scores_dir', 'log_dir',
                 'max_score_age_epochs', 'max_flag_rate', 'note'}
PAYOUT_RULE = 'always-pay-softmax-topk-v1'; PAYOUT_FIELDS = {'rule', 'temperature', 'top_k', 'min_units', 'fallback_top_n'}
SMOOTH_RULE = 'share-ema-v1'; SMOOTH_FIELDS = {'rule', 'half_life_epochs', 'lookback_epochs', 'min_share'}
canonical = lambda v: json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def _read(path, limit=64 * 1024 ** 2):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > limit: raise ValueError('bounded regular file')
        with os.fdopen(fd, 'rb', closefd=False) as f: return f.read(st.st_size + 1)
    finally: os.close(fd)
def _verify(doc, key):
    if type(doc) is not dict or set(doc) != {'payload', 'signer', 'signature'} or doc['signer'] != key: raise ValueError('signer')
    VerifyKey(bytes.fromhex(key)).verify(canonical(doc['payload']), base64.b64decode(doc['signature'], validate=True))
    return doc['payload']
def load_config(path, authority):
    c = _verify(json.loads(_read(path)), authority)
    if set(c) not in (CONFIG_FIELDS, CONFIG_FIELDS | {'payout'}, CONFIG_FIELDS | {'payout', 'smoothing'}) or c['version'] != CONFIG_VERSION or c['mode'] != 'v9-only-burn-uid0': raise ValueError('exact v9 live config')
    if not (0.0 <= float(c['miner_share']) <= 1.0) or (c['max_uid_share'] is not None and not (0.0 < float(c['max_uid_share']) <= 1.0)): raise ValueError('share range')
    if 'payout' in c:
        q = c['payout']
        if (type(q) is not dict or set(q) != PAYOUT_FIELDS or q['rule'] != PAYOUT_RULE or not (float(q['temperature']) > 0)
                or type(q['top_k']) is not int or q['top_k'] < 1 or type(q['min_units']) is not int or q['min_units'] < 1
                or type(q['fallback_top_n']) is not int or not (1 <= q['fallback_top_n'] <= q['top_k'])): raise ValueError('exact payout rule')
    if 'smoothing' in c:
        q = c['smoothing']
        if (type(q) is not dict or set(q) != SMOOTH_FIELDS or q['rule'] != SMOOTH_RULE or type(q['lookback_epochs']) is not int
                or not (1 <= q['lookback_epochs'] <= 48) or not (0.25 <= float(q['half_life_epochs']) <= 24)
                or not (0.0 <= float(q['min_share']) < 0.5)): raise ValueError('exact smoothing rule')
    return c
def always_pay(window_gate, q, miner_share=1.0):
    """Pure: -> (shares {miner: share}, details). softmax(z/T) over the top-K positive-z eligible miners (fallback top-F)."""
    el = [(float(r['stouffer_z']), m) for m, r in window_gate.items()
          if int(r.get('B') or 0) >= q['min_units'] and r.get('stouffer_z') is not None and math.isfinite(float(r['stouffer_z']))]
    el.sort(key=lambda t: (-t[0], t[1]))
    pos = [t for t in el if t[0] > 0]
    pool = pos[:q['top_k']] if pos else el[:q['fallback_top_n']]
    d = dict(eligible=len(el), positive_z=len(pos), fallback_no_positive=not pos and bool(el), pool=len(pool),
             z_max=el[0][0] if el else None, z_pool_min=pool[-1][0] if pool else None)
    if not pool: return {}, d
    T = float(q['temperature']); zmax = pool[0][0]
    e = {m: math.exp((z - zmax) / T) for z, m in pool}; tot = math.fsum(e.values())
    return {m: float(miner_share) * v / tot for m, v in e.items()}, d
def _score(c, E):
    raw = _read(Path(c['scores_dir']) / f'score-e{E}.json'); p = _verify(json.loads(raw), c['scorer_pubkey'])
    if p['version'] != SCORE_VERSION or p['epoch'] != E: raise ValueError('score binding e%d' % E)
    return raw, p
def smooth(c, E, gate, current):
    """Pure: EMA over per-epoch always-pay shares of the signed scores E, E-1, ..., E-N+1 -> (shares, details)."""
    q, po = c['smoothing'], c['payout']; ms = float(c['miner_share']); a = 1.0 - 2.0 ** (-1.0 / float(q['half_life_epochs']))
    acc, wt, used, skipped = {}, [], [], []
    for k in range(q['lookback_epochs']):
        if k == 0: s = current
        else:
            try: s = always_pay(_score(c, E - k)[1]['window_gate'], po, ms)[0]
            except Exception: skipped.append(E - k); continue   # missing / unsigned / mis-bound older score: skipped, never guessed
        w = a * (1.0 - a) ** k; wt.append(w); used.append(E - k)
        for m, v in s.items(): acc.setdefault(m, []).append(w * v)
    W = math.fsum(wt); ema = {m: math.fsum(v) / W for m, v in acc.items()}
    elig = {m for m, r in gate.items() if int(r.get('B') or 0) >= po['min_units'] and r.get('stouffer_z') is not None and math.isfinite(float(r['stouffer_z']))}
    keep = {m: v for m, v in ema.items() if m in elig and v >= float(q['min_share']) * ms}
    d = dict(rule=SMOOTH_RULE, half_life_epochs=q['half_life_epochs'], lookback_epochs=q['lookback_epochs'], min_share=q['min_share'],
             epochs_used=used, epochs_skipped=skipped, candidates=len(ema), dropped_ineligible=sum(1 for m in ema if m not in elig),
             dropped_below_min=sum(1 for m, v in ema.items() if m in elig and v < float(q['min_share']) * ms), paid=len(keep), fallback_unsmoothed=not keep)
    if not keep: return dict(current), d
    tot = math.fsum(keep.values()); return {m: ms * v / tot for m, v in sorted(keep.items())}, d
def latest_sealed(): return max(int(e.name) for e in EPOCHS.iterdir() if e.name.isdigit() and (e / 'FINALIZED.json').exists())

def v9_vector(c, L=None):
    """-> (shares {miner_pubkey: share}, burn share, info). Pure function of signed inputs."""
    L = latest_sealed() if L is None else L; info = dict(latest_sealed=L, reasons=[]); p = None
    try:
        for E in range(L, L - c['max_score_age_epochs'] - 1, -1):
            f = Path(c['scores_dir']) / f'score-e{E}.json'
            if f.exists():
                raw = _read(f); p = _verify(json.loads(raw), c['scorer_pubkey'])
                if p['version'] != SCORE_VERSION or p['epoch'] != E: raise ValueError('score binding e%d' % E)
                info.update(score_epoch=E, score_sha256=hashlib.sha256(raw).hexdigest(), seed_block=p['seed']['seed_block']); break
    except Exception as e:
        info['reasons'].append('score_invalid:%s' % type(e).__name__); p = None
    if p is None:
        info['reasons'].append('score_missing_or_stale'); return {}, 1.0, info
    fp = p['false_positive_checks']; alarms = []
    if fp['label_shuffle']['gated_max'] > 0: alarms.append('placebo_gated')
    if (p['window_summary']['flag_rate_stouffer_uncorrected'] or 0) > c['max_flag_rate']: alarms.append('calibration_flag_rate')
    V = {m: float(r['V']) for m, r in p['window_gate'].items() if r['gated'] and r['V'] > 0}
    info.update(gated=len(V), scored_uids=p['window_summary']['scored_uids'])
    if 'payout' in c:   # seq5 always-pay: gate + alarms are diagnostics only
        sh, d = always_pay(p['window_gate'], c['payout'], c['miner_share'])
        info.update(rule=PAYOUT_RULE, diagnostic_alarms=alarms, gate_diagnostic=dict(gated=len(V)), **d)
        if not sh: info['reasons'].append('no_eligible_miner'); return {}, 1.0, info
        if 'smoothing' in c: sh, info['smoothing'] = smooth(c, info['score_epoch'], p['window_gate'], sh)
        if c['max_uid_share'] is not None: sh = {m: min(v, float(c['max_uid_share'])) for m, v in sh.items()}
        return sh, max(0.0, 1.0 - math.fsum(sh.values())), info
    info['reasons'] += alarms
    if info['reasons'] or not V: return {}, 1.0, info
    tot = math.fsum(V.values()); sh = {m: float(c['miner_share']) * v / tot for m, v in V.items()}
    if c['max_uid_share'] is not None: sh = {m: min(v, float(c['max_uid_share'])) for m, v in sh.items()}   # excess burns
    return sh, max(0.0, 1.0 - math.fsum(sh.values())), info

def _patched_module(name, origin, edits):
    import sys, types
    src = Path(origin).read_text()
    for old, new in edits:
        if src.count(old) != 1: raise ValueError('unexpected source in %s' % origin)
        src = src.replace(old, new)
    m = types.ModuleType(name); m.__file__ = origin + '#v9-burn-uid0'; sys.modules[name] = m
    exec(compile(src, m.__file__, 'exec'), m.__dict__); return m

def isolated_journal_modules():
    """Exact copies of the installed journal modules; the ONLY edits let UID 0 (the owner/validator hotkey) carry weight.
    The original modules are untouched (the old writer path keeps forbidding owner weights)."""
    import importlib, sys
    if '_sn120_v9_weight_submission_transaction' in sys.modules:
        return sys.modules['_sn120_v9_weight_submission_transaction'], sys.modules['_sn120_v9_weight_submission_reconciliation']
    r0 = importlib.import_module('subnet.weight_submission_reconciliation'); t0 = importlib.import_module('subnet.weight_submission_transaction')
    wsr = _patched_module('_sn120_v9_weight_submission_reconciliation', r0.__file__, [(
        '    if any(uid == owner_uid for uid, _ in weights):\n        raise ValueError("owner weights are forbidden")\n',
        '    if any(uid == owner_uid for uid, _ in weights) and owner_uid != 0:\n        raise ValueError("v9: owner weight only as the UID 0 burn")\n')])
    wst = _patched_module('_sn120_v9_weight_submission_transaction', t0.__file__, [
        ('from subnet.weight_submission_reconciliation import (', 'from _sn120_v9_weight_submission_reconciliation import ('),
        ("            if preflight.uid in norm_uids:\n                raise ValueError('owner allocation is forbidden')\n",
         "            if preflight.uid in norm_uids and int(preflight.uid) != 0:\n                raise ValueError('v9: owner allocation only as the UID 0 burn')\n")])
    return wst, wsr

def make_rule(config_path, authority, owner):
    def evidence_loader(*_a, cutoff=None, **_k):
        c = load_config(config_path, authority); sh, burn, info = v9_vector(c)
        return dict(snapshots=dict(shares=sh, burn=burn, info=info, config_sequence=c['sequence']), committed_at_by_epoch={},
                    evidence_hashes=[info.get('score_sha256')] if info.get('score_sha256') else [], refused=[], excluded=[], deferred=[])
    def calculate(snap, _committed, cutoff):
        pts = {m: s for m, s in snap['shares'].items()}
        if snap['burn'] > 0: pts[BURN] = snap['burn']
        return dict(version=VERSION, cutoff=cutoff, points=pts, v9=snap['info'], config_sequence=snap['config_sequence'])
    def validate_assessment(a, _p, cutoff):
        if a.get('version') != VERSION or a['cutoff'] != cutoff or not math.isclose(math.fsum(a['points'].values()), 1.0, abs_tol=1e-9): raise ValueError('v9 assessment')
    def recipients(assessment, registrations):
        by_pub = {row['public_key']: (h, row) for h, row in registrations.items()}
        pts, sel, excluded, burn = {}, {}, [], float(assessment['points'].get(BURN, 0.0))
        ap = (assessment.get('v9') or {}).get('rule') == PAYOUT_RULE
        if ap:   # always-pay: drop unregistered/owner pool members and renormalise the miner mass over the registered rest
            mp = {m: s for m, s in assessment['points'].items() if m != BURN}; ok = {m: s for m, s in mp.items() if m in by_pub and by_pub[m][0] != owner}
            if ok and len(ok) < len(mp):
                f = math.fsum(mp.values()) / math.fsum(ok.values())
                assessment = dict(assessment, points=dict({m: s * f for m, s in ok.items()}, **({BURN: burn} if burn > 0 else {})))
                excluded = sorted(set(mp) - set(ok))
        for m, s in assessment['points'].items():
            if m == BURN: continue
            if m not in by_pub or by_pub[m][0] == owner:
                if m not in excluded: excluded.append(m)
                burn += s; continue   # unregistered -> burned
            h, row = by_pub[m]; pts[h] = max(1, int(s * UNITS)); sel[h] = row
        if burn > 1e-12: pts[owner] = max(1, int(burn * UNITS))
        log = dict(version=VERSION, at=time.time(), cutoff=assessment['cutoff'], recipients=len(pts), burn_share=round(burn, 12),
                   miners_paid=len(pts) - (owner in pts), v9=assessment.get('v9'), digest=hashlib.sha256(canonical(sorted(pts.items()))).hexdigest())
        try:
            d = Path(load_config(config_path, authority)['log_dir']); d.mkdir(parents=True, exist_ok=True, mode=0o700)
            t = d / ('.w-%s.tmp' % assessment['cutoff']); t.write_text(json.dumps(log, sort_keys=True)); os.replace(t, d / ('weights-%s.json' % assessment['cutoff']))
        except Exception: pass
        return pts, sel, sorted(set(excluded))
    return dict(evidence_loader=evidence_loader, calculate=calculate, validate_assessment=validate_assessment, recipients=recipients)

def make_adapter(ChainAdapter):
    """Identical submit_hour except UID 0 (owner hotkey) may receive the burn weight."""
    full = textwrap.dedent(inspect.getsource(ChainAdapter.submit_hour))
    body = full
    b1 = "    if points.get(OWNER, 0) or points.get(self.owner, 0):\n        raise ValueError('owner is never a reward recipient')\n"
    b2 = "            if not count:\n                continue\n"
    b3 = "        if any(h in (OWNER, self.owner) for _, h, _ in recipients):\n            raise ValueError('owner is never a reward recipient')\n"
    if not (body.count(b1) == 1 and body.count(b2) == 1 and body.count(b3) == 1): raise ValueError('unexpected submit_hour source')
    body = body.replace(b1, "    if OWNER != self.owner:\n        raise ValueError('v9 burn requires owner hotkey == validator')\n")
    body = body.replace(b2, b2 + "            if hotkey == self.owner:   # v9: burn weight to UID 0 (owner hotkey), verified at the snapshot block below\n"
                                 "                recipients.append((None, hotkey, count))\n                continue\n")
    body = body.replace(b3, "        recipients = [(int(self.query('Uids', [self.netuid, self.owner], block)) if u is None else u, h, p) for u, h, p in recipients]\n"
                            "        if any(h == self.owner and u != 0 for u, h, _ in recipients):\n            raise ValueError('v9 burn target must be UID 0')\n")
    wst, wsr = isolated_journal_modules()
    body = body.replace('from subnet.weight_submission_transaction import', 'from _sn120_v9_weight_submission_transaction import')
    if 'from subnet.weight_submission_transaction' in body: raise ValueError('unpatched journal import')
    g = dict(inspect.getmodule(ChainAdapter).__dict__); ns = {}
    exec(compile(body, inspect.getsourcefile(ChainAdapter) + '#v9-burn-uid0', 'exec'), g, ns)
    V9Adapter = type('V9ChainAdapter', (ChainAdapter,), {'submit_hour': ns['submit_hour']})
    return V9Adapter, wst

def make(config_path, authority, owner, ChainAdapter):
    """Replacements for the writer's run_once globals (assessment rule + recipients + adapter + journal)."""
    V9Adapter, wst = make_adapter(ChainAdapter)
    return dict(make_rule(config_path, authority, owner), ChainAdapter=V9Adapter, authenticated_assessment_sources=lambda *a, **k: ([], []),
                historical_assessments=lambda *a, **k: ([], []), best_fallback=lambda *a, **k: None,
                ASSESSMENT_DIRECTORY='current-assessment-v9-live-1', SubmissionJournal=wst.SubmissionJournal)
