"""Sign the SN120 v9 shadow release manifest with the on-box authority key (same key/flow as v7c sign_impact.py; seed never printed).
  sign_v9.py release [--note ...]   -> RELEASE-v9.seq<N>.signed.json, atomically made RELEASE-v9.signed.json (ACTIVE)
  sign_v9.py live-config            -> LIVE-CONFIG-v9.seq<N>.signed.json + LIVE-CONFIG-v9.ACTIVE.signed.json (v9-only weights, burn UID 0)
  sign_v9.py host [--disabled]      -> HOST-COMPOSITION.v9-live.ROOT-SIGNED.private.json (v7d writer base + v9 rule)
Pins sha256 of every Arbos-side v9 file, the patched publisher module + its drop-in, the v9 units, and the pod-side code."""
import argparse, base64, hashlib, json, os, subprocess, time
from pathlib import Path
from nacl.signing import SigningKey, VerifyKey
R = Path('/home/const/subnet120-rewrite/state/root-audits'); N = R / '20261020-measured-impact-v9'
AUTH = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
ACTIVE = N / 'RELEASE-v9.signed.json'; U = Path.home() / '.config/systemd/user'
canon = lambda v: json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
sha = lambda b: hashlib.sha256(b).hexdigest()
FILES = [N / 'scorer_driver_v9.py', N / 'driver_v9.json', N / 'seed_block.py', N / 'fetch_epoch.py', N / 'sign_v9.py',
         N / 'publisher-embargo/source/ops/mimo_finalized_evidence.py', U / 'affine-finalized-evidence-publication.service.d/zzzz~~v9-embargo.conf',
         U / 'sn120-impact-scorer-v9.service', U / 'sn120-impact-scorer-v9.timer',
         N / 'weights_v9.py', N / 'hourly_entry_v9.py', N / 'check_weight_freshness_v9.py', N / 'test_weights_v9.py', N / 'verify_onchain_v9.py',
         N / 'zzzz~~~~~~~~~~v9-live-weights.conf', N / 'zz-v9-freshness.conf', N / 'public_scores_v9.py']
POD = ['v9common.py', 'v9agg.py', 'gen_ref_v9.py', 'grad_docs_v9.py', 'score_v9.py', 'test_v9agg.py']
def key():
    life = json.loads((R / '20261015-mimo-v8.2-release-succession/release-v822-signed/lifecycle.v8.2.2.ACTIVE.private.json').read_text())
    k = SigningKey(bytes.fromhex(Path(life['authority_seed_file']).read_text().strip()))
    assert k.verify_key.encode().hex() == AUTH, 'authority mismatch'
    return k
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('what', choices=['release', 'live-config', 'host']); ap.add_argument('--note', default=''); ap.add_argument('--disabled', action='store_true')
    ap.add_argument('--smoothing', default='', help='live-config only (needs --payout): JSON share-EMA rule, e.g. {"rule":"share-ema-v1","half_life_epochs":4,"lookback_epochs":12,"min_share":0.01}; omit = no smoothing')
    ap.add_argument('--payout', default='', help='live-config only: JSON always-pay rule, e.g. {"rule":"always-pay-softmax-topk-v1","temperature":1.0,"top_k":10,"min_units":3,"fallback_top_n":3}; omit = legacy hard gate')
    ap.add_argument('--window-payout', default='', help='live-config only (seq7, excludes --payout/--smoothing): JSON, e.g. {"rule":"window-credit-v1","window_epochs":4,"min_window_groups":1,"min_newest_groups":1}')
    a = ap.parse_args()
    if a.what != 'release': return other(a)
    cfg = json.loads((N / 'driver_v9.json').read_text())
    r = subprocess.run(['ssh', '-n', '-o', 'BatchMode=yes', '-p', str(cfg['pod_port']), cfg['pod'], f'cd {cfg["pod_dir"]} && sha256sum ' + ' '.join(POD)],
                       capture_output=True, text=True, check=True, timeout=60)
    pod = {l.split()[1]: l.split()[0] for l in r.stdout.strip().splitlines()}; assert set(pod) == set(POD)
    prev = json.loads(ACTIVE.read_text())['payload']['sequence'] if ACTIVE.exists() else 0
    p = dict(version='sn120-measured-impact-v9-shadow-release', sequence=prev + 1, issued_at=int(time.time()), shadow_only=True,
             writer_blend_config_unchanged=str(R / '20261019-measured-impact-v1/BLEND-CONFIG.ACTIVE.signed.json'),
             seed_rule='sn120-v9-seed-rule-v1', embargo_epochs=3, window_epochs=cfg['window_epochs'], unit_groups=3, alpha=0.01,
             files=[dict(path=str(f), sha256=sha(f.read_bytes())) for f in FILES], pod_files=pod, note=a.note)
    k = key(); doc = dict(payload=p, signer=AUTH, signature=base64.b64encode(k.sign(canon(p)).signature).decode())
    VerifyKey(bytes.fromhex(AUTH)).verify(canon(p), base64.b64decode(doc['signature']))
    f = N / f'RELEASE-v9.seq{p["sequence"]}.signed.json'
    fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.write(fd, json.dumps(doc, indent=1, sort_keys=True).encode()); os.close(fd)
    tmp = N / '.RELEASE-v9.tmp'; tmp.unlink(missing_ok=True); tmp.write_bytes(f.read_bytes()); os.chmod(tmp, 0o600); os.replace(tmp, ACTIVE)
    print(json.dumps(dict(written=str(f), sequence=p['sequence'], sha256=sha(f.read_bytes()), signature_prefix=doc['signature'][:8])))
def _sign_write(p, f, active=None):
    k = key(); doc = dict(payload=p, signer=AUTH, signature=base64.b64encode(k.sign(canon(p)).signature).decode())
    VerifyKey(bytes.fromhex(AUTH)).verify(canon(p), base64.b64decode(doc['signature']))
    fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.write(fd, json.dumps(doc, indent=1, sort_keys=True).encode()); os.close(fd)
    if active is not None:
        tmp = N / ('.' + active.name + '.tmp'); tmp.unlink(missing_ok=True); tmp.write_bytes(f.read_bytes()); os.chmod(tmp, 0o600); os.replace(tmp, active)
    print(json.dumps(dict(written=str(f), sha256=sha(f.read_bytes()), signature_prefix=doc['signature'][:8], **({'sequence': p['sequence']} if 'sequence' in p else {}))))
pin = lambda f: dict(path=str(f), sha256=sha(Path(f).read_bytes()))
LIVE = N / 'LIVE-CONFIG-v9.ACTIVE.signed.json'; B = R / '20261020-writer-time-budget-v2'
def other(a):
    if a.what == 'live-config':
        prev = json.loads(LIVE.read_text())['payload']['sequence'] if LIVE.exists() else 0
        p = dict(version='sn120-v9-live-weights-config-v1', sequence=prev + 1, issued_at=int(time.time()), mode='v9-only-burn-uid0',
                 miner_share=1.0, max_uid_share=None, scorer_pubkey=json.loads((N / 'driver_v9.json').read_text())['scorer_pubkey'],
                 scores_dir=str(N / 'scores'), log_dir=str(N / 'live-log'), max_score_age_epochs=2, max_flag_rate=0.03,
                 note=a.note or 'Jacob 2026-10-20: weights 100% from v9; unearned weight burns to UID 0; audit evidence removed from the weight path')
        if a.window_payout:
            if a.payout or a.smoothing: raise SystemExit('--window-payout replaces --payout/--smoothing')
            p['window_payout'] = json.loads(a.window_payout)
        if a.payout: p['payout'] = json.loads(a.payout)
        if a.smoothing:
            if not a.payout: raise SystemExit('--smoothing requires --payout')
            p['smoothing'] = json.loads(a.smoothing)
        return _sign_write(p, N / f'LIVE-CONFIG-v9.seq{p["sequence"]}.signed.json', LIVE)
    f = N / 'HOST-COMPOSITION.v9-live.ROOT-SIGNED.private.json'
    if f.exists(): f.rename(N / ('HOST-COMPOSITION.v9-live.superseded-%d.private.json' % int(time.time())))
    p = dict(version='hourly-host-composition-v9-live-weights', execute_enabled=not a.disabled, entry=pin(N / 'hourly_entry_v9.py'),
             base_entry=pin(B / 'hourly_population_index_entry_v7d.py'), base_composition=pin(B / 'HOST-COMPOSITION.v8.2.2-accel-window-v7d-budget2.ROOT-SIGNED.private.json'),
             weights_module=pin(N / 'weights_v9.py'), config_path=str(LIVE), maximum_seconds=1600, evidence_timeout_seconds=1400)
    return _sign_write(p, f)
if __name__ == '__main__': main()
