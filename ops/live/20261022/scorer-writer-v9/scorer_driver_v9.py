"""SN120 v9 measured-impact scorer driver (Arbos, user timer every 2 min, flock). SHADOW ONLY: nothing here is read by the
hourly weight writer (its blend config / scores_dir still point at the v7c release, blend 0.0).
Startup: verify RELEASE-v9.signed.json (authority 3301134b...) and the sha256 of every pinned file; fail closed.
Per sealed epoch (FINALIZED.json) among the latest keep_epochs (>= first_epoch) without a verified v9 score:
  READ-ONLY R2 fetch of training documents (fetch_epoch.py, LIST+GET only) -> public post-commit seed from the finalized
  chain (seed_block.py, read-only) -> push both to the pod -> start score_v9.py (non-blocking) -> pull the signed score,
  verify against the pinned scorer key + epoch/seed binding, install atomically in scores/.  Status in status.json."""
import base64, fcntl, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from nacl.signing import VerifyKey
D = Path(__file__).resolve().parent; sys.path.insert(0, str(D))
AUTH = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
canon = lambda v: json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
def verify_release():
    doc = json.loads((D / 'RELEASE-v9.signed.json').read_text())
    if set(doc) != {'payload', 'signer', 'signature'} or doc['signer'] != AUTH: raise ValueError('release signer')
    VerifyKey(bytes.fromhex(AUTH)).verify(canon(doc['payload']), base64.b64decode(doc['signature'], validate=True))
    for f in doc['payload']['files']:
        if hashlib.sha256(Path(f['path']).read_bytes()).hexdigest() != f['sha256']: raise ValueError('pinned file changed: ' + f['path'])
    return doc['payload']
REL = verify_release(); CFG = json.loads((D / 'driver_v9.json').read_text())
S = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs'); SC = D / 'scores'; WK = D / 'work'; PD = CFG['pod_dir']
SSH = ['ssh', '-n', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-p', str(CFG['pod_port']), CFG['pod']]
SCP = ['scp', '-q', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-P', str(CFG['pod_port'])]
V7WORK = Path('/home/const/subnet120-rewrite/state/root-audits/20261019-measured-impact-v1/work')
def atomic(p, b):
    t = p.with_name(p.name + '.tmp'); fd = os.open(t, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'wb') as f: f.write(b)
    os.replace(t, p)
def pod(cmd, timeout=120): return subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)
def verify(raw, E, seed):
    d = json.loads(raw); p = d['payload']
    if set(d) != {'payload', 'signer', 'signature'} or d['signer'] != CFG['scorer_pubkey']: raise ValueError('scorer key')
    VerifyKey(bytes.fromhex(CFG['scorer_pubkey'])).verify(canon(p), base64.b64decode(d['signature'], validate=True))
    if p['epoch'] != E or p['version'] != 'sn120-measured-impact-score-v9-shadow' or p['shadow_only'] is not True: raise ValueError('epoch binding')
    if seed is not None and (p['seed']['seed'] != seed['seed'] or p['seed']['seed_block_hash'] != seed['seed_block_hash']): raise ValueError('seed binding')
    return p
def main():
    SC.mkdir(exist_ok=True, mode=0o700); WK.mkdir(exist_ok=True, mode=0o700)
    lock = open(D / 'driver.lock', 'w')
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: return
    st = json.loads((D / 'status.json').read_text()) if (D / 'status.json').exists() else {}
    st.setdefault('errors', {}); st.setdefault('started', {}); st.setdefault('installed', {})
    r = pod(f'cd {PD} && sha256sum ' + ' '.join(sorted(REL['pod_files'])))
    got = {l.split()[1]: l.split()[0] for l in r.stdout.strip().splitlines()}
    if r.returncode != 0 or got != REL['pod_files']: raise SystemExit('pod v9 code does not match the signed release')
    sealed = sorted(int(e.name) for e in S.iterdir() if e.name.isdigit() and (e / 'FINALIZED.json').exists())
    sealed = [e for e in sealed if e >= CFG['first_epoch']][-CFG['keep_epochs']:]
    st.update(at=time.time(), latest_sealed=max(sealed) if sealed else None, release_sequence=REL['sequence'])
    for E in sorted(sealed):        # oldest first: the window gate of E reads E-1..E-3 on the pod
        out = SC / f'score-e{E}.json'
        if out.exists(): continue
        k = str(E); seedf = WK / f'seed-e{E}.json'
        try:
            r = pod(f'test -f {PD}/out/score-e{E}.json && echo DONE || (pgrep -f "[s]core_v9.py --epoch {E} " >/dev/null && echo RUNNING || echo IDLE)')
            state = r.stdout.strip()
            if state == 'DONE':
                tmp = WK / f'score-e{E}.json'
                subprocess.run(SCP + [f'{CFG["pod"]}:{PD}/out/score-e{E}.json', str(tmp)], check=True, timeout=120)
                raw = tmp.read_bytes(); p = verify(raw, E, json.loads(seedf.read_text()) if seedf.exists() else None); atomic(out, raw); tmp.unlink()
                st['errors'].pop(k, None)
                st['installed'][k] = dict(at=time.time(), miners=p['miners'], seed_block=p['seed']['seed_block'], gated=p['window_summary']['gated'],
                                          scored_uids=p['window_summary']['scored_uids'], fp=p['false_positive_checks']['label_shuffle']['gated_max'])
                continue
            if state == 'RUNNING':
                st['errors'].pop(k, None); st['started'].setdefault(k, time.time()); continue
            if st['started'].get(k) and time.time() - st['started'][k] < CFG['retry_seconds']:
                tail = pod(f'tail -3 {PD}/logs/score-e{E}.log').stdout[-400:]
                st['errors'][k] = dict(at=time.time(), error='scorer exited without output', tail=tail); continue
            if any(int(x) < E for x in st['started'] if not (SC / f'score-e{x}.json').exists() and str(x) not in st['errors']):
                continue                    # serialize epochs (one GPU job chain at a time)
            docs = WK / f'docs-e{E}.jsonl.gz'
            if not docs.exists():
                if (V7WORK / docs.name).exists(): atomic(docs, (V7WORK / docs.name).read_bytes())   # identical read-only fetch already on disk
                else: subprocess.run([sys.executable, str(D / 'fetch_epoch.py'), str(E), str(docs)], check=True, timeout=900, capture_output=True)
            if not seedf.exists():
                subprocess.run([sys.executable, '-B', str(D / 'seed_block.py'), str(E), str(seedf)], check=True, timeout=300, capture_output=True)
            pod(f'mkdir -p {PD}/data/e{E} {PD}/logs')
            subprocess.run(SCP + [str(docs), f'{CFG["pod"]}:{PD}/data/e{E}/docs.jsonl.gz'], check=True, timeout=600)
            subprocess.run(SCP + [str(seedf), f'{CFG["pod"]}:{PD}/data/e{E}/seed.json'], check=True, timeout=60)
            pod(f'cd {PD} && (setsid nohup /root/sampler-venv/bin/python score_v9.py --epoch {E} --max-gpus {CFG["max_gpus"]} --window {CFG["window_epochs"]} '
                f'> logs/score-e{E}.log 2>&1 < /dev/null &)', timeout=60)
            st['started'][k] = time.time(); st['errors'].pop(k, None)
        except Exception as e:
            st['errors'][k] = dict(at=time.time(), error=f'{type(e).__name__}: {str(e)[:300]}')
    atomic(D / 'status.json', json.dumps(st, indent=1, sort_keys=True).encode())
if __name__ == '__main__': main()
