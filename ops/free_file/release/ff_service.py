#!/usr/bin/env python3
"""SN120 free-file service (systemd --user affine-ff.service). Jacob 2026-10-30 pt898u, RA/20261030-free-file.

  links (every LINK_EVERY s): for every hotkey in the current epoch opening's capabilities (registered, activated Ed25519,
    not yet served) a private upload link for ONE stable object private/ff/v1/miners/<hotkey>/batch.json, sealed to the
    hotkey with the same SealedBox method as the epoch capability (subnet.storage.encrypt), written publicly to
    website/ff/links/<hotkey>.json. Fresh presigned PUT/GET URLs every time, LINK_TTL validity, same object key.
  readiness (every TICK s): lists the miner objects; each new ETag is fetched and format-checked (ff_format, live batch cap)
    -> private state/FF-READY.json (early-seal prediction, queue preview) and public website/ff/status.json
    (valid / reason / counts per hotkey; never content). The turn itself re-copies and re-checks the file.
"""
import hashlib, json, os, sys, time, traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ff_format
import ff_lifecycle as FL

SRC = Path('/home/const/subnet120-rewrite/state/root-audits/20261029-v829u-125task/deployment-final-source-v823u')
LIFE = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1')
LIFE_CFG = Path('/home/const/subnet120-rewrite/state/root-audits/20261017-v824/release-v824-signed/lifecycle.v8.2.4.ACTIVE.private.json')
WEB = Path('/home/const/subnet120/affine/website')
LINKS = WEB / 'ff' / 'links'
STATUS = WEB / 'ff' / 'status.json'
TICK, LINK_EVERY, LINK_TTL = 20, 1800, 86400
LINK_VERSION = 'affine-ff-link-v1'


def cr(ts):
    return time.strftime('%Y-%m-%d %H:%M:%S CR (UTC-6)', time.gmtime(ts - 6 * 3600))


def log(**kw):
    kw['at_cr'] = cr(time.time()); print(json.dumps(kw, default=str), flush=True)


def wj(path, value, mode):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, 'w') as f:
        f.write(json.dumps(value, indent=1, sort_keys=True) + '\n')
    os.chmod(tmp, mode); os.replace(tmp, path)


def bucket():
    sys.path.insert(0, str(SRC))
    from subnet.storage import Bucket
    return Bucket(json.loads(LIFE_CFG.read_text())['bucket'])


def current():
    c = json.loads((LIFE / 'CURRENT.json').read_bytes())
    return c['epoch_index'], c


def eligible():
    """hotkey -> identity hex, from the newest epoch opening (capabilities = activated registered identities)."""
    eps = sorted((int(p.name) for p in (LIFE / 'epochs').iterdir() if p.name.isdigit() and (p / 'coordinator' / 'OPENING.json').is_file()))
    chain = json.loads(FL.L_K1_CHAIN.read_bytes())
    m = json.loads((LIFE / 'epochs' / str(eps[-1]) / 'coordinator' / 'OPENING.json').read_bytes())['payload']
    caps = set(m['capabilities'])
    return {row['hotkey']: row['miner_hex'] for u, row in chain['uids'].items() if int(u) != 0 and row.get('miner_hex') in caps}


def issue_links(b, state):
    from subnet.storage import encrypt
    now = int(time.time())
    hks = eligible()
    LINKS.mkdir(parents=True, exist_ok=True)
    for hk, ident in hks.items():
        key = FL.object_key(hk)
        link = dict(version=LINK_VERSION, hotkey=hk, object_key=key, issued_at=now, expires_at=now + LINK_TTL,
                    put_url=b.presign(key, 'put_object', LINK_TTL), put_headers={'Content-Type': 'application/octet-stream'},
                    get_url=b.presign(key, 'get_object', LINK_TTL), format=ff_format.FORMAT,
                    spec='https://affine.io/mining.json (training.batch_file)')
        wj(LINKS / (hk + '.json'), dict(version=LINK_VERSION, hotkey=hk, issued_at=now, expires_at=now + LINK_TTL,
                                        sealed_to=ident, sealed=encrypt(ident, link)), 0o644)
    for p in LINKS.glob('*.json'):
        if p.stem not in hks:
            p.unlink()
    wj(WEB / 'ff' / 'links.json', dict(version=LINK_VERSION, issued_at=now, issued_at_cr=cr(now), expires_at=now + LINK_TTL,
                                       hotkeys=sorted(hks), reissue_every_seconds=LINK_EVERY), 0o644)
    state['links_at'] = now
    log(event='links', n=len(hks))


def readiness(b, state):
    f = FL.flag()
    if f is None:
        return
    e, cur = current()
    tok = FL.tokenizer(f, cur['checkpoint']['files'])
    max_tasks, max_rollouts = FL.caps()
    capkey = (max_tasks, max_rollouts, cur['checkpoint']['files']['tokenizer.json'])
    cache = state.setdefault('cache', {})
    objs = FL.listing(b)
    now = time.time()
    miners = {}
    for hk, (etag, size) in sorted(objs.items()):
        hit = cache.get(hk)
        if not hit or hit['etag'] != etag or hit['capkey'] != capkey:
            if size > ff_format.MAX_FILE_BYTES:
                ok, info = False, 'file larger than %d bytes' % ff_format.MAX_FILE_BYTES
            else:
                r = b.client.get_object(Bucket=b.name, Key=FL.object_key(hk))
                raw, etag = r['Body'].read(), r['ETag'].strip('"')
                ok, info, _ = ff_format.check(raw, hk, tok, max_tasks, max_rollouts)
            hit = dict(etag=etag, capkey=capkey, size=size, valid=ok, checked_at=now,
                       **(dict(groups=info['groups'], rollouts=info['rollouts'], dropped_groups=info['dropped_groups'],
                               file_sha256=info['file_sha256']) if ok else dict(reason=info)))
            cache[hk] = hit
            log(event='checked', hotkey=hk, valid=ok, **({'groups': info['groups'], 'rollouts': info['rollouts']} if ok else {'reason': info}))
        miners[hk] = {k: v for k, v in hit.items() if k != 'capkey'}
    for hk in list(cache):
        if hk not in objs:
            del cache[hk]
    wj(FL.FF_READY, dict(version='sn120-ff-ready-v1', checked_at=now, epoch_index=e, caps=dict(max_tasks=max_tasks, max_rollouts=max_rollouts),
                         miners=miners), 0o600)
    wj(STATUS, dict(version='sn120-ff-status-v1', generated_at=int(now), generated_at_cr=cr(now), format=ff_format.FORMAT,
                    caps=dict(max_tasks=max_tasks, max_rollouts=max_rollouts),
                    note='Format check of each miner file as uploaded (no content). The turn copies and re-checks the file.',
                    miners={hk: {k: v[k] for k in ('valid', 'reason', 'groups', 'rollouts', 'dropped_groups', 'checked_at', 'size') if k in v}
                            for hk, v in miners.items()}), 0o644)


def main():
    once = '--once' in sys.argv
    b = bucket(); state = {}
    while True:
        try:
            if time.time() - state.get('links_at', 0) >= LINK_EVERY:
                issue_links(b, state)
        except Exception:
            log(event='links_error', error=traceback.format_exc()[-600:])
        try:
            readiness(b, state)
        except Exception:
            log(event='ready_error', error=traceback.format_exc()[-600:])
        if once:
            return
        time.sleep(TICK)


if __name__ == '__main__':
    main()
