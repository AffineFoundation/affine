#!/usr/bin/env python3
"""Read-only exporter: published SN120 MiMo checkpoints -> a small index file.

Runs on the control host (Arbos). It only READS state/<run>/epochs/*/coordinator/OPENING.json
(the signed epoch openings that miners receive) and copies out, per checkpoint:
id, file hashes, the input checkpoint it was trained from, optimizer step, and the
presigned GET read_urls already published to miners (read-only, ~24h expiry).
It never copies miner capabilities or any other miner data, never writes to the
state tree, and uses no credentials.
"""
import argparse, glob, hashlib, json, os, re, sys, time
from pathlib import Path


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def url_expiry(url):
    d = re.search(r'X-Amz-Date=(\d{8}T\d{6}Z)', url); e = re.search(r'X-Amz-Expires=(\d+)', url)
    if not d or not e:
        return None
    import calendar
    t = calendar.timegm(time.strptime(d.group(1), '%Y%m%dT%H%M%SZ'))
    return t + int(e.group(1))


def export(state, cache):
    out = {}
    pool = None
    for path in sorted(glob.glob(str(Path(state) / 'epochs/*/coordinator/OPENING.json')),
                       key=lambda p: int(Path(p).parts[-3])):
        st = os.stat(path)
        key = f'{path}:{st.st_mtime_ns}:{st.st_size}'
        row = cache.get(key)
        if row is None:
            o = json.loads(Path(path).read_bytes())['payload']
            cp = o['checkpoint']
            auth = (o.get('mimo_checkpoint_authorization') or {}).get('payload', {})
            env = [e for e in o['environments'] if e['env_id'] == 'affine_math']
            row = dict(opening_epoch=int(Path(path).parts[-3]), epoch=o.get('epoch'),
                       checkpoint_id=cp['id'], files=cp['files'],
                       input_checkpoint=auth.get('input_checkpoint'),
                       optimizer_step=auth.get('optimizer_step'),
                       read_urls=cp.get('read_urls', {}),
                       opening_start=o.get('start'),
                       pool=dict(env_id='affine_math', indices=env[0]['indices'], spec=env[0]['spec'],
                                 harness=env[0]['harness']) if env else None,
                       mimo_runtime={k: o['mimo_runtime'][k] for k in (
                           'model_id', 'temperature', 'top_p', 'top_k', 'enable_thinking',
                           'eos_token_ids', 'max_context_tokens')},
                       vocab_size=o['group_transport_policy']['vocab_size'],
                       answer_projection=o.get('mimo_answer_projection'))
            cache[key] = row
        if hashlib.sha256(canonical(row['files'])).hexdigest() != row['checkpoint_id']:
            print('skip: filemap identity mismatch', path, file=sys.stderr); continue
        exp = [url_expiry(u) for u in row['read_urls'].values()]
        r = {k: v for k, v in row.items() if k != 'pool'}
        r['read_urls_expire_at'] = min([e for e in exp if e] or [0])
        # A checkpoint can open several epochs (e.g. an empty epoch); keep the newest URLs.
        prev = out.get(r['checkpoint_id'])
        if prev is None or r['read_urls_expire_at'] >= prev['read_urls_expire_at']:
            first = prev['opening_epoch'] if prev else r['opening_epoch']
            out[r['checkpoint_id']] = dict(r, first_opening_epoch=min(first, r['opening_epoch']))
        pool = row['pool'] or pool
    cps = sorted(out.values(), key=lambda r: (r['first_opening_epoch'], r['optimizer_step'] or 0))
    return dict(version='trusted-sampler-checkpoint-index-v1', exported_at=time.time(),
                checkpoints=cps, pool=pool)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--state', default=os.path.expanduser('~/subnet120-rewrite/state/mimo-mainnet-v1'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--watch', type=float, default=0, help='re-export every N seconds')
    ap.add_argument('--push', action='append', default=[],
                    help='scp destination(s) for the index, e.g. root@host:/root/trusted-sampler/index.json (with --ssh-port)')
    ap.add_argument('--ssh-opts', default='')
    a = ap.parse_args()
    cache = {}
    while True:
        idx = export(a.state, cache)
        tmp = a.out + '.tmp'
        Path(tmp).write_bytes(canonical(idx)); os.replace(tmp, a.out)
        for dest in a.push:
            os.system(f'scp -q {a.ssh_opts} {a.out} {dest}.tmp && ssh {a.ssh_opts.replace("-P ", "-p ")} '
                      f'{dest.split(":")[0]} mv {dest.split(":")[1]}.tmp {dest.split(":")[1]}')
        print(json.dumps(dict(t=time.strftime('%H:%M:%SZ', time.gmtime()), checkpoints=len(idx['checkpoints']),
                              latest=idx['checkpoints'][-1]['checkpoint_id'][:8])), flush=True)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == '__main__':
    main()
