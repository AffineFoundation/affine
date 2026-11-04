#!/usr/bin/env python3
"""Download the fp32 master of a GREEN head and verify it (sha256 per file + the master digest the trainer committed).

  python3 fetch_fp32_master.py --out ./head-fp32-master                 # master of the current head (mining.json checkpoint)
  python3 fetch_fp32_master.py --head <checkpoint id> --out DIR          # a specific GREEN head
  python3 fetch_fp32_master.py --out DIR --verify-only                   # re-check a downloaded master, no network for shards

Every GREEN turn publishes the head twice: the bf16 checkpoint (fetch_head.py) and the fp32 master the trainer steps
(M_C; bf16 head = round-to-nearest-even(M_C)). The next turn starts from the master, not from fp32(bf16 head), so a local
step from the bf16 upcast drifts from the gate's candidate. Pass the directory to train_step.py --master.

Chain of trust: https://affine.io/k1/fp32-master/<head>.json lists files {name: {sha256, bytes}}, tensor_sha256 and
master_digest = sha256(canonical JSON {tensor name: sha256(little-endian fp32 bytes)}); the same master_digest is in the
GREEN turn's turn.json training.optimizer.sgd.fp32_master.output_master_digest. Download links are presigned R2 GETs in the
manifest's `download` block (7 days, re-signed; the newest few GREEN heads only).

Keep the master OUTSIDE the bf16 head directory: vLLM and from_pretrained load every *.safetensors in a model directory.
~36 GB on disk (9 shards, 427 float32 tensors, weights only)."""
import argparse, hashlib, json, sys, urllib.request
from pathlib import Path
SITE = 'https://affine.io'
UA = {'User-Agent': 'affine-miner-kit/1'}


def get(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def download(url, dst):
    tmp = dst.with_suffix(dst.suffix + '.part')
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r, open(tmp, 'wb') as f:
        while (b := r.read(1 << 24)):
            f.write(b)
    tmp.replace(dst)


def master_digest(out, man):
    """Recompute sha256(fp32 bytes) per tensor from the shards and the canonical master digest."""
    from safetensors import safe_open
    got = {}
    for f in sorted(man['files']):
        with safe_open(str(out / f), framework='numpy') as s:
            for n in s.keys():
                a = s.get_tensor(n)
                if str(a.dtype) != 'float32':
                    sys.exit(f'{f}:{n} is {a.dtype}, expected float32')
                got[n] = hashlib.sha256(a.astype('<f4', copy=False).tobytes()).hexdigest()
    return got, hashlib.sha256(json.dumps(got, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='directory for the shards (NOT the bf16 head directory)')
    ap.add_argument('--head', help='GREEN checkpoint id (default: mining.json checkpoint = the current head)')
    ap.add_argument('--verify-only', action='store_true')
    ap.add_argument('--skip-tensor-digest', action='store_true', help='file sha256s only (skip re-hashing every tensor)')
    ap.add_argument('--jobs', type=int, default=4, help='parallel shard downloads (default 4)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        sys.exit(f'{out} looks like a model directory (config.json); keep the master in its own directory')
    head = a.head or json.loads(get(SITE + '/mining.json'))['checkpoint']
    try:
        man = json.loads(get(f'{SITE}/k1/fp32-master/{head}.json'))
    except Exception as x:
        sys.exit(f'no fp32 master manifest for {head} ({x}); the head may not be GREEN-published yet: '
                 f'see {SITE}/k1/fp32-master/index.json and current.json')
    if man['checkpoint'] != head:
        sys.exit('manifest checkpoint != requested head')
    if not a.verify_only:
        dl = man.get('download') or {}
        if not dl.get('files'):
            sys.exit('this manifest has no download links (only the newest GREEN heads keep live links)')
        def one(item):
            name, want = item; dst = out / name
            if dst.exists() and dst.stat().st_size == want['bytes'] and sha256_file(dst) == want['sha256']:
                return
            print('download', name, f"{want['bytes'] / 1e9:.2f} GB", flush=True)
            download(dl['files'][name], dst)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max(1, a.jobs)) as ex:
            list(ex.map(one, sorted(man['files'].items())))
    bad = [n for n, w in man['files'].items() if not (out / n).exists() or sha256_file(out / n) != w['sha256']]
    if bad:
        sys.exit(f'sha256 mismatch or missing: {bad}')
    res = dict(ok=True, checkpoint=head, epoch=man['epoch'], files=len(man['files']), bytes=man.get('bytes'),
               master_digest=man['master_digest'], dir=str(out), file_sha256='ok')
    if not a.skip_tensor_digest:
        per, dig = master_digest(out, man)
        if per != man['tensor_sha256'] or dig != man['master_digest']:
            sys.exit(f'master digest mismatch: got {dig}, manifest {man["master_digest"]}')
        res['master_digest_check'] = 'ok'
    (out / 'FP32-MASTER.json').write_text(json.dumps(dict(checkpoint=head, epoch=man['epoch'], master_digest=man['master_digest'],
                                                          files={n: w['sha256'] for n, w in man['files'].items()}), indent=1))
    print(json.dumps(res))


if __name__ == '__main__':
    main()
