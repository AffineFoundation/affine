#!/usr/bin/env python3
"""Download the current head checkpoint (and optionally the KL reference) and verify it byte-for-byte.

  python3 fetch_head.py --out ./head                 # current head, the model the next turn trains from
  python3 fetch_head.py --out ./ref --reference      # the KL reference = initial checkpoint (Hugging Face base)

Chain of trust: https://affine.io/mining.json -> current_url (signed current.json) -> manifest_url (opening.json,
checked against current.json manifest_sha256) -> payload.checkpoint {files: {name: sha256}, id, read_urls}.
Every file is checked against its sha256 and the checkpoint id is recomputed as
sha256(canonical JSON of {filename: sha256}) - the same descriptor id the validator, trainer and gate use.
The links are short-lived presigned R2 GETs handed out publicly by mining.json; nothing here stores them.
The reference has no R2 links: it is the public base XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B, verified against
opening.json payload.mimo_release_policy.payload.scope.initial_checkpoint.files (id 87de8304...).
"""
import argparse, hashlib, json, os, sys, urllib.request
from pathlib import Path
MINING = 'https://affine.io/mining.json'
UA = {'User-Agent': 'affine-miner-kit/1'}


def get(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def descriptor_id(files):
    return hashlib.sha256(json.dumps(dict(sorted(files.items())), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


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


def opening():
    m = json.loads(get(MINING))
    cur = json.loads(get(m['current_url']))['payload']
    raw = get(cur['manifest_url'])
    if hashlib.sha256(raw).hexdigest() != cur['manifest_sha256']:
        sys.exit('opening.json sha256 does not match current.json manifest_sha256')
    return m, cur, json.loads(raw)['payload']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--reference', action='store_true', help='fetch the KL reference (HF base) instead of the head')
    ap.add_argument('--tokenizer-only', action='store_true', help='head tokenizer/config files only (no weights; for CPU tests)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    m, cur, op = opening()
    if a.reference:
        ck = op['mimo_release_policy']['payload']['scope']['initial_checkpoint']
        from huggingface_hub import hf_hub_download
        for name in ck['files']:
            p = Path(hf_hub_download('XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B', name, local_dir=str(out)))
            if p.resolve() != (out / name).resolve():
                os.replace(p, out / name)
    else:
        ck = op['checkpoint']
        if ck['id'] != cur['checkpoint']:
            sys.exit('opening.json checkpoint id != current.json checkpoint')
        for name, want in ck['files'].items():
            if a.tokenizer_only and name.endswith('.safetensors'):
                continue
            dst = out / name
            if dst.exists() and sha256_file(dst) == want:
                continue
            print('download', name, flush=True)
            download(ck['read_urls'][name], dst)
    bad = [n for n, want in ck['files'].items() if (out / n).exists() and sha256_file(out / n) != want]
    missing = [n for n in ck['files'] if not (out / n).exists()]
    if bad or (missing and not a.tokenizer_only):
        sys.exit(f'sha256 mismatch: {bad} missing: {missing}')
    if a.tokenizer_only:
        print(json.dumps({'ok': True, 'role': 'tokenizer', 'checkpoint': ck['id'], 'epoch': cur['epoch'], 'dir': str(out)}))
        return
    if descriptor_id(ck['files']) != ck['id']:
        sys.exit('descriptor id mismatch')
    (out / 'CHECKPOINT.json').write_text(json.dumps({'id': ck['id'], 'files': ck['files'], 'epoch': cur['epoch'],
                                                     'role': 'reference' if a.reference else 'head'}, indent=1))
    print(json.dumps({'ok': True, 'role': 'reference' if a.reference else 'head', 'checkpoint': ck['id'], 'epoch': cur['epoch'], 'dir': str(out)}))


if __name__ == '__main__':
    main()
