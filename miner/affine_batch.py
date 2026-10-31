#!/usr/bin/env python3
"""Affine SN120 batch-file helper (reference, Python standard library only). NOT a mining client.

  python3 affine_batch.py link     --hotkey <ss58> --key <seed file>       # fetch + open your private upload link
  python3 affine_batch.py check    --hotkey <ss58> batch.json              # local format check (structure + caps)
  python3 affine_batch.py upload   --hotkey <ss58> --key <seed file> batch.json
  python3 affine_batch.py download --hotkey <ss58> --key <seed file> out.json   # read back what is stored
  python3 affine_batch.py status   --hotkey <ss58>                         # the validator's format check of your stored file

--key: a file holding your activated hotkey's 32-byte Ed25519 seed as 64 hex characters (the same key file the old miner used).
Your link is https://affine.io/ff/links/<hotkey>.json: a SealedBox (libsodium crypto_box_seal) sealed to your hotkey's Ed25519
public key (converted to X25519). Only your seed opens it. Format and rules: https://affine.io/mining.json -> batch_file.
"""
import argparse, base64, hashlib, json, math, sys, time, urllib.request

SITE = 'https://affine.io'
FORMAT = 'affine-batch-v1'

# ---- libsodium crypto_box_seal_open in pure Python (X25519 + XSalsa20-Poly1305, nonce = blake2b(epk || pk, 24)) ----
P = 2 ** 255 - 19


def _x25519(k, u):
    k = bytearray(k); k[0] &= 248; k[31] &= 127; k[31] |= 64
    k = int.from_bytes(k, 'little'); u = int.from_bytes(u, 'little') & ((1 << 255) - 1)
    x1, x2, z2, x3, z3, swap = u, 1, 0, u, 1, 0
    for t in reversed(range(255)):
        b = (k >> t) & 1; swap ^= b
        if swap: x2, x3, z2, z3 = x3, x2, z3, z2
        swap = b
        a, aa = x2 + z2, (x2 + z2) ** 2 % P; bb_, bb = x2 - z2, (x2 - z2) ** 2 % P
        e = aa - bb; c, d = x3 + z3, x3 - z3
        da, cb = d * a % P, c * bb_ % P
        x3, z3 = (da + cb) ** 2 % P, x1 * (da - cb) ** 2 % P
        x2, z2 = aa * bb % P, e * (aa + 121665 * e) % P
    if swap: x2, z2 = x3, z3
    return (x2 * pow(z2, P - 2, P) % P).to_bytes(32, 'little')


def _rotl(v, c):
    return ((v << c) & 0xffffffff) | (v >> (32 - c))


def _salsa_rounds(x):
    for _ in range(10):
        for a, b, c, d in ((0, 4, 8, 12), (5, 9, 13, 1), (10, 14, 2, 6), (15, 3, 7, 11),
                           (0, 1, 2, 3), (5, 6, 7, 4), (10, 11, 8, 9), (15, 12, 13, 14)):
            x[b] ^= _rotl((x[a] + x[d]) & 0xffffffff, 7); x[c] ^= _rotl((x[b] + x[a]) & 0xffffffff, 9)
            x[d] ^= _rotl((x[c] + x[b]) & 0xffffffff, 13); x[a] ^= _rotl((x[d] + x[c]) & 0xffffffff, 18)
    return x


def _words(b):
    return [int.from_bytes(b[i:i + 4], 'little') for i in range(0, len(b), 4)]


SIGMA = _words(b'expand 32-byte k')


def _state(key, n16):
    k, n = _words(key), _words(n16)
    return [SIGMA[0], k[0], k[1], k[2], k[3], SIGMA[1], n[0], n[1], n[2], n[3], SIGMA[2], k[4], k[5], k[6], k[7], SIGMA[3]]


def _hsalsa20(key, n16):
    x = _salsa_rounds(_state(key, n16))
    return b''.join(x[i].to_bytes(4, 'little') for i in (0, 5, 10, 15, 6, 7, 8, 9))


def _salsa20_stream(key, n8, length):
    out = bytearray()
    for counter in range((length + 63) // 64):
        s = _state(key, n8 + counter.to_bytes(8, 'little'))
        x = _salsa_rounds(list(s))
        out += b''.join(((x[i] + s[i]) & 0xffffffff).to_bytes(4, 'little') for i in range(16))
    return bytes(out[:length])


def _poly1305(key, msg):
    r = int.from_bytes(key[:16], 'little') & 0x0ffffffc0ffffffc0ffffffc0fffffff; s = int.from_bytes(key[16:], 'little')
    p, acc = (1 << 130) - 5, 0
    for i in range(0, len(msg), 16):
        acc = (acc + int.from_bytes(msg[i:i + 16] + b'\x01', 'little')) * r % p
    return ((acc + s) % (1 << 128)).to_bytes(16, 'little')


def seal_open(sealed, seed):
    sk = hashlib.sha512(seed).digest()[:32]                     # Ed25519 seed -> X25519 secret (libsodium conversion)
    pk = _x25519(sk, (9).to_bytes(32, 'little'))
    epk, box = sealed[:32], sealed[32:]
    nonce = hashlib.blake2b(epk + pk, digest_size=24).digest()
    k = _hsalsa20(_x25519(sk, epk), bytes(16))                  # crypto_box_beforenm
    sub = _hsalsa20(k, nonce[:16])
    stream = _salsa20_stream(sub, nonce[16:], 32 + len(box) - 16)
    mac, ct = box[:16], box[16:]
    if _poly1305(stream[:32], ct) != mac:
        raise SystemExit('cannot open the link: wrong key file for this hotkey')
    return bytes(a ^ b for a, b in zip(ct, stream[32:]))


# ---- the protocol ----
def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'affine-batch/1'}), timeout=60) as r:
        return json.loads(r.read())


def seed_of(path):
    raw = open(path).read().strip()
    if len(raw) != 64:
        raise SystemExit('key file must hold the 32-byte Ed25519 seed as 64 hex characters')
    return bytes.fromhex(raw)


def link(hotkey, key):
    doc = get_json('%s/ff/links/%s.json' % (SITE, hotkey))
    value = json.loads(seal_open(base64.b64decode(doc['sealed']), seed_of(key)))
    if value.get('hotkey') != hotkey or value['expires_at'] < time.time() + 60:
        raise SystemExit('link is for another hotkey or expired; links are re-issued every 30 minutes')
    return value


def check(raw, hotkey, caps):
    """Local structural check (the validator also tokenizes: <= 8192 output tokens, <= 12288 prompt+output, no chat
    control tokens). Returns (ok, message)."""
    try:
        v = json.loads(raw.decode('utf-8'), parse_constant=lambda c: (_ for _ in ()).throw(ValueError('non-finite number')))
    except Exception as e:
        return False, 'not UTF-8 JSON: %s' % e
    if not isinstance(v, dict) or set(v) != {'version', 'hotkey', 'groups'}:
        return False, 'top level keys must be exactly version, hotkey, groups'
    if v['version'] != FORMAT: return False, 'version must be ' + FORMAT
    if v['hotkey'] != hotkey: return False, 'hotkey does not match'
    gs = v['groups']
    if not isinstance(gs, list) or not gs: return False, 'groups must be a non-empty list'
    seen, n, kept = set(), 0, 0
    for i, g in enumerate(gs):
        if not isinstance(g, dict) or set(g) != {'prompt', 'rollouts'}: return False, 'groups[%d] keys must be prompt, rollouts' % i
        if not isinstance(g['prompt'], str) or not g['prompt'].strip(): return False, 'groups[%d].prompt must be a non-empty string' % i
        if g['prompt'] in seen: return False, 'groups[%d].prompt duplicates an earlier prompt' % i
        seen.add(g['prompt'])
        rs = g['rollouts']
        if not isinstance(rs, list) or not rs: return False, 'groups[%d].rollouts must be a non-empty list' % i
        for j, r in enumerate(rs):
            if not isinstance(r, dict) or set(r) != {'output', 'reward'}: return False, 'groups[%d].rollouts[%d] keys must be output, reward' % (i, j)
            if not isinstance(r['output'], str): return False, 'groups[%d].rollouts[%d].output must be a string' % (i, j)
            w = r['reward']
            if type(w) not in (int, float) or not math.isfinite(w): return False, 'groups[%d].rollouts[%d].reward must be a finite number' % (i, j)
            for bad in ('<|im_start|>', '<|im_end|>', '<|endoftext|>'):
                if bad in r['output'] or bad in g['prompt']: return False, 'groups[%d]: %s is not allowed' % (i, bad)
        if kept < caps['max_tasks_per_miner'] and n + len(rs) <= caps['max_rollouts_per_miner']:
            kept += 1; n += len(rs)
        elif kept == 0:
            return False, 'the first group alone exceeds the rollout cap'
    return True, 'ok: %d prompts, %d rollouts will be trained (cap %d prompts / %d rollouts)%s' % (
        kept, n, caps['max_tasks_per_miner'], caps['max_rollouts_per_miner'], '' if kept == len(gs) else '; %d prompts beyond the cap are dropped' % (len(gs) - kept))


def caps_now():
    m = get_json(SITE + '/mining.json')
    return m['training']['batch_cap']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=('link', 'check', 'upload', 'download', 'status'))
    ap.add_argument('file', nargs='?')
    ap.add_argument('--hotkey', required=True)
    ap.add_argument('--key')
    a = ap.parse_args()
    if a.command == 'link':
        v = link(a.hotkey, a.key)
        print(json.dumps(dict(object_key=v['object_key'], expires_at=v['expires_at']), indent=1)); return
    if a.command == 'status':
        st = get_json(SITE + '/ff/status.json')
        print(json.dumps(st['miners'].get(a.hotkey, 'no file stored'), indent=1)); return
    if a.command == 'download':
        v = link(a.hotkey, a.key)
        with urllib.request.urlopen(v['get_url'], timeout=300) as r:
            open(a.file, 'wb').write(r.read())
        print('saved', a.file); return
    raw = open(a.file, 'rb').read()
    ok, msg = check(raw, a.hotkey, caps_now())
    print(msg)
    if not ok:
        sys.exit(1)
    if a.command == 'upload':
        v = link(a.hotkey, a.key)
        req = urllib.request.Request(v['put_url'], data=raw, method='PUT', headers=v['put_headers'])
        with urllib.request.urlopen(req, timeout=600) as r:
            print('uploaded %d bytes (HTTP %d). The validator checks it within a minute: affine_batch.py status --hotkey %s'
                  % (len(raw), r.status, a.hotkey))


if __name__ == '__main__':
    main()
