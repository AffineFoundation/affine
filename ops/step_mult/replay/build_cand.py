#!/usr/bin/env python3
"""RA/20261102-step-mult pod side: build step-multiplier candidates from published fp32 masters.
For a GREEN turn H -> C (both fp32 masters published), the live step is M_C = M_H - Delta with Delta_t = s * n_t * g_t/||g_t||.
Under the proposed rule (step multiplier m scales lr and the per-tensor cap together; global cap 1.0 never fires: ||n|| ~ 0.023,
x8 = 0.18) the step is exactly m * Delta, so M_C(m) = M_H + m * (M_C - M_H); candidate = bf16_rne(M_C(m)) on the trainable tensors,
all other tensors from the bf16 template (never trained). m=0 / m=1 reproduce the published heads (checked via bf16_sha256).
usage: build_cand.py H C OUTBASE TEMPLATE_DIR MANIFEST_DIR m[,m...]"""
import hashlib, json, os, shutil, sys, time
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
H, C, OUTB, TPL, MAN, MS = sys.argv[1:7]
ms = [float(x) for x in MS.split(',')]
def load_master(ck):
    d = {}
    for f in sorted(Path('/data/smult/fp32', ck).glob('*.safetensors')): d.update(load_file(str(f)))
    return d
t0 = time.time(); MH = load_master(H); MC = load_master(C); print('loaded masters', len(MH), len(MC), '%.0fs' % (time.time() - t0), flush=True)
assert set(MH) == set(MC)
man = {ck: json.load(open(os.path.join(MAN, ck + '.json'))) for ck in (H, C)}
def bf16_sha(tensors):
    m = {n: hashlib.sha256(t.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest() for n, t in tensors.items()}
    return {repr(s): hashlib.sha256(json.dumps(m, sort_keys=True, separators=s).encode()).hexdigest() for s in ((',', ':'), (', ', ': '))}
tpl_shards = sorted(Path(TPL).glob('model-*.safetensors'))
for m in ms:
    tag = ('%g' % m)
    out = Path(OUTB + '-m' + tag); tmp = Path(str(out) + '.partial')
    if (out / '.verified').exists(): print('exists', out); continue
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    t1 = time.time(); cand = {}
    for n in MH:
        mh = MH[n].float(); mc = MC[n].float()
        cand[n] = (mh + m * (mc - mh)).to(torch.bfloat16) if m != 0 else mh.to(torch.bfloat16)
    sh = bf16_sha(cand); check = None
    if m == 0: check = man[H]['bf16_sha256'] in sh.values()
    if m == 1: check = man[C]['bf16_sha256'] in sh.values()
    # per-tensor relative step norm stats (fp32)
    num = sum(float(((MC[n].double() - MH[n].double()) * m).pow(2).sum()) for n in MH) ** 0.5
    for f in tpl_shards:
        ts = load_file(str(f)); hit = 0
        for n in list(ts):
            if n in cand: ts[n] = cand[n]; hit += 1
        save_file(ts, str(tmp / f.name), metadata={'format': 'pt'})
    for f in Path(TPL).iterdir():
        if f.is_file() and not f.name.startswith('model-') and not f.name.endswith(('.ok', '.part')):
            shutil.copy(f, tmp / f.name)
    meta = dict(H=H, C=C, m=m, master_step_norm=num, bf16_sha=sh, bf16_check=check, seconds=time.time() - t1)
    (tmp / 'smult-meta.json').write_text(json.dumps(meta, indent=1)); (tmp / '.verified').write_text('ok')
    if out.exists(): shutil.rmtree(out)
    os.replace(tmp, out)
    print(json.dumps(dict(out=str(out), m=m, step_norm=num, bf16_check=check, s=round(time.time() - t1))), flush=True)
