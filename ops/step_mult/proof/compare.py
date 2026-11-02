import hashlib, json, sys
from pathlib import Path
import torch
from safetensors.torch import load_file
P = Path('/data/smult/prove'); names = sys.argv[1:]
def fsha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(d):
    t = {}
    for f in sorted((P / ('out-' + d)).glob('model*.safetensors')): t.update(load_file(str(f)))
    return t, load_file(str(P / ('out-' + d) / 'fp32-master.safetensors'))
info = {}
for n in names:
    o = P / ('out-' + n)
    files = {p.name: fsha(p) for p in sorted(o.iterdir()) if p.is_file() and not p.name.startswith('.')}
    rep = json.loads((P / ('rep-' + n + '.json')).read_text())
    info[n] = dict(model_files={k: v for k, v in files.items() if k.startswith('model')}, fp32_master_sha256=files.get('fp32-master.safetensors'),
                   train_params=rep.get('train_params') if isinstance(rep, dict) else None, report_keys=sorted(rep)[:30] if isinstance(rep, dict) else None)
base_b, base_m = load(names[0]); res = {}
for n in names[1:]:
    b, m = load(n)
    res[n + '_vs_' + names[0]] = dict(
        same_tensor_names=set(b) == set(base_b) and set(m) == set(base_m),
        bf16_identical=all(torch.equal(b[k].view(torch.uint16) if b[k].dtype == torch.bfloat16 else b[k], base_b[k].view(torch.uint16) if base_b[k].dtype == torch.bfloat16 else base_b[k]) for k in base_b),
        fp32_identical=all(torch.equal(m[k].view(torch.int32), base_m[k].view(torch.int32)) for k in base_m),
        tensors_differing_bf16=sum(not torch.equal(b[k], base_b[k]) for k in base_b),
        model_file_shas_equal=info[n]['model_files'] == info[names[0]]['model_files'],
        fp32_master_file_sha_equal=info[n]['fp32_master_sha256'] == info[names[0]]['fp32_master_sha256'])
ok = all(v['bf16_identical'] and v['fp32_identical'] for v in res.values())
print(json.dumps(dict(runs=info, compare=res, all_identical=ok), indent=1, default=str))
