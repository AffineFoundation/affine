#!/usr/bin/env python3
"""Trainer-side, weights-only merge using authenticated fp32 master inputs.

Pure filesystem/tensor operation: no signer, credentials, R2, CURRENT mutation,
optimizer update, or live install. The publisher/controller must verify the
input master digests before calling, then publish the returned model inventory
through the admitted publisher and create a restorable generation/keep.

Unlike converting bf16 files to float(), this preserves sub-bf16 master deltas.
"""
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<24),b''): h.update(b)
    return h.hexdigest()


def inventory(root):
    return {p.name:dict(size=p.stat().st_size,sha256=sha(p)) for p in sorted(Path(root).iterdir()) if p.is_file()}


def merge(head_model, candidate_model, training_model, head_master, candidate_master,
          training_master, out_model, out_master):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    def tmap(root):
        result={}
        for path in sorted(Path(root).glob('*.safetensors')):
            if path.is_symlink(): raise ValueError('symlink input shard')
            with safe_open(str(path),'pt') as f:
                for key in f.keys():
                    if key in result: raise ValueError('duplicate tensor '+key)
                    result[key]=path
        if not result: raise ValueError('no weight tensors')
        return result

    def get(m,key):
        with safe_open(str(m[key]),'pt') as f: return f.get_tensor(key)

    inputs=list(map(Path,(head_model,candidate_model,training_model,head_master,candidate_master,training_master)))
    H,C,T,HM,CM,TM=map(tmap,inputs)
    if not (H.keys()==C.keys()==T.keys()): raise ValueError('model tensor sets differ')
    if not (HM.keys()==CM.keys()==TM.keys()): raise ValueError('master tensor sets differ')
    if not set(HM).issubset(H): raise ValueError('master tensor absent from model')
    outputs=[Path(out_model),Path(out_master)]
    if outputs[0].resolve()==outputs[1].resolve(): raise ValueError('model/master output alias')
    for out in outputs:
        for inp in inputs:
            if out.resolve()==inp.resolve() or out.resolve().is_relative_to(inp.resolve()) or inp.resolve().is_relative_to(out.resolve()):
                raise ValueError('input/output paths overlap')
    # Unique temp dirs; completed destinations are immutable and idempotent.
    tmps=[p.with_name('.'+p.name+'.pe-%d.tmp'%os.getpid()) for p in outputs]
    for p in tmps:
        if p.exists(): raise FileExistsError(str(p))
        p.mkdir(parents=True)
    stats=dict(tensors=0,master_tensors=0,frozen_tensors=0)
    try:
        for shard in sorted(Path(head_model).glob('*.safetensors')):
            dst={}; masters={}
            with safe_open(str(shard),'pt') as sf:
                metadata=sf.metadata()
                for key in sf.keys():
                    h,c,t=get(H,key),get(C,key),get(T,key)
                    if not (h.shape==c.shape==t.shape and h.dtype==c.dtype==t.dtype):
                        raise ValueError('model shape/dtype mismatch '+key)
                    if key in HM:
                        hs,cs,ts=get(HM,key),get(CM,key),get(TM,key)
                        for master,bf16 in [(hs,h),(cs,c),(ts,t)]:
                            if master.dtype!=torch.float32 or master.shape!=bf16.shape or not torch.isfinite(master).all():
                                raise ValueError('non-fp32/nonfinite/shape master '+key)
                            if not torch.equal(master.to(bf16.dtype),bf16):
                                raise ValueError('master does not round to model '+key)
                        value=hs+(cs-ts)
                        if not torch.isfinite(value).all(): raise ValueError('nonfinite merge '+key)
                        dst[key]=value.to(h.dtype).contiguous(); masters[key]=value.contiguous()
                        stats['master_tensors']+=1
                    else:
                        if not torch.equal(c,t): raise ValueError('changed tensor missing fp32 master '+key)
                        dst[key]=h.contiguous(); stats['frozen_tensors']+=1
                    stats['tensors']+=1
            save_file(dst,str(tmps[0]/shard.name),metadata=metadata)
            if masters: save_file(masters,str(tmps[1]/shard.name),metadata={'format':'pt'})
        # Preserve head's model topology/tokenizer; config must not silently come
        # from an unrelated candidate. Only model-only ordinary files are copied.
        for f in sorted(Path(head_model).iterdir()):
            if f.is_symlink(): raise ValueError('symlink model file')
            if f.is_file() and not f.name.endswith('.safetensors') and not f.name.startswith('.'):
                if f.suffix not in ('.json','.txt','.model','.jinja','.tiktoken') or 'optimizer' in f.name.lower():
                    raise ValueError('non-model file '+f.name)
                shutil.copyfile(f,tmps[0]/f.name)
        inventories=[inventory(p) for p in tmps]
        # Check BOTH destinations before renaming either; never remove/overwrite
        # an existing output. A crash between renames is safely rerunnable.
        for dst,inv in zip(outputs,inventories):
            if dst.exists() and inventory(dst)!=inv: raise FileExistsError('different output at '+str(dst))
        for tmp,dst in zip(tmps,outputs):
            if dst.exists(): shutil.rmtree(tmp)
            else: os.rename(tmp,dst)
        files={k:v['sha256'] for k,v in inventories[0].items()}
        checkpoint=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        return dict(checkpoint=checkpoint,model_files=inventories[0],master_files=inventories[1],stats=stats,
                    formula='head_fp32 + (candidate_fp32 - training_head_fp32)',published=False)
    finally:
        for p in tmps:
            if p.exists(): shutil.rmtree(p)
