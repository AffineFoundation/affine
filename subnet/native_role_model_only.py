"""Prospective model-only CPU role loader; no environment or harness import.

Numerical operations preserve qualified CPU v2 semantics, but this separate
source revision requires fresh honest/full-LP/TOPLOC qualification before use.
"""
import hashlib,re
from pathlib import Path

REVISION='native-role-model-only-cpu-float32-eager-toploc4-v1'
SAFE_SUFFIXES={'.json','.safetensors','.txt','.model','.jinja','.tiktoken'}
MODEL_SUFFIXES=SAFE_SUFFIXES|{'.bin','.pt'}
EXECUTABLE_SUFFIXES={'.py','.pyc','.so','.dll','.dylib','.sh'}

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def validate_checkpoint(checkpoint,files):
    if not isinstance(files,dict) or not files:raise ValueError('empty or invalid checkpoint allowlist')
    for name,value in files.items():
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*',name) or Path(name).name!=name or Path(name).suffix not in SAFE_SUFFIXES:raise ValueError('unsafe model checkpoint filename/type')
        if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('checkpoint SHA256 pin')
    if 'config.json' not in files or not any(n.endswith('.safetensors') for n in files):raise ValueError('config and safetensors required')
    root=Path(checkpoint)
    if root.is_symlink() or not root.is_dir():raise ValueError('real checkpoint directory required')
    actual={}
    for path in root.iterdir():
        if path.is_symlink() or path.is_dir():raise ValueError('checkpoint must contain flat regular files')
        if path.suffix in EXECUTABLE_SUFFIXES:raise ValueError('uploaded executable checkpoint content')
        if path.suffix in MODEL_SUFFIXES:
            if path.is_symlink() or not path.is_file():raise ValueError('model-relevant checkpoint file is not regular')
            actual[path.name]=file_hash(path)
    if set(actual)!=set(files):raise ValueError('unexpected or missing model-relevant checkpoint files')
    if actual!=files:raise ValueError('untrusted checkpoint hash')
    return root

class ModelOnlyCPURuntime:
    def __init__(self,checkpoint,files,threads=4):
        if type(threads) is not int or threads!=4:raise ValueError('approved four-thread CPU role policy')
        root=validate_checkpoint(checkpoint,files)
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        torch.set_num_threads(threads)
        self.tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(root,local_files_only=True,dtype=torch.float32,attn_implementation='eager',trust_remote_code=False,use_safetensors=True).eval()
        from toploc import build_proofs_base64,verify_proofs_base64
        self.build_proofs,self.verify_proofs=build_proofs_base64,verify_proofs_base64
        import toploc.poly as toploc_poly
        from toploc.C.csrc.utils import get_fp_parts
        self.toploc_threads=threads
        toploc_poly.get_fp_parts=lambda tensor:get_fp_parts(tensor,num_threads=threads)
        self.files=dict(files);self.checkpoint=root;self.revision=REVISION

    def compute(self,prompt,output):
        # Exact forward, hidden cast/contiguity, prediction-row slicing, and
        # activation segmentation from the qualified CPU Runtime.compute.
        import torch
        with torch.inference_mode():
            result=self.model(torch.tensor([prompt+output]),output_hidden_states=True,use_cache=False)
            hidden=result.hidden_states[-1][0].to(torch.bfloat16).contiguous()
            logprobs=torch.log_softmax(result.logits[0,len(prompt)-1:len(prompt)+len(output)-1].float(),-1).numpy()
        acts=[hidden[:len(prompt)]]+[hidden[i:i+1] for i in range(len(prompt),len(prompt)+len(output))]
        return acts,logprobs
