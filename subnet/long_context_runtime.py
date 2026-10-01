"""Separate, pinned CUDA long-context computation; never a live runtime selector."""
import base64,hashlib,json,os,pathlib,time

REVISION='cuda-bf16-sdpa-flash-sm86-selective-head-v1'
AUTHORITY='d54a3a345d0de3e2c7898f30c0942d78f931f8c4b8036ffdc6adffcd2525062f'
POLICY={'revision':REVISION,'dtype':'bfloat16','device_capability':[8,6],
        'attention_implementation':'sdpa','sdpa_backend':'FLASH_ATTENTION',
        'deterministic_algorithms':True,'tf32':False,'use_cache':False,
        'max_context':32768,'lm_head':'prediction-rows-only-full-vocabulary',
        'activations':'all-last-layer-rows-bfloat16','toploc_threads':2,
        'logprob_atol':1e-5,'logprob_rtol':0.,'toploc_errors':0}

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def authenticate(envelope,authority):
    from nacl.signing import VerifyKey
    if authority!=AUTHORITY or envelope.get('signer')!=authority:raise ValueError('long-context operator authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(envelope['payload']),base64.b64decode(envelope['signature'],validate=True))
    return envelope['payload']
def runtime_environment():
    from importlib.metadata import version
    import sys
    return {'python_version':sys.version,'interpreter_sha256':file_sha(pathlib.Path(sys.executable).resolve()),'packages':{n:version(n) for n in ('torch','transformers','toploc','numpy')}}
def validate_job(envelope,authority):
    job=authenticate(envelope,authority) # Before job-specified artifacts or model reads.
    if job.get('role')!='long-context-proof-probe' or job.get('policy')!=POLICY or job.get('payable') is not False or job.get('chain_transactions') is not False:raise ValueError('long-context role/profile')
    if job.get('runtime_source_sha256')!=file_sha(__file__):raise ValueError('long-context source closure')
    if job.get('min_free_vram_mib')!=12288 or job.get('wait_seconds')!=1800:raise ValueError('long-context VRAM policy')
    cp=job.get('checkpoint',{})
    if cp.get('model_id')!='Qwen/Qwen2.5-0.5B-Instruct' or cp.get('revision')!='7ae557604adf67be50417f59c2c2f167def9a775' or not cp.get('files'):raise ValueError('long-context approved model revision')
    if job.get('proof_helper_sha256')!=file_sha(pathlib.Path(__file__).parent/'proofs.py'):raise ValueError('long-context proof helper source')
    if job.get('runtime_environment')!=runtime_environment():raise ValueError('long-context interpreter/packages')
    return job
def validate_tokens(prompt,output,vocab,context=32768):
    if not isinstance(prompt,list) or not isinstance(output,list) or not prompt or not output or len(prompt)+len(output)>context:raise ValueError('long-context token/context framing')
    if any(type(t)!=int or not 0<=t<vocab for t in prompt+output):raise ValueError('long-context token ids')
def prediction_rows(hidden,prompt_length,output_length):
    return hidden[prompt_length-1:prompt_length+output_length-1]
def wait_vram(minimum=12288,seconds=1800):
    import subprocess
    until=time.monotonic()+seconds
    while True:
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if free>=minimum:return free
        if time.monotonic()>=until:raise TimeoutError('long-context free VRAM guard; existing jobs preserved')
        time.sleep(5)

class LongContextRuntime:
    def __init__(self,checkpoint,files):
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('long-context CUBLAS profile')
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(8,6):raise ValueError('long-context approved CUDA device')
        checkpoint=pathlib.Path(checkpoint)
        if {p.name:file_sha(p) for p in checkpoint.iterdir() if p.is_file()}!=files:raise ValueError('long-context checkpoint exact files')
        self.checkpoint_files=dict(files)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        self.tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
        if self.model.config.max_position_embeddings!=32768 or self.model.config.attention_dropout!=0 or self.model.config._attn_implementation!='sdpa':raise ValueError('long-context model configuration')
        from toploc import build_proofs_base64,verify_proofs_base64
        from toploc.C.csrc.utils import get_fp_parts
        import toploc.poly as poly
        poly.get_fp_parts=lambda tensor:get_fp_parts(tensor,num_threads=2)
        self.build_proofs,self.verify_proofs=build_proofs_base64,verify_proofs_base64
    def profile(self):
        import torch
        from importlib.metadata import version
        return {'policy':POLICY,'runtime_source_sha256':file_sha(__file__),
                'proof_helper_sha256':file_sha(pathlib.Path(__file__).parent/'proofs.py'),
                'checkpoint_files':self.checkpoint_files,'checkpoint_id':digest(self.checkpoint_files),
                'python_version':__import__('sys').version,'interpreter_sha256':file_sha(pathlib.Path(__import__('sys').executable).resolve()),
                'packages':{n:version(n) for n in ('torch','transformers','toploc')},
                'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
                'environment':{n:os.environ.get(n) for n in ('CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM')}}
    def compute(self,prompt,output):
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        validate_tokens(prompt,output,self.model.config.vocab_size)
        # All prefix rows enter the transformer. Only prediction rows enter the
        # output head; no tool/message/token truncation and no prefix-vocab tensor.
        with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            result=self.model.base_model(torch.tensor([prompt+output],device='cuda'),use_cache=False)
            hidden=result.last_hidden_state[0]
            selected=prediction_rows(hidden,len(prompt),len(output))
            logits=self.model.lm_head(selected)
            lp=torch.log_softmax(logits.float(),-1).cpu().numpy()
            full_hidden=hidden.to(torch.bfloat16).cpu().contiguous()
        acts=[full_hidden[:len(prompt)]]+[full_hidden[i:i+1] for i in range(len(prompt),len(prompt)+len(output))]
        return acts,lp
    def greedy(self,prompt,count):
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        if type(count)!=int or count<1 or len(prompt)+count>32768:raise ValueError('long-context generation budget')
        output=[]
        with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for _ in range(count):
                hidden=self.model.base_model(torch.tensor([prompt+output],device='cuda'),use_cache=False).last_hidden_state[0,-1:]
                output.append(int(self.model.lm_head(hidden).argmax(-1)))
                if output[-1]==self.tokenizer.eos_token_id:break
        return output
    def verify(self,artifact,claimed):
        import numpy as np,math
        from .proofs import validate_framing
        if artifact.get('profile')!=self.profile():raise ValueError('long-context runtime profile binding')
        acts,actual=self.compute(artifact['prompt'],artifact['output'])
        if claimed.dtype!=np.float32 or claimed.shape!=actual.shape or not np.isfinite(claimed).all() or not np.allclose(actual,claimed,atol=1e-5,rtol=0):raise ValueError('long-context log probabilities')
        validate_framing(artifact['proofs'],1+math.ceil(len(artifact['output'])/16))
        checks=self.verify_proofs(acts,artifact['proofs'],decode_batching_size=16,topk=128)
        if any(r.exp_mismatches or r.mant_err_mean or r.mant_err_median for r in checks):raise ValueError('long-context TOPLOC')
        return True
