"""Isolated native-agent CUDA adapter for the approved 32K Qwen profile.

No environment, default task, native simulator or auxiliary runtime is created.
Different model/tokenizer/profile from the historical 135M CPU agent controls.
"""
import copy,hashlib,pathlib
from .long_context_runtime import LongContextRuntime,POLICY,file_sha,digest,wait_vram,canonical

REVISION='native-agent-qwen05-cuda-bf16-sm86-sdpa-flash-role-v1'
RENDERER='native-tau2-complete-chat-tools-v2'

def validate_descriptor(descriptor,root=None):
    root=pathlib.Path(root) if root else pathlib.Path(__file__).resolve().parent.parent
    if descriptor.get('kind')!='agent' or descriptor.get('training_eligible') is not True:raise ValueError('approved agent-only CUDA role')
    if descriptor.get('native_role_revision')!=REVISION or descriptor.get('renderer')!=RENDERER:raise ValueError('native CUDA adapter/complete renderer revision')
    if descriptor.get('max_context')!=32768 or type(descriptor.get('max_context')) is not int or descriptor.get('vocab_size')!=151936 or type(descriptor.get('vocab_size')) is not int:raise ValueError('approved Qwen role geometry')
    if type(descriptor.get('max_output_tokens')) is not int or not 1<=descriptor['max_output_tokens']<=512:raise ValueError('bounded native CUDA output')
    if canonical(descriptor.get('runtime_profile'))!=canonical({'policy':POLICY,'native_role_revision':REVISION}):raise ValueError('strict native CUDA profile')
    if canonical(descriptor.get('numerical_policy'))!=canonical({'logprobs_atol':1e-5,'logprobs_rtol':0,'TOPLOC_errors':0}):raise ValueError('strict role numerical tolerances')
    sources=descriptor.get('source_files',{})
    required=('subnet/native_role_cuda.py','subnet/long_context_runtime.py','subnet/proofs.py','subnet/native_tau2_model.py','subnet/native_tau2_probe.py','subnet/harness.py')
    if not set(required)<=set(sources):raise ValueError('native CUDA source closure')
    for name,expected in sources.items():
        if not isinstance(name,str) or name.startswith('/') or '..' in name.split('/') or not name.endswith('.py'):raise ValueError('native role source path')
        path=root/name
        if path.is_symlink() or file_sha(path)!=expected:raise ValueError('native role approved source bytes')
    if sources['subnet/harness.py']!=descriptor.get('harness_source_sha256'):raise ValueError('native role harness source binding')
    cp=descriptor.get('checkpoint',{})
    if not isinstance(cp.get('files'),dict) or not cp['files'] or cp.get('id')!=digest(cp['files']):raise ValueError('native role exact checkpoint descriptor')
    return descriptor

class NativeAgentCUDARuntime:
    def __init__(self,checkpoint,descriptor,runtime_factory=None,source_validator=None):
        validate=(source_validator or validate_descriptor)
        validate(descriptor)
        self.descriptor=copy.deepcopy(descriptor)
        if runtime_factory is None:
            wait_vram(12288,1800)
            runtime_factory=LongContextRuntime
        self.runtime=runtime_factory(checkpoint,descriptor['checkpoint']['files'])
        self.tokenizer=self.runtime.tokenizer;self.model=self.runtime.model
        self.compute=self.runtime.compute;self.build_proofs=self.runtime.build_proofs;self.verify_proofs=self.runtime.verify_proofs
        if self.model.config.vocab_size!=151936 or self.model.config.max_position_embeddings!=32768:raise ValueError('actual native CUDA model geometry')
    def approved_descriptor(self):return copy.deepcopy(self.descriptor)
    def profile(self):return {**self.runtime.profile(),'native_role_revision':REVISION,'native_role_source_sha256':file_sha(__file__)}
    def render(self,request):
        from .native_tau2_model import render
        return render(self.tokenizer,request)
    def sample(self,prompt,seed,temperature,top_p,max_tokens):
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        if type(seed) is not int or seed<0 or type(max_tokens) is not int or not 1<=max_tokens<=self.descriptor['max_output_tokens'] or temperature<=0 or not 0<top_p<=1 or len(prompt)+max_tokens>32768:raise ValueError('native CUDA signed sampling budget')
        generator=torch.Generator(device='cuda').manual_seed(seed);output=[]
        with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for _ in range(max_tokens):
                hidden=self.model.base_model(torch.tensor([prompt+output],device='cuda'),use_cache=False).last_hidden_state[0,-1:]
                probabilities=torch.softmax(self.model.lm_head(hidden)[0].float()/temperature,-1)
                if top_p<1:
                    values,ids=probabilities.sort(descending=True);remove=values.cumsum(-1)>top_p;remove[1:]=remove[:-1].clone();remove[0]=False;values[remove]=0;probabilities=torch.zeros_like(probabilities).scatter(0,ids,values)
                token=int(torch.multinomial(probabilities,1,generator=generator));output.append(token)
                if token==self.tokenizer.eos_token_id:break
        return output
