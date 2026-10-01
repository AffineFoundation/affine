"""New numerical approval for fixed-weight 135M native user CUDA computation.

Never claims CPU numerical equivalence or creates a native environment/session.
"""
import copy,math,os,pathlib
from .long_context_runtime import LongContextRuntime,file_sha,digest,canonical,wait_vram,validate_tokens,prediction_rows
REVISION='native-fixed-user-smol135-cuda-bf16-sm86-sdpa-flash-role-v1'
POLICY={'revision':REVISION,'dtype':'bfloat16','device_capability':[8,6],'attention_implementation':'sdpa','sdpa_backend':'FLASH_ATTENTION','deterministic_algorithms':True,'tf32':False,'use_cache':False,'max_context':8192,'vocab_size':49152,'lm_head':'prediction-rows-only-full-vocabulary','activations':'all-last-layer-rows-bfloat16','toploc_threads':2,'logprob_atol':1e-5,'logprob_rtol':0.,'toploc_errors':0}
CHECKPOINT='39818e714a6e4e47b3fdd07e4eeb9cac619cf010fcc83a30068708531eac7d06'

def validate_descriptor(descriptor,root=None):
    root=pathlib.Path(root) if root else pathlib.Path(__file__).resolve().parent.parent
    if descriptor.get('kind')!='auxiliary' or descriptor.get('training_eligible') is not False:raise ValueError('fixed auxiliary-only loss exclusion')
    expected={'native_role_revision':REVISION,'model_runtime_revision':REVISION,'max_context':8192,'vocab_size':49152,'renderer':'native-tau2-complete-chat-tools-v2','runtime_profile':{'policy':POLICY,'native_role_revision':REVISION},'numerical_policy':{'logprobs_atol':1e-5,'logprobs_rtol':0,'TOPLOC_errors':0}}
    if any(canonical(descriptor.get(k))!=canonical(v) for k,v in expected.items()):raise ValueError('explicit auxiliary CUDA numerical profile')
    if type(descriptor.get('max_output_tokens')) is not int or not 1<=descriptor['max_output_tokens']<=512:raise ValueError('auxiliary output budget')
    cp=descriptor.get('checkpoint',{})
    if cp.get('id')!=CHECKPOINT or digest(cp.get('files'))!=CHECKPOINT:raise ValueError('immutable fixed auxiliary checkpoint')
    sources=descriptor.get('source_files',{})
    if not {'subnet/native_auxiliary_cuda.py','subnet/long_context_runtime.py','subnet/proofs.py','subnet/native_tau2_model.py','subnet/harness.py'}<=set(sources):raise ValueError('auxiliary source closure')
    for name,expected_sha in sources.items():
        if not isinstance(name,str) or name.startswith('/') or '..' in name.split('/') or not name.endswith('.py'):raise ValueError('source path')
        path=root/name
        if path.is_symlink() or file_sha(path)!=expected_sha:raise ValueError('approved source bytes')
    if sources['subnet/harness.py']!=descriptor.get('harness_source_sha256'):raise ValueError('auxiliary complete renderer binding')
    return descriptor

class NativeAuxiliaryCUDARuntime(LongContextRuntime):
    def __init__(self,checkpoint,descriptor):
        validate_descriptor(descriptor);self.descriptor=copy.deepcopy(descriptor)
        if any(os.environ.get(k)!=v for k,v in {'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}.items()):raise ValueError('isolated auxiliary CUDA environment')
        wait_vram(12288,1800)
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(8,6):raise ValueError('approved auxiliary CUDA device')
        directory=pathlib.Path(checkpoint);files=descriptor['checkpoint']['files']
        if any(p.is_symlink() or not p.is_file() for p in directory.iterdir()) or {p.name:file_sha(p) for p in directory.iterdir()}!=files:raise ValueError('fixed auxiliary exact checkpoint files')
        self.checkpoint_files=dict(files)
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
        self.tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(directory,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
        config=self.model.config
        if config.vocab_size!=49152 or config.max_position_embeddings!=8192 or config.attention_dropout!=0 or config._attn_implementation!='sdpa':raise ValueError('actual fixed auxiliary model geometry')
        from toploc import build_proofs_base64,verify_proofs_base64
        from toploc.C.csrc.utils import get_fp_parts
        import toploc.poly as poly
        poly.get_fp_parts=lambda tensor:get_fp_parts(tensor,num_threads=2)
        self.build_proofs,self.verify_proofs=build_proofs_base64,verify_proofs_base64
    def approved_descriptor(self):return copy.deepcopy(self.descriptor)
    def profile(self):
        from .long_context_runtime import runtime_environment
        import torch
        return {'policy':POLICY,'runtime_source_sha256':file_sha(__file__),'proof_helper_sha256':file_sha(pathlib.Path(__file__).parent/'proofs.py'),'checkpoint_files':self.checkpoint_files,'checkpoint_id':CHECKPOINT,'runtime_environment':runtime_environment(),'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),'environment':{n:os.environ.get(n) for n in ('CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM')}}
    def render(self,request):
        from .native_tau2_model import render
        return render(self.tokenizer,request)
    def compute(self,prompt,output):
        validate_tokens(prompt,output,49152,8192)
        return super().compute(prompt,output)
    def sample(self,prompt,seed,temperature,top_p,max_tokens):
        import torch
        from torch.nn.attention import sdpa_kernel,SDPBackend
        if type(seed) is not int or not 0<=seed<2**63 or type(max_tokens) is not int or not 1<=max_tokens<=self.descriptor['max_output_tokens'] or len(prompt)+max_tokens>8192 or any(type(v) not in (int,float) or not math.isfinite(v) for v in (temperature,top_p)) or temperature<=0 or not 0<top_p<=1:raise ValueError('auxiliary sampling policy')
        output=[];generator=torch.Generator(device='cuda').manual_seed(seed)
        with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for _ in range(max_tokens):
                hidden=self.model.base_model(torch.tensor([prompt+output],device='cuda'),use_cache=False).last_hidden_state[0,-1:]
                probabilities=torch.softmax(self.model.lm_head(hidden)[0].float()/temperature,-1)
                if top_p<1:
                    values,ids=probabilities.sort(descending=True);remove=values.cumsum(-1)>top_p;remove[1:]=remove[:-1].clone();remove[0]=False;values[remove]=0;probabilities=torch.zeros_like(probabilities).scatter(0,ids,values)
                token=int(torch.multinomial(probabilities,1,generator=generator));output.append(token)
                if token==self.tokenizer.eos_token_id:break
        return output
