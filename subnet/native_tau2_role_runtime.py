"""Prospective model-only fixed-auxiliary CPU runtime for mixed Tau2 roles.

This source is distinct from the frozen CPU v2a runner. It preserves numerical
operations qualified by native_role_model_only; cross-host admission is separate.
"""
import copy,hashlib,sys
from importlib.metadata import version
from pathlib import Path
from .native_tau2_common_search_contract import role_descriptor
from .native_tau2_common_endpoint import verify_sources
from .native_tau2_model import PROFILE,profile,sample_cached
from .native_role_model_only import ModelOnlyCPURuntime,REVISION

class ModelOnlyRoleCPU:
    def __init__(self,checkpoint,descriptor):
        role_descriptor(descriptor,descriptor['kind']);profile()
        if descriptor.get('model_runtime_revision')!=REVISION or descriptor.get('runtime_profile')!=PROFILE or descriptor.get('renderer')!='native-tau2-complete-chat-tools-v2':
            raise ValueError('approved model-only native CPU role profile')
        if descriptor['interpreter_sha256']!=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest() or any(version(n)!=v for n,v in descriptor['runtime_versions'].items()):
            raise ValueError('native CPU role interpreter/package identity')
        root=Path(__file__).resolve().parent.parent
        for required in ('subnet/native_role_model_only.py','subnet/native_tau2_role_runtime.py'):
            if descriptor['source_files'].get(required)!=hashlib.sha256((root/required).read_bytes()).hexdigest():raise ValueError('model-only loader/wrapper source pin')
        verify_sources(descriptor)
        self.runtime=ModelOnlyCPURuntime(checkpoint,descriptor['checkpoint']['files'],threads=4)
        self.model=self.runtime.model;self.tokenizer=self.runtime.tokenizer
        if self.model.config.vocab_size!=descriptor['vocab_size'] or descriptor['max_context']>self.model.config.max_position_embeddings:
            raise ValueError('native CPU role model geometry')
        self.descriptor=copy.deepcopy(descriptor)
    def approved_descriptor(self):return copy.deepcopy(self.descriptor)
    def compute(self,*args):return self.runtime.compute(*args)
    def sample(self,prompt,seed,temperature,top_p,max_tokens):
        if temperature!=.7 or top_p!=1.:raise ValueError('approved native CPU sampling policy')
        return sample_cached(self.runtime,prompt,seed,max_tokens)
    def build_proofs(self,*args,**kwargs):return self.runtime.build_proofs(*args,**kwargs)
    def verify_proofs(self,*args,**kwargs):return self.runtime.verify_proofs(*args,**kwargs)

CUDA_REVISION='native-agent-qwen05-cuda-bf16-sm86-sdpa-flash-role-v1'
AUX_CUDA_REVISION='native-fixed-user-smol135-cuda-bf16-sm86-sdpa-flash-role-v1'

def load_role(checkpoint,descriptor):
    """Fixed operator-code registry; manifest cannot name an import path."""
    revision=descriptor.get('model_runtime_revision')
    if revision==REVISION:return ModelOnlyRoleCPU(checkpoint,descriptor)
    if revision==CUDA_REVISION:
        if descriptor.get('kind')!='agent' or descriptor.get('training_eligible') is not True:raise ValueError('CUDA native adapter is agent-only')
        from .native_role_cuda import NativeAgentCUDARuntime
        return NativeAgentCUDARuntime(checkpoint,descriptor)
    if revision==AUX_CUDA_REVISION:
        if descriptor.get('kind')!='auxiliary' or descriptor.get('training_eligible') is not False:raise ValueError('fixed-user CUDA adapter is auxiliary-only')
        from .native_auxiliary_cuda import NativeAuxiliaryCUDARuntime
        return NativeAuxiliaryCUDARuntime(checkpoint,descriptor)
    raise ValueError('unsupported explicitly qualified native role runtime')
