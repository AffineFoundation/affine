"""Operator-owned CPU role runtime and disclosed first-task curated policy.

The public policy is derived from original visible messages/tools and does not
read databases, graders or task collections. Its roaming heuristic is qualified
only on the existing first-task controls, not the whole original telecom suite.
"""
import copy,hashlib,os,sys
from pathlib import Path
from importlib.metadata import version
from .native_tau2_common_contract import canonical,exact,role_descriptor
from .native_tau2_model import PROFILE,profile,sample_cached
from .native_tau2_public_policy import public_action,format_candidates,POSITIVE_GUIDANCE,NEGATIVE_GUIDANCE

RUNTIME_REVISION='native-tau2-common-cpu-float32-two-runtime-v1'
POLICY_REVISION='native-tau2-first-roaming-public-role-policy-v1'

def hash_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class CPURoleRuntime:
    def __init__(self,checkpoint,descriptor):
        from .native_tau2_common_search_contract import role_descriptor as search_role_descriptor,AGENT_SEED_POLICY as SEARCH_SEED_POLICY
        (search_role_descriptor if descriptor.get('seed_policy')==SEARCH_SEED_POLICY else role_descriptor)(descriptor,descriptor['kind']);profile()
        if descriptor.get('runtime_profile')!=PROFILE or descriptor.get('model_runtime_revision')!='cpu-float32-eager-v2-bounded-toploc' or descriptor.get('renderer')!='native-tau2-complete-chat-tools-v2':
            raise ValueError('qualified full CPU role profile/renderer')
        if descriptor['interpreter_sha256']!=hash_file(Path(sys.executable).resolve()) or any(version(name)!=expected for name,expected in descriptor['runtime_versions'].items()):
            raise ValueError('CPU role interpreter/package identity')
        from .native_tau2_common_endpoint import verify_sources
        verify_sources(descriptor)
        from .model import Runtime
        self.runtime=Runtime(checkpoint,descriptor['checkpoint']['files'],threads=4)
        self.model=self.runtime.model;self.tokenizer=self.runtime.tokenizer
        if self.model.config.vocab_size!=descriptor['vocab_size'] or descriptor['max_context']>self.model.config.max_position_embeddings:
            raise ValueError('approved model vocabulary/context geometry')
        self.descriptor=copy.deepcopy(descriptor)
    def approved_descriptor(self):return copy.deepcopy(self.descriptor)
    def sample(self,prompt,seed,temperature,top_p,max_tokens):
        if temperature!=.7 or top_p!=1.:raise ValueError('qualified CPU cached sampling settings')
        return sample_cached(self.runtime,prompt,seed,max_tokens)
    def compute(self,*args):return self.runtime.compute(*args)
    def build_proofs(self,*args,**kwargs):return self.runtime.build_proofs(*args,**kwargs)
    def verify_proofs(self,*args,**kwargs):return self.runtime.verify_proofs(*args,**kwargs)

class FirstTaskPublicPolicy:
    """Curated target-model computations, not policy sampling provenance.

    The agent search attempt selects diagnostic guidance without claiming it was
    sampled. Auxiliary decisions use the same immutable function and fixed seeds
    for every attempt. All returned tokens still require full model verification.
    """
    def __init__(self,descriptor,role,request_model):
        if role not in ('agent','user') or descriptor.get('revision')!=POLICY_REVISION or descriptor.get('scope')!='public-request-derived-curated-output' or descriptor.get('role')!=role or descriptor.get('request_model')!=request_model or descriptor.get('agent_seed_stride')!=256:
            raise ValueError('explicit disclosed public role policy')
        root=Path(__file__).resolve().parent.parent
        required=('subnet/native_tau2_public_policy.py','subnet/native_tau2_common_cpu.py')
        if not set(required)<=set(descriptor.get('source_files',{})):raise ValueError('complete public policy sources')
        for name,expected in descriptor['source_files'].items():
            path=root/name
            if path.is_symlink() or not path.is_file() or hash_file(path)!=expected:raise ValueError('public policy source bytes')
        self.descriptor=copy.deepcopy(descriptor);self.role=role;self.request_model=request_model
    def approved_descriptor(self):return copy.deepcopy(self.descriptor)
    def select(self,request,seed):
        if request.get('model')!=self.request_model:raise ValueError('public role request')
        local=copy.deepcopy(request);local['model']='native-model-'+self.role
        guidance=POSITIVE_GUIDANCE if (seed//256)%2==0 else NEGATIVE_GUIDANCE
        text,_=public_action(local,guidance)
        # Seeded equivalent formatting is public and independently reproducible.
        return format_candidates(text)[seed%2]
