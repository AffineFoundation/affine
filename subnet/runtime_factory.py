"""Select only operator-authenticated, explicitly versioned inference backends."""
CPU_REVISION='cpu-float32-eager-v2-bounded-toploc'
GPU_REVISION='cuda-bf16-eager-sm86-v1'
GPU_POLICY=dict(logprob_atol=1e-5,logprob_rtol=0,toploc_exp_mismatches=0,toploc_mant_err_mean=0,toploc_mant_err_median=0)
GPU_PROFILE=dict(device='cuda',dtype='bfloat16',attention='eager',sm=[8,6],tf32=False,deterministic_algorithms=True,cublas_workspace_config=':4096:8',native_toploc_threads=2,torch_threads=2)


def validate_backend(manifest):
 revision=manifest.get('model_runtime_revision',CPU_REVISION)
 if revision=='qwen3_5-mimo-v1':
  from .mimo_runtime import validate_runtime_spec
  from .batch_contract import for_manifest, CompleteGroupContract, REQUIRED_CLIENT_FEATURES
  validate_runtime_spec(manifest.get('mimo_runtime'))
  contract=for_manifest(manifest,supported_features=REQUIRED_CLIENT_FEATURES)
  if not isinstance(contract,CompleteGroupContract) or contract.sampling_version!='forced-inverse-cdf-complete-group-mimo-v1':
   raise ValueError('MiMo runtime requires its complete-group sampling contract')
 elif revision==CPU_REVISION:
  if manifest.get('backend_profile') not in (None,{'device':'cpu','dtype':'float32','attention':'eager'}):raise ValueError('CPU backend profile mismatch')
  if manifest.get('numerical_policy','cpu-float32-eager-exact-toploc-logprob-atol1e-5')!='cpu-float32-eager-exact-toploc-logprob-atol1e-5':raise ValueError('CPU numerical policy mismatch')
 elif revision in (GPU_REVISION,'cuda-bf16-eager-sm90-v1'):
  from .backend_profiles import resolve
  resolve(manifest)
 else:raise ValueError('unsupported numerical runtime revision')
 return revision


def runtime(checkpoint,manifest,environment,harness=None,*,miner=None):
 from .forced_sampling import bind_runtime
 revision=validate_backend(manifest)
 if revision=='qwen3_5-mimo-v1':
  if not isinstance(miner,str) or len(miner)!=64 or any(c not in '0123456789abcdef' for c in miner):
   raise ValueError('authenticated miner identity required for MiMo sampling')
  from .complete_group_runtime import MimoGroupRuntime
  return MimoGroupRuntime(checkpoint,manifest,environment,harness,miner=miner)
 if revision==CPU_REVISION:
  from .model import Runtime
  return bind_runtime(Runtime(checkpoint,manifest['checkpoint']['files'],environment=environment,harness=harness),manifest)
 from .gpu_runtime import GPURuntime
 return bind_runtime(GPURuntime(checkpoint,manifest['checkpoint']['files'],environment,harness,runtime_revision=revision),manifest)
