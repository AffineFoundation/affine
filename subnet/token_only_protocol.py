"""Explicit prospective token transport. Default absent; no historical promotion."""
import hashlib,json,io,zipfile
VERSION='prescribed-token-artifacts-v1'
TRANSPORT='small-commitment-token-pairs-v3'
POLICY={'version':VERSION,'verification':'calibrated-prefill-threeway-native',
        'probability_upload':False,'TOPLOC_upload':False,'autoregressive_fallback':False,
        'historical_execution_proof':False}
canonical=lambda v:json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def validate_policy(value):
    if type(value)is not dict or canonical(value)!=canonical(POLICY):
        raise ValueError('exact prospective token artifact policy')
    return dict(value)

def for_manifest(manifest):
    if 'token_artifact_policy' not in manifest:
        if manifest.get('submission_transport_policy')==TRANSPORT:raise ValueError('token transport requires explicit token policy')
        return None
    value=manifest['token_artifact_policy']
    validate_policy(value)
    from .fast_prefill_audit import THREEWAY_VERSION
    if (manifest.get('sampling_contract',{}).get('version')!=THREEWAY_VERSION or
            manifest.get('submission_transport_policy')!=TRANSPORT or
            'probability_artifact_policy' in manifest):
        raise ValueError('token-only requires new transport and calibrated threeway contract')
    from .forced_sampling import binding
    if binding(manifest)is None:raise ValueError('authenticated prescribed draw binding required')
    return dict(value)

def bind_runtime(runtime,manifest):
    value=for_manifest(manifest)
    if value is not None:
        from .forced_sampling import binding
        if getattr(runtime,'sampling_context',None)!=binding(manifest):
            raise ValueError('token-only runtime draw binding')
        from .fast_prefill_audit import bind
        if getattr(runtime,'fast_sampling_calibration',None)!=bind(manifest,runtime.harness):
            raise ValueError('token-only current checkpoint calibration')
    runtime.token_artifact_manifest=manifest if value is not None else None
    return runtime

def framing(batch,arrays):
    if type(batch)is not dict or type(batch.get('rollouts'))is not list or not 1<=len(batch['rollouts'])<=32:
        raise ValueError('token batch framing')
    if arrays!=[[] for _ in batch['rollouts']]:raise ValueError('token transport forbids probability arrays')
    for rollout in batch['rollouts']:
        if type(rollout)is not dict or type(rollout.get('turns'))is not list or not 1<=len(rollout['turns'])<=32:
            raise ValueError('token trajectory framing')
        for turn in rollout['turns']:
            if type(turn)is not dict or any(k in turn for k in ('proofs','probabilities','logprobs')):
                raise ValueError('token transport forbids probability/TOPLOC claims')

def pack(records,*,budget,compression_level=6,stable=True):
    if stable is not True:raise ValueError('token framing is always stable')
    if type(compression_level)is not int or not 0<=compression_level<=9:raise ValueError('token compression policy')
    if not 0<len(records)<=32:raise ValueError('token batch budget')
    rows=[]
    for batch,arrays in records:
        framing(batch,arrays);rows.append(batch)
    data=canonical({'version':VERSION,'batches':rows})
    if len(data)>min(2_000_000,budget['raw_bytes']):raise ValueError('token document budget')
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED)as archive:
        info=zipfile.ZipInfo('tokens.json',(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
        info.create_system=3;info.external_attr=0o600<<16
        archive.writestr(info,data,compresslevel=compression_level)
    data=stream.getvalue()
    if len(data)>budget['compressed_bytes']:raise ValueError('token upload budget')
    return data

def unpack(data,*,budget,max_batches):
    if type(data)is not bytes or len(data)>budget['compressed_bytes']:raise ValueError('token compressed budget')
    with zipfile.ZipFile(io.BytesIO(data))as archive:
        members=archive.infolist()
        if len(members)!=1 or members[0].filename!='tokens.json' or members[0].flag_bits&1 or members[0].compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED) or members[0].file_size>min(2_000_000,budget['raw_bytes']):
            raise ValueError('token archive framing')
        raw=archive.read('tokens.json');value=json.loads(raw)
    if canonical(value)!=raw or type(value)is not dict or set(value)!={'version','batches'} or value['version']!=VERSION or type(value['batches'])is not list or not 0<len(value['batches'])<=min(max_batches,32):
        raise ValueError('token document framing')
    result=[]
    for batch in value['batches']:
        arrays=[[] for _ in batch.get('rollouts',[])];framing(batch,arrays);result.append((batch,arrays))
    return result

NATIVE_POLICY={'version':'immutable-job-native-source-validation-v1'}

def prepare_native_validations(job,manifest,authority):
    policy=manifest.get('native_source_validation_policy')
    scopes=job.get('native_source_validation_scopes')
    if policy is None:
        if scopes is not None:raise ValueError('unsigned native validation opt-in')
        return {}
    if canonical(policy)!=canonical(NATIVE_POLICY)or for_manifest(manifest)is None:
        raise ValueError('explicit token job native validation policy')
    if type(scopes)is not dict:raise ValueError('signed native environment scopes required')
    from .protocol import entries
    from .native_session_validation import JobSourceValidation
    from .environments import EnvironmentSpec
    from pathlib import Path
    definitions=entries(manifest);ids={row['env_id']for row in definitions}
    if type(scopes)is not dict or set(scopes)!=ids:raise ValueError('exact native environment scopes')
    root=Path(__file__).resolve().parent.parent;result={}
    for row in definitions:
        scope=scopes[row['env_id']];payload=scope['payload']
        if Path(payload['source_root'])!=root or any(payload['source_files'].get(name)!=digest for name,digest in job['source_files'].items()):
            raise ValueError('native cache exact admitted source inventory')
        result[row['env_id']]=JobSourceValidation(scope,authority=authority,job_id=job['job_id'],spec=EnvironmentSpec.from_dict(row['spec']))
    return result
