"""CPU-only, default-off disk admission before verifier backend execution.

A size grant is operational metadata, never a model/proof validity shortcut.
Backend still hashes every member and applies the original scientific contract.
"""
import hashlib,json,os,re,stat,subprocess,sys
from pathlib import Path
from subnet.distributed_roles import authenticate
from subnet.cache_lifecycle import snapshot
from subnet.artifact_budget import for_manifest
VERSION='owned-verifier-download-capacity-v1'

class CapacityDeferred(Exception):
    """Infrastructure admission only; never an invalid scientific report."""


def verify_envelope_limit(job,authority):
    """Byte allowance only; source, capacity and proof checks remain mandatory."""
    import base64
    from nacl.signing import VerifyKey
    envelope=job['manifest']
    if envelope.get('signer')!=authority:raise ValueError('original manifest authority')
    raw=json.dumps(envelope['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    VerifyKey(bytes.fromhex(authority)).verify(raw,base64.b64decode(envelope['signature'],validate=True))
    manifest=envelope['payload'];cap=manifest.get('training_task_capacity')
    if (job.get('role')=='verify' and type(cap)is dict and set(cap)=={'version','max_tasks'}
            and cap['version']=='signed-training-task-capacity-v1' and type(cap['max_tasks'])is int and cap['max_tasks']==512
            and type(manifest.get('max_batches'))is int and manifest['max_batches']==9
            and type(manifest.get('K'))is int and manifest['K']==4 and type(manifest.get('L'))is int and manifest['L']==4
            and manifest.get('training_policy')=='bf16-cpu-fp32-master-task-normalized-persistent-v4'
            and manifest.get('training_input_policy')=='committed-unaudited-training-v1'
            and ('samples_per_batch'not in manifest or type(manifest['samples_per_batch'])is int and manifest['samples_per_batch']==8)
            and 'training_startup_recovery'not in manifest):return 32_000_000
    return 4_000_000

def budget(job,envelope,authority,*,protocol_source=None):
    # The worker's initial protocol package can be historical. Validate input
    # transport with the exact registry-selected job source, without replacing
    # cached modules in a multithreaded worker or importing a model.
    if protocol_source is not None:
        return source_budget(job,envelope,authority,protocol_source)

    policy=authenticate(envelope,authority)
    expected={'version','disk_floor_bytes','extra_temporary_bytes','max_submissions','poll_seconds'}
    policy_keys=set(policy)-{'runtime_sidecars'}
    validate_sidecar_grants(policy.get('runtime_sidecars',{}))
    if policy_keys not in (expected|{'checkpoint_inventories'},expected|{'checkpoint_inventory_directory'})or policy['version']!=VERSION:raise ValueError('explicit signed capacity policy')
    for key,minimum,maximum in [('disk_floor_bytes',2*1024**3,100*1024**3),('extra_temporary_bytes',0,100*1024**3),('max_submissions',1,256),('poll_seconds',1,60)]:
        if type(policy[key])is not int or not minimum<=policy[key]<=maximum:raise ValueError('bounded capacity '+key)
    inventories=policy.get('checkpoint_inventories')
    if inventories is not None and (not isinstance(inventories,dict)or not 1<=len(inventories)<=1024):raise ValueError('bounded signed checkpoint sizes')
    if job.get('role')!='verify':raise ValueError('only original verify jobs')
    manifest=authenticate(job['manifest'],authority);cp=manifest['checkpoint']
    if inventories is not None:row=inventories.get(cp['id'])
    else:
        directory=Path(policy['checkpoint_inventory_directory'])
        if not directory.is_absolute()or directory.is_symlink()or directory.resolve()!=directory or not re.fullmatch('[0-9a-f]{64}',cp['id']):raise ValueError('exact owned inventory directory')
        path=directory/(cp['id']+'.json')
        if not path.exists():raise CapacityDeferred('checkpoint-size-inventory-not-yet-authoritative')
        st=path.lstat()
        if not stat.S_ISREG(st.st_mode)or path.is_symlink()or st.st_nlink!=1 or st.st_uid!=os.getuid()or st.st_size>1_000_000:raise ValueError('owned signed size grant')
        document=authenticate(json.loads(path.read_bytes()),authority)
        if set(document)!={'version','checkpoint_id','descriptor_sha256','files'}or document['version']!='authenticated-model-byte-inventory-v1'or document['checkpoint_id']!=cp['id']:raise ValueError('exact published size grant')
        row={k:document[k]for k in ('descriptor_sha256','files')}
    if row is None:raise CapacityDeferred('checkpoint-size-inventory-not-yet-authoritative')
    if set(row)!={'descriptor_sha256','files'}or not re.fullmatch('[0-9a-f]{64}',row['descriptor_sha256']or'')or not isinstance(row['files'],dict)or set(row['files'])!=set(cp['files']):raise ValueError('exact signed checkpoint byte inventory')
    from subnet.training_receipts import sha
    if row['descriptor_sha256']!=sha(dict(id=cp['id'],files=cp['files'])):raise ValueError('exact checkpoint descriptor payload binding')
    sizes={}
    for name,value in row['files'].items():
        if Path(name).name!=name or name in ('.','..')or set(value)!={'size','sha256'}or value['sha256']!=cp['files'][name]or type(value['size'])is not int or not 0<value['size']<=20_000_000_000:raise ValueError('exact approved model byte size')
        sizes[name]=value['size']
    inputs=job.get('submissions')
    if not isinstance(inputs,list)or not 1<=len(inputs)<=policy['max_submissions']:raise ValueError('signed input count capacity limit')
    # Existing backend verifies using this cap, even when child commitment.size
    # is smaller. Reserve the enforceable signed cap, not an optimistic claim.
    cap=for_manifest(manifest)['compressed_bytes']
    limits=input_limits(job,manifest)
    return policy,cp,sizes,sum(limits),max(limits)

_SOURCE_BUDGET_CODE = r"""
import importlib.util,json,pathlib,sys
source=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(source))
spec=importlib.util.spec_from_file_location('operator_capacity_source_budget',sys.argv[2])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
try:
    v=json.load(sys.stdin);result=module.budget(v['job'],v['policy'],v['authority'])
    if 'torch' in sys.modules:raise ValueError('capacity validation must remain CPU metadata only')
    print(json.dumps({'budget':result},separators=(',',':'),allow_nan=False))
except Exception as error:
    print(json.dumps({'error':type(error).__name__},separators=(',',':')))
"""

def validate_sidecar_grants(grants):
    if not isinstance(grants,dict) or len(grants)>16:raise ValueError('bounded explicit runtime sidecar grants')
    for bundle,members in grants.items():
        if not isinstance(bundle,str) or not re.fullmatch('[0-9a-f]{64}',bundle) or not isinstance(members,dict) or set(members)!={'subnet/source_sampling_admission.py'}:
            raise ValueError('exact source-bound runtime sidecar grant')
        if any(not isinstance(sha,str) or not re.fullmatch('[0-9a-f]{64}',sha) for sha in members.values()):raise ValueError('exact runtime sidecar digest')
    return grants


def validate_runtime_inventory(job,envelope,authority,source):
    """Keep runtime exact; a separate ROOT grant may admit one CPU sidecar.

    Sidecars are never supplied by a miner and never added to the scientific
    inventory. Their name, source archive and bytes must all match ROOT metadata.
    """
    policy=authenticate(envelope,authority)
    grants=validate_sidecar_grants(policy.get('runtime_sidecars',{}))
    manifest=authenticate(job['manifest'],authority)
    bundle=manifest.get('source_bundle',{}).get('sha256')
    approved=grants.get(bundle,{})
    files=job.get('source_files');source=Path(source)
    actual={str(p.relative_to(source))for p in (source/'subnet').glob('*.py')}
    if not isinstance(files,dict) or not 1<=len(files)<=256 or set(files)&set(approved) or set(files)|set(approved)!=actual:
        raise ValueError('complete capacity protocol runtime inventory')
    for name,expected in approved.items():
        path=source/name
        if path!=path.resolve() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('ROOT-approved runtime sidecar changed')


def source_budget(job,envelope,authority,source):
    source=Path(source)
    if not source.is_absolute() or source!=source.resolve(strict=True) or not source.is_dir():raise ValueError('exact registry-selected capacity source')
    files=job.get('source_files')
    validate_runtime_inventory(job,envelope,authority,source)
    for name,expected in files.items():
        p=source/name
        if not isinstance(expected,str)or not re.fullmatch('[0-9a-f]{64}',expected)or p!=p.resolve()or not p.is_file()or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('capacity protocol source changed')
    raw=json.dumps(dict(job=job,policy=envelope,authority=authority),separators=(',',':'),allow_nan=False).encode()
    if len(raw)>verify_envelope_limit(job,authority):raise ValueError('bounded signed capacity input')
    try:
        result=subprocess.run([sys.executable,'-I','-B','-c',_SOURCE_BUDGET_CODE,str(source),str(Path(__file__).resolve())],input=raw,capture_output=True,timeout=30,check=False,cwd=source)
    except subprocess.TimeoutExpired as error:raise CapacityDeferred('source-bound CPU capacity validation timed out')from error
    if result.returncode or len(result.stdout)>1_000_000:raise CapacityDeferred('source-bound CPU capacity validation unavailable')
    value=json.loads(result.stdout)
    if value.get('error')=='CapacityDeferred':raise CapacityDeferred('source-bound checkpoint inventory not yet authoritative')
    if 'error'in value:raise ValueError('source-bound capacity protocol refused ('+str(value['error'])+')')
    if set(value)!={'budget'}or not isinstance(value['budget'],list)or len(value['budget'])!=5:raise ValueError('source-bound capacity result')
    return tuple(value['budget'])

def admit(job,envelope,authority,*,lifecycle,selected_cache=None,free_bytes=None,credit_lifecycle=None,protocol_source=None):
    if envelope is None:return dict(status='disabled')
    policy,cp,sizes,inputs,cap=budget(job,envelope,authority,protocol_source=protocol_source)
    credited_lifecycle=credit_lifecycle or lifecycle
    if credited_lifecycle.root.stat().st_dev!=lifecycle.root.stat().st_dev:raise ValueError('capacity credit must share filesystem')
    receipt_path=credited_lifecycle._receipt(cp['id']);receipt={}
    if receipt_path.exists():
        if receipt_path.is_symlink():raise ValueError('capacity receipt symlink')
        receipt=json.loads(receipt_path.read_bytes())
    cache=credited_lifecycle.root/'checkpoints'/cp['id']
    # External caches are not silently adopted. Their original verification and
    # leases remain unchanged; admission conservatively reserves a cold model.
    reusable=selected_cache is None or Path(selected_cache).absolute()==cache
    credited=0
    if reusable and receipt.get('cp')==cp['id']and receipt.get('files')==cp['files']and receipt.get('path',str(Path('checkpoints')/cp['id']))==str(Path('checkpoints')/cp['id']):
        for name,size in sizes.items():
            member=receipt.get('members',{}).get(name);path=cache/name
            if not member or member.get('sha256')!=cp['files'][name]or member.get('origin')not in ('authenticated-job-ACK','authenticated-model-map','durable-publication-ACK'):continue
            try:current=snapshot(path)
            except (FileNotFoundError,ValueError):continue
            if current==member.get('stat')and current['size']==size:credited+=size
    missing=sum(sizes.values())-credited
    # Final artifact files coexist. Also reserve one largest temporary/replaced
    # object; existing .partial bytes already reduce actual free space and gain
    # no credit. Full changed originals remain present until atomic replacement.
    temporary=max(max(sizes.values())if missing else 0,cap)+policy['extra_temporary_bytes']
    required=missing+inputs+temporary+policy['disk_floor_bytes']
    def free():
        if free_bytes is not None:return free_bytes()
        st=os.statvfs(lifecycle.root);return st.f_bavail*st.f_frsize
    before=free();removed=[]
    if before<required:
        removed=lifecycle.evict_checkpoints(exclude=[cp['id']],keep=0,required_free_bytes=required)
    available=free()
    result=dict(status='admitted'if available>=required else'deferred',checkpoint=cp['id'],model_total_bytes=sum(sizes.values()),authenticated_reusable_bytes=credited,model_missing_bytes=missing,input_cap_bytes=inputs,temporary_bytes=temporary,disk_floor_bytes=policy['disk_floor_bytes'],required_free_bytes=required,free_bytes=available,retired_owned_checkpoints=removed)
    return result


def input_limits(job,manifest):
    cap=for_manifest(manifest)['compressed_bytes']
    if manifest.get('submission_transport_policy') is None:return [cap]*len(job['submissions'])
    from subnet.distributed_roles import validate_frozen_submissions
    validate_frozen_submissions(manifest,job['submissions'])
    # Validator binds this exact size to the original signed commitment and
    # captured frozen object; external bootstrap enforces the same byte cap.
    if any(type(obj['commitment_ref']['size'])is not int or obj['commitment_ref']['size']<=0 for obj in job['submissions']):raise ValueError('positive authenticated input size')
    return [min(cap,obj['commitment_ref']['size'])for obj in job['submissions']]
