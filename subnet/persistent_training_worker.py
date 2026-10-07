"""V4 worker transport: one input ZIP and one optimizer shard at a time."""
import gc
import copy
import hashlib
import json
import time
from pathlib import Path

from .persistent_cpu_adamw import parameter_inventory, sha
from .persistent_training_protocol import validate_job, validate_output
from .persistent_training_state import (MAX_SHARD_BYTES, resource_plan,
    admit_resources, restore_state, export_state,transport_concurrency)
from .storage import canonical


def report_updates(diagnostics,job,manifest):
    """Keep the common report update list separate from v4 state diagnostics."""
    updates=diagnostics.get('updates')
    if (not isinstance(updates,list)or len(updates)!=job['steps']or
            any(not isinstance(row,dict)for row in updates)or
            diagnostics.get('training_policy')!=job['training_policy']or
            diagnostics.get('epoch')!=manifest['epoch']or
            diagnostics.get('input_checkpoint')!=manifest['checkpoint']['id']):
        raise ValueError('persistent diagnostic optimizer update list/report binding')
    summary={k:copy.deepcopy(v)for k,v in diagnostics.items()if k!='updates'}
    return copy.deepcopy(updates),summary


def admitted_submission(path,obj,manifest,authority):
    """Retire only byte-authenticated input after verifier receipt admission."""
    if manifest.get('training_input_policy')=='committed-unaudited-training-v1':
        from .committed_training_inputs import admitted_submission as admit
        return admit(path,obj,manifest,authority,retire=True)
    if (manifest.get('training_input_policy') == 'authenticated-verifier-compact-inputs-v2'):
        from .compact_training_inputs import admitted_submission as admit
    else:
        from .training_receipts import admitted_submission as admit
    return admit(path,obj,manifest,authority,retire=True)


def read_chunks(url, *, limit):
    from .backend_jobs import r2_url
    import requests
    count=0
    with requests.get(r2_url(url,'GET'),stream=True,timeout=600,allow_redirects=False,
                      headers={'Accept-Encoding':'identity'})as response:
        if response.status_code!=200 or response.headers.get('Content-Encoding','identity')!='identity':
            raise ValueError('state GET status/encoding')
        for part in response.iter_content(1024**2):
            if not part:continue
            count+=len(part)
            if count>limit:raise ValueError('bounded state GET size')
            yield part


def put_file(url,path):
    """Retry only transient failures, rewinding the same owned immutable file.

    PUT is idempotent for these exact signed object bytes. Never refresh a grant,
    follow redirects, print response bodies/URLs, or retry authorization failures.
    """
    from .backend_jobs import r2_url
    import os, stat, requests
    target=r2_url(url,'PUT')
    descriptor=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(descriptor,'rb') as body:
        before=os.fstat(body.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=MAX_SHARD_BYTES:
            raise ValueError('actual state PUT regular object cap')
        def unchanged():
            now=os.fstat(body.fileno())
            if (now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns,now.st_ctime_ns)!=(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns):
                raise ValueError('state PUT input changed during publication')
        for attempt in range(4):
            unchanged();body.seek(0);response=None
            try:
                response=requests.put(target,data=body,headers={'Content-Type':'application/octet-stream'},
                    timeout=1800,allow_redirects=False)
                unchanged();status=response.status_code
                if status in (200,201,204):return
                if status not in (408,429,500,502,503,504) or attempt==3:
                    raise ValueError('persistent state PUT HTTP '+str(status)+' after '+str(attempt+1)+' attempt(s)')
            except (requests.ConnectionError,requests.Timeout) as error:
                unchanged()
                if attempt==3:
                    raise ValueError('persistent state PUT transient transport exhausted after 4 attempts') from None
            finally:
                if response is not None:response.close()
            time.sleep(.5*(2**attempt))


def train(runtime,pairs,out,manifest,job,authority,*,approved_checkpoint=None):
    """Stage durable state before returning; operator authority commits later."""
    from .backend_jobs import get_object, file_map
    from .model import model_files
    from .task_normalized_training import train_epoch
    binding,parent=validate_job(job,manifest,authority)
    _,inventory=parameter_inventory(runtime.model.named_parameters())
    if inventory!=binding['parameters']:raise ValueError('actual model approved parameter inventory')
    export_bytes=max(sum(p.stat().st_size for p in Path(approved_checkpoint).iterdir()if p.is_file())
                     if approved_checkpoint is not None else 0, sum(r['numel']for r in inventory)*2+1024**3)
    # Actual model is already loaded and ZIP arrays released. The margin is
    # additional to existing input weights and measured available resources.
    concurrency=transport_concurrency(manifest)
    plan=resource_plan(inventory,bf16_export_bytes=export_bytes,concurrency=concurrency)
    from contextlib import nullcontext
    local_cache=None
    if manifest.get('optimizer_state_local_cache')is not None:
        from .optimizer_state_cache import policy,StateCache
        policy(manifest)
        local_cache=StateCache(Path(out).parent.parent,job,manifest,authority)
    with local_cache if local_cache else nullcontext():
        cache_prepare_started=time.monotonic()
        cache_bytes=local_cache.prepare_parent(parent,manifest['source_bundle']['sha256'])if local_cache else 0
        cache_budget=local_cache.admit(plan,reclaimable_parent_bytes=cache_bytes)if local_cache else None
        admission=admit_resources(out,plan);restored=None;restore_evidence=[]
        cache_prepare_seconds=time.monotonic()-cache_prepare_started
        transport=job['persistent_training']
        restore_started=time.monotonic()
        if parent is not None:
            shards={s['name']:s for s in parent['shards']}
            def fetch(name,path):
                row=shards[name]
                def cold(name,path):get_object(transport['parent_read_urls'][name],row['sha256'],path,row['size'])
                if local_cache:return local_cache.fetch(name,path,cold)
                return cold(name,path)
            restored,restore_evidence=restore_state(parent,binding['parent']['descriptor_sha256'],
                binding['input_checkpoint'],inventory,workspace=out,fetch_shard=fetch,resource_admission=admission,concurrency=concurrency,
                owned_cache=local_cache if local_cache and local_cache.policy['version']=='sole-current-fp32-state-cache-stat-v2' else None)
        restore_seconds=time.monotonic()-restore_started
        train_started=time.monotonic()
        destination,optimizer,diagnostics=train_epoch(runtime,pairs,out,
            input_checkpoint=binding['input_checkpoint'],epoch=manifest['epoch'],
            seed=manifest['training_coverage']['seed'],steps=job['steps'],
            approved_genesis=binding['genesis'],approved_genesis_sha256=binding['genesis_sha256']if parent is None else None,
            restored_state=restored,resource_admission=admission,
            **({'required_pairs_per_task':manifest['K']} if type(manifest.get('K'))is int and manifest.get('K')==manifest.get('L') and manifest['K']>=2 else {}))
        training_and_checkpoint_seconds=time.monotonic()-train_started
        files=model_files(destination);checkpoint=file_map(files)
        def publish(name,path):put_file(transport['output_shards'][name]['put_url'],path)
        def readback(name):return read_chunks(transport['output_shards'][name]['get_url'],limit=MAX_SHARD_BYTES)
        def stage_descriptor(document):
            validate_output(document,job,manifest)
            data=canonical(document)
            if len(data)>4_000_000:raise ValueError('bounded staged state descriptor')
            path=Path(out)/'staged-state.json';path.write_bytes(data);path.chmod(0o600)
            put_file(transport['descriptor_put_url'],path)
            data=b''.join(read_chunks(transport['descriptor_read_url'],limit=4_000_000))
            if data!=canonical(document):raise ValueError('durable staged descriptor readback')
            path.unlink()
            return dict(descriptor_sha256=sha(document),durable_readback_verified=True,authority_committed=False)
        from .persistent_publication import export_policy
        readback_mode=export_policy(manifest)
        if local_cache:local_cache.begin_candidate()
        export_started=time.monotonic()
        descriptor,evidence=export_state(optimizer,epoch=manifest['epoch'],inference_checkpoint=checkpoint,
            workspace=out,publish_shard=publish,readback_shard=readback,
            commit_descriptor=stage_descriptor,resource_admission=admission,concurrency=concurrency,readback_mode=readback_mode,retain_shard=local_cache.retain if local_cache else None)
        diagnostics['transport_phase_seconds']=dict(parent_state_restore=restore_seconds,
            parent_cache_validation_and_admission=cache_prepare_seconds,
            parent_cache_and_restore_total=cache_prepare_seconds+restore_seconds,
            training_and_checkpoint=training_and_checkpoint_seconds,
            **({'state_export_upload_only':time.monotonic()-export_started}if readback_mode!='trainer-full'else {'state_export_and_trainer_full_readback':time.monotonic()-export_started}),
            state_transfer_concurrency=concurrency,parent_restore_performed=parent is not None)
        diagnostics.update(state_staged=True,authority_commit_required=True,complete=False)
        state=dict(namespace=transport['output_namespace'],descriptor_sha256=sha(descriptor),descriptor=descriptor,
            authority_committed=False,restore_evidence=restore_evidence,publication_evidence=evidence,
            resource_admission=admission)
        if local_cache:
            state['local_optimizer_cache_candidate']=local_cache.finish(descriptor)
            state['local_optimizer_cache_candidate']['disk_admission']=cache_budget
        del optimizer,restored;gc.collect()
        return destination,diagnostics,state


def capacity_probe(workspace,cache=None):
    """Actual Linux/cgroup RAM and existing filesystem bytes, no GPU claim."""
    import shutil
    from .persistent_training_state import available_ram_bytes
    path=Path(workspace)
    if not path.is_dir()or path.is_symlink():raise ValueError('actual existing trainer workspace')
    result=dict(free_bytes=shutil.disk_usage(path).free,available_ram_bytes=available_ram_bytes(),
        checkpoint_bytes=0,input_cache=False,workspace=str(path.resolve()))
    if cache is not None:
        target=Path(cache)
        if not target.is_dir()or target.is_symlink()or any(p.is_symlink()for p in target.iterdir()):
            raise ValueError('actual regular checkpoint cache inventory')
        result.update(checkpoint_bytes=sum(p.stat().st_size for p in target.iterdir()if p.is_file()),input_cache=True)
    return result


def capacity_requirement(manifest,probe,*,checkpoint_bytes,missing_input):
    """Bounded streaming disk, all pairs retained, actual resource observations."""
    from .artifact_budget import for_manifest
    binding=manifest['trainer_state_binding'];budget=for_manifest(manifest)
    if manifest.get('training_input_policy')in ('authenticated-verifier-compact-inputs-v2','committed-unaudited-training-v1'):
        from .compact_training_inputs import MAX_BYTES,DECODE_WORKING_BYTES
        budget=dict(compressed_bytes=256*MAX_BYTES,raw_bytes=MAX_BYTES)
    if type(checkpoint_bytes)is not int or checkpoint_bytes<=0 or type(missing_input)is not bool:
        raise ValueError('measured checkpoint hydration size')
    export=max(checkpoint_bytes,sum(r['numel']for r in binding['parameters'])*2+1024**3)
    plan=resource_plan(binding['parameters'],bf16_export_bytes=export,concurrency=transport_concurrency(manifest))
    disk=plan['additional_disk_required_bytes']+(checkpoint_bytes if missing_input else 0)+budget['compressed_bytes']+budget['raw_bytes']
    # Model is not loaded yet during the coordinator probe. Reserve one BF16
    # input load separately; worker repeats admission after loading the model.
    working_ram=DECODE_WORKING_BYTES if (manifest.get('training_input_policy')in ('authenticated-verifier-compact-inputs-v2','committed-unaudited-training-v1')) else budget['raw_bytes']
    if manifest.get('training_input_policy')=='committed-unaudited-training-v1':working_ram*=256
    ram=plan['cpu_additional_ram_required_bytes']+checkpoint_bytes+working_ram
    if probe['free_bytes']<disk:raise ValueError('persistent trainer bounded stream/input/export/artifact disk reserve')
    if probe['available_ram_bytes']<ram:raise ValueError('persistent trainer actual CPU/cgroup memory reserve')
    return dict(probe,plan=plan,required_bytes=disk,required_available_ram_bytes=ram,
        checkpoint_bytes=checkpoint_bytes,input_cache=not missing_input,
        retained_step_checkpoints=0,final_exports=1,temporary_export_copies=0,
        download_reserve_bytes=budget['compressed_bytes'],raw_working_reserve_bytes=budget['raw_bytes'],
        state_streaming=True,full_state_disk_hydration=False,gpu_forward_backward_capacity_qualified=False)
