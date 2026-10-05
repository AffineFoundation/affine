"""Prospective authenticated streaming FP32 trainer-state transport.

No pickle; no automatic moment reset; no full state hydration on local disk.
The integration supplies authority-authenticated descriptor hashes and storage
callbacks. A shard is retired only after verified restore or durable readback.
"""
import hashlib
import math
import os
import shutil
import uuid
import copy
import time
import threading
import json
from pathlib import Path

from .persistent_cpu_adamw import POLICY, HYPERPARAMETERS, checkpoint_id, finite, sha

VERSION = 'sharded-fp32-master-adamw-state-v1'
MAX_SHARD_BYTES = 4_000_000_000
SLOTS = ('master', 'exp_avg', 'exp_avg_sq')
HEADER_RESERVE = 1_048_576
TRANSPORT_VERSION='bounded-parallel-fp32-state-v1'
EIGHT_STREAM_TRANSPORT_VERSION='bounded-eight-fp32-state-v1'
SUPPORTED_CONCURRENCY=(1,2,3,4,8)

def transport_concurrency(manifest):
    value=manifest.get('optimizer_state_transport')
    if value is None:return 1
    if not isinstance(value,dict) or set(value)!={'version','concurrency'}:
        raise ValueError('signed bounded optimizer state transport')
    concurrency=value['concurrency'];version=value['version']
    if type(concurrency) is not int or not (
            (version==TRANSPORT_VERSION and 1<=concurrency<=4) or
            (version==EIGHT_STREAM_TRANSPORT_VERSION and concurrency==8)):
        raise ValueError('optimizer state transport concurrency bound')
    return concurrency


def resource_plan(inventory, *, bf16_export_bytes, transfer_bytes=MAX_SHARD_BYTES,
                  disk_reserve_bytes=8*1024**3, ram_reserve_bytes=8*1024**3,
                  concurrency=1):
    """Additional RAM/disk beyond already loaded model and existing files.

    CPU state is three FP32 buffers. RAM also reserves a shard serialization
    buffer and six FP32-equivalent largest-parameter temporaries. Disk keeps
    at most concurrency transfer shards and one complete BF16 export; state input is streamed.
    No promise is made about model forward/backward GPU capacity.
    """
    if (not inventory or type(bf16_export_bytes) is not int or bf16_export_bytes < 1 or
            type(transfer_bytes) is not int or not 1 <= transfer_bytes <= MAX_SHARD_BYTES or
            type(disk_reserve_bytes) is not int or disk_reserve_bytes < 0 or
            type(ram_reserve_bytes) is not int or ram_reserve_bytes < 0 or
            type(concurrency) is not int or concurrency not in SUPPORTED_CONCURRENCY):
        raise ValueError('explicit bounded resource plan')
    _inventory_valid(inventory)
    count = sum(r['numel'] for r in inventory); largest = max(r['numel'] for r in inventory)
    if bf16_export_bytes < count*2:
        raise ValueError('BF16 export reserve below parameter payload')
    return dict(cpu_state_bytes=count*12, bounded_transfer_bytes=transfer_bytes,
        state_transfer_concurrency=concurrency,bounded_inflight_transfer_bytes=transfer_bytes*concurrency,
        cpu_additional_ram_required_bytes=count*12 + largest*24 + transfer_bytes*concurrency + ram_reserve_bytes,
        additional_disk_required_bytes=bf16_export_bytes + transfer_bytes*concurrency + disk_reserve_bytes,
        bf16_export_bytes=bf16_export_bytes, ram_reserve_bytes=ram_reserve_bytes,
        disk_reserve_bytes=disk_reserve_bytes, full_state_disk_hydration=False,
        gpu_forward_backward_capacity_qualified=False)


def cgroup_headroom(maximum,current,stats):
    """Conservative automatic-reclaim allowance; never count active file pages.

    The charged cgroup file cache can exceed new anonymous-state requirements.
    Only clean inactive file pages count, subtracting all mapped, dirty,
    writeback, unevictable and shmem pages conservatively, even if those fields
    overlap. No cache eviction is requested and no pinned page is assumed free.
    """
    if (type(maximum)is not int or maximum<0 or type(current)is not int or current<0 or
            not isinstance(stats,dict)or any(type(v)is not int or v<0 for v in stats.values())):
        raise ValueError('actual nonnegative cgroup memory accounting')
    excluded=sum(stats.get(k,0)for k in ('file_dirty','file_writeback','file_mapped','unevictable','shmem'))
    eligible=max(0,min(stats.get('inactive_file',0),stats.get('file',0))-excluded)
    # Do not exceed charged usage or the finite cgroup limit.
    eligible=min(eligible,current,maximum)
    return dict(hard_headroom_bytes=max(0,maximum-current),
        conservative_clean_inactive_file_bytes=eligible,
        usable_bytes=min(maximum,max(0,maximum-current)+eligible),
        active_file_cache_counted=False,drop_caches_requested=False)


def available_ram_bytes():
    """Actual MemAvailable bounded by finite, cache-aware cgroup headroom."""
    fields = {line.split(':', 1)[0]: line.split(':', 1)[1].strip().split()
              for line in Path('/proc/meminfo').read_text().splitlines()}
    if 'MemAvailable' not in fields or fields['MemAvailable'][1:] != ['kB']:
        raise ValueError('actual Linux available RAM observation required')
    available = int(fields['MemAvailable'][0])*1024
    for root in (Path('/sys/fs/cgroup'),):
        maximum, current = root/'memory.max', root/'memory.current'
        if maximum.is_file() and current.is_file():
            text = maximum.read_text().strip()
            if text != 'max':
                stats_path=root/'memory.stat'
                stats={k:int(v)for k,v in (line.split()for line in stats_path.read_text().splitlines())}if stats_path.is_file()else {}
                available = min(available,cgroup_headroom(int(text),int(current.read_text()),stats)['usable_bytes'])
    legacy = Path('/sys/fs/cgroup/memory')
    limit, usage = legacy/'memory.limit_in_bytes', legacy/'memory.usage_in_bytes'
    if limit.is_file() and usage.is_file():
        stats_path=legacy/'memory.stat'
        values={k:int(v)for k,v in (line.split()for line in stats_path.read_text().splitlines())}if stats_path.is_file()else {}
        # V1 uses hierarchical total_* fields where present.
        names=('inactive_file','file_dirty','file_writeback','file_mapped','unevictable','shmem')
        stats={k:values.get('total_'+k,values.get(k,0))for k in names}
        stats['file']=values.get('total_cache',values.get('cache',0))
        stats['file_mapped']=values.get('total_mapped_file',values.get('mapped_file',stats['file_mapped']))
        stats['file_dirty']=values.get('total_dirty',values.get('dirty',stats['file_dirty']))
        stats['file_writeback']=values.get('total_writeback',values.get('writeback',stats['file_writeback']))
        available = min(available,cgroup_headroom(int(limit.read_text()),int(usage.read_text()),stats)['usable_bytes'])
    return available


def admit_resources(workspace, plan, *, ram_available=None, disk_available=None):
    path = Path(workspace)
    if not path.is_dir() or path.is_symlink():
        raise ValueError('real existing trainer workspace required')
    ram = available_ram_bytes() if ram_available is None else ram_available
    disk = shutil.disk_usage(path).free if disk_available is None else disk_available
    if type(ram) is not int or ram < plan['cpu_additional_ram_required_bytes']:
        raise ValueError('persistent CPU optimizer RAM reserve')
    if type(disk) is not int or disk < plan['additional_disk_required_bytes']:
        raise ValueError('bounded state-transfer and BF16-export disk reserve')
    return dict(plan, workspace=str(path.resolve()), observed_available_ram_bytes=ram, observed_free_disk_bytes=disk,
                admitted=True)


def _inventory_valid(inventory):
    if (not isinstance(inventory, list) or not 1 <= len(inventory) <= 4096 or
            any(not isinstance(r, dict) or not isinstance(r.get('name'), str) for r in inventory) or
            len({r.get('name') for r in inventory}) != len(inventory)):
        raise ValueError('state parameter inventory names')
    for row in inventory:
        if (set(row) != {'name', 'shape', 'numel'} or not isinstance(row['name'], str) or
                not row['name'] or len(row['name']) > 1024 or
                not isinstance(row['shape'], list) or
                any(type(x) is not int or x < 1 for x in row['shape']) or
                type(row['numel']) is not int or row['numel'] < 1 or
                math.prod(row['shape']) != row['numel']):
            raise ValueError('state parameter shape/name')


def _hash_file(path):
    h = hashlib.sha256(); count = 0
    with path.open('rb') as source:
        for part in iter(lambda: source.read(1024**2), b''):
            h.update(part); count += len(part)
    return h.hexdigest(), count


def validate_descriptor(descriptor, approved_sha256, input_checkpoint, inventory):
    checkpoint_id(approved_sha256); checkpoint_id(input_checkpoint)
    if sha(descriptor) != approved_sha256:
        raise ValueError('authority-approved state descriptor digest')
    fields = {'version', 'policy', 'hyperparameters', 'parameters', 'parameters_sha256',
              'input_checkpoint', 'inference_checkpoint', 'epoch', 'parent_state_sha256',
              'genesis_sha256', 'optimizer_steps', 'parameter_steps', 'shards'}
    if set(descriptor) != fields:
        raise ValueError('state descriptor fields')
    _inventory_valid(inventory)
    _inventory_valid(descriptor['parameters'])
    if (descriptor['version'] != VERSION or descriptor['policy'] != POLICY or
            sha(descriptor['hyperparameters']) != sha(HYPERPARAMETERS) or
            sha(descriptor['parameters']) != sha(inventory) or
            descriptor['parameters_sha256'] != sha(inventory) or
            descriptor['inference_checkpoint'] != input_checkpoint):
        raise ValueError('state policy, parameter inventory or input model lineage')
    checkpoint_id(descriptor['input_checkpoint']); checkpoint_id(descriptor['genesis_sha256'])
    parent = descriptor['parent_state_sha256']
    if parent is not None: checkpoint_id(parent)
    if (not isinstance(descriptor['epoch'], str) or not descriptor['epoch'] or
            len(descriptor['epoch']) > 200 or type(descriptor['optimizer_steps']) is not int or
            not 1 <= descriptor['optimizer_steps'] < 2**31):
        raise ValueError('state completed epoch/counter binding')
    expected_steps = {r['name']: descriptor['optimizer_steps'] for r in inventory}
    if (not isinstance(descriptor['parameter_steps'], dict) or
            any(type(v) is not int for v in descriptor['parameter_steps'].values()) or
            descriptor['parameter_steps'] != expected_steps):
        raise ValueError('state per-parameter/global counter mismatch')
    shards = descriptor['shards']
    if not isinstance(shards, list) or not 1 <= len(shards) <= 65536:
        raise ValueError('bounded state shard list')
    file_names = set(); tensor_keys = set(); spans = {}
    expected = {(r['name'], s): r['numel'] for r in inventory for s in SLOTS}
    for shard in shards:
        if not isinstance(shard, dict) or set(shard) != {'name', 'sha256', 'size', 'tensors'}:
            raise ValueError('state shard fields')
        name = shard['name']
        if (not isinstance(name, str) or '/' in name or '\\' in name or
                not name.startswith('state-') or not name.endswith('.safetensors') or
                name in file_names or type(shard['size']) is not int or
                not 1 <= shard['size'] <= MAX_SHARD_BYTES):
            raise ValueError('state shard path/size/duplicate')
        checkpoint_id(shard['sha256']); file_names.add(name)
        tensors = shard['tensors']
        if not isinstance(tensors, list) or not 1 <= len(tensors) <= 512:
            raise ValueError('state shard tensor budget')
        for row in tensors:
            if not isinstance(row, dict) or set(row) != {'key', 'parameter', 'slot', 'start', 'count'}:
                raise ValueError('state tensor metadata fields')
            key = row['key']; pair = row['parameter'], row['slot']
            if (not isinstance(key, str) or not key or len(key) > 80 or key in tensor_keys or
                    pair not in expected or type(row['start']) is not int or row['start'] < 0 or
                    type(row['count']) is not int or row['count'] < 1 or
                    row['start'] + row['count'] > expected[pair]):
                raise ValueError('state tensor span/name/dtype binding')
            tensor_keys.add(key); spans.setdefault(pair, []).append((row['start'], row['count']))
    if set(spans) != set(expected):
        raise ValueError('incomplete optimizer state slots')
    for pair, length in expected.items():
        offset = 0
        for start, count in sorted(spans[pair]):
            if start != offset:
                raise ValueError('overlapping or missing state tensor spans')
            offset += count
        if offset != length: raise ValueError('incomplete state tensor coverage')
    return descriptor


def _transfer_directory(workspace):
    root = Path(workspace)
    if not root.is_dir() or root.is_symlink():
        raise ValueError('real transfer workspace required')
    path = root/('.fp32-state-transfer-' + uuid.uuid4().hex)
    path.mkdir(mode=0o700)
    return path


def restore_state(descriptor, approved_sha256, input_checkpoint, inventory, *,
                  workspace, fetch_shard, resource_admission, concurrency=None):
    """Restore approved disjoint spans; never expose partially materialized rows."""
    import torch
    from safetensors import safe_open
    validate_descriptor(descriptor, approved_sha256, input_checkpoint, inventory)
    if resource_admission.get('admitted') is not True:
        raise ValueError('actual RAM/disk admission required before state allocation')
    concurrency=resource_admission.get('state_transfer_concurrency',1) if concurrency is None else concurrency
    if (type(concurrency) is not int or concurrency not in SUPPORTED_CONCURRENCY or
            concurrency!=resource_admission.get('state_transfer_concurrency',1)):
        raise ValueError('signed restore concurrency/admission binding')
    required = resource_plan(inventory, bf16_export_bytes=resource_admission['bf16_export_bytes'],
                             transfer_bytes=resource_admission['bounded_transfer_bytes'],
                             disk_reserve_bytes=resource_admission['disk_reserve_bytes'],
                             ram_reserve_bytes=resource_admission['ram_reserve_bytes'],
                             concurrency=concurrency)
    if any(resource_admission.get(k) != v for k, v in required.items()):
        raise ValueError('state admission/inventory mismatch')
    if max(s['size'] for s in descriptor['shards']) > required['bounded_transfer_bytes']:
        raise ValueError('state shard exceeds admitted transfer workspace')
    admit_resources(workspace, required)
    rows = {r['name']: {s: torch.empty(r['shape'], dtype=torch.float32, device='cpu')
                       for s in SLOTS} for r in inventory}
    for row in rows.values(): row['step'] = descriptor['optimizer_steps']
    transfer = _transfer_directory(workspace)
    counters={'inflight':0,'maximum':0};counter_lock=threading.Lock()
    def restore_one(number,shard):
        started=time.time()
        with counter_lock:
            counters['inflight']+=1;counters['maximum']=max(counters['maximum'],counters['inflight'])
        try:
            path = transfer/shard['name']; fetch_shard(shard['name'], path)
            if not path.is_file() or path.is_symlink():
                raise ValueError('downloaded state regular file required')
            actual_sha, size = _hash_file(path)
            if (actual_sha, size) != (shard['sha256'], shard['size']):
                raise ValueError('downloaded state shard digest/size')
            with safe_open(path, framework='pt', device='cpu') as source:
                if set(source.keys()) != {r['key'] for r in shard['tensors']}:
                    raise ValueError('state shard tensor allowlist')
                for metadata in shard['tensors']:
                    value = source.get_tensor(metadata['key'])
                    if value.dtype != torch.float32 or list(value.shape) != [metadata['count']]:
                        raise ValueError('restored state tensor dtype/shape')
                    finite(torch, value, nonnegative=metadata['slot'] == 'exp_avg_sq')
                    target = rows[metadata['parameter']][metadata['slot']].reshape(-1)
                    start, count = metadata['start'], metadata['count']
                    # validate_descriptor established exact, disjoint coverage
                    # before allocation. Concurrent tasks cannot write an
                    # overlapping approved slice or publish these private rows.
                    target[start:start + count].copy_(value)
                    del value, target
            path.unlink()
            receipt=dict(name=shard['name'],sha256=actual_sha,size=size,
                         verified_materialization=True,local_shard_retired=True,
                         started_at=started,completed_at=time.time())
            if concurrency>1:
                path=transfer/('restore-evidence-'+format(number,'06d')+'.json')
                path.write_text(json.dumps(receipt,sort_keys=True));path.chmod(0o600)
            return number,receipt
        except Exception as error:
            if concurrency>1:
                path=transfer/('restore-failure-'+format(number,'06d')+'.json')
                path.write_text(json.dumps(dict(name=shard['name'],error_type=type(error).__name__,
                    started_at=started,completed_at=time.time()),sort_keys=True));path.chmod(0o600)
            raise
        finally:
            with counter_lock:counters['inflight']-=1
    results={}
    if concurrency==1:
        for number,shard in enumerate(descriptor['shards']):
            index,receipt=restore_one(number,shard);results[index]=receipt
    else:
        from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            remaining=iter(enumerate(descriptor['shards']));pending={}
            def submit_next():
                item=next(remaining,None)
                if item is not None:pending[pool.submit(restore_one,*item)]=item[0]
            for _ in range(concurrency):submit_next()
            try:
                while pending:
                    done,_=wait(pending,return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future);index,receipt=future.result();results[index]=receipt
                    for _ in done:submit_next()
            except Exception:
                for future in pending:future.cancel()
                # Executor scope waits every active task. No incomplete row
                # object escapes; failed bytes and receipts remain evidence.
                raise
    evidence=[results[number] for number in range(len(descriptor['shards']))]
    for receipt in evidence:
        receipt.update(restore_concurrency=concurrency,actual_maximum_inflight_shards=counters['maximum'])
    if concurrency>1:
        for path in transfer.glob('restore-evidence-*.json'):path.unlink()
    transfer.rmdir()
    return (descriptor, rows, approved_sha256), evidence


def _plans(inventory, shard_bytes):
    if type(shard_bytes) is not int or not HEADER_RESERVE + 4 <= shard_bytes <= MAX_SHARD_BYTES:
        raise ValueError('state shard payload cap')
    payload = shard_bytes - HEADER_RESERVE; plans = []; group = []; used = 0
    serial = 0
    for parameter in inventory:
        for slot in SLOTS:
            offset = 0
            while offset < parameter['numel']:
                count = min(parameter['numel'] - offset, (payload - used)//4)
                if not count or len(group) >= 512:
                    plans.append(group); group = []; used = 0; continue
                group.append(dict(key='tensor-' + format(serial, '08d'),
                    parameter=parameter['name'], slot=slot, start=offset, count=count))
                serial += 1; used += count*4; offset += count
    if group: plans.append(group)
    return plans


def _export_state(optimizer, *, epoch, inference_checkpoint, workspace,
                 publish_shard, readback_shard, commit_descriptor,
                 resource_admission, shard_bytes=MAX_SHARD_BYTES, concurrency=1, readback_mode='trainer-full',retain_shard=None):
    """Upload/read back bounded concurrent shards; transport the descriptor last.

    readback_shard(name) yields actual bounded byte chunks from durable storage.
    commit_descriptor(doc) publishes/reads back actual bytes and returns its
    payload digest plus durable_readback_verified=True. A GPU worker stages an
    unsigned descriptor and returns authority_committed=False; the coordinator
    must independently hash durable state and sign authority before admission.
    Only a callback that actually completed authority commit may return True.
    """
    import torch
    from safetensors.torch import save_file
    if readback_mode not in ('trainer-full','upload-only-independent-full-v1'):raise ValueError('explicit export readback mode')
    checkpoint_id(inference_checkpoint)
    if not isinstance(epoch, str) or not epoch or len(epoch) > 200:
        raise ValueError('completed epoch binding required before shard export')
    if (type(optimizer.global_step) is not int or optimizer.global_step < 1 or
            any(row['step'] != optimizer.global_step for row in optimizer.rows.values())):
        raise ValueError('only completed full-parameter updates may publish state')
    if (type(concurrency) is not int or concurrency not in SUPPORTED_CONCURRENCY or
            resource_admission.get('state_transfer_concurrency',1)!=concurrency or
            resource_admission.get('admitted') is not True or
            shard_bytes > resource_admission['bounded_transfer_bytes']):
        raise ValueError('actual bounded transfer admission required')
    expected=resource_plan(optimizer.inventory,bf16_export_bytes=resource_admission['bf16_export_bytes'],
        transfer_bytes=resource_admission['bounded_transfer_bytes'],disk_reserve_bytes=resource_admission['disk_reserve_bytes'],
        ram_reserve_bytes=resource_admission['ram_reserve_bytes'],concurrency=concurrency)
    if any(resource_admission.get(k)!=v for k,v in expected.items()):
        raise ValueError('parallel export resource admission binding')
    if (not Path(workspace).is_dir() or Path(workspace).is_symlink() or
            shutil.disk_usage(workspace).free < shard_bytes*concurrency + resource_admission['disk_reserve_bytes']):
        raise ValueError('actual transfer disk reserve at publication')
    for name, parameter in optimizer.parameters:
        if not torch.equal(optimizer.rows[name]['master'].to(torch.bfloat16), parameter.detach().cpu()):
            raise ValueError('export master/BF16 inference projection')
    if concurrency>1 and available_ram_bytes()<shard_bytes*concurrency+resource_admission['ram_reserve_bytes']:
        raise ValueError('actual parallel publication RAM reserve')
    transfer = _transfer_directory(workspace); shards = []; evidence = []
    counters={'inflight':0,'maximum':0,'transport_inflight':0,'transport_maximum':0}; counter_lock=threading.Lock()
    def transfer_one(number,plan):
        started=time.time()
        with counter_lock:
            counters['inflight']+=1;counters['maximum']=max(counters['maximum'],counters['inflight'])
        try:
            result=materialize(number,plan,started)
            if concurrency>1:
                receipt=transfer/('evidence-'+format(number,'06d')+'.json')
                receipt.write_text(json.dumps(result[2],sort_keys=True));receipt.chmod(0o600)
            return result
        except Exception as error:
            if concurrency>1:
                receipt=transfer/('failure-'+format(number,'06d')+'.json')
                receipt.write_text(json.dumps(dict(name='state-'+format(number,'06d')+'.safetensors',
                    error_type=type(error).__name__,started_at=started,completed_at=time.time()),sort_keys=True));receipt.chmod(0o600)
            raise
        finally:
            with counter_lock:counters['inflight']-=1
    def materialize(number,plan,started):
        name = 'state-' + format(number, '06d') + '.safetensors'; path = transfer/name
        tensors = {}
        for metadata in plan:
            value = optimizer.rows[metadata['parameter']][metadata['slot']]
            if value.device.type != 'cpu' or value.dtype != torch.float32 or not value.is_contiguous():
                raise ValueError('export CPU FP32 tensor profile')
            # Plans partition every slot into disjoint, exhaustive slices. Check
            # the bytes actually serialized once, rather than rescanning an
            # entire large parameter for each shard that contains part of it.
            part = value.reshape(-1)[metadata['start']:metadata['start'] + metadata['count']]
            finite(torch, part, nonnegative=metadata['slot'] == 'exp_avg_sq')
            tensors[metadata['key']] = part
        save_file(tensors, str(path)); path.chmod(0o600); del tensors
        actual_sha, size = _hash_file(path)
        if not 1 <= size <= shard_bytes: raise ValueError('actual state shard exceeds object cap')
        transport_started=time.time()
        with counter_lock:
            counters['transport_inflight']+=1
            counters['transport_maximum']=max(counters['transport_maximum'],counters['transport_inflight'])
        try:
            publish_shard(name, path)
            if readback_mode=='trainer-full':
                h = hashlib.sha256(); actual_size = 0
                for part in readback_shard(name):
                    if not isinstance(part, bytes) or not part or len(part) > 16*1024**2:
                        raise ValueError('bounded actual durable readback chunks required')
                    actual_size += len(part)
                    if actual_size > size: raise ValueError('durable state readback oversized')
                    h.update(part)
                if (h.hexdigest(), actual_size) != (actual_sha, size):
                    raise ValueError('durable state shard readback mismatch')
        finally:
            with counter_lock:counters['transport_inflight']-=1
        transport_completed=time.time()
        retained=retain_shard(name,path,actual_sha,size)if retain_shard is not None else False
        if type(retained)is not bool or (retained and (path.exists()or path.is_symlink())):
            raise ValueError('actual owned optimizer candidate retention required')
        if not retained:path.unlink()
        return number,dict(name=name,sha256=actual_sha,size=size,tensors=plan),dict(name=name,sha256=actual_sha,size=size,
            durable_readback_verified=readback_mode=='trainer-full',local_shard_retired=not retained,started_at=started,completed_at=time.time(),
            transport_started_at=transport_started,transport_completed_at=transport_completed)
    plans=_plans(optimizer.inventory,shard_bytes);results={}
    if concurrency==1:
        for number,plan in enumerate(plans):
            index,shard,receipt=transfer_one(number,plan);results[index]=(shard,receipt)
    else:
        from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            remaining=iter(enumerate(plans));pending={}
            def submit_next():
                item=next(remaining,None)
                if item is not None:pending[pool.submit(transfer_one,*item)]=item[0]
            for _ in range(concurrency):submit_next()
            try:
                while pending:
                    done,_=wait(pending,return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        index,shard,receipt=future.result();results[index]=(shard,receipt)
                    for _ in done:submit_next()
            except Exception:
                for future in pending:future.cancel()
                # Running transfers finish under the publication freeze. Failed
                # shard files remain forensic evidence; descriptor is never called.
                raise
    for number in range(len(plans)):
        shard,receipt=results[number];shards.append(shard);evidence.append(receipt)
    descriptor = dict(version=VERSION, policy=POLICY,
        hyperparameters=copy.deepcopy(HYPERPARAMETERS), parameters=copy.deepcopy(optimizer.inventory),
        parameters_sha256=sha(optimizer.inventory), input_checkpoint=optimizer.input_checkpoint,
        inference_checkpoint=inference_checkpoint, epoch=epoch,
        parent_state_sha256=optimizer.parent_state_sha256, genesis_sha256=optimizer.genesis_sha256,
        optimizer_steps=optimizer.global_step,
        parameter_steps={name: row['step'] for name, row in optimizer.rows.items()}, shards=shards)
    if readback_mode!='trainer-full':
        for receipt in evidence:receipt.update(export_verification='uploaded-local-sha-only',local_sha_verified=True,upload_completed=True,independent_full_readback_required=True)
    digest = sha(descriptor)
    validate_descriptor(descriptor, digest, inference_checkpoint, optimizer.inventory)
    acknowledgement = commit_descriptor(descriptor)
    if (not isinstance(acknowledgement, dict) or acknowledgement.get('descriptor_sha256') != digest or
            acknowledgement.get('durable_readback_verified') is not True):
        raise ValueError('authenticated descriptor-last publication/readback acknowledgement')
    if readback_mode!='trainer-full' and acknowledgement.get('authority_committed')is not False:
        raise ValueError('upload-only descriptor cannot claim authority commit')
    if concurrency>1:
        for path in transfer.glob('evidence-*.json'):path.unlink()
    transfer.rmdir()
    return descriptor, dict(shards=evidence, descriptor_sha256=digest,
        descriptor_published_last=True,descriptor_committed_last=acknowledgement.get('authority_committed')is True,
        authority_commit_required=acknowledgement.get('authority_committed')is not True,
        no_full_state_disk_hydration=retain_shard is None,transport_concurrency=concurrency,
        actual_maximum_inflight_shards=counters['maximum'],actual_maximum_inflight_transfers=counters['transport_maximum'],
        transport_timing_measured=True,**(dict(optimizer_state_export_policy=readback_mode,trainer_full_readback_performed=False,independent_full_readback_required=True)if readback_mode!='trainer-full'else {}))


def export_state(optimizer, *, epoch, inference_checkpoint, workspace,
                 publish_shard, readback_shard, commit_descriptor,
                 resource_admission, shard_bytes=MAX_SHARD_BYTES, concurrency=1, readback_mode='trainer-full',retain_shard=None):
    """Seal normal optimizer mutations until descriptor-last publication ends."""
    with optimizer.freeze_for_publication():
        return _export_state(optimizer, epoch=epoch, inference_checkpoint=inference_checkpoint,
            workspace=workspace, publish_shard=publish_shard, readback_shard=readback_shard,
            commit_descriptor=commit_descriptor, resource_admission=resource_admission,
            shard_bytes=shard_bytes,concurrency=concurrency,readback_mode=readback_mode,retain_shard=retain_shard)
