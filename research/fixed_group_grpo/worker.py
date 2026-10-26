"""ROOT-authorized, private, fresh full-model fixed-group learning experiment.

CLI: python -m research.fixed_group_grpo.worker --scope ROOT-SIGNED.json
The launcher owns host rentals, wall-clock supervision, capture and deletion.
This worker has no chain, live controller, bucket or publishing credentials.
"""
import argparse
import base64
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from .core import Policy, Stamp, canonical, sha, require, group_eligibility
from .objectives import Objective

AUTHORITY = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
UPSTREAM = '2173b3e0119380ab813c9588bedff6b3edd5d3dcb01565490fa33ac67b2debf6'


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''): h.update(block)
    return h.hexdigest()


def authenticate(envelope):
    from nacl.signing import VerifyKey
    require(envelope['signer'] == AUTHORITY, 'ROOT scope authority')
    VerifyKey(bytes.fromhex(AUTHORITY)).verify(canonical(envelope['payload']), base64.b64decode(envelope['signature'], validate=True))
    return envelope['payload']


def original(path, value):
    with Path(path).open('xb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())


def run(scope_path):
    s = authenticate(json.loads(Path(scope_path).read_bytes()))
    require(s['version'] == 'private-fixed-group-grpo-study-v1', 'private scope version')
    root = Path(s['workspace'])
    require(root.resolve() == root and root.is_dir(), 'owned workspace')
    require(socket.gethostname() == s['hostname'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == s['boot_id'], 'assigned host boot')
    require(s['created_at'] <= time.time() < s['expires_at'], 'bounded current original scope')
    for prefix, mapping in [('', s['files']), ('source', s['source_files'])]:
        for name, expected in mapping.items():
            p = root/prefix/name
            require(p.resolve().is_relative_to(root) and p.is_file() and not p.is_symlink()
                    and file_hash(p) == expected, 'pinned source/input '+name)
    plan = json.loads((root/'PLAN.json').read_bytes())
    inputs = json.loads((root/'INPUTS.json').read_bytes())
    require(file_hash(root/'PLAN.json') == s['files']['PLAN.json'] and
            file_hash(root/'INPUTS.json') == s['files']['INPUTS.json'], 'frozen scientific inputs')
    policy = Policy(**plan['policy']); policy.validate()
    objective = Objective(**plan['objective']); objective.validate()
    require(objective.mode in ('grpo', 'group_preference'), 'one declared arm')
    require(plan['steps'] > 0 and 2 <= plan['min_tasks_per_step'] <= plan['tasks_per_step'] <= plan['max_candidate_tasks_per_step'], 'bounded collection/update counts')
    require(plan['group_size'] >= 2 and 0 < plan['initial_ratio_max_token_logratio'] <= .5, 'explicit group and drift bound')
    tasks = inputs['tasks']
    require(len(tasks) == plan['steps']*plan['max_candidate_tasks_per_step']
            and len({t['task_hash'] for t in tasks}) == len(tasks), 'complete frozen unique TRAIN task schedule')
    for name, expected in s['runtime_versions'].items():
        require(importlib.metadata.version(name) == expected, 'qualified package '+name)
    cp = s['checkpoint']; base = root/'base'
    require(cp['id'] == UPSTREAM and sha(cp['files']) == UPSTREAM, 'genuine original upstream checkpoint')
    require(set(p.name for p in base.iterdir()) == set(cp['files']), 'complete original model files')
    for name, digest in cp['files'].items():
        require(file_hash(base/name) == digest and not (base/name).is_symlink(), 'upstream file digest')
    output = root/'output'
    require(not output.exists(), 'one original numerical invocation; no resume/retry')
    output.mkdir(mode=0o700)
    original(output/'SCOPE.original.json', json.loads(Path(scope_path).read_bytes()))
    original(output/'PLAN.original.json', plan)
    sys.path.insert(0, str(root/'source'))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .collector import collect_group
    from .training import train_update
    from subnet.backend_jobs import parameter_value_digest
    from subnet.persistent_cpu_adamw import PersistentCPUAdamW, parameter_inventory
    from subnet.persistent_training_state import resource_plan, admit_resources
    from subnet.fp32_gradient_accumulation import admit_capacity
    from subnet.native_math_grader import isolated_argv, ASSET, runtime_lock
    from subnet.math_completion import final_box

    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    require(torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0), 'single H200 profile')
    require(str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-') == s['gpu_uuid'].removeprefix('GPU-'), 'assigned GPU')
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    require(not {int(v.strip()) for v in active.stdout.splitlines() if v.strip()}-{os.getpid()},
            'empty assigned GPU except this worker CUDA initialization')
    started = time.monotonic()
    log = (output/'observations.jsonl').open('xb')
    def record(row):
        value = dict(at=time.time(), elapsed_seconds=time.monotonic()-started, **row)
        log.write(canonical(value)+b'\n'); log.flush(); os.fsync(log.fileno())
    def guard():
        require(time.time() < s['expires_at'] and time.monotonic()-started < plan['duration_seconds'], 'fixed experiment deadline')
        require(shutil.disk_usage(root).free >= plan['disk_reserve_bytes'], 'preserved scratch reserve')
    def native(arguments, *, reference_preflight=False):
        guard()
        argv = isolated_argv(s['native_interpreter'], ASSET, arguments); argv.insert(1, '-B')
        result = subprocess.run(['/usr/bin/prlimit', '--cpu=8:9', '--as=805306368:805306368', '--core=0:0', '--', *argv],
            capture_output=True, text=True, timeout=15, cwd=root,
            env={'PATH':'/usr/bin:/bin','UV_OFFLINE':'1','HF_HUB_OFFLINE':'1'})
        if reference_preflight and result.returncode != 0:
            try: failure = json.loads(result.stderr.strip())
            except (ValueError, TypeError): failure = {}
            if failure.get('reason') == 'grader_reference_unparseable':
                return 'reference_unparseable'
        require(result.returncode == 0, 'grader infrastructure error aborts original, never negative reward')
        return result.stdout.strip()
    model = reference = optimizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
        require(len(tokenizer) == policy.support_size and tokenizer.eos_token_id == policy.eos_token_id, 'exact tokenizer support/EOS')
        for task in tasks:
            data = task['native_snapshot_row']['data']
            messages = [dict(role='system',content=data['system_prompt']),dict(role='user',content=data['prompt'])]
            require(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=False)==task['prompt'], 'native prompt without answer leakage')
            require(len(task['prompt'])+policy.max_output_tokens <= policy.max_context_tokens, 'full untruncated horizon')
        health = json.loads(native(['--runtime-check']))
        require(health['status']=='ready' and health['runtime']==runtime_lock(), 'native grader qualification')
        for role in ('model','reference'):
            guard()
            loaded = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, dtype=torch.bfloat16, attn_implementation='eager').to('cuda').eval()
            native_context = loaded.config.max_position_embeddings
            if policy.max_context_tokens > native_context:
                require(plan.get('extend_context_without_rope_scaling') is True, 'explicit research context extension')
                loaded.config.max_position_embeddings = policy.max_context_tokens
            record(dict(stage='model_loaded',role=role,native_context=native_context,declared_context=policy.max_context_tokens,
                        rope_parameters=getattr(loaded.config,'rope_parameters',None)))
            if role=='model': model=loaded
            else: reference=loaded
        for p in reference.parameters(): p.requires_grad_(False)
        initial_digest = parameter_value_digest(model)
        require(initial_digest == parameter_value_digest(reference), 'identical independent frozen reference')
        named, inventory = parameter_inventory(model.named_parameters())
        resource = resource_plan(inventory,bf16_export_bytes=sum(p.numel()*2 for _,p in named),
            transfer_bytes=4_000_000_000,ram_reserve_bytes=8*1024**3,disk_reserve_bytes=plan['disk_reserve_bytes'])
        admission=admit_resources(root,resource)
        optimizer=PersistentCPUAdamW(model.named_parameters(), cp['id'], approved_genesis=s['genesis'],
            approved_genesis_sha256=sha(s['genesis']),resource_admission=admission,
            learning_rate_authorization=s['learning_rate_authorization'],learning_rate_authority=s['authority'],
            epoch=s['epoch'],job_id=s['run_id'],steps=plan['steps'])
        require(optimizer.global_step==0 and optimizer.hyperparameters['lr']==plan['learning_rate'], 'fresh exact Adam dose')
        admit_capacity(torch,model.named_parameters())
        receipts=[]
        for update in range(plan['steps']):
            guard()
            directory=output/f'update-{update:03d}';directory.mkdir()
            parent=parameter_value_digest(model)
            stamp=Stamp(plan['sampling_run_id'],parent,update,s['numerical_runtime_sha256'],policy)
            offset=update*plan['max_candidate_tasks_per_step']
            block=tasks[offset:offset+plan['max_candidate_tasks_per_step']]
            groups=[]; attempted=0
            original(directory/'INTENT.json',dict(update=update,parent=parent,optimizer_step=optimizer.global_step,
                candidate_tasks=[t['task_hash'] for t in block]))
            for task in block:
                guard();model.eval();model.gradient_checkpointing_disable()
                attempted+=1
                answer=task['native_snapshot_row']['data']['answer']
                preflight=native(['--json-arguments',json.dumps([answer,'\\boxed{'+answer+'}'],ensure_ascii=True)],
                                 reference_preflight=True)
                require(preflight in ('0.0','1.0','reference_unparseable'), 'known-reference preflight schema')
                if preflight != '1.0':
                    original(directory/f'{task["task_hash"]}.reference-skip.json',dict(task_id=task['task_hash'],
                        reason=preflight,model_calls=0,whole_candidate_slot_consumed=True))
                    record(dict(stage='reference_preflight_skip',update=update,task_id=task['task_hash'],reason=preflight))
                    continue
                def grade(tokens):
                    text=tokenizer.decode(tokens[:-1],skip_special_tokens=True)
                    if final_box(text) is None: return 'unresolved'
                    reward=native(['--json-arguments',json.dumps([task['native_snapshot_row']['data']['answer'],text],ensure_ascii=True)])
                    require(reward in ('0.0','1.0'), 'native binary verdict')
                    return 'correct' if reward=='1.0' else 'wrong'
                with (directory/f'{task["task_hash"]}.attempts.jsonl').open('xb') as attempts:
                    def record_attempt(row):
                        attempts.write(canonical(asdict(row))+b'\n');attempts.flush();os.fsync(attempts.fileno())
                    collection=collect_group(model,tuple(task['prompt']),stamp=stamp,task_id=task['task_hash'],
                        grade_completed=grade,record_attempt=record_attempt,group_size=plan['group_size'])
                group=collection.group
                eligibility=group_eligibility(group,plan['group_size'])
                original(directory/f'{task["task_hash"]}.group.json',dict(group=asdict(group),eligibility=eligibility))
                record(dict(stage='group_collected',update=update,task_id=task['task_hash'],
                    seconds=collection.collection_seconds,tokens=collection.generated_tokens,**eligibility))
                if eligibility['eligible']: groups.append(group)
                if len(groups)==plan['tasks_per_step']: break
            require(len(groups)>=plan['min_tasks_per_step'], 'bounded search exhausted: incomplete study, no changing tasks/seeds')
            require(parameter_value_digest(model)==parent, 'collection preserved parent')
            receipt=train_update(model,reference,optimizer,groups,objective=objective,group_size=plan['group_size'],
                guard=guard,record=lambda row:record(dict(update=update,**row)),
                initial_ratio_max_token_logratio=plan['initial_ratio_max_token_logratio'])
            receipt.update(input_parameter_digest=parent,output_parameter_digest=parameter_value_digest(model),
                candidate_groups=attempted,admitted_task_ids=[g.task_id for g in groups])
            original(directory/'COMPLETE.json',receipt);receipts.append(receipt)
            record(dict(stage='update_complete',update=update,**{k:v for k,v in receipt.items() if k not in ('optimizer_precision','optimizer_threading','admitted_task_ids')}))
        guard()
        require(parameter_value_digest(reference)==initial_digest, 'frozen reference unchanged')
        endpoint=output/'endpoint';endpoint.mkdir()
        model.save_pretrained(endpoint,safe_serialization=True,max_shard_size='4GB');tokenizer.save_pretrained(endpoint)
        files={p.name:file_hash(p) for p in sorted(endpoint.iterdir()) if p.is_file()}
        result=dict(version='private-fixed-group-grpo-result-v1',objective=asdict(objective),
            policy=asdict(policy),input_checkpoint=cp['id'],output_checkpoint=dict(id=sha(files),files=files),
            updates=len(receipts),optimizer_steps=optimizer.global_step,group_count=sum(r['group_count'] for r in receipts),
            total_rollouts=sum(r['rollout_count'] for r in receipts),trained_rollouts=sum(r['trained_rollouts'] for r in receipts),
            masked_rollouts=sum(r['masked_rollouts'] for r in receipts),seconds=time.monotonic()-started,
            production_changes=False,optimizer_exported=False,efficacy_claimed=False)
        original(output/'RESULT.json',result);record(dict(stage='complete',**result))
    except BaseException as exc:
        record(dict(stage='original_failed',exception=type(exc).__name__,message=str(exc)[:600],
                    automatic_retry=False,partial_endpoint_publishable=False))
        raise
    finally:
        log.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scope',required=True)
    run(parser.parse_args().scope)
