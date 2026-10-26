"""ROOT-signed, single-use, no-update six-profile worker on a separate H200.

python -m research.group_numerical_qualification.worker --scope ROOT-SIGNED.json
Reads INPUTS.json and its authenticated scientific plan and capture receipt,
then four exact original-groups/<task_hash>.group.json files. No remote clients.
"""
import argparse
from dataclasses import asdict
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import sys
import time

from research.fixed_group_grpo.core import Policy, canonical, require, sha
from research.fixed_group_grpo.worker import UPSTREAM, authenticate, file_hash, original
from .profiles import buffer_inventory, cached_replay, compare_rows, decode_group, prefill, prepare_dtype


PROFILES = [
    'BF16-cached-batch8-forced-token-replay', 'BF16-full-prefill-eval-no-grad',
    'BF16-full-prefill-train-gradient-checkpointed', 'FP32-cached-batch8-forced-token-replay',
    'FP32-full-prefill-eval-no-grad', 'FP32-full-prefill-train-gradient-checkpointed',
]


def run(scope_path):
    envelope = json.loads(Path(scope_path).read_bytes())
    s = authenticate(envelope)
    require(s['version'] == 'private-group-numerical-qualification-v1', 'diagnostic scope version')
    root = Path(s['workspace'])
    require(root.resolve() == root and root.is_dir(), 'owned canonical workspace')
    require(socket.gethostname() == s['hostname'] and
            Path('/proc/sys/kernel/random/boot_id').read_text().strip() == s['boot_id'], 'assigned host boot')
    require(s['created_at'] <= time.time() < s['expires_at'], 'bounded current scope')
    for prefix, mapping in [('', s['files']), ('source', s['source_files'])]:
        for name, expected in mapping.items():
            path = root/prefix/name
            require(path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink()
                    and file_hash(path) == expected, 'pinned source/input '+name)
    require(Path(__file__).resolve().is_relative_to(root/'source'), 'executing assigned source tree')
    require('INPUTS.json' in s['files'], 'pinned input inventory')
    inputs = json.loads((root/'INPUTS.json').read_bytes())
    require(inputs['version'] == 'fixed-original-train-groups-v1', 'fixed input inventory')
    require(inputs['selection_plan']['path'] == 'SCIENTIFIC-PLAN.ROOT-SIGNED.json'
            and inputs['original_capture']['path'] == 'ORIGINAL-INPUT-CAPTURE.ROOT-SIGNED.json',
            'fixed signed provenance paths')
    plan_path = root/inputs['selection_plan']['path']
    capture_path = root/inputs['original_capture']['path']
    require(all(p.name in s['files'] for p in (plan_path, capture_path)), 'pinned signed plans')
    require(file_hash(plan_path) == inputs['selection_plan']['sha256']
            and file_hash(capture_path) == inputs['original_capture']['sha256'], 'input provenance hashes')
    plan = authenticate(json.loads(plan_path.read_bytes()))
    capture = authenticate(json.loads(capture_path.read_bytes()))
    require(plan['version'] == 'no-update-policy-numerics-v2' and plan['profiles'] == PROFILES,
            'exact six frozen diagnostic profiles')
    require(plan['optimizer_steps'] == plan['parameter_updates'] == 0
            and plan['sampling'] is False and plan['model_exports'] is False
            and plan['heldout_used'] is False and plan['new_numeric_threshold'] is None,
            'no-update diagnostic without a tolerance claim')
    require(capture['selection_plan_sha256'] == file_hash(plan_path)
            and capture['selected_group_count'] == len(capture['selected']) == 4,
            'exact first-four selection receipt')
    require(plan['selection']['arm'] == 'control' and plan['selection']['update'] == 0
            and plan['selection']['groups'] == 4, 'original control first update inputs')
    policy_fields = dict(plan['policy'])
    require(policy_fields.pop('top_p') == 1., 'unchanged full supported distribution')
    policy = Policy(**policy_fields); policy.validate()
    require(policy == Policy(), 'original T0.7/support/context/horizon profile')
    groups = []
    require(len(inputs['groups']) == 4, 'exact four captured groups')
    for ordinal, (item, selected) in enumerate(zip(inputs['groups'], capture['selected'])):
        name = f"original-groups/{item['task_hash']}.group.json"
        require(item['original_ordinal'] == ordinal and item['path'] == name
                and item['task_hash'] == selected['task_hash'] and item['sha256'] == selected['sha256']
                and selected['relative_path'] == f"output/update-000/{item['task_hash']}.group.json",
                'original selection order and hashes without outcome filtering')
        path = root/name
        require(name in s['files'] and s['files'][name] == item['sha256']
                and file_hash(path) == item['sha256'], 'exact original captured group')
        group = decode_group(json.loads(path.read_bytes()))
        require(group.task_id == item['task_hash'] and group.stamp.optimizer_step == 0
                and group.stamp.policy == policy, 'unchanged first-update task and policy')
        groups.append(group)
    require(len({g.task_id for g in groups}) == 4
            and all(g.stamp == groups[0].stamp for g in groups), 'one unchanged original policy')
    for name, version in s['runtime_versions'].items():
        require(importlib.metadata.version(name) == version, 'pinned runtime '+name)
    base = root/'base'; cp = s['checkpoint']
    require(cp['id'] == UPSTREAM and sha(cp['files']) == UPSTREAM, 'original upstream checkpoint')
    require(set(p.name for p in base.iterdir()) == set(cp['files']), 'complete checkpoint files')
    for name, digest in cp['files'].items():
        require(not (base/name).is_symlink() and file_hash(base/name) == digest, 'original model bytes')
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True)
    require(not {int(p.strip()) for p in active.stdout.splitlines() if p.strip()}-{os.getpid()},
            'separate empty GPU before initialization')
    sys.path.insert(0, str(root/'source'))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from subnet.backend_jobs import parameter_value_digest
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    require(torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (9, 0), 'single H200')
    require(str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
            == s['gpu_uuid'].removeprefix('GPU-'), 'assigned GPU UUID')
    output = root/'numerical-output'
    require(not output.exists(), 'single original diagnostic; no resume or retry')
    output.mkdir(mode=0o700)
    original(output/'SCOPE.original.json', envelope)
    started = time.monotonic()
    log = (output/'observations.jsonl').open('xb')
    def record(value):
        row = dict(at=time.time(), elapsed_seconds=time.monotonic()-started, **value)
        log.write(canonical(row)+b'\n'); log.flush(); os.fsync(log.fileno())
    def guard():
        require(time.time() < s['expires_at'] and time.monotonic()-started < s['duration_seconds'],
                'fixed diagnostic deadline')
        require(shutil.disk_usage(root).free >= s['disk_reserve_bytes'], 'preserved scratch reserve')
    def memory():
        return dict(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    rss_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    model = None
    digests, comparisons = [], []
    try:
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
        require(len(tokenizer) == policy.support_size and tokenizer.eos_token_id == policy.eos_token_id,
                'original tokenizer support and EOS')
        for dtype_name, dtype in [('BF16', torch.bfloat16), ('FP32', torch.float32)]:
            guard(); torch.cuda.reset_peak_memory_stats()
            load_started = time.monotonic()
            model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
                trust_remote_code=False, use_safetensors=True, dtype=torch.bfloat16,
                attn_implementation='eager')
            model = prepare_dtype(model, dtype, device='cuda')
            require(model.config.model_type == 'qwen2', 'pinned Qwen2 family')
            require(all(p.dtype == dtype for p in model.parameters()), 'one declared parameter dtype')
            require(getattr(model.config, 'rope_scaling', None) is None, 'no extra RoPE scaling')
            native_context = model.config.max_position_embeddings
            model.config.max_position_embeddings = policy.max_context_tokens
            before = parameter_value_digest(model)
            buffers_before = buffer_inventory(model)
            if dtype_name == 'BF16':
                require(before == groups[0].stamp.checkpoint_sha256, 'same original sampled parameter digest')
            record(dict(stage='model_loaded', dtype=dtype_name, parameter_digest=before,
                        weight_origin='original upstream loaded BF16 then losslessly cast to declared dtype',
                        native_context=native_context, declared_context=policy.max_context_tokens,
                        rope_parameters=getattr(model.config, 'rope_parameters', None),
                        buffers=buffers_before,
                        seconds=time.monotonic()-load_started, **memory()))
            for ordinal, group in enumerate(groups):
                witnesses = [list(r.behavior_logprobs) for r in group.rollouts]
                profiles = {}
                for profile, gradients in [('cached', None), ('prefill_eval', False), ('prefill_train', True)]:
                    guard(); torch.cuda.reset_peak_memory_stats()
                    if profile == 'cached': values, timing = cached_replay(model, group, guard=guard)
                    else: values, timing = prefill(model, group, gradients=gradients, guard=guard)
                    # Persist EVERY selected LP before making comparisons.
                    filename = f'{dtype_name}-{ordinal}-{profile}.json'
                    document = dict(dtype=dtype_name, profile=profile, task_id=group.task_id,
                        original_ordinal=ordinal, fixed_original_tokens=True, sampling=False,
                        original_stops=[r.stop for r in group.rollouts],
                        original_outcomes=[r.outcome for r in group.rollouts],
                        original_lengths=[len(r.output) for r in group.rollouts],
                        selected_logprobs=values, policy=asdict(policy), **timing, **memory())
                    original(output/filename, document)
                    record(dict(stage='profile_recorded', dtype=dtype_name, profile=profile,
                        ordinal=ordinal, tokens=sum(map(len, values)), file=filename,
                        sha256=file_hash(output/filename), **timing, **memory()))
                    profiles[profile] = values
                pairs = [('cached', 'prefill_eval'), ('cached', 'prefill_train'),
                         ('prefill_eval', 'prefill_train')]
                if dtype_name == 'BF16': pairs.insert(0, ('original', 'cached'))
                for reference_name, target_name in pairs:
                    reference = witnesses if reference_name == 'original' else profiles[reference_name]
                    comparison = dict(dtype=dtype_name, original_ordinal=ordinal, task_id=group.task_id,
                        reference=reference_name, compared=target_name,
                        original_on_policy_claimed=False, **compare_rows(group, reference, profiles[target_name]))
                    comparisons.append(comparison)
                    original(output/f'{dtype_name}-{ordinal}-{reference_name}-vs-{target_name}.json', comparison)
                if dtype_name == 'BF16':
                    for target in ('prefill_eval', 'prefill_train'):
                        comparison = dict(dtype=dtype_name, original_ordinal=ordinal, task_id=group.task_id,
                            reference='original', compared=target, original_on_policy_claimed=False,
                            **compare_rows(group, witnesses, profiles[target]))
                        comparisons.append(comparison)
                        original(output/f'{dtype_name}-{ordinal}-original-vs-{target}.json', comparison)
            after = parameter_value_digest(model)
            require(before == after and all(p.grad is None for p in model.parameters()),
                    'unchanged parameters and no accumulated gradients')
            buffers_after = buffer_inventory(model)
            row = dict(dtype=dtype_name, before=before, after=after, unchanged=True,
                       buffers_before=buffers_before, buffers_after=buffers_after,
                       buffers_unchanged=buffers_before == buffers_after)
            digests.append(row); original(output/f'{dtype_name}-PARAMETERS.json', row)
            del model; model = None; gc.collect(); torch.cuda.empty_cache()
        result = dict(version='private-group-numerical-qualification-result-v1', complete=True,
            groups=4, profiles=PROFILES, parameter_digests=digests, comparisons=comparisons,
            optimizer_steps=0, parameter_updates=0, sampling=False, heldout_used=False,
            model_exports=False, qualification_pass_claimed=False, seconds=time.monotonic()-started)
        original(output/'RESULT.json', result); record(dict(stage='complete', groups=4, optimizer_steps=0))
    except BaseException as exc:
        record(dict(stage='original_failed', exception=type(exc).__name__, message=str(exc)[:700],
                    automatic_retry=False, optimizer_steps=0))
        raise
    finally:
        log.close()
        if model is not None: del model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--scope', required=True)
    run(parser.parse_args().scope)
