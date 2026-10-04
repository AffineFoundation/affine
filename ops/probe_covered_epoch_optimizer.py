"""Operator-signed isolated GPU control, never a live epoch or payout worker.

Uses explicitly curated native success/failure traces. This checks numerical
execution/coverage and proof replay, not mining provenance or held-out gains.
"""
import argparse
import base64
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from nacl.signing import VerifyKey

sys.dont_write_bytecode = True


def canonical(v):
    return json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def inventory(source):
    files = list(Path(source).rglob('*'))
    if any(f.is_symlink() for f in files):
        raise ValueError('isolated regular source inventory')
    return {str(f.relative_to(source)): digest(f) for f in files if f.is_file()}


def admit_gpu(torch):
    # Querying device properties initializes CUDA. Check process freshness
    # BEFORE that query, then admit the actual numerical hardware profile.
    if torch.cuda.is_initialized():
        raise ValueError('fresh control before CUDA initialization')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise ValueError('Hopper GPU control required')


def run(document, authority):
    if document['signer'] != authority:
        raise ValueError('original control authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']),
        base64.b64decode(document['signature'], validate=True))
    plan = document['payload']
    if (not plan['created_at'] <= time.time() < plan['expires_at'] or
            not 0 < plan['expires_at']-plan['created_at'] <= 7200 or
            plan.get('revision') != 'isolated-covered-training-control-v1' or
            plan.get('chain_transactions') is not False or
            plan.get('live_epoch_changed') is not False or
            plan.get('trace_origin') != 'curated-native-controls' or
            plan['helper_sha256'] != digest(__file__)):
        raise ValueError('bounded isolated covered control scope')
    indices = plan['indices']
    if (not isinstance(indices, list) or len(indices) != 5 or len(set(indices)) != 5
            or any(type(i) is not int or i < 0 for i in indices)
            or not set(indices) <= set(plan['approved_mining_indices'])):
        raise ValueError('five distinct approved non-heldout task controls')
    source = Path(plan['source']); output = Path(plan['output'])
    if (not source.is_absolute() or source.resolve() != source or output.exists() or
            source in output.parents or inventory(source) != plan['source_files']):
        raise ValueError('original source readback before imports')
    versions = {n: importlib.metadata.version(n) for n in plan['runtime_versions']}
    if versions != plan['runtime_versions'] or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('actual immutable runtime versions')
    sys.path.insert(0, str(source))
    from subnet.backend_jobs import parameter_value_digest
    from subnet.covered_epoch_optimizer import POLICY, coverage_schedule, train_epoch
    from subnet.gpu_runtime import GPURuntime
    from subnet.model import model_files
    from subnet.batches import pack
    import torch
    if plan['training_policy'] != POLICY or plan['steps'] != 3:
        raise ValueError('exact prospective training policy')
    # Admission refuses concurrent GPU jobs; the launching parent independently
    # checks occupancy before starting this original process.
    admit_gpu(torch)
    checkpoint = Path(plan['checkpoint_path'])
    if model_files(checkpoint) != plan['checkpoint']['files']:
        raise ValueError('input checkpoint byte identity')
    rows = json.loads((source/plan['environment']['config']['task_snapshot']).read_text())
    output.mkdir(parents=True, mode=0o700)
    report = dict(training_policy=POLICY, source_sha256=plan['source_sha256'],
                  plan_sha256=hashlib.sha256(canonical(document)).hexdigest(),
                  original_pid=os.getpid(), original_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                  input_checkpoint=plan['checkpoint']['id'], trace_origin=plan['trace_origin'],
                  chain_transactions=False, live_epoch_changed=False,
                  heldout_improvement_verified=False, public_activation=False)

    def save(stage):
        report.update(stage=stage, observed_at=time.time())
        (output/'progress.private.json').write_bytes(canonical(report))
        print(json.dumps({'stage':stage}), flush=True)

    runtime = None
    try:
        save('loading-pinned-model')
        runtime = GPURuntime(checkpoint, plan['checkpoint']['files'], plan['environment'],
            plan['harness'], runtime_revision=plan['model_runtime_revision'])
        pairs = []; controls = []
        definition = dict(env_id=plan['environment']['id'], spec=plan['environment'],
                          harness=plan['harness'], indices=plan['indices'])
        original_sample = runtime.sample
        save('building-curated-proofs-and-native-pairs')
        for i in plan['indices']:
            rollouts = []; arrays = []
            for text, label in [('\\boxed{'+str(rows[i]['data']['answer'])+'}', 'positive'),
                                ('No boxed answer.', 'negative')]:
                tokens = runtime.tokenizer.encode(text, add_special_tokens=False)
                if not tokens or len(tokens) > plan['harness']['max_output_tokens']:
                    raise ValueError('bounded curated output')
                runtime.sample = lambda *args, tokens=tokens: list(tokens)
                rollout, probabilities = runtime.rollout(i, 20261004+i)
                if rollout['classification'] != label or runtime.verify(rollout, probabilities) is not True:
                    raise ValueError('original curated native/full-probability/proof control')
                rollouts.append(rollout); arrays.append(probabilities)
            batch = dict(schema=2, env_id=definition['env_id'], index=i, epoch='isolated-covered-control',
                         checkpoint=plan['checkpoint']['id'], rollouts=rollouts)
            from subnet.artifact_budget import LONG
            artifact = pack([(batch, arrays)], budget=LONG)
            (output/('curated-'+str(i)+'.zip')).write_bytes(artifact)
            controls.append(dict(index=i, artifact_sha256=hashlib.sha256(artifact).hexdigest(),
                                 bytes=len(artifact), success_and_failure_fully_verified=True))
            pairs.append((definition, *rollouts))
        runtime.sample = original_sample
        groups, identities = coverage_schedule(pairs, plan['steps'], plan['coverage_seed'])
        if len(pairs) != 5 or [len(g) for g in groups] != [2, 2, 1]:
            raise ValueError('five distinct task controls, three accumulated updates')
        before = parameter_value_digest(runtime.model)
        save('training-all-five-verified-pairs')
        destination, updates = train_epoch(runtime, pairs, output/'training',
                                          seed=plan['coverage_seed'], steps=plan['steps'])
        after = parameter_value_digest(runtime.model)
        if before == after or updates[-1]['cumulative_unique_gradient_pairs'] != 5:
            raise ValueError('actual covered parameter update required')
        files = model_files(destination); new_id = hashlib.sha256(canonical(files)).hexdigest()
        report.update(controls=controls, updates=updates, parameter_values_sha256_before=before,
                      parameter_values_sha256_after=after,
                      new_checkpoint=dict(id=new_id, files=files, path=str(destination)),
                      covered_pair_sha256=identities, all_five_pairs_contributed=True)
        del original_sample; runtime = None; gc.collect(); torch.cuda.empty_cache()
        save('fresh-successor-model-proof-construction')
        fresh = GPURuntime(destination, files, plan['environment'], plan['harness'],
                           runtime_revision=plan['model_runtime_revision'])
        text = '\\boxed{'+str(rows[plan['indices'][0]]['data']['answer'])+'}'
        tokens = fresh.tokenizer.encode(text, add_special_tokens=False)
        fresh.sample = lambda *args: list(tokens)
        rollout, probabilities = fresh.rollout(plan['indices'][0], 20261004)
        fresh = None; gc.collect(); torch.cuda.empty_cache()
        save('independent-reloaded-successor-proof-verification')
        verifier = GPURuntime(destination, files, plan['environment'], plan['harness'],
                              runtime_revision=plan['model_runtime_revision'])
        if verifier.verify(rollout, probabilities) is not True:
            raise ValueError('fresh successor full proof/native replay')
        from subnet.audit_policy import InvalidSample
        import copy
        mutation = copy.deepcopy(rollout); mutation['turns'][0]['proofs'][0] = 'AAAA'
        try: verifier.verify(mutation, probabilities)
        except InvalidSample: pass
        else: raise ValueError('tampered successor proof accepted')
        verifier = None; gc.collect(); torch.cuda.empty_cache()
        if model_files(checkpoint) != plan['checkpoint']['files'] or inventory(source) != plan['source_files']:
            raise ValueError('immutable input/source changed during control')
        report.update(success=True, fresh_successor_full_proof_verified=True,
                      successor_proof_mutation_rejected=True, original_inputs_unchanged=True,
                      GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(), completed_at=time.time())
        save('complete'); (output/'completion.private.json').write_bytes(canonical(report))
    except Exception as error:
        report.update(success=False, error_type=type(error).__name__)
        save('failed'); raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True)
    parser.add_argument('--authority', required=True); args = parser.parse_args()
    run(json.loads(Path(args.plan).read_text()), args.authority)


if __name__ == '__main__':
    main()
