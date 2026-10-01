"""Comparable operator-owned held-out evaluation, separate from miner rewards."""
import argparse
import datetime
import hashlib
import json
import math
import time
from pathlib import Path
from .model import check_runtime_profile
from .runtime_factory import runtime as make_runtime,validate_backend
from .protocol import entry
from .storage import canonical


def wilson(successes, count):
    if not count:
        return None
    z = 1.96; p = successes/count
    center=(p+z*z/(2*count))/(1+z*z/count)
    half=z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return [max(0.,center-half),min(1.,center+half)]


def evaluate(manifest, checkpoint_path, definition, heldout, destination):
    check_runtime_profile(manifest)
    if set(heldout['indices']) & set(definition['indices']):
        raise ValueError('held-out tasks overlap training challenge')
    if not heldout['indices'] or len(heldout['indices']) > 64 or not 1 <= heldout.get('repeats',1) <= 16:
        raise ValueError('evaluation budget')
    frozen = dict(env_id=definition['env_id'],environment=definition['spec'],harness=definition['harness'],
                  indices=heldout['indices'],seed=heldout['seed'],repeats=heldout.get('repeats',1),runtime_profile=manifest.get('runtime_profile',{}),model_runtime_revision=validate_backend(manifest))
    dataset_id = hashlib.sha256(canonical(frozen)).hexdigest()
    runtime=make_runtime(checkpoint_path,manifest,definition['spec'],definition['harness'])
    successes, rewards, failures, task_hashes = 0, [], [], []
    for index in heldout['indices']:
        for repeat in range(heldout.get('repeats',1)):
            seed=heldout['seed']+index*1000+repeat
            try:
                rollout,arrays=runtime.rollout(index,seed)
                runtime.verify(rollout,arrays)
                rewards.append(rollout['reward']);successes+=rollout['classification']=='positive'
                task_hashes.append(rollout['task_hash'])
            except Exception as exc:
                failures.append(dict(index=index,repeat=repeat,error=str(exc)))
    count=len(rewards)
    record=dict(run_id=heldout['run_id'],experiment_id=heldout.get('experiment_id',heldout['run_id']),epoch_id=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],
                model_runtime_revision=validate_backend(manifest),model=heldout.get('model',manifest.get('model_id','HuggingFaceTB/SmolLM2-135M-Instruct')),env_id=definition['env_id'],
                environment_version=definition['spec'].get('version','legacy-v1'),harness=definition['harness']['version']+':'+definition['harness']['policy'],
                harness_version=definition['harness']['version'],harness_config=definition['harness'],policy_kind='autoregressive' if definition['harness']['policy']=='autoregressive' else 'curated-control',cpu_affinity_count=len(__import__('os').sched_getaffinity(0)),
                dataset_id=dataset_id,seed=heldout['seed'],heldout_indices=heldout['indices'],
                count=count,completed_count=count,requested_count=len(heldout['indices'])*heldout.get('repeats',1),attempted_count=len(heldout['indices'])*heldout.get('repeats',1),successes=successes,
                mean_reward=sum(rewards)/count if count and not failures else None,timestamp=time.time(),timestamp_iso=datetime.datetime.now(datetime.timezone.utc).isoformat(),fixed_task_ids=[f"{definition['env_id']}:{i}" for i in heldout['indices']],taskset_hash=dataset_id,
                training_steps=heldout.get('training_steps',0),status='complete' if not failures else 'error',
                evaluation_failures=failures,uncertainty=wilson(successes,count) if not failures else None,task_hashes=task_hashes,
                runtime_profile=manifest.get('runtime_profile',{}),payable=False,weight_submission=False)
    if failures:
        record['error']='one or more held-out evaluations failed; failures excluded from success denominator'
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix('.tmp');temporary.write_bytes(canonical(record));temporary.replace(destination)
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('manifest');parser.add_argument('checkpoint');parser.add_argument('heldout');parser.add_argument('output');parser.add_argument('--env-id')
    args=parser.parse_args();manifest=json.loads(Path(args.manifest).read_text());definition=entry(manifest,args.env_id)
    heldout=json.loads(Path(args.heldout).read_text())
    evaluate(manifest,args.checkpoint,definition,heldout,args.output)

if __name__=='__main__':main()
