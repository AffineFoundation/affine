"""Independent full-audit worker; trust policy is supplied by operator, not ZIP."""
import argparse
import json
from pathlib import Path
from .batches import unpack
from .model import check_runtime_profile
from .runtime_factory import runtime as make_runtime
from .storage import sha
from .protocol import entry, entries, sample_key, classification
from .auditing import select, assurance

def verify(data, manifest, checkpoint):
    check_runtime_profile(manifest)
    definitions = entries(manifest)
    first = definitions[0]
    runtime = make_runtime(checkpoint, manifest, first['spec'], first.get('harness'))
    runtimes = {first['env_id']: runtime}
    records = unpack(data)
    if len(records)>manifest.get('max_batches',4):
        raise ValueError('epoch batch budget')
    policy=manifest.get('audit_policy',{'mode':'full'})
    if isinstance(policy,str):policy={'mode':'full','version':1}
    selected_indices=select(len(records),policy,manifest.get('audit_seed'),sha(data))
    outcomes, accepted, seen = [], [], set()
    for bi, (batch, arrays) in enumerate(records):
        try:
            if batch['epoch'] != manifest['epoch'] or batch['checkpoint'] != manifest['checkpoint']['id']:
                raise ValueError('epoch or checkpoint')
            definition = entry(manifest, batch.get('env_id'))
            env_id = definition['env_id']
            if batch.get('schema',1)>=2 and (batch.get('env_id')!=env_id or batch.get('sample_index')!=batch.get('index')):
                raise ValueError('required batch environment binding')
            if env_id not in runtimes:
                runtimes[env_id] = runtime.for_environment(definition['spec'], definition.get('harness'))
            selected = runtimes[env_id]
            if batch.get('schema',1)>=2 and batch.get('environment_version')!=selected.spec.version:
                raise ValueError('batch environment version')
            key = sample_key(batch)
            index = key[1]
            if type(index) is not int or index not in definition['indices'] or key in seen:
                raise ValueError('index or duplicate batch')
            seen.add(key)
            rolls = batch['rollouts']
            if len(rolls) != manifest['K']+manifest['L'] or len(arrays) != len(rolls):
                raise ValueError('sample count')
            fingerprints = set()
            for rollout, tensors in zip(rolls, arrays):
                if rollout['index'] != index or rollout.get('env_id',env_id) != env_id:
                    raise ValueError('rollout index')
                signature = tuple(tuple(t['output']) for t in rollout['turns'])
                if signature in fingerprints:
                    raise ValueError('duplicate sample')
                fingerprints.add(signature)
                if bi in selected_indices:
                    selected.verify(rollout, tensors)
            if sum(classification(r) == 'positive' for r in rolls) != manifest['K'] or sum(classification(r) == 'negative' for r in rolls) != manifest['L']:
                raise ValueError('positive negative counts')
            if bi in selected_indices:
                accepted.append(batch)
            outcomes.append(dict(batch=bi,env_id=env_id,index=index,valid=True if bi in selected_indices else None,structural_valid=True,fully_audited=bi in selected_indices))
        except Exception as e:
            outcomes.append(dict(batch=bi,index=batch.get('index'),valid=False,reason=str(e)))
    return dict(epoch=manifest['epoch'],submission_sha256=sha(data),policy=policy,audit_seed=manifest.get('audit_seed'),selected_batches=selected_indices,assurance=assurance(len(records),len(selected_indices)),outcomes=outcomes,accepted=accepted,training_eligibility='fully-audited-only')

def main():
    p=argparse.ArgumentParser();p.add_argument('artifact');p.add_argument('manifest');p.add_argument('checkpoint');p.add_argument('report')
    a=p.parse_args()
    try:
        report=verify(Path(a.artifact).read_bytes(),json.loads(Path(a.manifest).read_text()),a.checkpoint)
    except Exception as e:
        report=dict(valid=False,reason=str(e),accepted=[])
    Path(a.report).write_text(json.dumps(report))

if __name__=='__main__':main()
