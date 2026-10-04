"""Operator-signed real GPU sampling controls; never publish or set weights."""
import argparse
import base64
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from nacl.signing import VerifyKey

sys.dont_write_bytecode = True

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()

def inventory(source):
    paths = list(Path(source).rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('regular immutable source required')
    return {str(p.relative_to(source)): digest(p) for p in paths if p.is_file()}

def run(document, authority):
    if document['signer'] != authority:
        raise ValueError('operator authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']), base64.b64decode(document['signature'], validate=True))
    p = document['payload']
    if (p['revision'] != 'isolated-forced-sampling-control-v1' or
        not p['created_at'] <= time.time() < p['expires_at'] or
        not 0 < p['expires_at']-p['created_at'] <= 7200 or
        p['chain_transactions'] is not False or p['live_epoch_changed'] is not False or
        p['helper_sha256'] != digest(__file__)):
        raise ValueError('bounded control scope')
    source, output = Path(p['source']), Path(p['output'])
    if output.exists() or inventory(source) != p['source_files']:
        raise ValueError('source identity/output freshness')
    if {n:importlib.metadata.version(n) for n in p['runtime_versions']} != p['runtime_versions']:
        raise ValueError('runtime versions')
    if not 1 <= len(p['indices']) <= 5 or not set(p['indices']) <= set(p['approved_mining_indices']):
        raise ValueError('approved non-heldout tasks')
    if type(p['search_attempts']) is not int or not 2 <= p['search_attempts'] <= 16:
        raise ValueError('bounded control attempts')
    sys.path.insert(0, str(source))
    from subnet.gpu_runtime import GPURuntime
    from subnet.forced_sampling import bind_runtime, receipt, binding
    from subnet.model import model_files
    from subnet.audit_policy import InvalidSample
    from subnet.batches import pack
    from subnet.artifact_budget import LONG
    import torch
    if torch.cuda.is_initialized() or torch.cuda.get_device_capability() != (9,0):
        raise ValueError('fresh Hopper process')
    if model_files(p['checkpoint_path']) != p['manifest']['checkpoint']['files']:
        raise ValueError('full checkpoint byte identity')
    output.mkdir(parents=True, mode=0o700)
    report = dict(plan_sha256=hashlib.sha256(canonical(document)).hexdigest(),
                  source_sha256=p['source_sha256'], checkpoint=p['manifest']['checkpoint']['id'],
                  historical_execution_proven=False, chain_transactions=False, live_epoch_changed=False,
                  honest_controls=[], attacks=[], original_pid=os.getpid(),
                  original_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    def save(stage):
        report.update(stage=stage, observed_at=time.time())
        (output/'progress.private.json').write_bytes(canonical(report))
        print(json.dumps({'stage':stage, 'honest':len(report['honest_controls']), 'attacks':len(report['attacks'])}), flush=True)
    def runtime():
        return bind_runtime(GPURuntime(p['checkpoint_path'], p['manifest']['checkpoint']['files'],
             p['environment'], p['harness'], runtime_revision=p['model_runtime_revision']), p['manifest'])
    start=time.monotonic(); miner=None; verifier=None
    try:
        save('loading-pinned-miner')
        miner=runtime(); selected=None; arrays=None
        for index in p['indices']:
            found={}
            for attempt in range(p['search_attempts']):
                before=time.monotonic(); rollout, probs=miner.rollout(index, attempt)
                report['honest_controls'].append(dict(index=index, attempt=attempt,
                    classification=rollout['classification'], tokens=sum(len(t['output']) for t in rollout['turns']),
                    generation_seconds=time.monotonic()-before))
                found.setdefault(rollout['classification'], (rollout, probs)); save('genuine-sampling')
                if 'positive' in found and 'negative' in found:
                    selected=[found[c][0] for c in ['positive','negative']];arrays=[found[c][1] for c in ['positive','negative']];break
            if selected is not None:break
        if selected is None:
            raise ValueError('no genuine success/failure pair in bounded controls')
        batch=dict(schema=2, env_id=p['environment']['id'], environment_version=p['environment']['version'],
                   index=index, sample_index=index, epoch=p['manifest']['epoch'],
                   checkpoint=p['manifest']['checkpoint']['id'], rollouts=selected)
        body=pack([(batch, arrays)], budget=LONG);(output/'honest.zip').write_bytes(body)
        report['honest_artifact_sha256']=hashlib.sha256(body).hexdigest();report['honest_artifact_bytes']=len(body)
        # Deliberately copy the dataset answer and produce real target-model
        # conditional probabilities and TOPLOC evidence. This is the attack,
        # never the honest generation control.
        rows=json.loads((source/p['environment']['config']['task_snapshot']).read_text())
        tokens=miner.tokenizer.encode('\\boxed{'+str(rows[index]['data']['answer'])+'}', add_special_tokens=False)
        original=miner.sample; miner.sampling_context=None;miner.sample=lambda *args:list(tokens)
        attack, attack_arrays=miner.rollout(index, selected[0]['seed'])
        assert miner.verify(attack, attack_arrays) is True
        attack['sampling']=receipt(binding(p['manifest']), attack['seed'])
        miner.sample=original;original=None;miner=None;gc.collect();torch.cuda.empty_cache()
        save('independent-pinned-verifier-reload')
        verifier=runtime()
        for rollout, probs in zip(selected,arrays):
            before=time.monotonic();assert verifier.verify(rollout,probs) is True
            report.setdefault('independent_replays',[]).append(dict(classification=rollout['classification'], seconds=time.monotonic()-before))
        try:verifier.verify(attack,attack_arrays)
        except InvalidSample as error:
            if str(error) != 'sampling replay mismatch':raise
            report['attacks'].append(dict(kind='copied-answer-fresh-genuine-probabilities-and-TOPLOC', legacy_passed=True, forced_rejected=True))
        else:raise ValueError('copied-answer attack accepted')
        for name, edit in [('out-of-range-attempt',lambda r:r.update(seed=p['manifest']['sampling_contract']['max_attempts'])),
                           ('missing-sampling-receipt',lambda r:r.pop('sampling'))]:
            bad=copy.deepcopy(selected[0]);edit(bad)
            try:verifier.verify(bad,arrays[0])
            except InvalidSample:report['attacks'].append(dict(kind=name, forced_rejected=True))
            else:raise ValueError(name+' accepted')
        verifier=None;gc.collect();torch.cuda.empty_cache()
        assert inventory(source)==p['source_files'] and model_files(p['checkpoint_path'])==p['manifest']['checkpoint']['files']
        report.update(success=True, honest_false_rejections=0, elapsed_seconds=time.monotonic()-start,
                      actual_independent_model_reload=True, genuine_success_and_failure=True)
        save('complete');(output/'completion.private.json').write_bytes(canonical(report))
    except Exception as error:
        report.update(success=False,error_type=type(error).__name__);save('failed');raise

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--authority',required=True)
    args=parser.parse_args();run(json.loads(Path(args.plan).read_text()),args.authority)
