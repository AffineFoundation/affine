#!/usr/bin/env python3
"""Bounded private MiMo canonical-miner diagnostic; no upload, signing or weights.

This uses the actual Miner, MiMo backend, native math session and compact codec.
The zero-tolerance calibration is observational, not a production qualification.
Every completed original attempt is retained privately, even filtered groups.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subnet.storage import canonical, sha, Identity
from subnet.complete_group_transport import CompleteGroupTransport
from subnet.complete_group_codec import encode, limits
from subnet.complete_group_mining import GroupFiltered, GroupIndeterminate, _rollout_from_runtime
from subnet.complete_group_sampling import need


def write_new(path, body):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('xb') as stream:
        import os
        os.fchmod(stream.fileno(), 0o600); stream.write(body); stream.flush(); os.fsync(stream.fileno())


class LocalDiagnosticTransport(CompleteGroupTransport):
    """Only capability delivery is replaced; actual prepared codec/state is used."""
    def validate_capability(self, cap, manifest, miner):
        need(cap == {'private_diagnostic':True}, 'explicit local-only diagnostic capability')
    def upload(self, *args, **kwargs):
        raise RuntimeError('qualification has no network upload authority')


class RecordingRuntime:
    def __init__(self, backend, manifest, output):
        self.backend, self.manifest, self.output = backend, manifest, Path(output)
    def __getattr__(self, name): return getattr(self.backend, name)
    @property
    def complete_group_sample_evidence(self): return self.backend.complete_group_sample_evidence
    @complete_group_sample_evidence.setter
    def complete_group_sample_evidence(self, value): self.backend.complete_group_sample_evidence = value
    def for_environment(self, environment, harness=None):
        return type(self)(self.backend.for_environment(environment, harness), self.manifest, self.output)
    def rollout(self, index, nonce):
        began = time.monotonic()
        base = self.output / 'attempts' / str(index) / str(nonce)
        try:
            row, arrays = self.backend.rollout(index, nonce)
            projected = _rollout_from_runtime(self.backend, row, nonce)
            data = encode([({'original_rollout':projected}, [arrays])], limits(self.manifest))
            write_new(base.with_suffix('.zip'), data)
            record = dict(nonce=nonce, index=index, seconds=time.monotonic()-began, status='completed',
                classification=projected['classification'], stop_reason=projected['turns'][0]['stop_reason'],
                output_tokens=len(projected['turns'][0]['output']), sha256=sha(data), size=len(data))
            write_new(base.with_suffix('.json'), canonical(record))
            return row, arrays
        except BaseException as error:
            if not base.with_suffix('.json').exists():
                write_new(base.with_suffix('.json'), canonical(dict(nonce=nonce,index=index,
                    seconds=time.monotonic()-began,status='error',error_type=type(error).__name__)))
            raise


def environment(args):
    """The operator supplies the TRAIN index list; no automatic split inference."""
    from subnet.environments import EnvironmentSpec, build_spec
    if args.env_descriptor:
        value=json.loads(Path(args.env_descriptor).read_text())
        rows=value.get('environments') if type(value) is dict else None
        if rows is not None:
            need(type(rows) is list and len(rows)==1,'one original math environment for this diagnostic')
            value=rows[0]
        need(type(value) is dict and 'spec' in value and 'indices' in value,'native environment descriptor and TRAIN indices')
        result=deepcopy(value)
        spec=EnvironmentSpec.from_dict(result['spec'])
        need(spec.id in ('affine_math','math'),'original MATH environment only')
        result['spec']=dict(spec.to_dict(),max_output_tokens=max(spec.max_output_tokens,args.max_output_tokens))
    else:
        need(args.train_indices is not None,'--math-snapshot also requires explicit --train-indices')
        snapshot=Path(args.math_snapshot).resolve();rows=json.loads(snapshot.read_text())
        need(type(rows) is list and rows,'trusted native task snapshot list')
        spec=build_spec('affine_math',config={'task_snapshot':str(snapshot)}, num_samples=len(rows),
                        max_turns=1,max_output_tokens=args.max_output_tokens)
        result={'env_id':spec.id,'spec':spec.to_dict(),'indices':[]}
    if args.train_indices:
        indices=json.loads(Path(args.train_indices).read_text())
        need(type(indices) is list and indices and all(type(i) is int for i in indices), 'explicit TRAIN-only task list')
        if args.env_descriptor:
            need(set(indices)<=set(result['indices']), 'diagnostic tasks must be in approved TRAIN registry')
        result['indices']=indices
    indices=result['indices']
    need(type(indices) is list and indices and len(set(indices))==len(indices)
         and all(type(i) is int and 0<=i<result['spec']['num_samples'] for i in indices), 'bounded unique TRAIN indices')
    # Full registry stays canonical/sorted; the bounded attempt schedule is
    # separately explicit, preserving an operator's boundary-task ordering.
    schedule=indices[:args.max_tasks]
    result['indices']=sorted(schedule)
    return result,schedule


def run(args):
    from transformers import AutoTokenizer
    from subnet.mimo_runtime import validate_runtime_spec
    from subnet.mimo_manifest import build
    from subnet.complete_group_runtime import MimoGroupRuntime
    from subnet.miner import Miner
    from subnet.fast_prefill_audit import CALIBRATION
    import torch
    target=Path(args.output).resolve()
    need(not target.exists(), 'fresh private diagnostic output directory required')
    target.mkdir(parents=True,mode=0o700)
    pin=json.loads(Path(args.pin).read_text()); spec=validate_runtime_spec(pin['mimo_runtime'])
    checkpoint=Path(args.checkpoint).resolve()
    tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True,trust_remote_code=False)
    definition,schedule=environment(args)
    need(type(args.source_sha) is str and len(args.source_sha)==64
         and set(args.source_sha)<=set('0123456789abcdef'),'actual staged source SHA required')
    backend={'device':'cuda','dtype':spec.parameter_dtype,'attention':spec.attention,
             'recurrent_state_dtype':spec.recurrent_state_dtype}
    h=dict(version='text-tools-mimo-complete-group-v1',policy='autoregressive',max_output_tokens=args.max_output_tokens,
           max_context_tokens=spec.max_context_tokens,temperature=1.,top_p=.95,top_k=20,enable_thinking=spec.enable_thinking)
    observational=dict(version='private-observational-zero-tolerance-v1',qualification=False,
                       checkpoint=pin['checkpoint']['id'],source_sha256=args.source_sha,task_schedule=schedule)
    calibration=dict(version=CALIBRATION,checkpoint=pin['checkpoint']['id'],model_runtime_revision=spec.version,
        backend_profile_sha256=sha(canonical(backend)),harness_sha256=sha(canonical(h)),
        report_sha256=sha(canonical(observational)),cdf_abs_error=0.,logprob_atol=0.,
        toploc_exp_mismatches=0,toploc_mant_err_mean=0.,toploc_mant_err_median=0.)
    start=int(time.time()); run_id='private-mimo-mining-'+str(start)
    manifest=build(run_id=run_id,epoch=run_id+'-0',checkpoint=pin['checkpoint'],
        source_bundle={'sha256':args.source_sha},environments=[definition],runtime_spec=spec.to_dict(),
        vocab_size=len(tokenizer),backend_profile=backend,calibration=calibration,start=start,duration=3600,
        max_output_tokens=args.max_output_tokens)
    write_new(target/'OBSERVATIONAL-POLICY.json',canonical(observational))
    write_new(target/'manifest.json',canonical(manifest))
    identity=Identity()
    # This disposable diagnostic identity can later exercise the real signed
    # upload path with these original attempts. It is private operator state,
    # never part of a public proof or source release.
    write_new(target/'miner-identity.private.seed', bytes(identity.key))
    report=dict(version='mimo-canonical-miner-diagnostic-v1',qualification=False,
        network_upload_executed=False, checkpoint=pin['checkpoint']['id'],source_sha256=args.source_sha,
        manifest_sha256=sha(canonical(manifest)),runtime_profile=spec.to_dict(),task_schedule=schedule,
        group_size=8,nonce_range=[0,7],gpu=torch.cuda.get_device_name(),tasks=[],successful_groups=0)
    actor=None;runtime=None;began=time.monotonic()
    try:
        actor=Miner(identity,manifest,checkpoint,capability={'private_diagnostic':True},
            state_path=target/'miner-state.json',complete_group_transport=LocalDiagnosticTransport())
        runtime=MimoGroupRuntime(checkpoint,manifest,definition['spec'],manifest['environments'][0]['harness'],miner=identity.id)
        actor.runtime=RecordingRuntime(runtime,manifest,target)
        for index in schedule:
            at=time.monotonic()
            try:
                batch=actor.search(index,max_attempts=8,env_id=definition['env_id'])
                artifact=actor._group_prepared[-1][1]
                write_new(target/'accepted-group.zip',artifact)
                report['tasks'].append(dict(index=index,status='eligible',seconds=time.monotonic()-at,
                    artifact_sha256=sha(artifact),artifact_size=len(artifact)))
                report['successful_groups']=1
                break
            except (GroupFiltered,GroupIndeterminate) as error:
                report['tasks'].append(dict(index=index,status='filtered' if isinstance(error,GroupFiltered) else 'indeterminate',
                                             error_type=type(error).__name__,seconds=time.monotonic()-at))
        report['status']='complete' if report['successful_groups'] else 'insufficient-mixed-groups'
    except BaseException as error:
        report.update(status='error',error_type=type(error).__name__)
        raise
    finally:
        report['seconds']=time.monotonic()-began
        report['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        write_new(target/'REPORT.json',canonical(report))
        if actor is not None:actor._group_journal.close()
        if runtime is not None:runtime.backend.close()
    print(json.dumps({'status':report['status'],'successful_groups':report['successful_groups'],
                      'output':str(target),'qualification':False,'seconds':report['seconds']}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--pin',required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--env-descriptor');source.add_argument('--math-snapshot')
    p.add_argument('--train-indices');p.add_argument('--source-sha',required=True);p.add_argument('--output',required=True)
    p.add_argument('--max-tasks',type=int,default=4)
    p.add_argument('--max-output-tokens','--max-output-token',type=int,default=512)
    a=p.parse_args()
    if not 1<=a.max_tasks<=16 or not 1<=a.max_output_tokens<=8192:p.error('bounded diagnostic: tasks1..16, tokens1..8192')
    run(a)


if __name__=='__main__':main()
