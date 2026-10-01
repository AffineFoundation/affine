#!/usr/bin/env python3
"""Operator seals complete seven-turn EOG model evidence under a new audit kind."""
import argparse,base64,hashlib,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from subnet.long_context_runtime import authenticate,AUTHORITY,POLICY,file_sha,digest,canonical

def validate_fetched(job,report,out,public_bytes):
    import numpy as np
    if report.get('terminal_model_proof') is not True or report.get('terminal_text')!=job.get('terminal_text'):raise ValueError('complete terminal model binding')
    if job.get('role')!='long-context-proof-probe' or job.get('experiment')!='native-public-eog-terminal-v3' or job.get('policy')!=POLICY:raise ValueError('approved native model role')
    if report.get('job_hash')!=digest(job) or report.get('completed') is not True or report.get('full_model_recompute') is not True or len(report.get('records',[]))!=7 or not all(r.get('full_proof_verified') is True for r in report.get('records',[])):raise ValueError('complete fresh model evidence')
    if report['public_trace_sha256']!=hashlib.sha256(public_bytes).hexdigest() or report['public_trace_sha256']!=job['public_trace_sha256'] or report['checkpoint_files']!=job['checkpoint']['files'] or report['checkpoint']!=digest(job['checkpoint']['files']):raise ValueError('model checkpoint/public bytes binding')
    if report['runtime_profile']['runtime_source_sha256']!=job['runtime_source_sha256'] or report['runtime_profile']['policy']!=job['policy'] or report['harness_source_sha256']!=job['harness_source_sha256']:raise ValueError('approved model source/profile')
    if not {f'turn-{i}.{suffix}' for i in range(7) for suffix in ('json','npz')}<=set(report['artifact_files']):raise ValueError('all model raw files must be hash-bound')
    for name,row in report['artifact_files'].items():
        if pathlib.Path(name).name!=name or file_sha(out/name)!=row['sha256'] or (out/name).stat().st_size!=row['size']:raise ValueError('fetched model artifact integrity')
    from ops.probe_long_context_eog_terminal import trace_rows
    from subnet import harness
    if file_sha(ROOT/'subnet/harness.py')!=job['harness_source_sha256']:raise ValueError('operator exact harness version')
    _,expected=trace_rows(json.loads(public_bytes),harness,job['terminal_text'])
    if len(expected)!=7 or len(report['records'])!=7:raise ValueError('all native six tool turns plus terminal')
    for row,claimed in zip(expected,report['records']):
        for field in ('turn_index','messages_sha256','action_sha256','observation_sha256'):
            if row[field]!=claimed[field]:raise ValueError('full native public-history binding')
        i=row['turn_index'];artifact=json.loads((out/f'turn-{i}.json').read_text());array=np.load(out/f'turn-{i}.npz',allow_pickle=False)['logprobs']
        if artifact['profile']!=report['runtime_profile'] or array.dtype!=np.float32 or array.shape!=(claimed['output_tokens'],151936) or not np.isfinite(array).all() or len(artifact['prompt'])!=claimed['prompt_tokens'] or len(artifact['output'])!=claimed['output_tokens']:raise ValueError('full vocabulary probability framing')
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--out',required=True);p.add_argument('--public',required=True);p.add_argument('--seed-file',required=True);a=p.parse_args()
    job=authenticate(json.loads(pathlib.Path(a.job).read_text()),AUTHORITY);out=pathlib.Path(a.out);raw=pathlib.Path(a.public).read_bytes();report=json.loads((out/'report.json').read_text());validate_fetched(job,report,out,raw)
    from nacl.signing import SigningKey
    key=SigningKey(bytes.fromhex(pathlib.Path(a.seed_file).read_text()))
    if key.verify_key.encode().hex()!=AUTHORITY:raise ValueError('operator model audit signer')
    approved={k:report[k] for k in ('checkpoint','checkpoint_files','runtime_profile','harness_source_sha256')}
    audit={'kind':'controlled-original-eog-model-audit-v3-terminal',**approved,'public_trace_sha256':report['public_trace_sha256'],
           'numerical_tolerances':{'TOPLOC_errors':0,'logprobs_atol':1e-5,'logprobs_rtol':0},'full_model_recompute':True,
           'curated_target_model_computation':True,'originally_sampled':False,'records':report['records'],
           'terminal_text':job['terminal_text'],'terminal_model_proof':True,
           'artifact_files':report['artifact_files'],'report_sha256':file_sha(out/'report.json'),'approved_job_hash':digest(job),
           'sealer_source_sha256':file_sha(__file__),'native_grader_verified_here':False,'payable':False,'chain_transactions':False}
    for name,value in (('model-audit.json',{'payload':audit,'signer':AUTHORITY,'signature':base64.b64encode(key.sign(canonical(audit)).signature).decode()}),('approved-model.json',approved)):
        with (out/name).open('x') as f:f.write(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'model_audit':str(out/'model-audit.json'),'approved_model':str(out/'approved-model.json'),'native_grade_pending':True}),flush=True)
if __name__=='__main__':main()
