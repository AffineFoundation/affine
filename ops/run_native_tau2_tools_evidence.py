#!/usr/bin/env python3
"""Bounded local native evidence continuation; never retries an active job."""
import argparse,json,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--positive-pid',type=int,required=True);p.add_argument('--positive-start',required=True);a=p.parse_args()
    root=ROOT/'state/native-tau2-probe';positive=root/'public-tools-model-positive';negative=root/'public-tools-model-negative'
    proc=pathlib.Path(f'/proc/{a.positive_pid}/stat');deadline=time.monotonic()+3900
    while proc.exists():
        parts=proc.read_text().split()
        if parts[21]!=a.positive_start:raise ValueError('positive PID identity changed')
        if parts[2]=='Z':break
        if time.monotonic()>deadline:raise TimeoutError('positive process still active; no retry')
        time.sleep(5)
    report=json.loads((positive/'report.json').read_text())
    if report['exit_code']!=0 or report['errors']:raise ValueError('positive native generation did not complete')
    cp=ROOT/'state/service-conformance/checkpoints/nonpayable-service-conformance-1790827619-9';data=root/'upstream/data';seed=ROOT/'state/service-conformance/authority.seed'
    authority='d54a3a345d0de3e2c7898f30c0942d78f931f8c4b8036ffdc6adffcd2525062f'
    common=[sys.executable,'-m','subnet.native_tau2_curated','--data',str(data),'--checkpoint',str(cp),'--seed-file',str(seed)]
    def job(args,label):
        with (root/'public-tools-model-logs'/f'{label}.log').open('wb') as log:subprocess.run(common+args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3900)
    job(['--verify','--out',str(positive),'--authority',authority],'positive-independent')
    if negative.exists():raise ValueError('negative namespace exists; no blind rerun')
    job(['--out',str(negative),'--negative','--manifest',str(ROOT/'state/service-conformance/nonpayable-service-conformance-1790828232-10-manifest.json')],'negative')
    job(['--verify','--out',str(negative),'--authority',authority],'negative-independent')
    from subnet.native_role_batch import describe_sample,describe_batch
    batch=describe_batch([describe_sample(positive,authority,0),describe_sample(negative,authority,0)],1,1)
    (root/'public-tools-model-K1L1.json').write_text(json.dumps(batch,indent=2)+'\n')
    print(json.dumps({'status':'complete','K':batch['K'],'L':batch['L'],'payable':False,'production_admitted':False}),flush=True)
if __name__=='__main__':main()
