#!/usr/bin/env python3
"""Queued controlled CPU optimizer -> immutable R2 CP -> fresh native audit."""
import argparse,hashlib,json,os,pathlib,signal,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
AUTHORITY='d54a3a345d0de3e2c7898f30c0942d78f931f8c4b8036ffdc6adffcd2525062f'

def publication(checkpoint_path,receipt,out,key):
    from subnet.storage import Bucket
    from subnet.native_role_optimizer import canonical,digest,signed,authenticate
    from botocore.exceptions import ClientError
    bucket=Bucket(json.loads((ROOT/'state/r2-direct.json').read_text()));cp=receipt['checkpoint'];verified={}
    def read_object(name):
        try:return bucket.client.get_object(Bucket=bucket.name,Key=name)
        except ClientError as e:
            if str(e.response.get('Error',{}).get('Code')) not in ('404','NoSuchKey','NotFound'):raise
            return None
    def check_object(name,expected):
        response=read_object(name)
        if response is None:return None
        body=response['Body'];h=hashlib.sha256();size=0
        try:
            for block in iter(lambda:body.read(1024*1024),b''):h.update(block);size+=len(block)
        finally:body.close()
        if h.hexdigest()!=expected:raise ValueError('immutable native R2 checkpoint content collision')
        return {'sha256':h.hexdigest(),'size':size}
    for name,expected in cp['files'].items():
        object_key=f"public/checkpoints/{cp['id']}/{name}"
        found=check_object(object_key,expected)
        if found is None:
            bucket.upload(object_key,checkpoint_path/name);found=check_object(object_key,expected)
        if found is None or found['size']!=(checkpoint_path/name).stat().st_size:raise ValueError('native independent R2 size check')
        verified[name]=found
    descriptor={'id':cp['id'],'files':cp['files']};envelope=signed(descriptor,key)
    descriptor_key=f"public/checkpoints/{cp['id']}/authorities/{AUTHORITY}/checkpoint.json"
    response=read_object(descriptor_key)
    if response is None:bucket.json(descriptor_key,envelope)
    else:
        body=response['Body']
        try:existing=json.loads(body.read())
        finally:body.close()
        if authenticate(existing,AUTHORITY)!=descriptor:raise ValueError('native checkpoint descriptor collision')
    if authenticate(json.loads(bucket.get(descriptor_key)),AUTHORITY)!=descriptor:raise ValueError('native checkpoint descriptor readback')
    record={'checkpoint':cp,'descriptor_key':descriptor_key,'verified_files':verified,'independent_r2_stream_hash_verified':True,'training_receipt_hash':digest(receipt),'completed_at':time.time(),'payable':False,'chain_transactions':False}
    (out/'checkpoint-publication.json').write_text(json.dumps(signed(record,key),indent=2)+'\n');return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence-invocation',required=True);a=p.parse_args()
    from nacl.signing import SigningKey
    from subnet.native_role_optimizer import signed,authenticate,source_map,runtime_environment,digest,OPTIMIZER,REVISION
    from subnet.native_role_batch import describe_sample,describe_batch,preference_pair
    deadline=time.monotonic()+14400
    while True:
        value=subprocess.check_output(['systemctl','--user','show','affine-native-public-tools-evidence.service','-p','InvocationID','-p','ActiveState','-p','Result'],text=True)
        state=dict(line.split('=',1) for line in value.splitlines() if '=' in line)
        active=state.get('ActiveState') in ('active','activating')
        invocation=state.get('InvocationID')
        if (active or invocation) and invocation!=a.evidence_invocation:raise ValueError('native evidence invocation identity changed')
        if not active:
            # Successful transient units may be garbage-collected. Admission
            # below still requires every terminal signed report and exact K/L.
            if state.get('Result') not in ('success',None,''):raise ValueError('native K/L evidence unit failed')
            if not (ROOT/'state/native-tau2-probe/public-tools-model-K1L1.json').is_file():raise ValueError('native terminal K/L evidence missing')
            break
        if time.monotonic()>deadline:raise TimeoutError('native K/L unit still active; no retry')
        time.sleep(10)
    base=ROOT/'state/native-tau2-probe';positive=base/'public-tools-model-positive';negative=base/'public-tools-model-negative'
    kldata=json.loads((base/'public-tools-model-K1L1.json').read_text())
    pos=describe_sample(positive,AUTHORITY,0);neg=describe_sample(negative,AUTHORITY,0);batch=describe_batch([pos,neg],1,1)
    if digest(kldata)!=digest(batch):raise ValueError('terminal native K/L descriptor mismatch')
    pair=preference_pair(pos,neg)
    plan=authenticate(json.loads((positive/'plan.json').read_text()),AUTHORITY)
    out=ROOT/'state/native-tau2-training'/f'full-agent-pair-{int(time.time())}'
    out.mkdir(parents=True,exist_ok=False);out.chmod(0o700)
    seed=ROOT/'state/service-conformance/authority.seed';key=SigningKey(bytes.fromhex(seed.read_text()))
    if key.verify_key.encode().hex()!=AUTHORITY:raise ValueError('native training signer')
    job={'role':'native-agent-full-optimizer','revision':REVISION,'source_files':source_map(),'runtime_environment':runtime_environment(),'optimizer':OPTIMIZER,'budget':{'max_parameters':200000000,'max_context':8192,'max_peak_rss_bytes':64*1024**3,'min_disk_free_bytes':2*1024**3},'positive':{'path':str(positive),'descriptor_hash':digest(pos)},'negative':{'path':str(negative),'descriptor_hash':digest(neg)},'checkpoint':dict(plan['checkpoint'],path=str(ROOT/'state/service-conformance/checkpoints/nonpayable-service-conformance-1790827619-9')),'destination':str(out/'checkpoint'),'environment_index':0,'seed':20261001,'payable':False,'chain_transactions':False}
    (out/'optimizer-job.json').write_text(json.dumps(signed(job,key),indent=2)+'\n');(out/'preference-pair.json').write_text(json.dumps(pair,indent=2)+'\n')
    def run(command,label,timeout):
        with (out/(label+'.log')).open('wb') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=process.wait(timeout=timeout)
                if code:raise subprocess.CalledProcessError(code,command)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait();raise
    run([sys.executable,'-m','subnet.native_role_optimizer','--job',str(out/'optimizer-job.json'),'--authority',AUTHORITY,'--seed-file',str(seed)],'optimizer',3900)
    receipt=authenticate(json.loads((out/'training-receipt.json').read_text()),AUTHORITY)
    pub=publication(out/'checkpoint',receipt,out,key)
    fresh=out/'fresh-native-positive';data=base/'upstream/data'
    common=[sys.executable,'-m','subnet.native_tau2_curated','--out',str(fresh),'--data',str(data),'--checkpoint',str(out/'checkpoint'),'--seed-file',str(seed)]
    run(common+['--manifest',str(out/'checkpoint-manifest.json')],'fresh-native-generation',3900)
    run(common+['--verify','--authority',AUTHORITY],'fresh-native-independent',3900)
    report=json.loads((fresh/'independent-full-verification.json').read_text());audit=authenticate(json.loads((fresh/'role-audit.json').read_text()),AUTHORITY)
    if audit['native_report_hash']!=digest(report):raise ValueError('fresh native audit binding')
    group={'revision':'native-post-full-optimizer-control-v1','task_hash':report['task_hash'],'curated_sources':plan['curated_sources'],'runtime_profile':plan['runtime_profile'],'renderer':plan['renderer'],'auxiliary_checkpoint':receipt['checkpoint']['id'],'training_objective':receipt['objective']}
    summary={'completed':True,'optimizer_receipt':str(out/'training-receipt.json'),'previous_checkpoint':receipt['previous_checkpoint'],'checkpoint':receipt['checkpoint']['id'],'changed_parameter_elements':receipt['changed_parameter_elements'],'full_model_finetune':True,'auxiliary_tokens_in_loss':False,'independent_r2_stream_hash_verified':True,'fresh_native_audit':str(fresh/'role-audit.json'),'fresh_native_verified':True,'fresh_native_reward':report['reward'],'baseline_group':digest(group),'baseline_group_definition':group,'quality_improvement_claimed':False,'production_admitted':False,'payable':False,'chain_transactions':False,'completed_at':time.time()}
    (out/'completion.json').write_text(json.dumps(signed(summary,key),indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
