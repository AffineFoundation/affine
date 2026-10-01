"""Scoped operator orchestration for isolated, permanently nonpayable GPU jobs."""
import base64,copy,hashlib,json,shlex,subprocess,tarfile,time
from pathlib import Path
from subnet.backend_jobs import SOURCE_FILES,REVISION,NUMERICAL_POLICY,BACKEND_PROFILE,canonical
from subnet.storage import Bucket,Gateway,Identity,sha
from subnet.controller import Controller

ROOT=Path(__file__).resolve().parent.parent
SSH=['ssh','-o','BatchMode=yes','-o','UserKnownHostsFile='+str(ROOT/'state/registered-pod-retained-known-hosts'),'-p','20059','root@90.95.12.246']
SCP=['scp','-q','-o','BatchMode=yes','-o','UserKnownHostsFile='+str(ROOT/'state/registered-pod-retained-known-hosts'),'-P','20059']
REMOTE='/root/gpu-backend-jobs-code';WORK='/root/gpu-backend-role-state'
MINER='598fa5ced6b34e5123ba0033c0af4536c0f53c480e3143bbda14f851486e7d90'

def ssh(command):return subprocess.check_output(SSH+[command],text=True)
def main():
    stamp=str(int(time.time()));state=ROOT/'state/multi-environment/gpu-service'/stamp;state.mkdir(parents=True);state.chmod(0o700)
    seed=(ROOT/'state/service-conformance/authority.seed').read_text();(state/'authority.seed').write_text(seed);(state/'authority.seed').chmod(0o600)
    archive=ROOT/'state/source-bundles/source-before-resource-adapter-25bb0bdb.tar.gz'
    with tarfile.open(archive) as tar:
        for item in tar.getmembers():
            if item.issym() or item.islnk() or item.name.startswith('/') or '..' in Path(item.name).parts:raise ValueError('unsafe approved source archive')
    subprocess.run(SCP+[str(archive),'root@90.95.12.246:/root/gpu-backend-source.tar.gz'],check=True)
    ssh('mkdir -p '+REMOTE+'; tar -xzf /root/gpu-backend-source.tar.gz -C '+REMOTE+'; cp -a /root/affine-service-miner/subnet/vendor/. '+REMOTE+'/subnet/vendor/')
    subprocess.run(SCP+[str(ROOT/'subnet/backend_jobs.py'),'root@90.95.12.246:'+REMOTE+'/subnet/backend_jobs.py'],check=True)
    meta=ssh('cd '+REMOTE+' && /root/miner-venv/bin/python -B -c '+shlex.quote("import json,hashlib;from pathlib import Path;from importlib.metadata import version;from subnet.harness import source_hash;print(json.dumps(dict(source_files={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in "+repr(SOURCE_FILES)+"},runtime_versions={n:version(n) for n in ['torch','transformers','toploc']},harness_source_hash=source_hash())))"))
    meta=json.loads(meta)
    bucket=Bucket(json.loads((ROOT/'state/r2-direct.json').read_text()));gateway=Gateway(bucket,state_path=state/'gateway.json',public_url='http://unused-gpu-job.invalid',direct_r2=True)
    controller=Controller(bucket,gateway,state);receipt=json.loads((ROOT/'state/multi-environment/gpu-training-pilot/checkpoint-receipt.json').read_text())
    cp=dict(id=receipt['filemap_sha256'],files=receipt['filemap'],descriptor_key='public/checkpoints/'+receipt['filemap_sha256']+'/authorities/'+controller.authority.id+'/checkpoint.json')
    spec=json.loads((ROOT/'state/original-task-snapshots/oolong-date-fixed4.spec.json').read_text());harness=json.loads((ROOT/'state/multi-environment/oolong-date-balanced-harness.json').read_text())
    epochs=[];results={}
    def manifest(checkpoint,number):
        epoch='nonpayable-gpu-service-'+stamp+'-'+str(number)
        doc=controller.open(epoch,checkpoint,[MINER],duration=1800,environments=[dict(spec=spec,harness=harness,indices=[0])],audit_policy={'mode':'full','version':1},model_runtime_revision=REVISION,numerical_policy=NUMERICAL_POLICY,backend_profile=BACKEND_PROFILE,model_id='SmolLM2-1.7B-Instruct')
        doc['harness_source_hash']=meta['harness_source_hash'];doc['source_bundle_sha256']=sha(archive.read_bytes());doc['operator_authorized_experiment']=True
        (state/(epoch+'-manifest.json')).write_bytes(canonical(doc));bucket.json('public/'+epoch+'/manifest.json',controller.signed(doc));epochs.append(epoch)
        original=ROOT/'state/service-conformance/nonpayable-service-conformance-1790823563-5-registrations.json'
        snapshot=json.loads(original.read_text());(state/(epoch+'-registrations.json')).write_bytes(canonical(snapshot))
        return doc
    def run(role,doc,cache,**fields):
        identifier=stamp+'-'+role+'-'+str(len(results));now=time.time()
        job=dict(schema=1,job_id=identifier,role=role,created_at=now,expires_at=now+3600,manifest=controller.signed(doc),**meta)
        job.pop('harness_source_hash');job.update(fields);envelope=controller.signed(job)
        path=state/(identifier+'-job.json');path.write_bytes(canonical(envelope));path.chmod(0o600)
        subprocess.run(SCP+[str(path),'root@90.95.12.246:/root/gpu-backend-job.json'],check=True)
        command='cd '+REMOTE+' && CUBLAS_WORKSPACE_CONFIG=:4096:8 /root/miner-venv/bin/python -B -m subnet.backend_jobs /root/gpu-backend-job.json --authority '+controller.authority.id+' --workspace '+WORK+' --checkpoint-cache '+shlex.quote(cache)
        print(json.dumps(dict(stage='start',role=role,job=identifier)),flush=True);print(ssh(command),flush=True)
        subprocess.run(SCP+['root@90.95.12.246:'+WORK+'/jobs/'+identifier+'/report.json',str(state/(identifier+'-report.json'))],check=True)
        report=json.loads((state/(identifier+'-report.json')).read_text());results[identifier]=report
        (state/'progress.json').write_bytes(canonical(dict(epochs=epochs,reports=results,source_archive_sha256=sha(archive.read_bytes()),chain_transactions=False)))
        return report
    def mine(doc,cache,start):
        capability=dict(put_url=bucket.presign('private/'+doc['epoch']+'/staging/'+MINER+'.zip','put_object',1800),headers={'Content-Type':'application/octet-stream'})
        report=run('mine',doc,cache,miner_id=MINER,capability=capability,search_budget=24,seed_start=start)
        frozen=gateway.freeze(doc['epoch']);entry=frozen[MINER]
        if entry['sha256']!=report['submission_sha256']:raise ValueError('frozen upload hash')
        (state/(doc['epoch']+'-freeze-policy.json')).write_bytes(canonical(dict(payable=False,early_freeze=True,reason='operator-authorized nonpayable role conformance',receipt=entry)))
        return dict(url=entry['read_url'],sha256=entry['sha256'])
    doc=manifest(cp,0);cache=receipt['remote_checkpoint_path'];submission=mine(doc,cache,500)
    verify=run('verify',doc,cache,submissions=[submission])
    if not verify['audits'][0]['accepted']:raise ValueError('no fully verified batch')
    train=run('train',doc,cache,submissions=[submission],steps=1);newcp=train['new_checkpoint'];newcache=newcp.pop('path')
    newdoc=copy.deepcopy(doc);newdoc['checkpoint']=dict(newcp,read_urls={n:bucket.presign('public/checkpoints/'+newcp['id']+'/'+n) for n in newcp['files']})
    autoreg={**harness,'policy':'autoregressive','max_output_tokens':16,'turn_overrides':{}}
    run('evaluate',newdoc,newcache,heldout=[dict(env_id=spec['id'],indices=[2,3],seeds=[202,203],harness=autoreg)])
    run('upload',newdoc,newcache,put_urls={n:bucket.presign('public/checkpoints/'+newcp['id']+'/'+n,'put_object',3600) for n in newcp['files']})
    import requests
    for name,expected in newcp['files'].items():
        check=hashlib.sha256()
        with requests.get(bucket.presign('public/checkpoints/'+newcp['id']+'/'+name),stream=True,timeout=180,allow_redirects=False) as response:
            response.raise_for_status()
            for part in response.iter_content(1024*1024):check.update(part)
        if check.hexdigest()!=expected:raise ValueError('operator independent R2 checkpoint digest')
    bucket.json('public/checkpoints/'+newcp['id']+'/authorities/'+controller.authority.id+'/checkpoint.json',controller.signed(newcp))
    fresh=manifest(newcp,1);freshsubmission=mine(fresh,newcache,700);run('verify',fresh,newcache,submissions=[freshsubmission])
    (state/'complete.json').write_bytes(canonical(dict(success=True,epochs=epochs,new_checkpoint=newcp,chain_transactions=False,full_model_finetune=False,operator_independent_R2_rehash_passed=True)))
    print(json.dumps(dict(stage='complete',state=str(state),checkpoint=newcp['id'])),flush=True)
if __name__=='__main__':main()
