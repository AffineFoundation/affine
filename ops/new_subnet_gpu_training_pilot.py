"""Retained-GPU honest pair, independent audit, real update and fresh audit."""
import copy,gc,json,time,hashlib,os,subprocess
from pathlib import Path
from importlib.metadata import version
import numpy as np
import torch
from subnet.gpu_runtime import GPURuntime,PROFILE_VERSION
from subnet.model import model_files,file_hash

def main():
    out=Path('state/multi-environment/gpu-training-pilot');out.mkdir(parents=True,exist_ok=True)
    trust=json.loads(Path('/root/gpu-pilot-trust.json').read_text());checkpoint='/root/gpu-pilot-1.7b';files=trust['files']
    spec=json.loads(Path('state/original-task-snapshots/oolong-date-fixed4.spec.json').read_text())
    harness=json.loads(Path('state/multi-environment/oolong-date-balanced-harness.json').read_text())
    report={'success':False,'stage':'begin','model':trust['model'],'original_model_revision':trust['revision'],
        'approved_checkpoint_files':files,'approved_checkpoint_id':hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'environment':spec,'harness':harness,'training_indices':[0],'original_training_task_index':215,'heldout_indices':[2,3],
        'source_pins':{'gpu_runtime.py':file_hash('subnet/gpu_runtime.py'),'environments.py':file_hash('subnet/environments.py')},
        'profile':{'version':PROFILE_VERSION,'device':'cuda','dtype':'bfloat16','attention':'eager','tf32':False,'deterministic_algorithms':True,'sm':[8,6],
        'torch':version('torch'),'transformers':version('transformers'),'toploc':version('toploc'),'cuda':torch.version.cuda,'native_toploc_threads':2,
        'environment':{k:os.environ.get(k) for k in ['CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM']}}}
    started=time.time();runtime=None;artifacts=[];pairs={}
    def save():
        report['seconds']=round(time.time()-started,2);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'stage':report['stage'],'seconds':report['seconds']}),flush=True)
    def artifact(name,rollout,arrays):
        (out/(name+'.json')).write_text(json.dumps(rollout)+'\n');np.savez_compressed(out/(name+'.npz'),**{'turn_'+str(i):a for i,a in enumerate(arrays)})
        return {'name':name,'classification':rollout['classification'],'reward':rollout['reward'],'turns':len(rollout['turns']),
                'output_tokens':[len(t['output']) for t in rollout['turns']],'task_hash':rollout['task_hash'],'sha256':{suffix:file_hash(out/(name+suffix)) for suffix in ['.json','.npz']}}
    def release():
        nonlocal runtime
        runtime=None;gc.collect();torch.cuda.empty_cache()
    try:
        report['gpu']=torch.cuda.get_device_name();report['driver']=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()
        runtime=GPURuntime(checkpoint,files,spec,harness);report['stage']='candidate_rollouts';save()
        for seed in range(100,116):
            rollout,arrays=runtime.rollout(0,seed);row=artifact('before-'+str(seed),rollout,arrays);row['sampling']='curated public Counter+0/+1 under target GPU model';artifacts.append(row)
            pairs.setdefault(rollout['classification'],(rollout,arrays));report['artifacts']=artifacts;save()
            if {'positive','negative'}<=pairs.keys():break
        if not {'positive','negative'}<=pairs.keys():raise ValueError('bounded target sampling did not find both classes')
        release();runtime=GPURuntime(checkpoint,files,spec,harness);report['stage']='fresh_independent_pair_audit';save()
        report['pair_audits']={label:runtime.verify(doc,arrays) for label,(doc,arrays) in pairs.items()}
        pos=pairs['positive'][0];neg=pairs['negative'][0]
        forged=copy.deepcopy(pos);forged['reward']=.5
        try:report['reward_tamper_rejected']=not runtime.verify(forged,pairs['positive'][1])
        except ValueError:report['reward_tamper_rejected']=True
        if not all(report['pair_audits'].values()) or not report['reward_tamper_rejected']:raise ValueError('GPU pair audit failed')
        autoregressive={**harness,'policy':'autoregressive','max_output_tokens':16,'turn_overrides':{}}
        # These are the original distinct temporal-label questions, not the date
        # Counter task. No Counter policy or hidden answer is used here.
        runtime.configure(spec,autoregressive);report['heldout_before']=[]
        for index in [2,3]:
            doc,arrays=runtime.rollout(index,200+index);row=artifact('heldout-before-'+str(index),doc,arrays)
            row.update(index=index,sampling='free target-model autoregressive',verified=runtime.verify(doc,arrays));report['heldout_before'].append(row)
        runtime.configure(spec,harness);report['stage']='real_head_training';save()
        destination='/root/gpu-training-state/oolong-head-one-step-'+str(int(time.time()))
        report['training']=runtime.train([(pos,neg)],destination,steps=1)
        newfiles=model_files(destination);report['new_checkpoint']={'path':destination,'files':newfiles,'id':hashlib.sha256(json.dumps(newfiles,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
        if newfiles.get('model.safetensors')==files['model.safetensors']:raise ValueError('checkpoint weight SHA did not change')
        release();runtime=GPURuntime(destination,newfiles,spec,harness);doc,arrays=runtime.rollout(0,400)
        report['after_artifact']=artifact('after-new-checkpoint',doc,arrays);release()
        runtime=GPURuntime(destination,newfiles,spec,harness);report['after_independent_verified']=runtime.verify(doc,arrays)
        runtime.configure(spec,autoregressive);report['heldout_after']=[]
        for index in [2,3]:
            doc,arrays=runtime.rollout(index,200+index);row=artifact('heldout-after-'+str(index),doc,arrays)
            row.update(index=index,sampling='free target-model autoregressive',verified=runtime.verify(doc,arrays));report['heldout_after'].append(row)
        report['success']=report['after_independent_verified'] and report['training']['weights_changed'] and all(r['verified'] for r in report['heldout_before']+report['heldout_after'])
        report['stage']='complete'
    except Exception as e:
        report.update(stage='error',error=type(e).__name__+': '+str(e)[:800]);raise
    finally:
        release();report['GPU_peak_allocated_bytes']=torch.cuda.max_memory_allocated();report['GPU_allocated_bytes_after_cleanup']=torch.cuda.memory_allocated();save()
if __name__=='__main__':main()
