"""Bounded real checkpoint calibration before a prospective fast opening.

A completed GPU report is a measurement, not a policy admission. All numerical
bounds still pass fast_prefill_audit's strict validator; failures hold opening.
"""
from .storage import canonical
import hashlib
digest=lambda v:hashlib.sha256(canonical(v)).hexdigest()
VERSION='bounded-successor-calibration-v1'
PROMPTS=('Solve 3x + 7 = 22. Explain each algebraic step.',
         'Find the positive integer n such that n squared equals 144. Explain the result.')

def request(value):
    if type(value)is not dict or set(value)!={'version','env_id','harness','prompts','max_tokens'} or value['version']!=VERSION:
        raise ValueError('exact successor calibration request')
    if type(value['env_id'])is not str or not value['env_id']:raise ValueError('calibration environment')
    h=value['harness']
    if type(h)is not dict:raise ValueError('calibration harness object')
    if h['policy']!='autoregressive' or h.get('turn_overrides'):raise ValueError('calibration sampling harness')
    if type(value['prompts'])is not list or not 2<=len(value['prompts'])<=4 or any(type(p)is not str or not 1<=len(p)<=1024 for p in value['prompts']):raise ValueError('bounded calibration prompts')
    if type(value['max_tokens'])is not int or not 8<=value['max_tokens']<=64 or value['max_tokens']>h['max_output_tokens']:raise ValueError('bounded calibration output')
    return dict(value,harness=h)

def execute(runtime, manifest, value):
    from . import fast_prefill_audit as fast,forced_sampling as forced
    import torch
    value=request(value)
    runtime.sampling_context=forced.binding(manifest)
    reports=[];native=[]
    for i,text in enumerate(value['prompts']):
        prompt=runtime.prompt([{'role':'user','content':text}],[]);task_hash=digest({'calibration_prompt':text})
        reports.append(fast.measure_cached_prefill(runtime,prompt,checkpoint=manifest['checkpoint']['id'],index=i,task_hash=task_hash,max_tokens=value['max_tokens']))
        context=runtime.sampling_context;tokens=torch.tensor([prompt],device=next(runtime.model.parameters()).device);cache=None;output=[];draws=[]
        with torch.inference_mode():
            for pos in range(value['max_tokens']):
                v=runtime.model(tokens,past_key_values=cache,use_cache=True);u=forced.uniform(context,runtime.spec.id,task_hash,i,0,0,pos)
                token=forced.pick(v.logits[0,-1],u,runtime.harness['temperature'],runtime.harness['top_p']);output.append(token);draws.append(u)
                cache=v.past_key_values;tokens=torch.tensor([[token]],device=tokens.device)
                if token==runtime.tokenizer.eos_token_id:break
        acts,lp=runtime.compute(prompt,output);proofs=runtime.build_proofs(acts,decode_batching_size=16,topk=128)
        replay,_=runtime.compute(prompt,output);stats=[dict(exp_mismatches=int(r.exp_mismatches),mant_err_mean=float(r.mant_err_mean),mant_err_median=float(r.mant_err_median))for r in runtime.verify_proofs(replay,proofs,16,128)]
        if not stats or any(v['exp_mismatches'] or v['mant_err_mean'] or v['mant_err_median'] for v in stats):raise ValueError('honest calibration native proof replay')
        native.append(stats)
    return dict(version=VERSION,checkpoint=manifest['checkpoint']['id'],request_sha256=digest(value),reports=reports,native_controls=native,assurance='executed-measurements-not-policy-admission')

def admitted_policy(result,manifest,value):
    from .fast_prefill_audit import policy_from_executed_controls
    value=request(value)
    if type(result)is not dict or result.get('version')!=VERSION or result.get('checkpoint')!=manifest['checkpoint']['id'] or result.get('request_sha256')!=digest(value) or result.get('assurance')!='executed-measurements-not-policy-admission':raise ValueError('original calibration report binding')
    controls=result.get('native_controls')
    if type(controls)is not list or len(controls)!=len(value['prompts']):raise ValueError('complete native calibration controls')
    for rows in controls:
        if type(rows)is not list or not rows:raise ValueError('native calibration proof controls')
        for row in rows:
            if type(row)is not dict or set(row)!={'exp_mismatches','mant_err_mean','mant_err_median'} or type(row['exp_mismatches'])is not int or row['exp_mismatches']!=0 or any(type(row[k])not in(int,float)or row[k]!=0 for k in ('mant_err_mean','mant_err_median')):raise ValueError('native calibration proof mismatch')
    if len(result.get('reports',[]))!=len(value['prompts']):raise ValueError('complete executed calibration reports')
    return policy_from_executed_controls(result['reports'],checkpoint=manifest['checkpoint']['id'],model_runtime_revision=manifest['model_runtime_revision'],backend_profile=manifest['backend_profile'],harness=value['harness'])

def before_open(controller,config,status,opening):
    """Use same original evaluated-role job on retry; never reuse another CP.

    Runs while no new mining epoch has been published. Its signed manifest is a
    qualification namespace and exact new checkpoint, not a historical epoch.
    """
    from .fast_prefill_audit import VERSION as FAST
    from .forced_sampling import VERSION as STRICT,new_contract,source_hash
    from .remote_backend import save
    import json
    if config.get('sampling_policy',{}).get('version')!=FAST:return opening
    opt=config.get('successor_calibration')
    if type(opt)is not dict or set(opt)!={'version','env_id','max_tokens'} or opt['version']!=VERSION:raise ValueError('fast openings require automatic successor calibration')
    rows=[dict(r,env_id=r['spec']['id'])for r in opening['environments']];row=next(r for r in rows if r['env_id']==opt['env_id'] and r['indices'])
    from .protocol import harness_for
    harness=harness_for(row,row['indices'][0]);req=request(dict(version=VERSION,env_id=row['env_id'],harness=harness,prompts=list(PROMPTS),max_tokens=opt['max_tokens']))
    manifest=dict(opening,epoch='nonpayable-successor-calibration-'+status['checkpoint']['id'][:16],checkpoint=controller.checkpoint_with_reads(status['checkpoint']),environments=rows,payable=False)
    from .harness import source_hash as harness_hash
    manifest['harness_source_hash']=harness_hash()
    manifest.pop('sampling_policy',None);manifest['sampling_contract']=new_contract(dict(version=STRICT,max_attempts=16));manifest['sampling_source_hash']=source_hash()
    # Stable nonce/original manifest is retained before remote dispatch.
    key=digest(dict(checkpoint=status['checkpoint'],request=req,source=opening['source_bundle']['sha256'],runtime=opening['model_runtime_revision'],profile=opening['backend_profile']))
    path=controller.state/'successor-calibration'/(key+'.json')
    if path.exists():
        record=json.loads(path.read_text());manifest=record['manifest']
        if record['request']!=req or record['checkpoint']!=status['checkpoint'] or record['source_sha256']!=opening['source_bundle']['sha256']:raise ValueError('immutable successor calibration original changed')
    else:
        record=dict(manifest=manifest,request=req,checkpoint=status['checkpoint'],source_sha256=opening['source_bundle']['sha256']);save(path,record)
    # The sole trainer is free only after its durable publication completed.
    # Do not compete with the independently running held-out evaluator.
    jobs=getattr(controller.jobs,'roles',{}).get('train',controller.jobs)
    cache=getattr(controller.jobs,'caches',{}).get('train',{}).get(status['checkpoint']['id'])
    report=jobs.run('successor-calibration-'+key[:32],'evaluate',manifest,cache,successor_calibration=req)
    policy=admitted_policy(report['successor_calibration'],manifest,req)
    # Only the operator-authenticated actual job report can reach this point.
    record.update(report_sha256=digest(report),original_job_id=report['job_id'],calibration=policy);save(path,record)
    result=dict(opening);result['sampling_policy']=dict(config['sampling_policy'],calibration=policy)
    return result
