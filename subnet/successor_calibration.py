"""Bounded real checkpoint calibration before a prospective fast opening.

A completed GPU report is a measurement, not a policy admission. All numerical
bounds still pass fast_prefill_audit's strict validator; failures hold opening.
"""
from .storage import canonical
import hashlib
digest=lambda v:hashlib.sha256(canonical(v)).hexdigest()
VERSION='bounded-successor-calibration-v1'
RECALIBRATION_VERSION='bounded-successor-confirmation-recalibration-v2'
MINER_CALIBRATION_VERSION='bounded-miner-bound-successor-calibration-v3'
PROMPTS=('Solve 3x + 7 = 22. Explain each algebraic step.',
         'Find the positive integer n such that n squared equals 144. Explain the result.')

def request(value):
    miner_bound=isinstance(value,dict)and value.get('version')==MINER_CALIBRATION_VERSION
    fields={'version','env_id','harness','task_indices','max_tokens','draw_contract'}|({'miner'}if miner_bound else set())
    if type(value)is not dict or set(value)!=fields or value['version']not in(VERSION,MINER_CALIBRATION_VERSION):
        raise ValueError('exact successor calibration request')
    if type(value['env_id'])is not str or not value['env_id']:raise ValueError('calibration environment')
    h=value['harness']
    if type(h)is not dict:raise ValueError('calibration harness object')
    if h['policy']!='autoregressive' or h.get('turn_overrides'):raise ValueError('calibration sampling harness')
    if type(value['task_indices'])is not list or len(value['task_indices'])!=2 or any(type(i)is not int or i<0 for i in value['task_indices'])or len(set(value['task_indices']))!=2:raise ValueError('two distinct real calibration tasks')
    if type(value['max_tokens'])is not int or value['max_tokens']!=h['max_output_tokens'] or not 8<=value['max_tokens']<=2048:raise ValueError('production calibration output budget')
    d=value['draw_contract']
    if miner_bound:
        # Bootstrap admission must remain pure before the worker installs its
        # authenticated source finder. Full sampler validation is draw_context.
        fields={'version','randomness','max_attempts','verification','generation','calibration','support_adjudication'}
        if type(d)is not dict or set(d)!=fields or d['version']!='forced-inverse-cdf-prefill-miner-bound-v5' or type(d['max_attempts'])is not int or d['max_attempts']!=1000 or d['verification']!='prefill-cdf-calibrated' or d['generation']!='cached-eager-inverse-cdf' or d['support_adjudication']!='exact-cached-replay-v1' or type(d['calibration'])is not dict:raise ValueError('miner-bound qualification draw contract')
        miner=value['miner']
        if type(miner)is not str or len(miner)!=64 or any(x not in '0123456789abcdef'for x in miner):raise ValueError('authenticated calibration miner identity')
    elif type(d)is not dict or d.get('version')not in('forced-inverse-cdf-replay-v1','forced-inverse-cdf-prefill-v2','forced-inverse-cdf-prefill-support-v3','forced-inverse-cdf-prefill-threeway-v4')or type(d.get('max_attempts'))is not int or not 2<=d['max_attempts']<=128:raise ValueError('qualification draw contract')
    if d['version']=='forced-inverse-cdf-prefill-threeway-v4':
        from .forced_sampling import validate
        validate(d)
    seed=d.get('randomness')
    if type(seed)is not str or len(seed)!=64 or any(x not in '0123456789abcdef'for x in seed):raise ValueError('qualification public draws')
    return dict(value,harness=h)


def draw_context(manifest,value):
    """Signed request owns miner identity; v5 uses the production draw recipe.

    Initial predecessor calibration seeds measurement draws only. It never
    admits that calibration for the new checkpoint; fresh measured controls and
    confirmation still gate its prospective opening.
    """
    value=request(value)
    context=dict(epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],contract=value['draw_contract'])
    if value['version']==MINER_CALIBRATION_VERSION:
        from .forced_sampling import validate
        validate(value['draw_contract'])
        if (type(manifest.get('K'))is not int or type(manifest.get('L'))is not int or type(manifest.get('max_batches'))is not int or (manifest['K'],manifest['L'],manifest['max_batches'])!=(2,2,3)):raise ValueError('miner-bound calibration K2 L2 max3')
        context['miner']=value['miner']
    return context

def preflight_native_spec(spec):
    """Authenticate code/data/dependencies on the worker before model loading."""
    from .environments import create_session
    session=create_session(spec)
    session.close()

def execute(runtime, manifest, value):
    from . import fast_prefill_audit as fast,forced_sampling as forced
    import torch
    value=request(value)
    runtime.sampling_context=draw_context(manifest,value)
    from .environments import create_session
    session=create_session(runtime.spec)
    reports=[];native=[]
    for i in value['task_indices']:
        initial=session.reset(i,int(runtime.spec.config.get('seed',0)));prompt=runtime.prompt(initial['messages'],initial.get('tools',[]));task_hash=initial['task_hash']
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
    session.close()
    return dict(version=value['version'],checkpoint=manifest['checkpoint']['id'],request_sha256=digest(value),reports=reports,native_controls=native,assurance='executed-measurements-not-policy-admission',**(dict(sampling_miner=value['miner'],sampling_context_sha256=digest(runtime.sampling_context))if value['version']==MINER_CALIBRATION_VERSION else {}))

def admitted_policy(result,manifest,value):
    from .fast_prefill_audit import policy_from_executed_controls
    value=request(value)
    if type(result)is not dict or result.get('version')!=value['version'] or result.get('checkpoint')!=manifest['checkpoint']['id'] or result.get('request_sha256')!=digest(value) or result.get('assurance')!='executed-measurements-not-policy-admission':raise ValueError('original calibration report binding')
    if value['version']==MINER_CALIBRATION_VERSION and (result.get('sampling_miner')!=value['miner']or result.get('sampling_context_sha256')!=digest(draw_context(manifest,value))):raise ValueError('miner-bound calibration result context')
    controls=result.get('native_controls')
    if type(controls)is not list or len(controls)!=len(value['task_indices']):raise ValueError('complete native calibration controls')
    for rows in controls:
        if type(rows)is not list or not rows:raise ValueError('native calibration proof controls')
        for row in rows:
            if type(row)is not dict or set(row)!={'exp_mismatches','mant_err_mean','mant_err_median'} or type(row['exp_mismatches'])is not int or row['exp_mismatches']!=0 or any(type(row[k])not in(int,float)or row[k]!=0 for k in ('mant_err_mean','mant_err_median')):raise ValueError('native calibration proof mismatch')
    if len(result.get('reports',[]))!=len(value['task_indices']):raise ValueError('complete executed calibration reports')
    return policy_from_executed_controls(result['reports'],checkpoint=manifest['checkpoint']['id'],model_runtime_revision=manifest['model_runtime_revision'],backend_profile=manifest['backend_profile'],harness=value['harness'])

def before_open(controller,config,status,opening):
    """Use same original evaluated-role job on retry; never reuse another CP.

    Runs while no new mining epoch has been published. Its signed manifest is a
    qualification namespace and exact new checkpoint, not a historical epoch.
    """
    from .fast_prefill_audit import VERSION as FAST
    from .forced_sampling import VERSION as STRICT,MINER_VERSION,new_contract,source_hash
    from .remote_backend import save
    import json
    if config.get('sampling_policy',{}).get('version')not in(FAST,'forced-inverse-cdf-prefill-support-v3','forced-inverse-cdf-prefill-threeway-v4',MINER_VERSION):return opening
    opt=config.get('successor_calibration')
    if type(opt)is not dict:raise ValueError('fast openings require automatic successor calibration')
    bounded=opt.get('version')==RECALIBRATION_VERSION
    if bounded:
        if set(opt)!={'version','env_id','max_confirmations','deadline_seconds'} or type(opt['max_confirmations'])is not int or not 1<=opt['max_confirmations']<=4 or type(opt['deadline_seconds'])is not int or not 60<=opt['deadline_seconds']<=3600:raise ValueError('bounded successor confirmation policy')
    elif set(opt)!={'version','env_id'} or opt['version']!=VERSION:raise ValueError('fast openings require automatic successor calibration')
    rows=[dict(r,env_id=r['spec']['id'])for r in opening['environments']];row=next(r for r in rows if r['env_id']==opt['env_id'] and r['indices'])
    from .protocol import harness_for
    harness=harness_for(row,row['indices'][0])
    if len(row['indices'])<2:raise ValueError('two approved calibration tasks required')
    from .harness import normalize
    harness=normalize(harness)
    for i in row['indices'][:2]:
        if normalize(harness_for(row,i))!=harness:raise ValueError('same calibration harness task pair')
    miner_bound=config['sampling_policy']['version']==MINER_VERSION
    request_fields={}
    if miner_bound:
        owned=config.get('owned_miner_identity_files')
        if type(owned)is not dict or not owned or any(type(k)is not str or len(k)!=64 or any(c not in '0123456789abcdef'for c in k)for k in owned):raise ValueError('signed owned miner identity required for v5 calibration')
        request_fields['miner']=sorted(owned)[0]
    req=request(dict(version=MINER_CALIBRATION_VERSION if miner_bound else VERSION,env_id=row['env_id'],harness=harness,task_indices=row['indices'][:2],max_tokens=harness['max_output_tokens'],draw_contract=new_contract(config['sampling_policy']),**request_fields))
    manifest=dict(opening,epoch='nonpayable-successor-calibration-'+status['checkpoint']['id'][:16],checkpoint=controller.checkpoint_with_reads(status['checkpoint']),environments=rows,payable=False)
    from .harness import source_hash as harness_hash
    manifest['harness_source_hash']=harness_hash()
    manifest.pop('sampling_policy',None);manifest['sampling_contract']=new_contract(dict(version=STRICT,max_attempts=16));manifest['sampling_source_hash']=source_hash()
    # Stable nonce/original manifest is retained before remote dispatch.
    key=digest(dict(calibration_environment=row,checkpoint=status['checkpoint'],request={k:v for k,v in req.items()if k!='draw_contract'},sampling_policy=config['sampling_policy'],source=opening['source_bundle']['sha256'],runtime=opening['model_runtime_revision'],profile=opening['backend_profile']))
    if bounded:key=digest(dict(original_key=key,recalibration_policy=opt))
    manifest['epoch']+='-'+key[:12]
    path=controller.state/'successor-calibration'/(key+'.json')
    if path.exists():
        record=json.loads(path.read_text());manifest=record['manifest']
        if canonical(record.get('calibration_environment'))!=canonical(row) or not any(canonical(r)==canonical(row)for r in manifest.get('environments',[])):
            raise ValueError('immutable successor native environment binding changed')
        if {k:v for k,v in record['request'].items()if k!='draw_contract'}!={k:v for k,v in req.items()if k!='draw_contract'} or record['checkpoint']!=status['checkpoint'] or record['source_sha256']!=opening['source_bundle']['sha256']:raise ValueError('immutable successor calibration original changed')
        req=record['request']
    else:
        record=dict(calibration_environment=row,manifest=manifest,request=req,checkpoint=status['checkpoint'],source_sha256=opening['source_bundle']['sha256']);save(path,record)
    # The sole trainer is free only after its durable publication completed.
    # Do not compete with the independently running held-out evaluator.
    jobs=getattr(controller.jobs,'roles',{}).get('train',controller.jobs)
    cache=getattr(controller.jobs,'caches',{}).get('train',{}).get(status['checkpoint']['id'])
    report=jobs.run('successor-calibration-'+key[:32],'evaluate',manifest,cache,successor_calibration=req)
    policy=admitted_policy(report['successor_calibration'],manifest,req)
    if bounded:return _bounded_confirmation(controller,config,opening,record,path,key,manifest,req,report,jobs,cache,opt)
    # The first request's predecessor calibration only seeds qualification draws.
    # Confirm the proposal using fresh final-v3-contract draws; do not fabricate
    # successor admission or reattach historical tokens to new uniforms.
    final_request=dict(req,draw_contract=new_contract(dict(config['sampling_policy'],calibration=policy)))
    if 'confirmation_request'in record:final_request=record['confirmation_request']
    else:record['confirmation_request']=final_request;save(path,record)
    confirmation=jobs.run('successor-confirm-'+key[:32],'evaluate',manifest,cache,successor_calibration=final_request)
    confirmed=confirmation['successor_calibration'];admitted_policy(confirmed,manifest,final_request)
    if any(r['measured_cdf_abs_error']>policy['cdf_abs_error']or r['measured_logprob_abs_error']>policy['logprob_atol']for r in confirmed['reports']):raise ValueError('final-context calibration confirmation outside proposed bounds')
    record.update(confirmation_sha256=digest(confirmation),confirmation_original_job_id=confirmation['job_id'])
    # Only the operator-authenticated actual job reports can reach this point.
    record.update(report_sha256=digest(report),original_job_id=report['job_id'],calibration=policy);save(path,record)
    result=dict(opening);result['sampling_policy']=dict(config['sampling_policy'],calibration=policy)
    return result


def _bounded_confirmation(controller,config,opening,record,path,key,manifest,req,report,jobs,cache,opt):
    """Admit only a fresh confirmation; reports from failed rounds seed bounds.

    Job labels and draw requests are journaled before dispatch. An exhausted or
    expired journal refuses permanently instead of replaying an outside-bound
    report forever. RemoteJobs authenticates every returned original report.
    """
    import time
    from .remote_backend import save
    from .forced_sampling import new_contract
    from .fast_prefill_audit import policy_from_executed_controls
    seed=report['successor_calibration'];admitted_policy(seed,manifest,req)
    if 'bounded_confirmations'not in record:
        record['bounded_confirmations']=dict(version=RECALIBRATION_VERSION,created_at=time.time(),deadline=time.time()+opt['deadline_seconds'],rounds=[])
        save(path,record)
    journal=record['bounded_confirmations']
    if journal.get('version')!=RECALIBRATION_VERSION or type(journal.get('rounds'))is not list or len(journal['rounds'])>opt['max_confirmations'] or type(journal.get('created_at'))not in(int,float) or type(journal.get('deadline'))not in(int,float) or journal['deadline']-journal['created_at']>opt['deadline_seconds']+1 or journal['deadline']<=journal['created_at']:raise ValueError('immutable bounded calibration journal')
    rounds=journal['rounds'];reports=list(seed['reports'])
    token_only=opening.get('token_artifact_policy') is not None
    if token_only:
        from .token_only_protocol import for_manifest
        # Validate the explicit contract, never infer LP absence from inputs.
        for_manifest(opening)
    for index in range(opt['max_confirmations']):
        policy=policy_from_executed_controls(reports,checkpoint=manifest['checkpoint']['id'],model_runtime_revision=manifest['model_runtime_revision'],backend_profile=manifest['backend_profile'],harness=req['harness'],safety_factor=4.)
        if index<len(rounds):
            row=rounds[index]
            if row['policy']!=policy or row['label']!='successor-reconfirm-'+key[:24]+'-'+str(index):raise ValueError('immutable bounded calibration proposal changed')
        else:
            if time.time()>=journal['deadline']:raise ValueError('bounded successor confirmation deadline exhausted')
            row=dict(policy=policy,request=dict(req,draw_contract=new_contract(dict(config['sampling_policy'],calibration=policy))),label='successor-reconfirm-'+key[:24]+'-'+str(index))
            rounds.append(row);save(path,record)
        if time.time()>=journal['deadline']:raise ValueError('bounded successor confirmation deadline exhausted')
        # Reobserve the immutable signed RemoteJobs record on retries; a local
        # journal alone cannot authenticate an edited saved report.
        actual=jobs.run(row['label'],'evaluate',manifest,cache,successor_calibration=row['request'])
        admitted_policy(actual['successor_calibration'],manifest,row['request'])
        if 'report'in row and canonical(row['report'])!=canonical(actual):raise ValueError('immutable bounded confirmation report changed')
        row['report']=actual;save(path,record)
        if time.time()>=journal['deadline']:raise ValueError('bounded successor confirmation deadline exhausted')
        confirmed=actual['successor_calibration']
        admitted_policy(confirmed,manifest,row['request'])
        passed=all(r['measured_cdf_abs_error']<=policy['cdf_abs_error'] and (token_only or r['measured_logprob_abs_error']<=policy['logprob_atol'])for r in confirmed['reports'])
        if passed:
            record.update(confirmation_sha256=digest(actual),confirmation_original_job_id=actual['job_id'],report_sha256=digest(report),original_job_id=report['job_id'],calibration=policy);save(path,record)
            result=dict(opening);result['sampling_policy']=dict(config['sampling_policy'],calibration=policy);return result
        reports.extend(confirmed['reports'])
    raise ValueError('bounded successor confirmations exhausted; opening held')
