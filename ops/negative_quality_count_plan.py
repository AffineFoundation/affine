"""CPU-only prospective selection/resource controls; no GPU or authority API."""
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def resource_budget(parameters, model_bytes, state_bytes, concurrency=4):
    if type(concurrency)is not int or concurrency!=4:
        raise ValueError('declared four-stream local parent restore')
    count=sum(x['numel']for x in parameters);largest=max(x['numel']for x in parameters)
    return {'parameter_elements':count,'cpu_state_bytes':count*12,
            'cpu_additional_RAM_after_model_bytes':count*12+largest*24+4_000_000_000*concurrency+8*1024**3,
            'additional_disk_after_restored_inputs_bytes':model_bytes+4_000_000_000*concurrency+8*1024**3,
            'cold_model_restore_bytes':model_bytes,'cold_parent_state_restore_bytes':state_bytes,
            'cold_total_GET_bytes':model_bytes+state_bytes,
            'fully_retained_parent_and_two_branch_models_disk_bytes':state_bytes+model_bytes*3+4_000_000_000*concurrency+8*1024**3,
            'output_optimizer_export_bytes':0,'durable_research_optimizer_claim':False}


def paired_quality_refs(cap,eos,tasks):
    if not len(cap)==len(eos)==len(tasks)or [x['index']for x in cap]!=tasks or [x['index']for x in eos]!=tasks:
        raise ValueError('same predeclared task population')
    for p,n in zip(cap,eos):
        if p['positive']!=n['positive']or p['gradient_task_weight']!=n['gradient_task_weight']or p['gradient_task_weight']!=1/len(tasks):
            raise ValueError('negative-only contrast requires same positive and task weight')
        if p['negative']['rollout_sha256']==n['negative']['rollout_sha256']:
            raise ValueError('distinct negative treatments')
    return {'cap_refs_sha256':digest(cap),'eos_refs_sha256':digest(eos),'same_positive_all_tasks':True,'K':1,'L':1}


def select_eos_quota(streams,expected_tasks,eos_id=151645,cap=1024,minimum_tasks=8):
    """Matched K1/K2 from SAME sixteen attempts; negatives fixed to EOS stratum.

    Caller must independently authenticate original native+sampler artifacts,
    model/context/task bindings. This function never manufactures verification.
    It returns references/scalar supply, not new public batches or token dumps.
    """
    if set(streams)!=set(expected_tasks)or len(expected_tasks)!=minimum_tasks:
        raise ValueError('exact predeclared task set; no replacement/narrowing')
    selections=[];supply=[]
    for index in expected_tasks:
        rows=streams[index]
        if len(rows)!=16 or [x['attempt']for x in rows]!=list(range(16)):
            raise ValueError('all sixteen original prescribed attempts required')
        unique={};task_hashes=set();positive=[];negative=[]
        for row in rows:
            if row['index']!=index or row['status']not in ('verified','numerical_unknown','native_error','confirmed_invalid'):
                raise ValueError('exact task/status framing')
            if row['status']!='verified':continue
            if row.get('native_verified')is not True or row.get('sampler_verified')is not True:
                raise ValueError('actual native+sampler verification required')
            r=row['rollout'];turns=r['turns'];label=r['classification'];reward=r['reward']
            if r['index']!=index or r['seed']!=row['attempt']or label not in ('positive','negative')or type(reward)not in(int,float)or reward!=int(label=='positive')or len(turns)!=1:
                raise ValueError('authentic binary single-turn rollout binding')
            task_hashes.add(r['task_hash']);t=turns[0];output=t['output']
            if not output or len(output)>cap or eos_id in output[:-1]:raise ValueError('unchanged stop/cap semantics')
            # No claimed rewards/labels/proof serialization in content identity.
            content=digest([{k:q[k]for k in ('prompt','output','text','observations')}for q in turns])
            if content in unique:
                if unique[content]!=label:raise ValueError('identical content conflicting native outcome')
                continue
            unique[content]=label
            ref={'attempt':row['attempt'],'content_sha256':content,'rollout_sha256':digest(r)}
            if label=='positive':positive.append(ref)
            elif output[-1]==eos_id:negative.append(ref)
        if len(task_hashes)>1:raise ValueError('same native task hash required')
        eligible=min(len(positive),len(negative))>=2
        supply.append({'index':index,'unique_positive':len(positive),'unique_EOS_negative':len(negative),'matched':eligible})
        if eligible:selections.append({'index':index,'K1':{'positive':positive[:1],'negative':negative[:1]},'K2':{'positive':positive[:2],'negative':negative[:2]},'task_weight':1/minimum_tasks,'pair_weight_K1':1/minimum_tasks,'pair_weight_K2':1/(2*minimum_tasks)})
    return {'dispatch_allowed':False,'training_supply_gate_met':len(selections)==minimum_tasks,
            'matched_tasks':len(selections),'supply':supply,'selections':selections,
            'negative_quality_fixed':'EOS-completed native wrong','attempts':16,
            'no_task_replacement':True,'no_K1_only_population_substitution':True}
