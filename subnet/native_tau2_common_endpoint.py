"""Prospective controlled two-runtime Tau2 endpoint; no server/process launcher.

All injected runtime/renderer/policy objects are operator-owned dependencies,
never deserialized from miner artifacts. Numerical/native admission is separate.
"""
import copy,hashlib,math,pathlib,threading,time,os
from .native_tau2_common_contract import validate_epoch,digest,exact,canonical,sha,source_map,authenticate

VERSION='native-tau2-common-two-runtime-endpoint-v1'

def verify_sources(descriptor):
    root=pathlib.Path(__file__).resolve().parent.parent
    files=descriptor['source_files']
    for required in ('subnet/native_tau2_model.py','subnet/harness.py','subnet/native_tau2_common_contract.py','subnet/native_tau2_common_endpoint.py','subnet/proofs.py','subnet/batches.py'):
        if required not in files:raise ValueError('complete native renderer/action source pins')
    if files['subnet/harness.py']!=descriptor['harness_source_sha256']:raise ValueError('harness descriptor/source binding')
    for name,expected in files.items():
        path=root/name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('approved endpoint source bytes')

def validate_response_binding(record):
    """Verifier-side pure derived response check; no inference/source launch."""
    from .native_tau2_model import derived_response_message
    if record.get('role') not in ('agent','user'):raise ValueError('declared native response role')
    message,finish=derived_response_message(record['text'],record['ordinal'])
    expected={'id':f"native-{record['ordinal']}",'object':'chat.completion','created':int(record['created_at']),'model':record['request']['model'],'choices':[{'index':0,'message':message,'finish_reason':finish}],'usage':{'prompt_tokens':len(record['prompt']),'completion_tokens':len(record['output']),'total_tokens':len(record['prompt'])+len(record['output'])}}
    if not exact(record.get('response'),expected) or record.get('response_sha256')!=digest(expected):raise ValueError('derived native response/tool-call binding')
    if record.get('request_sha256')!=digest(record['request']) or record.get('prompt_sha256')!=digest(record['prompt']) or record.get('output_sha256')!=digest(record['output']):raise ValueError('complete request/token binding')
    eligible=record['role']=='agent'
    if record.get('training_eligible') is not eligible or not exact(record.get('loss_mask'),[eligible]*len(record['output'])):raise ValueError('auxiliary role loss mask')
    return True

class CommonRoleEndpoint:
    """Runtime interface: approved_descriptor(), tokenizer, sample(), compute(),
    build_proofs(). sample accepts prompt, seed, temperature, top_p, max_tokens.
    compute returns activations and full float32 emitted-token log probabilities.
    Constructor validates both independent role descriptors before any inference.
    """
    def __init__(self,epoch_envelope,authority,fixed_user,runtimes,signing_key,task_index,
                 source_validator=None,renderer=None,candidate_policy=None,clock=None,artifact_dir=None,user_candidate_policy=None):
        self.manifest=copy.deepcopy(validate_epoch(epoch_envelope,authority,fixed_user))
        if signing_key.verify_key.encode().hex()!=authority:raise ValueError('endpoint signer authority')
        if set(runtimes)!={'agent','user'}:raise ValueError('two distinct approved role runtimes required')
        self.task=next((t for t in self.manifest['tasks'] if t['index']==task_index),None)
        if self.task is None or type(task_index) is not int:raise ValueError('approved task index')
        self.runtimes=dict(runtimes);self.key=signing_key;self.authority=authority
        self.source_validator=source_validator or verify_sources
        for name,runtime in self.runtimes.items():
            descriptor=self.manifest['roles'][name]
            if not exact(runtime.approved_descriptor(),descriptor):raise ValueError('runtime role/checkpoint/profile descriptor')
            self.source_validator(descriptor)
        # Imports the unchanged renderer only after authority/source validation.
        if renderer is None:
            from .native_tau2_model import render
            renderer=render
        self.renderer=renderer;self.candidate_policies={'agent':candidate_policy,'user':user_candidate_policy}
        for role_name,policy in self.candidate_policies.items():
            expected=self.manifest['roles'][role_name].get('candidate_policy')
            if policy is None:
                if expected is not None:raise ValueError('declared public role policy dependency missing')
                continue
            if not isinstance(expected,dict) or expected.get('scope')!='public-request-derived-curated-output' or not exact(policy.approved_descriptor(),expected):raise ValueError('explicit pinned public role candidate policy')
            source_map(expected.get('source_files'))
            for name,value in expected['source_files'].items():
                if self.manifest['roles'][role_name]['source_files'].get(name)!=value:raise ValueError('candidate policy source closure')
        self.artifact_dir=pathlib.Path(artifact_dir) if artifact_dir is not None else None
        if self.artifact_dir is not None:self.artifact_dir.mkdir(parents=True,exist_ok=False);self.artifact_dir.chmod(0o700)
        self.clock=clock or time.time;self.records=[];self.counters={'agent':0,'user':0};self.lock=threading.Lock()

    def response(self,request):
        import numpy as np
        from .native_tau2_model import derived_response_message
        with self.lock:
            if not isinstance(request,dict) or not isinstance(request.get('messages'),list) or not request['messages']:raise ValueError('complete native request required')
            matches=[name for name,role in self.manifest['roles'].items() if role['request_model']==request.get('model')]
            if len(matches)!=1:raise ValueError('unapproved request model role')
            name=matches[0];role=self.manifest['roles'][name];runtime=self.runtimes[name]
            # Recheck descriptor/source before every model call, including user.
            if not exact(runtime.approved_descriptor(),role):raise ValueError('runtime role drift')
            self.source_validator(role)
            raw=copy.deepcopy(request);started=self.clock();ordinal=len(self.records);local=self.counters[name]
            seed=role['seed_start']+self.task['seed']+local
            prompt=self.renderer(runtime.tokenizer,copy.deepcopy(raw))
            if not isinstance(prompt,list) or not prompt or any(type(t) is not int or not 0<=t<role['vocab_size'] for t in prompt):raise ValueError('complete rendered token context')
            count=role['max_output_tokens']
            if len(prompt)+count>role['max_context']:raise ValueError('complete context exceeds approved role budget')
            generation=role.get('generation_policy',{'temperature':.7,'top_p':1.})
            if set(generation)!={'temperature','top_p'} or any(type(generation[k]) not in (int,float) or not math.isfinite(generation[k]) for k in generation) or generation['temperature']<=0 or not 0<generation['top_p']<=1:raise ValueError('pinned role sampling policy')
            policy=self.candidate_policies[name]
            if policy is not None:
                if not exact(policy.approved_descriptor(),role['candidate_policy']):raise ValueError('public role candidate policy drift')
                text=policy.select(copy.deepcopy(raw),seed)
                if not isinstance(text,str):raise ValueError('public curated candidate text')
                output=runtime.tokenizer.encode(text,add_special_tokens=False)
                policy_scope=('fixed-user-' if name=='user' else 'agent-')+'public-request-derived-curated-output'
            else:
                output=runtime.sample(prompt,seed,generation['temperature'],generation['top_p'],count)
                policy_scope='controlled-autoregressive-runtime-output'
            if not isinstance(output,list) or not 0<len(output)<=count or any(type(t) is not int or not 0<=t<role['vocab_size'] for t in output):raise ValueError('model output tokens/budget')
            text=runtime.tokenizer.decode(output,skip_special_tokens=True)
            activations,probabilities=runtime.compute(prompt,output)
            if not isinstance(probabilities,np.ndarray) or probabilities.dtype!=np.float32 or probabilities.shape!=(len(output),role['vocab_size']) or not np.isfinite(probabilities).all():raise ValueError('complete float32 output log probabilities')
            proofs=runtime.build_proofs(activations,decode_batching_size=16,topk=128)
            if not isinstance(proofs,list) or len(proofs)!=1+math.ceil(len(output)/16) or any(not isinstance(p,str) or not p for p in proofs):raise ValueError('complete proof framing')
            message,finish=derived_response_message(text,ordinal)
            completed=self.clock()
            if not math.isfinite(started) or not math.isfinite(completed) or completed<started:raise ValueError('response clock interval')
            response={'id':f'native-{ordinal}','object':'chat.completion','created':int(started),'model':raw['model'],'choices':[{'index':0,'message':message,'finish_reason':finish}],'usage':{'prompt_tokens':len(prompt),'completion_tokens':len(output),'total_tokens':len(prompt)+len(output)}}
            # Hash full NPY bytes, exactly what operator artifact storage writes.
            import io,base64
            array=io.BytesIO();np.save(array,probabilities,allow_pickle=False);raw_array=array.getvalue()
            record={'endpoint_version':VERSION,'manifest_sha256':digest(self.manifest),'epoch':self.manifest['epoch'],'environment_id':self.manifest['environment']['id'],'task_hash':self.task['task_hash'],'environment_index':self.task['index'],'role_descriptor_sha256':digest(role),'ordinal':ordinal,'role_ordinal':local,'seed':seed,'role':name,'checkpoint':role['checkpoint'],'runtime_profile':role['runtime_profile'],'harness_source_sha256':role['harness_source_sha256'],'renderer':role['renderer'],'source_files':role['source_files'],'request':raw,'response':response,'request_sha256':digest(raw),'response_sha256':digest(response),'prompt':prompt,'output':output,'prompt_sha256':digest(prompt),'output_sha256':digest(output),'probabilities_sha256':hashlib.sha256(raw_array).hexdigest(),'probabilities_file':f'role-{ordinal}.npy','proofs':proofs,'text':text,'created_at':started,'completed_at':completed,'loss_mask':[name=='agent']*len(output),'training_eligible':name=='agent','generation_scope':policy_scope,'originally_sampled':False,'payable':False}
            validate_response_binding(record)
            envelope={'payload':record,'signer':self.authority,'signature':base64.b64encode(self.key.sign(canonical(record)).signature).decode()}
            if self.artifact_dir is not None:
                for filename,data in ((record['probabilities_file'],raw_array),(f'role-{ordinal}.json',canonical(envelope))):
                    fd=os.open(self.artifact_dir/filename,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                    with os.fdopen(fd,'wb') as stream:stream.write(data)
            self.records.append(envelope);self.counters[name]+=1
            return copy.deepcopy(response),copy.deepcopy(envelope),raw_array


def verify_receipt(epoch_envelope,receipt_envelope,array_bytes,authority,fixed_user,runtime,
                   task_index,source_validator=None,renderer=None,framing_validator=None,candidate_policy=None):
    """Independent role recomputation; original native replay is still separate.

    Optional dependency hooks are trusted test/operator code, never artifact
    fields. Default framing uses the qualified non-executable TOPLOC validator.
    A caller verifies the complete ordered sequence separately before signing
    the aggregate audit; one valid receipt does not verify a whole trajectory.
    """
    import numpy as np
    manifest=validate_epoch(epoch_envelope,authority,fixed_user)
    record=authenticate(receipt_envelope,authority)
    name=record.get('role');role=manifest['roles'].get(name)
    task=next((t for t in manifest['tasks'] if t['index']==task_index),None)
    if role is None or task is None or type(task_index) is not int:raise ValueError('approved role/task')
    for key in ('ordinal','role_ordinal'):
        if type(record.get(key)) is not int or not 0<=record[key]<128:raise ValueError('receipt ordinals')
    expected={'endpoint_version':VERSION,'manifest_sha256':digest(manifest),'epoch':manifest['epoch'],'environment_id':manifest['environment']['id'],'task_hash':task['task_hash'],'environment_index':task_index,'role_descriptor_sha256':digest(role),'seed':role['seed_start']+task['seed']+record['role_ordinal'],'checkpoint':role['checkpoint'],'runtime_profile':role['runtime_profile'],'harness_source_sha256':role['harness_source_sha256'],'renderer':role['renderer'],'source_files':role['source_files']}
    if any(not exact(record.get(key),value) for key,value in expected.items()):raise ValueError('role verification descriptor/seed lineage')
    if not exact(runtime.approved_descriptor(),role):raise ValueError('fresh verifier runtime role descriptor')
    (source_validator or verify_sources)(role)
    validate_response_binding(record)
    if record['request'].get('model')!=role['request_model']:raise ValueError('request model role')
    if renderer is None:
        from .native_tau2_model import render
        renderer=render
    prompt=renderer(runtime.tokenizer,copy.deepcopy(record['request']));output=record['output']
    if prompt!=record['prompt']:raise ValueError('fresh complete context mismatch')
    if not isinstance(prompt,list) or not isinstance(output,list) or not prompt or not output or len(prompt)+len(output)>role['max_context'] or len(output)>role['max_output_tokens'] or any(type(t) is not int or not 0<=t<role['vocab_size'] for t in prompt+output):raise ValueError('fresh role token budget')
    if runtime.tokenizer.decode(output,skip_special_tokens=True)!=record['text']:raise ValueError('fresh decoded response tokens')
    declared=role.get('candidate_policy')
    if declared is not None:
        if candidate_policy is None or not exact(candidate_policy.approved_descriptor(),declared):raise ValueError('fresh fixed public role policy dependency')
        for filename,value in declared['source_files'].items():
            if role['source_files'].get(filename)!=value:raise ValueError('fresh public role policy source closure')
        proposal=candidate_policy.select(copy.deepcopy(record['request']),record['seed'])
        if not isinstance(proposal,str) or runtime.tokenizer.encode(proposal,add_special_tokens=False)!=output:raise ValueError('fresh public role policy output mismatch')
        scope=('fixed-user-' if name=='user' else 'agent-')+'public-request-derived-curated-output'
    else:
        if candidate_policy is not None:raise ValueError('undeclared public role policy')
        scope='controlled-autoregressive-runtime-output'
    if record.get('generation_scope')!=scope:raise ValueError('declared role generation scope')
    if not isinstance(array_bytes,bytes) or len(array_bytes)>role['max_output_tokens']*role['vocab_size']*4+10000 or hashlib.sha256(array_bytes).hexdigest()!=record['probabilities_sha256']:raise ValueError('bounded complete probability bytes')
    from .batches import bounded_tensor
    claimed=bounded_tensor(array_bytes)
    if claimed.shape!=(len(output),role['vocab_size']) or not np.isfinite(claimed).all():raise ValueError('complete role probability shape/finiteness')
    activations,actual=runtime.compute(prompt,output)
    if not isinstance(actual,np.ndarray) or actual.dtype!=np.float32 or actual.shape!=claimed.shape or not np.isfinite(actual).all() or not np.allclose(actual,claimed,atol=1e-5,rtol=0):raise ValueError('strict role probability recomputation')
    count=1+math.ceil(len(output)/16)
    if framing_validator is None:
        from .proofs import validate_framing
        framing_validator=validate_framing
    framing_validator(record['proofs'],count)
    checks=runtime.verify_proofs(activations,record['proofs'],decode_batching_size=16,topk=128)
    if len(checks)!=count or any(getattr(row,key,None)!=0 for row in checks for key in ('exp_mismatches','mant_err_mean','mant_err_median')):raise ValueError('strict role TOPLOC recomputation')
    return {'signed_receipt_sha256':digest(receipt_envelope),'role':name,'model_computation_verified':True,'context_verified':True,'derived_response_verified':True,'probabilities_sha256':record['probabilities_sha256'],'native_replay_performed_here':False,'role_descriptor_sha256':digest(role),'originally_sampled':False}
