"""Prospective signed Tau2 epoch admission with an immutable auxiliary policy.

This checks authenticated audit lineage, not numerical/model or native replay.
Those expensive checks must precede the independent operator audit signature.
"""
import base64,hashlib,json,math,re
from nacl.signing import VerifyKey

VERSION='native-tau2-common-fixed-auxiliary-search-contract-v2'
AUDIT_VERSION='native-tau2-common-fixed-auxiliary-search-audit-v2'
OBJECTIVE='agent-only-native-outcome-preference-v1'
SEED_POLICY='task-seed-plus-role-ordinal-v1'
AGENT_SEED_POLICY='task-seed-plus-attempt-stride-plus-role-ordinal-v2'
AGENT_SEED_STRIDE=256

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def exact(a,b):return canonical(a)==canonical(b)
def authenticate(envelope,authority):
    if envelope.get('signer')!=authority:raise ValueError('trusted authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(envelope['payload']),base64.b64decode(envelope['signature'],validate=True))
    return envelope['payload']
def sha(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('SHA256 identity')
    return value
def source_map(value):
    if not isinstance(value,dict) or not value:raise ValueError('source closure')
    for name,value in value.items():
        if not isinstance(name,str) or name.startswith('/') or '..' in name.split('/'):raise ValueError('source path')
        sha(value)
def role_descriptor(role,kind):
    if role.get('kind')!=kind or role.get('training_eligible') is not (kind=='agent'):raise ValueError('role loss eligibility')
    cp=role.get('checkpoint',{});files=cp.get('files',{})
    source_map(files)
    if cp.get('id')!=digest(files) or 'config.json' not in files or not any(n.endswith('.safetensors') for n in files):raise ValueError('role checkpoint files')
    source_map(role.get('source_files'));sha(role.get('harness_source_sha256'));sha(role.get('interpreter_sha256'))
    if not isinstance(role.get('runtime_profile'),dict) or not role['runtime_profile'] or not isinstance(role.get('runtime_versions'),dict) or not role['runtime_versions']:raise ValueError('role runtime pins')
    if not isinstance(role.get('renderer'),str) or not role['renderer'] or not isinstance(role.get('request_model'),str) or not role['request_model']:raise ValueError('role renderer/model')
    for key,limit in [('max_context',32768),('max_output_tokens',512),('vocab_size',200000)]:
        if type(role.get(key)) is not int or not 1<=role[key]<=limit:raise ValueError('role context/token budget')
    if role.get('seed_policy')!=(AGENT_SEED_POLICY if kind=='agent' else SEED_POLICY) or type(role.get('seed_start')) is not int or not 0<=role['seed_start']<2**62:raise ValueError('role seed policy')
    if not exact(role.get('numerical_policy'),{'logprobs_atol':1e-5,'logprobs_rtol':0,'TOPLOC_errors':0}):raise ValueError('strict role numerical policy')

def validate_epoch(envelope,authority,approved_fixed_user):
    manifest=authenticate(envelope,authority)
    if manifest.get('version')!=VERSION or manifest.get('objective')!=OBJECTIVE or manifest.get('payable') is not False or manifest.get('chain_transactions') is not False:raise ValueError('controlled common contract scope')
    if not isinstance(manifest.get('epoch'),str) or not manifest['epoch']:raise ValueError('epoch identity')
    search=manifest.get('trajectory_search_policy',{})
    if set(search)!={'max_attempts','agent_seed_stride'} or type(search.get('max_attempts')) is not int or not 1<=search['max_attempts']<=1024 or type(search.get('agent_seed_stride')) is not int or search['agent_seed_stride']!=AGENT_SEED_STRIDE:raise ValueError('bounded trajectory search policy')
    roles=manifest.get('roles',{})
    if set(roles)!= {'agent','user'}:raise ValueError('exact Tau2 model role set')
    role_descriptor(roles['agent'],'agent');role_descriptor(roles['user'],'auxiliary')
    # This independently configured trust anchor is not taken from the sidecar.
    role_descriptor(approved_fixed_user,'auxiliary')
    if not exact(roles['user'],approved_fixed_user):raise ValueError('fixed auxiliary checkpoint/profile/source drift')
    if not exact(manifest.get('checkpoint'),roles['agent']['checkpoint']):raise ValueError('current agent checkpoint binding')
    env=manifest.get('environment',{})
    if not isinstance(env.get('id'),str) or not env['id'] or not isinstance(env.get('version'),str) or not env['version']:raise ValueError('environment identity')
    sha(env.get('taskset_sha256'));sha(env.get('data_inventory_sha256'));source_map(env.get('source_files'))
    tasks=manifest.get('tasks')
    if not isinstance(tasks,list) or not 1<=len(tasks)<=10000:raise ValueError('task manifest budget')
    seen=set()
    for task in tasks:
        if type(task.get('index')) is not int or task['index']<0 or task['index'] in seen or type(task.get('seed')) is not int or not 0<=task['seed']<2**62:raise ValueError('fixed task index/seed')
        sha(task.get('task_hash'));seen.add(task['index'])
    if manifest.get('sampler_provenance')!='curated-target-model-computation-only':raise ValueError('honest computation provenance scope')
    return manifest

def validate_attempt(manifest,attempt):
    if type(attempt) is not int or not 0<=attempt<manifest['trajectory_search_policy']['max_attempts']:raise ValueError('trajectory attempt bound')
    return attempt

def role_seed(role,task,attempt,ordinal):
    if type(ordinal) is not int or not 0<=ordinal<128:raise ValueError('role ordinal bound')
    return role['seed_start']+task['seed']+ordinal+(attempt*AGENT_SEED_STRIDE if role['kind']=='agent' else 0)

def admit_sample(epoch_envelope,receipt_envelopes,audit_envelope,report,authority,approved_fixed_user):
    manifest=validate_epoch(epoch_envelope,authority,approved_fixed_user)
    audit=authenticate(audit_envelope,authority)
    if audit.get('version')!=AUDIT_VERSION or audit.get('manifest_sha256')!=digest(manifest) or audit.get('signed_receipts_sha256')!=digest(receipt_envelopes) or audit.get('verification_report_sha256')!=digest(report):raise ValueError('audit sidecar lineage')
    attempt=validate_attempt(manifest,audit.get('trajectory_attempt'))
    env=manifest['environment'];task=next((t for t in manifest['tasks'] if t['index']==audit.get('environment_index')),None)
    if task is None or audit.get('epoch')!=manifest['epoch'] or audit.get('environment_id')!=env['id'] or audit.get('environment_version')!=env['version'] or audit.get('task_hash')!=task['task_hash']:raise ValueError('audit epoch/environment/task lineage')
    flags=('full_native_trajectory_verified','all_model_roles_verified','derived_responses_verified','source_closure_verified')
    if any(audit.get(k) is not True or report.get(k) is not True for k in flags):raise ValueError('complete independent native/model audit')
    if audit.get('originally_sampled') is not False or audit.get('payable') is not False or audit.get('sampler_provenance')!=manifest['sampler_provenance']:raise ValueError('audit computation scope')
    for key,value in [('trajectory_attempt',attempt),('epoch',manifest['epoch']),('environment_id',env['id']),('environment_version',env['version']),('task_hash',task['task_hash']),('environment_index',task['index'])]:
        if not exact(report.get(key),value):raise ValueError('verification report task lineage')
    reward=report.get('reward')
    if type(reward) not in (int,float) or not math.isfinite(reward) or reward not in (0,1) or not exact(audit.get('reward'),reward):raise ValueError('native reward binding')
    if not isinstance(receipt_envelopes,list) or not 1<=len(receipt_envelopes)<=128:raise ValueError('role receipt budget')
    records=[authenticate(r,authority) for r in receipt_envelopes]
    proofs=report.get('role_checks',[])
    if len(proofs)!=len(records):raise ValueError('complete role verification coverage')
    counters={'agent':0,'user':0};views=[]
    for ordinal,(record,check) in enumerate(zip(records,proofs)):
        role_name=record.get('role');role=manifest['roles'].get(role_name)
        if role is None:raise ValueError('undeclared role')
        for key,value in [('manifest_sha256',digest(manifest)),('epoch',manifest['epoch']),('environment_id',env['id']),('task_hash',task['task_hash']),('environment_index',task['index']),('role_descriptor_sha256',digest(role)),('ordinal',ordinal),('role_ordinal',counters[role_name]),('trajectory_attempt',attempt),('seed',role_seed(role,task,attempt,counters[role_name]))]:
            if not exact(record.get(key),value):raise ValueError('role/model/context/seed lineage')
        counters[role_name]+=1
        if not exact(record.get('checkpoint'),role['checkpoint']) or not exact(record.get('runtime_profile'),role['runtime_profile']) or record.get('harness_source_sha256')!=role['harness_source_sha256'] or record.get('renderer')!=role['renderer'] or not exact(record.get('source_files'),role['source_files']):raise ValueError('role model/profile/source binding')
        request=record.get('request');response=record.get('response')
        if not isinstance(request,dict) or request.get('model')!=role['request_model'] or record.get('request_sha256')!=digest(request) or record.get('response_sha256')!=digest(response):raise ValueError('complete request/response binding')
        prompt=record.get('prompt');output=record.get('output')
        if not isinstance(prompt,list) or not isinstance(output,list) or not prompt or not output or len(prompt)+len(output)>role['max_context'] or len(output)>role['max_output_tokens'] or any(type(t) is not int or not 0<=t<role['vocab_size'] for t in prompt+output):raise ValueError('complete context/token budget')
        if record.get('prompt_sha256')!=digest(prompt) or record.get('output_sha256')!=digest(output):raise ValueError('prompt/output commitment')
        sha(record.get('probabilities_sha256'))
        if not isinstance(record.get('proofs'),list) or not record['proofs']:raise ValueError('model proof artifacts required')
        eligible=role_name=='agent';mask=[eligible]*len(output)
        if 'loss_mask' in record and not exact(record['loss_mask'],mask):raise ValueError('auxiliary/agent loss mask')
        if check.get('signed_receipt_sha256')!=digest(receipt_envelopes[ordinal]) or check.get('role')!=role_name or any(check.get(k) is not True for k in ('model_computation_verified','context_verified','derived_response_verified')):raise ValueError('receipt-specific proof/context audit')
        views.append({'role':role_name,'prompt':prompt,'output':output,'loss_mask':mask,'training_eligible':eligible,'receipt_sha256':digest(receipt_envelopes[ordinal])})
    if not all(counters.values()):raise ValueError('both agent and auxiliary evidence required')
    return {'version':VERSION,'trajectory_attempt':attempt,'epoch':manifest['epoch'],'environment_id':env['id'],'environment_version':env['version'],'environment_index':task['index'],'task_hash':task['task_hash'],'checkpoint':manifest['checkpoint'],'fixed_user_sha256':digest(approved_fixed_user),'manifest_sha256':digest(manifest),'audit_sha256':digest(audit_envelope),'reward':reward,'classification':'positive' if reward==1 else 'negative','training_view':views,'sampler_provenance':manifest['sampler_provenance'],'originally_sampled':False,'payable':False,'production_admitted':False}

def preference_pair(positive,negative):
    if positive.get('version')!=VERSION or negative.get('version')!=VERSION or positive.get('classification')!='positive' or negative.get('classification')!='negative':raise ValueError('admitted preference labels')
    for key in ('epoch','environment_id','environment_version','environment_index','task_hash','checkpoint','fixed_user_sha256','manifest_sha256'):
        if not exact(positive.get(key),negative.get(key)):raise ValueError('preference task/role lineage')
    for sample in (positive,negative):
        if sample.get('production_admitted') is not False or sample.get('originally_sampled') is not False or sample.get('payable') is not False:raise ValueError('controlled preference scope')
        for view in sample['training_view']:
            expected=view['role']=='agent'
            if view.get('training_eligible') is not expected or not exact(view.get('loss_mask'),[expected]*len(view['output'])):raise ValueError('preference loss mask')
    # Select a same-context divergent agent decision; user tokens remain evidence.
    for chosen in positive['training_view']:
        if chosen['role']!='agent':continue
        for rejected in negative['training_view']:
            if rejected['role']=='agent' and chosen['prompt']==rejected['prompt'] and chosen['output']!=rejected['output']:
                return {'objective':OBJECTIVE,'checkpoint':positive['checkpoint'],'prompt':chosen['prompt'],'chosen':chosen['output'],'rejected':rejected['output'],'positive_trajectory_attempt':positive['trajectory_attempt'],'negative_trajectory_attempt':negative['trajectory_attempt'],'positive_audit_sha256':positive['audit_sha256'],'negative_audit_sha256':negative['audit_sha256'],'auxiliary_tokens_in_loss':False,'training_performed':False,'payable':False}
    raise ValueError('same-prompt divergent agent preference required')
