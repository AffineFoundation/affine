"""Prospective native-artifact adapter; no deployed epoch or chain integration.

Only independently audited role receipts may become masked training views.
All native artifacts remain in the descriptor, including auxiliary role evidence.
"""
import hashlib,json,math,pathlib
from .native_tau2_model import authenticate
from .native_tau2_probe import digest
from .native_auxiliary_roles import admit_records

def describe_sample(out,authority,environment_index):
    if type(environment_index)!=int or environment_index<0:raise ValueError('environment index')
    out=pathlib.Path(out)
    def load(name):return json.loads((out/name).read_text())
    plan=authenticate(load('plan.json'),authority)
    audit=authenticate(load('role-audit.json'),authority)
    report=load('independent-full-verification.json')
    if audit.get('native_report_hash')!=digest(report) or report.get('verified') is not True or report.get('full_native_trajectory_verified') is not True or report.get('checkpoint')!=plan['checkpoint']['id']:raise ValueError('native report binding')
    simulation=authenticate(load('simulation-receipt.json'),authority)
    reward=report['reward']
    if type(reward) not in (int,float) or not math.isfinite(reward) or report['task_hash']!=simulation['task_hash'] or reward!=simulation['simulation']['reward_info']['reward'] or report['termination']!=simulation['simulation']['termination_reason']:raise ValueError('native task/reward binding')
    contract=authenticate(load('role-contract.json'),authority)
    require_declared_roles(contract,plan)
    records=load('receipts.json');views=admit_records(load('role-contract.json'),records,load('role-audit.json'),authority)
    names={'plan.json','role-contract.json','role-audit.json','receipts.json','simulation-receipt.json','independent-full-verification.json','operator-source-closure.json','verifier-supplement.json','generation-native_tau2_model.py','generation-native_tau2_replay.py'}
    for signed in records:
        r=authenticate(signed,authority);name=r['probabilities_file']
        if pathlib.Path(name).name!=name:raise ValueError('array path')
        if hashlib.sha256((out/name).read_bytes()).hexdigest()!=r['probabilities_sha256']:raise ValueError('array digest')
        names.add(name)
    from .native_tau2_replay import message_view
    trajectory_hash=digest({'checkpoint':plan['checkpoint']['id'],'task_hash':simulation['task_hash'],'messages':message_view(simulation['simulation'])})
    files={name:{'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest(),'size':(out/name).stat().st_size} for name in sorted(names)}
    return {'schema':'native-role-sample-v1','trajectory_hash':trajectory_hash,'checkpoint':plan['checkpoint']['id'],'environment_index':environment_index,'task_hash':report['task_hash'],'reward':reward,'classification':'positive' if reward==1 else 'negative' if reward==0 else 'neutral','role_contract_hash':digest(authenticate(load('role-contract.json'),authority)),'files':files,'training_view':views,'objective':'agent-only-curated-supervised-v1','payable':False,'production_admitted':False}

def describe_batch(samples,k,l):
    if type(k)!=int or type(l)!=int or min(k,l)<1:raise ValueError('batch K/L')
    if not samples or len({(s['checkpoint'],s['environment_index'],s['task_hash']) for s in samples})!=1:raise ValueError('native batch environment/checkpoint')
    if any(s.get('schema')!='native-role-sample-v1' or s.get('payable') is not False or s.get('production_admitted') is not False for s in samples):raise ValueError('native controlled scope')
    # Duplicate proof artifacts are not distinct experience samples.
    identities=[s.get('trajectory_hash',digest(s['files'])) for s in samples]
    if len(set(identities))!=len(identities):raise ValueError('duplicate native sample')
    positive=sum(s['classification']=='positive' for s in samples);negative=sum(s['classification']=='negative' for s in samples)
    if positive<k or negative<l:raise ValueError('native K/L incomplete')
    return {'schema':'native-role-batch-v1','checkpoint':samples[0]['checkpoint'],'environment_index':samples[0]['environment_index'],'task_hash':samples[0]['task_hash'],'K':positive,'L':negative,'samples':samples,'payable':False,'production_admitted':False}

def preference_pair(positive,negative):
    """An outcome-conditioned agent pair; auxiliary outputs stay as evidence.

This does not apply an optimizer update. It avoids interpreting failed agent
outputs as positive supervised targets. Both prompts must match exactly.
"""
    if positive.get('classification')!='positive' or negative.get('classification')!='negative':raise ValueError('preference labels')
    if any(positive.get(k)!=negative.get(k) for k in ('checkpoint','environment_index','task_hash')):raise ValueError('preference task/checkpoint')
    agents=[]
    for sample in (positive,negative):
        eligible=[r for r in sample['training_view'] if r['role']=='agent' and r['training_eligible']]
        if not eligible:raise ValueError('agent target missing')
        agents.append(eligible[0])
    if agents[0]['prompt']!=agents[1]['prompt'] or agents[0]['output']==agents[1]['output']:raise ValueError('preference prompt/output')
    return {'objective':'agent-only-native-outcome-preference-v1','checkpoint':positive['checkpoint'],'prompt':agents[0]['prompt'],'chosen':agents[0]['output'],'rejected':agents[1]['output'],'positive_artifact_hash':digest(positive['files']),'negative_artifact_hash':digest(negative['files']),'auxiliary_tokens_in_loss':False,'training_performed':False,'payable':False}


def require_declared_roles(contract,plan):
    """Declared role provenance must equal the actually verified plan."""
    expected_source=plan['curated_sources']['subnet/native_tau2_curated.py']
    if set(contract['roles'])!={'agent','user'}:raise ValueError('native role set')
    for role in contract['roles'].values():
        if role['checkpoint']!=plan['checkpoint']['id'] or role['source_hash']!=expected_source or role['numerical_policy']!=plan['runtime_profile']:raise ValueError('declared role provenance')
