"""Operator-only original Tau2 simulation for fixed-auxiliary common epochs.

The child is trusted code, not a sandbox for uploaded miner code. Private tasks,
user instructions, database/grader outputs and simulation reports remain private.
Inference is supplied only by the separately authenticated loopback role endpoint.
"""
from __future__ import annotations
import argparse,hashlib,json,os,signal,subprocess,sys
from pathlib import Path
from urllib.parse import urlparse
from .native_tau2_common_contract import canonical,digest,validate_epoch
from .native_tau2_probe import REVISION,data_inventory,sanitized_env

REVISION_COMMON='original-tau2-fixed-auxiliary-simulation-v1'

def validator(epoch):
    # Fixed code dispatch only; signature authentication occurs in the selected validator.
    from .native_tau2_common_search_contract import VERSION as SEARCH_VERSION,validate_epoch as search_validate
    version=epoch.get('payload',{}).get('version')
    if version==SEARCH_VERSION:return search_validate
    return validate_epoch

def selected_task(collection,index,expected_hash):
    if collection.get('schema')!='original-affine-tau2-telecom-private-tasks-v1' or collection.get('data_revision')!=REVISION:
        raise ValueError('original private task collection')
    rows=collection.get('tasks',[])
    matches=[r for r in rows if r.get('index')==index and type(r.get('index')) is int]
    if len(matches)!=1:raise ValueError('unique approved native task index')
    row=matches[0]
    if row.get('task_hash')!=expected_hash or digest(row.get('task'))!=expected_hash:
        raise ValueError('original task commitment')
    return row['task']

def loopback_endpoint(value):
    parsed=urlparse(value)
    if parsed.scheme!='http' or parsed.hostname!='127.0.0.1' or parsed.path!='/v1' or parsed.query or parsed.fragment or parsed.username or parsed.password or parsed.port is None:
        raise ValueError('trusted local role endpoint required')
    return value

def prepare(epoch,authority,fixed_user,data,public_tasks,private_tasks,index,endpoint,output,trajectory_attempt=0):
    manifest=validator(epoch)(epoch,authority,fixed_user)
    if manifest.get('trajectory_search_policy') is not None:
        if type(trajectory_attempt) is not int or not 0<=trajectory_attempt<manifest['trajectory_search_policy']['max_attempts']:raise ValueError('bounded trajectory attempt')
    elif trajectory_attempt!=0:raise ValueError('v1 does not support trajectory search')
    task=next((t for t in manifest['tasks'] if t['index']==index),None)
    if task is None:raise ValueError('task absent from signed epoch')
    public=json.loads(Path(public_tasks).read_bytes());private=json.loads(Path(private_tasks).read_bytes())
    env=manifest['environment'];data=Path(data).resolve()
    if digest(public)!=env['taskset_sha256'] or public.get('data_revision')!=REVISION or private.get('data_revision')!=REVISION:
        raise ValueError('signed public taskset/original data revision')
    matches=[t for t in public.get('tasks',[]) if t.get('index')==index]
    original=selected_task(private,index,task['task_hash'])
    if len(matches)!=1 or matches[0].get('task_hash')!=task['task_hash'] or matches[0].get('task_id')!=original.get('id'):
        raise ValueError('public/private task commitment')
    if (data/'.tau2_revision').read_text().strip()!=REVISION:raise ValueError('original installed data revision')
    inventory=data_inventory(data)
    if digest(inventory)!=env['data_inventory_sha256'] or inventory!=private.get('data_inventory') or digest(inventory)!=public.get('data_inventory_sha256'):
        raise ValueError('complete original data inventory')
    # Authenticated environment source pins precede any native import.
    source=Path(__file__).resolve().parent.parent
    for name,expected in env['source_files'].items():
        path=source/name
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('approved native simulation source')
    if env['source_files'].get('subnet/native_tau2_common_simulation.py')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('simulation dispatcher source pin required')
    policy=manifest.get('native_execution_policy',{})
    if policy.get('revision')!=REVISION_COMMON or type(policy.get('max_steps')) is not int or not 1<=policy['max_steps']<=128 or type(policy.get('max_errors')) is not int or not 0<=policy['max_errors']<=8:
        raise ValueError('bounded original native execution policy')
    for kind in ('agent','user'):
        if '/' in manifest['roles'][kind]['request_model']:raise ValueError('local model identifier')
    return {'revision':REVISION_COMMON,'manifest_sha256':digest(manifest),'epoch':manifest['epoch'],'environment_id':env['id'],'environment_index':index,'task_hash':task['task_hash'],'task_id':original['id'],'seed':task['seed'],'trajectory_attempt':trajectory_attempt,'private_collection':str(Path(private_tasks).resolve()),'data_revision':REVISION,'data_inventory_sha256':digest(inventory),'endpoint':loopback_endpoint(endpoint),'models':{k:manifest['roles'][k]['request_model'] for k in ('agent','user')},'max_steps':policy['max_steps'],'max_errors':policy['max_errors'],'output':str(Path(output).resolve()),'fixed_user_sha256':digest(fixed_user),'payable':False,'chain_transactions':False}

def install_original_patches():
    # The same original Affine patches used by qualified native controls.
    from tau2.utils import llm_utils
    from tau2.user.base import UserState
    repo=Path(__file__).resolve().parent
    sys.path.insert(0,str(repo/'vendor/research/environments/tool_use/tau2_bench_v1'))
    sys.path.insert(0,str(repo/'vendor/legacy/rollouts/envs/affine_tau2_v1'))
    from affine_tau2_v1.harness import apply_example_values
    from tau2_bench_v1.harness import _to_litellm_messages,_flip_roles
    llm_utils.to_litellm_messages=_to_litellm_messages
    UserState.flip_roles=_flip_roles
    apply_example_values()

def execute_original(config,load_tasks,run_task,evaluation_type):
    loopback_endpoint(config['endpoint'])
    private=json.loads(Path(config['private_collection']).read_bytes())
    expected=selected_task(private,config['environment_index'],config['task_hash'])
    excluded={t.id for t in load_tasks('telecom','base')}
    matches=[t for t in load_tasks('telecom','full') if t.id==expected['id'] and t.id not in excluded]
    if len(matches)!=1 or digest(matches[0].model_dump(mode='json'))!=config['task_hash']:
        raise ValueError('original provider task/source drift')
    args={'api_base':config['endpoint'],'api_key':'native-local','timeout':600,'max_retries':0,'num_retries':0,'temperature':.7}
    simulation=run_task(domain='telecom',task=matches[0],agent='llm_agent',user='user_simulator',llm_agent='openai/'+config['models']['agent'],llm_args_agent=dict(args),llm_user='openai/'+config['models']['user'],llm_args_user=dict(args),max_steps=config['max_steps'],max_errors=config['max_errors'],seed=config['seed'],evaluation_type=evaluation_type)
    return {'revision':REVISION_COMMON,'manifest_sha256':config['manifest_sha256'],'epoch':config['epoch'],'environment_id':config['environment_id'],'environment_index':config['environment_index'],'task_hash':config['task_hash'],'fixed_user_sha256':config['fixed_user_sha256'],'trajectory_attempt':config.get('trajectory_attempt',0),'task':expected,'simulation':simulation.model_dump(mode='json'),'payable':False,'chain_transactions':False}

def child(config):
    from tau2.run import load_tasks,run_task
    from tau2.evaluator.evaluator import EvaluationType
    install_original_patches()
    result=execute_original(config,load_tasks,run_task,EvaluationType.ALL)
    path=Path(config['output']);path.write_bytes(canonical(result));path.chmod(0o600)

def run(config,data,private_directory,wall_seconds=3000):
    if type(wall_seconds) is not int or not 1<=wall_seconds<=3600:raise ValueError('native walltime budget')
    directory=Path(private_directory).resolve();directory.mkdir(parents=True,exist_ok=True);directory.chmod(0o700)
    cfg=directory/'simulation-config.json';cfg.write_bytes(canonical(config));cfg.chmod(0o600)
    with (directory/'native-child.log').open('wb') as log:
        proc=subprocess.Popen([sys.executable,'-B','-m','subnet.native_tau2_common_simulation','--child',str(cfg)],env=sanitized_env(data),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=proc.wait(timeout=wall_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise TimeoutError('bounded native simulation exceeded walltime')
    if code:raise RuntimeError('original native simulation child failed; inspect private operator log')
    result=json.loads(Path(config['output']).read_bytes())
    if result.get('manifest_sha256')!=config['manifest_sha256'] or result.get('task_hash')!=config['task_hash'] or result.get('fixed_user_sha256')!=config['fixed_user_sha256']:
        raise ValueError('native result epoch/task/auxiliary binding')
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child',type=Path,required=True);args=parser.parse_args();child(json.loads(args.child.read_bytes()))
if __name__=='__main__':main()
