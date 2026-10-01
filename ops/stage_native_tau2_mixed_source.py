"""Freeze a new private mixed-CUDA Tau2 prerequisite source/role namespace.

No deployment, subprocess/model inference, transactions or private-data upload.
Remote workers receive source and private startup descriptors only; native data
and operator signing key remain local. SSH argv are operator-owned, not artifacts.
"""
import argparse,base64,copy,hashlib,json,pathlib,re,shlex,sys,tarfile
ROOT=pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from subnet.long_context_runtime import canonical,digest,file_sha,AUTHORITY,authenticate
from subnet.native_tau2_common_search_contract import validate_epoch,AGENT_SEED_POLICY,SEED_POLICY
from subnet.native_auxiliary_cuda import REVISION as USER_REVISION,POLICY as USER_POLICY,CHECKPOINT as USER_CHECKPOINT
from subnet.native_role_cuda import REVISION as AGENT_REVISION
from subnet.native_tau2_probe import data_inventory,REVISION as DATA_REVISION

FILES=('subnet/__init__.py','subnet/batches.py','subnet/storage.py','subnet/harness.py','subnet/proofs.py','subnet/long_context_runtime.py','subnet/native_role_cuda.py','subnet/native_auxiliary_cuda.py','subnet/native_role_process.py','subnet/native_role_worker.py','subnet/native_role_model_only.py','subnet/native_tau2_role_runtime.py','subnet/native_tau2_common_contract.py','subnet/native_tau2_common_endpoint.py','subnet/native_tau2_common_search_contract.py','subnet/native_tau2_common_search_endpoint.py','subnet/native_tau2_common_cpu.py','subnet/native_tau2_public_policy.py','subnet/native_tau2_common_simulation.py','subnet/native_tau2_common_replay.py','subnet/native_tau2_common_mixed_driver.py','subnet/native_tau2_model.py','subnet/native_tau2_probe.py','ops/stage_native_tau2_mixed_source.py')
ENV={'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}

def sign(payload,key):return {'payload':payload,'signer':key.verify_key.encode().hex(),'signature':base64.b64encode(key.sign(canonical(payload)).signature).decode()}
def write(path,value):
    path=pathlib.Path(path);path.write_bytes(canonical(value));path.chmod(0o600)
def source_files(root=ROOT):
    root=pathlib.Path(root);names=list(FILES)
    for directory in ('subnet/vendor/legacy/rollouts/envs/affine_tau2_v1','subnet/vendor/research/environments/tool_use/tau2_bench_v1'):
        names.extend(str(p.relative_to(root)) for p in (root/directory).rglob('*.py') if '__pycache__' not in p.parts)
    result={}
    for name in sorted(set(names)):
        path=root/name
        if path.is_symlink() or not path.is_file():raise ValueError('regular approved source closure')
        result[name]=file_sha(path)
    return result

def bind_roles(agent,user,base_roles,sources,runtime_environment):
    roles={}
    for name,descriptor in (('agent',agent),('user',user)):
        row=copy.deepcopy(descriptor)
        for field in ('candidate_policy','seed_start','generation_policy','request_model'):
            row[field]=copy.deepcopy(base_roles[name][field])
        # Public policy bytes must equal previously qualified algorithm pins.
        if any(sources.get(path)!=expected for path,expected in row['candidate_policy']['source_files'].items()):raise ValueError('unchanged approved public policy source')
        row.update(source_files=copy.deepcopy(sources),interpreter_sha256=runtime_environment['interpreter_sha256'],runtime_versions=runtime_environment['packages'],seed_policy=AGENT_SEED_POLICY if name=='agent' else SEED_POLICY)
        row['model_runtime_revision']=row['native_role_revision']=AGENT_REVISION if name=='agent' else USER_REVISION
        roles[name]=row
    if roles['user']['checkpoint']['id']!=USER_CHECKPOINT or roles['user']['training_eligible'] is not False:raise ValueError('fixed auxiliary weights and mask')
    if roles['agent']['checkpoint']['id']==USER_CHECKPOINT:raise ValueError('independent approved agent checkpoint')
    return roles

def prepare(out,remote_root,data,public_tasks,private_tasks,key,known_hosts,root=ROOT):
    root=pathlib.Path(root);out=pathlib.Path(out).resolve();data=pathlib.Path(data).resolve()
    if not re.fullmatch(r'/root/native-tau2-mixed-[0-9]+',remote_root):raise ValueError('new isolated remote namespace')
    if key.verify_key.encode().hex()!=AUTHORITY:raise ValueError('scoped operator authority')
    public=json.loads(pathlib.Path(public_tasks).read_bytes());private=json.loads(pathlib.Path(private_tasks).read_bytes())
    if public.get('split_policy')!='original-tau2-user-instruction-group-disjoint-v1' or len(public.get('tasks',[]))!=32 or public.get('mining_indices')!=list(range(16)) or public.get('heldout_indices')!=list(range(16,32)):raise ValueError('scenario-disjoint32 original inventory')
    if public.get('data_revision')!=DATA_REVISION or private.get('data_revision')!=DATA_REVISION or (data/'.tau2_revision').read_text().strip()!=DATA_REVISION:raise ValueError('original data revision')
    inventory=data_inventory(data)
    if digest(inventory)!=public.get('data_inventory_sha256') or inventory!=private.get('data_inventory'):raise ValueError('complete original data inventory')
    if {t['scenario_group_sha256'] for t in public['tasks'][:16]}&{t['scenario_group_sha256'] for t in public['tasks'][16:]}:raise ValueError('scenario group overlap')
    for row in public['tasks']:
        matches=[p for p in private['tasks'] if p['index']==row['index']]
        if len(matches)!=1 or digest(matches[0]['task'])!=row['task_hash'] or matches[0]['task_hash']!=row['task_hash']:raise ValueError('private original task commitment')
    agent_job=authenticate(json.loads((root/'state/native-role-cuda/1790862267/job.json').read_bytes()),AUTHORITY)
    user_completion=authenticate(json.loads((root/'state/native-auxiliary-cuda/1790864280/completion.json').read_bytes()),AUTHORITY)
    base=json.loads((root/'state/native-tau2-common/role-control-v2a/signed-epoch.json').read_bytes())['payload']
    sources=source_files(root);roles=bind_roles(agent_job['descriptor'],user_completion['approved_descriptor'],base['roles'],sources,agent_job['runtime_environment'])
    manifest=copy.deepcopy(base);manifest.update(epoch='nonpayable-native-tau2-mixed-'+remote_root.rsplit('-',1)[1],roles=roles,checkpoint=roles['agent']['checkpoint'],tasks=[{'index':r['index'],'task_hash':r['task_hash'],'seed':20260930+r['index']} for r in public['tasks']],heldout_indices=public['heldout_indices'],mining_indices=public['mining_indices'])
    manifest['environment'].update(version='original-telecom-mixed-cuda-fixed-user-source-v1',source_files=sources,taskset_sha256=digest(public),data_inventory_sha256=digest(inventory),split_policy=public['split_policy'])
    epoch=sign(manifest,key);validate_epoch(epoch,AUTHORITY,roles['user'])
    out.mkdir(parents=True,exist_ok=False);out.chmod(0o700);stage=out/'source';stage.mkdir(mode=0o700)
    for name,expected in sources.items():
        target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((root/name).read_bytes());target.chmod(0o600)
        if file_sha(target)!=expected:raise ValueError('frozen source changed during copy')
    with tarfile.open(out/'remote-source.tar.gz','w:gz') as archive:
        for name in sources:archive.add(stage/name,arcname=name)
    write(out/'source-inventory.json',sources);write(out/'signed-epoch.json',epoch);write(out/'fixed-user.json',roles['user']);write(out/'public-tasks.json',public);write(out/'private-tasks.json',private)
    checkpoint_paths={'agent':agent_job['checkpoint']['path'],'user':user_completion['checkpoint']['path']};workers={}
    for name in ('agent','user'):
        config={'version':'native-role-process-json-v1','checkpoint':checkpoint_paths[name],'descriptor':roles[name]};write(out/f'{name}-worker.json',config)
        command='cd '+shlex.quote(remote_root+'/source')+' && env '+ ' '.join(shlex.quote(k+'='+v) for k,v in ENV.items())+' /root/miner-venv/bin/python -m subnet.native_role_worker --config '+shlex.quote(remote_root+'/'+name+'-worker.json')
        workers[name]={'argv':['ssh','-T','-p','20059','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(pathlib.Path(known_hosts).resolve()),'root@90.95.12.246',command],'environment':{},'stderr_path':str(out/(name+'-worker.private.log'))}
    write(out/'worker-configs.json',workers)
    report={'epoch':manifest['epoch'],'authority':AUTHORITY,'remote_root':remote_root,'source_root':str(stage),'source_inventory_sha256':digest(sources),'source_archive_sha256':file_sha(out/'remote-source.tar.gz'),'file_count':len(sources),'agent_checkpoint':roles['agent']['checkpoint']['id'],'user_checkpoint':roles['user']['checkpoint']['id'],'fixed_user_new_numerical_profile':USER_REVISION,'original_data':str(data),'scenario_group_overlap':0,'task_count':32,'inference_started':False,'whole_native_admission':False,'training_performed':False,'payable':False,'chain_transactions':False}
    write(out/'staging-report.json',report);return report

def main():
    from nacl.signing import SigningKey
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('out','data','public-tasks','private-tasks','seed-file','known-hosts'):parser.add_argument('--'+name,type=pathlib.Path,required=True)
    parser.add_argument('--remote-root',required=True);args=parser.parse_args()
    key=SigningKey(bytes.fromhex(args.seed_file.read_text().strip()))
    print(json.dumps(prepare(args.out,args.remote_root,args.data,args.public_tasks,args.private_tasks,key,args.known_hosts)))
if __name__=='__main__':main()
