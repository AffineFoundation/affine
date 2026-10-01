"""Fresh original Tau2 replay of authenticated fixed-user role responses.

This module does native replay only. It cannot replace independent model/proof
recomputation, and never declares all_model_roles_verified itself.
"""
import json,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from .native_tau2_common_contract import authenticate,canonical,digest,validate_epoch
from .native_tau2_common_simulation import prepare,run,validator


def messages(simulation):
    # Original wall-clock metadata is excluded; every native message/tool field
    # and the complete native reward/termination report remain exact.
    return [{k:v for k,v in m.items() if k!='timestamp'} for m in simulation['messages']]

def checked_records(epoch,receipts,authority,fixed_user,index,trajectory_attempt=0):
    manifest=validator(epoch)(epoch,authority,fixed_user)
    search=manifest.get('trajectory_search_policy')
    if search is not None:
        if type(trajectory_attempt) is not int or not 0<=trajectory_attempt<search['max_attempts']:raise ValueError('bounded replay trajectory attempt')
    elif trajectory_attempt!=0:raise ValueError('v1 does not support search attempts')
    task=next((t for t in manifest['tasks'] if t['index']==index),None)
    if task is None or not isinstance(receipts,list) or not 1<=len(receipts)<=128:
        raise ValueError('complete replay task/role budget')
    records=[];counts={'agent':0,'user':0}
    for ordinal,envelope in enumerate(receipts):
        record=authenticate(envelope,authority);role_name=record.get('role');role=manifest['roles'].get(role_name)
        if role is None:raise ValueError('replay role')
        expected={'manifest_sha256':digest(manifest),'epoch':manifest['epoch'],'environment_id':manifest['environment']['id'],'environment_index':index,'task_hash':task['task_hash'],'ordinal':ordinal,'role_ordinal':counts[role_name],'role_descriptor_sha256':digest(role),'checkpoint':role['checkpoint'],'seed':role['seed_start']+task['seed']+counts[role_name],'runtime_profile':role['runtime_profile'],'source_files':role['source_files'],'harness_source_sha256':role['harness_source_sha256'],'renderer':role['renderer']}
        if search is not None:
            expected['trajectory_attempt']=trajectory_attempt
            if role_name=='agent':expected['seed']+=trajectory_attempt*search['agent_seed_stride']
        if any(canonical(record.get(k))!=canonical(v) for k,v in expected.items()):
            raise ValueError('replay receipt current-agent/fixed-user/context lineage')
        request=record.get('request');response=record.get('response')
        if not isinstance(request,dict) or request.get('model')!=role['request_model'] or record.get('request_sha256')!=digest(request) or not isinstance(response,dict) or record.get('response_sha256')!=digest(response):
            raise ValueError('replay authenticated native request/response')
        records.append(record);counts[role_name]+=1
    if not all(counts.values()):raise ValueError('both model roles required')
    return manifest,records

class ReplayResponses:
    def __init__(self,records):self.records=records;self.cursor=0;self.lock=threading.Lock()
    def response(self,request):
        with self.lock:
            if self.cursor>=len(self.records) or digest(request)!=self.records[self.cursor]['request_sha256']:
                raise ValueError('fresh original request/context/role mismatch')
            response=self.records[self.cursor]['response'];self.cursor+=1;return response

def compare_native(original,actual,manifest,index,trajectory_attempt=0):
    for key,value in [('manifest_sha256',digest(manifest)),('epoch',manifest['epoch']),('environment_id',manifest['environment']['id']),('environment_index',index),('fixed_user_sha256',digest(manifest['roles']['user']))]:
        if original.get(key)!=value or actual.get(key)!=value:raise ValueError('native result signed epoch/auxiliary/task lineage')
    if manifest.get('trajectory_search_policy') is not None and (original.get('trajectory_attempt')!=trajectory_attempt or actual.get('trajectory_attempt')!=trajectory_attempt):raise ValueError('native trajectory attempt binding')
    if original['task_hash']!=actual['task_hash'] or digest(original['task'])!=original['task_hash'] or digest(actual['task'])!=actual['task_hash']:
        raise ValueError('original task equality')
    if messages(original['simulation'])!=messages(actual['simulation']):raise ValueError('original tool/message observations mismatch')
    for field in ('termination_reason','reward_info'):
        if original['simulation'][field]!=actual['simulation'][field]:raise ValueError('original grader/termination mismatch')
    return True

def replay(epoch,receipts,original_envelope,authority,fixed_user,index,data,public_tasks,private_tasks,output_directory,trajectory_attempt=0):
    manifest,records=checked_records(epoch,receipts,authority,fixed_user,index,trajectory_attempt)
    original=authenticate(original_envelope,authority);responses=ReplayResponses(records);errors=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                size=int(self.headers.get('Content-Length','0'))
                if self.path!='/v1/chat/completions' or self.headers.get('Authorization')!='Bearer native-local' or not 0<size<=2_000_000:
                    raise ValueError('bounded loopback replay request')
                result=responses.response(json.loads(self.rfile.read(size)));status=200
            except Exception as e:
                errors.append(type(e).__name__);result={'error':{'message':'Native replay rejected','type':'original_replay_error'}};status=400
            raw=canonical(result);self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    directory=Path(output_directory).resolve();directory.mkdir(parents=True,exist_ok=True);directory.chmod(0o700)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        config=prepare(epoch,authority,fixed_user,data,public_tasks,private_tasks,index,f'http://127.0.0.1:{server.server_port}/v1',directory/'fresh-original-simulation.json',trajectory_attempt)
        actual=run(config,data,directory,wall_seconds=180)
    finally:server.shutdown();server.server_close();thread.join(timeout=2)
    if errors or responses.cursor!=len(records):raise ValueError('native replay incomplete or extra role request')
    compare_native(original,actual,manifest,index,trajectory_attempt)
    report={'epoch':manifest['epoch'],'environment_id':manifest['environment']['id'],'environment_version':manifest['environment']['version'],'environment_index':index,'trajectory_attempt':trajectory_attempt,'task_hash':actual['task_hash'],'reward':actual['simulation']['reward_info']['reward'],'full_native_trajectory_verified':True,'all_model_roles_verified':False,'source_closure_verified':False,'derived_responses_verified':False,'reason':'Original tools/database/grader replay only; separately recompute every model role before combined admission.','signed_receipts_sha256':digest(receipts),'simulation_sha256':digest(actual),'payable':False,'chain_transactions':False}
    path=directory/'native-replay-report.json';path.write_bytes(canonical(report));path.chmod(0o600);return report
