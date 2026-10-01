"""Bounded native tau2 conformance probe; mock transport is never a model solve.

The child imports the approved installed tau2 only, uses original tasks/database,
original orchestrator and ALL grader. It runs in a separate process. This is not
an execution sandbox for miner-uploaded code and cannot authorize such code.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, signal, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REVISION = '337326e62d8e0ca74c353b004a9c5d748e0ba914'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def data_inventory(root):
    root=pathlib.Path(root)
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file() and not p.name.startswith('.')}

def sanitized_env(data):
    # Child receives neither account credentials nor model API credentials.
    result={k:os.environ[k] for k in ('PATH','LANG','HOME','LD_LIBRARY_PATH') if k in os.environ}
    result.update(TAU2_DATA_DIR=str(pathlib.Path(data).resolve()),LITELLM_LOCAL_MODEL_COST_MAP='True',LITELLM_LOG='ERROR',PYTHONUNBUFFERED='1',PYTHONPATH=str(pathlib.Path(__file__).resolve().parent.parent))
    return result

def child(config):
    from importlib.metadata import version
    from tau2.run import load_tasks,run_task
    from tau2.evaluator.evaluator import EvaluationType
    excluded={t.id for t in load_tasks('telecom','base')}
    pool=[t for t in load_tasks('telecom','full') if t.id not in excluded]
    task=pool[config['index']]
    # Same Litellm message/role preservation patches as the approved wrapper.
    from tau2.data_model.message import AssistantMessage
    from tau2.user.base import UserState
    from tau2.utils import llm_utils
    convert=llm_utils.to_litellm_messages;flip=UserState.flip_roles
    def to_messages(messages):
        out=convert(messages)
        for i,m in enumerate(messages):
            if isinstance(m,AssistantMessage) and m.raw_data:out[i]=m.raw_data['message']
        return out
    def flip_roles(self):
        out=flip(self)
        for a,b in zip(self.messages,out,strict=True):
            if isinstance(b,AssistantMessage):b.raw_data=a.raw_data
        return out
    llm_utils.to_litellm_messages=to_messages;UserState.flip_roles=flip_roles
    # Faithful Affine example-value tool-schema transformation.
    from tau2.domains.telecom.tools import TelecomTools
    examples={'phone_number':'555-123-4567','customer_id':'C1234','line_id':'L5678','plan_id':'P1002','id':'C1234','full_name':'Jane Doe'}
    for name in dir(TelecomTools):
        fn=getattr(TelecomTools,name,None);doc=getattr(fn,'__doc__',None)
        if not callable(fn) or not doc or name.startswith('_'):continue
        lines=[]
        for line in doc.splitlines():
            key,sep,rest=line.strip().partition(':')
            if sep and key in examples and rest.strip() and 'such as' not in rest and 'e.g.' not in rest and line.startswith(' '):line=line.rstrip().rstrip('.')+f", such as '{examples[key]}'."
            lines.append(line)
        fn.__doc__='\n'.join(lines)
    args={'api_base':config['endpoint'],'api_key':'local-conformance-only','timeout':10,'max_retries':0,'temperature':0}
    simulation=run_task(domain='telecom',task=task,agent='llm_agent',user='user_simulator',llm_agent='openai/native-probe-agent',llm_args_agent=args,llm_user='openai/native-probe-user',llm_args_user=args,max_steps=config['max_steps'],max_errors=3,seed=config['seed'],evaluation_type=EvaluationType.ALL)
    import tau2.run,tau2.orchestrator.orchestrator,tau2.evaluator.evaluator,tau2.domains.telecom.tools,tau2.utils.llm_utils
    modules=[tau2.run,tau2.orchestrator.orchestrator,tau2.evaluator.evaluator,tau2.domains.telecom.tools,tau2.utils.llm_utils]
    package_sources={m.__name__:hashlib.sha256(pathlib.Path(m.__file__).read_bytes()).hexdigest() for m in modules}
    result={'package_sources':package_sources,'task':task.model_dump(mode='json'),'task_hash':digest(task.model_dump(mode='json')),'pool_count':len(pool),'excluded_base_count':len(excluded),'simulation':simulation.model_dump(mode='json'),'package_versions':{n:version(n) for n in ('tau2','litellm','verifiers')},'transport':'mock-http-conformance','genuine_model_solve':False,'model_proofs_captured':False,'original_grader':'tau2.evaluator.evaluate_simulation/ALL'}
    pathlib.Path(config['result']).write_text(json.dumps(result,indent=2)+'\n')

class MockTransport:
    def __init__(self):self.calls=[];self.agent_calls=0;self.user_calls=0;self.lock=threading.Lock()
    def response(self,request):
        with self.lock:
            role='agent' if request.get('model')=='native-probe-agent' else 'user'
            if role=='agent':
                self.agent_calls+=1
                if self.agent_calls==1:message={'role':'assistant','content':'What is your phone number?'};finish='stop'
                elif self.agent_calls==2:message={'role':'assistant','content':None,'tool_calls':[{'id':'probe-call-1','type':'function','function':{'name':'get_customer_by_phone','arguments':json.dumps({'phone_number':'555-123-4567'})}}]};finish='tool_calls'
                else:message={'role':'assistant','content':'###STOP###'};finish='stop'
            else:
                self.user_calls+=1
                message={'role':'assistant','content':'###STOP###' if self.user_calls>=3 else 'My phone number is 555-123-4567. Please help.'};finish='stop'
            response={'id':'local-probe-'+str(len(self.calls)),'object':'chat.completion','created':int(time.time()),'model':request.get('model'),'choices':[{'index':0,'message':message,'finish_reason':finish}],'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}}
            self.calls.append({'role':role,'request':request,'request_hash':digest(request),'response':response,'response_hash':digest(response),'received_at':time.time(),'authenticated_model_receipt':False})
            return response

def run_probe(data,out,index=0,max_steps=8,wall_seconds=90,seed=20260930):
    if not 0<=index<2171 or not 1<=max_steps<=12 or not 1<=wall_seconds<=180:raise ValueError('bounded native probe budget')
    out=pathlib.Path(out).resolve();out.mkdir(parents=True,exist_ok=True);out.chmod(0o700)
    revision=pathlib.Path(data)/'.tau2_revision'
    if not revision.exists() or revision.read_text().strip()!=REVISION:raise ValueError('unapproved original data revision')
    inventory=data_inventory(data)
    if 'tau2/domains/telecom/tasks_full.json' not in inventory:raise ValueError('missing original telecom data')
    transport=MockTransport()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            size=int(self.headers.get('Content-Length','0'))
            if self.path!='/v1/chat/completions' or not 0<size<=2_000_000:self.send_error(400);return
            response=transport.response(json.loads(self.rfile.read(size)));raw=json.dumps(response).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    config={'endpoint':f'http://127.0.0.1:{server.server_port}/v1','index':index,'max_steps':max_steps,'seed':seed,'result':str(out/'simulation.json')}
    config_path=out/'child-config.json';config_path.write_text(json.dumps(config));started=time.time()
    with (out/'child.log').open('wb') as log:
        process=subprocess.Popen([sys.executable,'-m','subnet.native_tau2_probe','--child',str(config_path)],env=sanitized_env(data),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=wall_seconds);status='complete' if code==0 else 'child_failed'
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait();code=process.returncode;status='wall_timeout'
    server.shutdown();server.server_close();thread.join(timeout=2)
    (out/'transport.json').write_text(json.dumps(transport.calls,indent=2)+'\n')
    report={'status':status,'exit_code':code,'started_at':started,'completed_at':time.time(),'wall_budget_seconds':wall_seconds,'max_steps':max_steps,'data_revision_required':REVISION,'data_files':inventory,'data_inventory_hash':digest(inventory),'agent_requests':transport.agent_calls,'user_requests':sum(c['role']=='user' for c in transport.calls),'mock_transport':True,'genuine_model_solve':False,'model_proofs_captured':False,'user_observations_authenticated':False,'credentials_exported':False,'uploaded_code_sandbox':False,'process_wall_timeout_enforced':True,'network_sandbox_enforced':False,'chain_transactions':False,'child_source_hash':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'blockers':['Targetagent token/logprob/TOPLOC bridge is not integrated.','User simulator observations require independent authenticated receipts or replayable pinned model.','ENGY user key is not configured in probe environment.'],'notes':'Uses real original orchestrator/tools/database/grader with fabricated HTTP model messages solely for transport conformance.'}
    if (out/'simulation.json').exists():
        sim=json.loads((out/'simulation.json').read_text());report.update(task_hash=sim['task_hash'],task_id=sim['task']['id'],package_versions=sim['package_versions'],package_sources=sim['package_sources'],termination=sim['simulation']['termination_reason'],reward=sim['simulation']['reward_info']['reward'],pool_count=sim['pool_count'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--child');p.add_argument('--data');p.add_argument('--out');p.add_argument('--index',type=int,default=0);args=p.parse_args()
    if args.child:child(json.loads(pathlib.Path(args.child).read_text()))
    else:print(json.dumps(run_probe(args.data,args.out,args.index),indent=2))
if __name__=='__main__':main()
