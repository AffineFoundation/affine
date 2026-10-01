"""Versioned CPU-model endpoint for bounded native Tau2 agent AND user requests.

Complete canonical requests are input; no tool truncation. This operator-owned
probe is separate from the deployed GPU worker and is not an uploaded-code sandbox.
"""
from __future__ import annotations
import argparse,base64,hashlib,json,math,os,pathlib,signal,subprocess,sys,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from nacl.signing import SigningKey,VerifyKey
from .native_tau2_probe import REVISION,data_inventory,digest,sanitized_env

PROFILE={'MKL_CBWR':'COMPATIBLE','ATEN_CPU_CAPABILITY':'default','ONEDNN_MAX_CPU_ISA':'SSE41','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}
RENDERER='native-tau2-complete-chat-tools-v2'

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def envelope(value,key):return {'payload':value,'signer':key.verify_key.encode().hex(),'signature':base64.b64encode(key.sign(canonical(value)).signature).decode()}
def authenticate(value,authority):
    if value['signer']!=authority:raise ValueError('authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(value['payload']),base64.b64decode(value['signature'],validate=True));return value['payload']
def profile():
    if any(os.environ.get(k)!=v for k,v in PROFILE.items()):raise ValueError('strict CPU runtime profile')
def render(tokenizer,request):
    from .harness import _chat_render
    messages=[]
    for message in request['messages']:
        role=message['role'];content=message.get('content')
        if role=='tool' or set(message)-{'role','content'} or not isinstance(content,str):
            content='Native structured message: '+canonical(message).decode()
        messages.append({'role':'user' if role=='tool' else role,'content':content})
    return _chat_render(tokenizer,messages,request.get('tools',[]))

def sample_cached(runtime,prompt,seed,count):
    # Explicit versioned generation policy; proof computation remains full CPUv2.
    import torch
    generator=torch.Generator().manual_seed(seed);tokens=[];past=None;input_ids=torch.tensor([prompt])
    with torch.inference_mode():
        for _ in range(count):
            result=runtime.model(input_ids,past_key_values=past,use_cache=True)
            past=result.past_key_values;probs=torch.softmax(result.logits[0,-1].float()/.7,-1)
            token=int(torch.multinomial(probs,1,generator=generator));tokens.append(token)
            if token==runtime.tokenizer.eos_token_id:break
            input_ids=torch.tensor([[token]])
    return tokens

def derived_response_message(text,number):
    from . import harness
    action=harness.action(text,{'policy':'autoregressive','version':'text-tools-v1','temperature':.7,'top_p':1.,'max_output_tokens':8})
    message={'role':'assistant','content':text}
    if action['tool_calls']:
        message={'role':'assistant','content':None,'tool_calls':[{'id':f'native-{number}-{i}','type':'function','function':{'name':t['name'],'arguments':json.dumps(t['arguments'],sort_keys=True)}} for i,t in enumerate(action['tool_calls'])]}
    return message,'tool_calls' if action['tool_calls'] else 'stop'

def validate_derived_response(record,number):
    message,finish=derived_response_message(record['text'],number)
    created=record['response'].get('created')
    if type(created)!=int or not int(record['created_at'])<=created<=int(record['completed_at']):raise ValueError('response timestamp binding')
    expected={'id':f'native-{number}','object':'chat.completion','created':created,'model':record['request']['model'],'choices':[{'index':0,'message':message,'finish_reason':finish}],'usage':{'prompt_tokens':len(record['prompt']),'completion_tokens':len(record['output']),'total_tokens':len(record['prompt'])+len(record['output'])}}
    if record['response']!=expected:raise ValueError('derived response content/tool calls/usage/model binding')
    return True

def make_runtime(path,plan):
    profile()
    from .model import Runtime,check_runtime_profile
    check_runtime_profile(plan)
    return Runtime(path,plan['checkpoint']['files'],threads=4)

class ModelEndpoint:
    def __init__(self,runtime,plan,key,out):self.runtime=runtime;self.plan=plan;self.key=key;self.out=pathlib.Path(out);self.records=[];self.lock=threading.Lock()
    def response(self,request):
        from . import harness
        import numpy as np
        with self.lock:
            role={'native-model-agent':'agent','native-model-user':'user'}.get(request.get('model'))
            if not role:raise ValueError('unapproved role model')
            if len(self.records)>=self.plan['max_requests']:raise ValueError('request budget')
            number=len(self.records);seed=self.plan['seed']+number
            prompt=render(self.runtime.tokenizer,request);limit=min(getattr(self.runtime.model.config,'max_position_embeddings',8192),8192)
            record={'request':request,'request_hash':digest(request),'role':role,'seed':seed,'checkpoint':self.plan['checkpoint']['id'],'plan_hash':digest(self.plan),'renderer':RENDERER,'prompt':prompt,'prompt_tokens':len(prompt),'output_budget':self.plan['output_tokens'],'context_limit':limit,'tools_count':len(request.get('tools',[])),'created_at':time.time()}
            if len(prompt)+self.plan['output_tokens']>limit:
                record.update(status='context_blocked',output=[],proofs=[]);self.save(record);raise ValueError(f'full native {role} prompt {len(prompt)} + output {self.plan["output_tokens"]} exceeds {limit}')
            config={'policy':'autoregressive','version':'text-tools-v1','temperature':.7,'top_p':1.,'max_output_tokens':self.plan['output_tokens']}
            output=sample_cached(self.runtime,prompt,seed,self.plan['output_tokens'])
            text=self.runtime.tokenizer.decode(output,skip_special_tokens=True)
            acts,lp=self.runtime.compute(prompt,output);proofs=self.runtime.build_proofs(acts,decode_batching_size=16,topk=128)
            if not proofs or any(p is None for p in proofs):raise ValueError('proof construction failed')
            array_name=f'request-{number}-probabilities.npy';np.save(self.out/array_name,lp,allow_pickle=False)
            message={'role':'assistant','content':text};action=harness.action(text,config)
            if action['tool_calls']:
                message={'role':'assistant','content':None,'tool_calls':[{'id':f'native-{number}-{i}','type':'function','function':{'name':t['name'],'arguments':json.dumps(t['arguments'],sort_keys=True)}} for i,t in enumerate(action['tool_calls'])]}
            response={'id':f'native-{number}','object':'chat.completion','created':int(time.time()),'model':request['model'],'choices':[{'index':0,'message':message,'finish_reason':'tool_calls' if action['tool_calls'] else 'stop'}],'usage':{'prompt_tokens':len(prompt),'completion_tokens':len(output),'total_tokens':len(prompt)+len(output)}}
            record.update(status='generated',output=output,text=text,proofs=proofs,probabilities_file=array_name,probabilities_sha256=hashlib.sha256((self.out/array_name).read_bytes()).hexdigest(),response=response,response_hash=digest(response),completed_at=time.time());self.save(record);return response
    def save(self,record):
        self.records.append(envelope(record,self.key));(self.out/'receipts.json').write_text(json.dumps(self.records)+'\n')

def native_child(config):
    from tau2.run import load_tasks,run_task
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.utils import llm_utils
    from tau2.user.base import UserState
    # Approved bundled wrapper implements the exact original raw-message/example patches.
    repo=pathlib.Path(__file__).resolve().parent
    sys.path.insert(0,str(repo/'vendor/research/environments/tool_use/tau2_bench_v1'))
    sys.path.insert(0,str(repo/'vendor/legacy/rollouts/envs/affine_tau2_v1'))
    from affine_tau2_v1.harness import apply_example_values
    from tau2_bench_v1.harness import _to_litellm_messages,_flip_roles
    llm_utils.to_litellm_messages=_to_litellm_messages;UserState.flip_roles=_flip_roles;apply_example_values()
    excluded={t.id for t in load_tasks('telecom','base')};task=next(t for t in load_tasks('telecom','full') if t.id not in excluded)
    args={'api_base':config['endpoint'],'api_key':'native-local','timeout':600,'max_retries':0,'num_retries':0,'temperature':.7}
    simulation=run_task(domain='telecom',task=task,agent='llm_agent',user='user_simulator',llm_agent='openai/native-model-agent',llm_args_agent=args,llm_user='openai/native-model-user',llm_args_user=args,max_steps=config['max_steps'],max_errors=3,seed=config['seed'],evaluation_type=EvaluationType.ALL)
    pathlib.Path(config['result']).write_text(json.dumps({'task':task.model_dump(mode='json'),'task_hash':digest(task.model_dump(mode='json')),'simulation':simulation.model_dump(mode='json')},indent=2)+'\n')

def verify_receipts(out,checkpoint,authority):
    profile();out=pathlib.Path(out);plan=authenticate(json.loads((out/'plan.json').read_text()),authority)
    if plan['renderer']!=RENDERER:raise ValueError('renderer')
    from .native_tau2_attestation import require_source_closure
    require_source_closure(out,plan,authority)
    receipts=json.loads((out/'receipts.json').read_text())
    for number,record in enumerate(receipts):
        checked=authenticate(record,authority)
        if checked['status']=='generated':validate_derived_response(checked,number)
    runtime=make_runtime(checkpoint,plan)
    import numpy as np
    from .proofs import validate_framing
    receipts=json.loads((out/'receipts.json').read_text());verified=[]
    for n,signed in enumerate(receipts):
        r=authenticate(signed,authority)
        if r['plan_hash']!=digest(plan) or r['checkpoint']!=plan['checkpoint']['id'] or r['seed']!=plan['seed']+n or r['request_hash']!=digest(r['request']) or r['role']!={'native-model-user':'user','native-model-agent':'agent'}.get(r['request'].get('model')):raise ValueError('request/model/seed binding')
        prompt=render(runtime.tokenizer,r['request'])
        if prompt!=r['prompt']:raise ValueError('prompt')
        if r['status']=='context_blocked':
            if len(prompt)+plan['output_tokens']<=8192 or r['output'] or r['proofs']:raise ValueError('false context blocker')
            verified.append({'role':r['role'],'status':'context_blocked','prompt_tokens':len(prompt)});continue
        output=r['output']
        if not 0<len(output)<=plan['output_tokens'] or any(type(t)!=int or not 0<=t<runtime.model.config.vocab_size for t in output):raise ValueError('output tokens')
        if runtime.tokenizer.decode(output,skip_special_tokens=True)!=r['text'] or digest(r['response'])!=r['response_hash']:raise ValueError('text/response binding')
        validate_derived_response(r,n)
        file=out/r['probabilities_file']
        if pathlib.Path(r['probabilities_file']).name!=r['probabilities_file'] or hashlib.sha256(file.read_bytes()).hexdigest()!=r['probabilities_sha256']:raise ValueError('array binding')
        claimed=np.load(file,allow_pickle=False);acts,actual=runtime.compute(prompt,output)
        if actual.shape!=claimed.shape or not np.isfinite(claimed).all() or not np.allclose(actual,claimed,atol=1e-5,rtol=0):raise ValueError('probabilities')
        count=1+math.ceil(len(output)/16);validate_framing(r['proofs'],count);checks=runtime.verify_proofs(acts,r['proofs'],decode_batching_size=16,topk=128)
        if len(checks)!=count or any(t.exp_mismatches or t.mant_err_mean or t.mant_err_median for t in checks):raise ValueError('TOPLOC')
        verified.append({'role':r['role'],'status':'model_computation_verified','prompt_tokens':len(prompt),'output_tokens':len(output)})
    return {'verified':verified,'full_native_trajectory_verified':False,'reason':'Inference checks alone do not authenticate complete native environment trajectory or reward.'}

def run(out,data,checkpoint,manifest,seed_path):
    profile();out=pathlib.Path(out).resolve();out.mkdir(parents=True,exist_ok=True);out.chmod(0o700)
    if (pathlib.Path(data)/'.tau2_revision').read_text().strip()!=REVISION:raise ValueError('data revision')
    key=SigningKey(bytes.fromhex(pathlib.Path(seed_path).read_text()));authority=key.verify_key.encode().hex()
    cp=json.loads(pathlib.Path(manifest).read_text())['checkpoint'];cp={k:cp[k] for k in ['id','files']}
    plan={'revision':'native-tau2-controlled-two-model-cpu-v1','checkpoint':cp,'runtime_profile':PROFILE,'model_runtime_revision':'cpu-float32-eager-v2-bounded-toploc','renderer':RENDERER,'source_hash':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'data_revision':REVISION,'data_inventory':data_inventory(data),'max_steps':3,'max_requests':3,'output_tokens':8,'generation_policy':'cpu-fp32-eager-kv-autoregressive-v1','harness_source_hash':hashlib.sha256((pathlib.Path(__file__).parent/'harness.py').read_bytes()).hexdigest(),'seed':20260930,'role_models':{'agent':cp['id'],'user':cp['id']},'temperature':.7,'top_p':1.,'user_simulator_policy':'same-approved-model-controlled-experiment','payable':False}
    (out/'plan.json').write_text(json.dumps(envelope(plan,key))+'\n');runtime=make_runtime(checkpoint,plan);endpoint=ModelEndpoint(runtime,plan,key,out)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                size=int(self.headers.get('Content-Length','0'))
                if self.path!='/v1/chat/completions' or self.headers.get('Authorization')!='Bearer native-local' or not 0<size<=2_000_000:raise ValueError('local request')
                response=endpoint.response(json.loads(self.rfile.read(size)));status=200
            except Exception as e:response={'error':{'message':str(e),'type':'native_probe_blocker'}};status=400
            raw=json.dumps(response).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    config={'endpoint':f'http://127.0.0.1:{server.server_port}/v1','max_steps':plan['max_steps'],'seed':plan['seed'],'result':str(out/'simulation.json')};config_path=out/'child-config.json';config_path.write_text(json.dumps(config));started=time.time()
    with (out/'child.log').open('wb') as log:
        process=subprocess.Popen([sys.executable,'-m','subnet.native_tau2_model','--child',str(config_path)],env=sanitized_env(data),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=1000);status='complete' if code==0 else 'native_child_failed'
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait();code=process.returncode;status='wall_timeout'
    server.shutdown();server.server_close();thread.join(timeout=2)
    # Wait for any already-running model handler before tearing down native Torch.
    with endpoint.lock:pass
    report={'status':status,'exit_code':code,'started_at':started,'completed_at':time.time(),'checkpoint':cp['id'],'genuine_model_roles':[r['payload']['role'] for r in endpoint.records if r['payload']['status']=='generated'],'context_blockers':[{k:r['payload'][k] for k in ['role','prompt_tokens','output_budget','context_limit','tools_count']} for r in endpoint.records if r['payload']['status']=='context_blocked'],'full_native_trajectory_verified':False,'chain_transactions':False,'gpu_used':False,'external_user_api':False}
    if (out/'simulation.json').exists():
        simulation=json.loads((out/'simulation.json').read_text());report.update(termination=simulation['simulation']['termination_reason'],reward=simulation['simulation']['reward_info']['reward'],simulation_hash=digest(simulation))
        (out/'simulation-receipt.json').write_text(json.dumps(envelope(simulation,key))+'\n')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--child');p.add_argument('--verify');p.add_argument('--out');p.add_argument('--data');p.add_argument('--checkpoint');p.add_argument('--manifest');p.add_argument('--seed-file');p.add_argument('--authority');args=p.parse_args()
    if args.child:native_child(json.loads(pathlib.Path(args.child).read_text()))
    elif args.verify:print(json.dumps(verify_receipts(args.verify,args.checkpoint,args.authority)))
    else:print(json.dumps(run(args.out,args.data,args.checkpoint,args.manifest,args.seed_file),indent=2))
if __name__=='__main__':main()
