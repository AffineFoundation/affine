"""Separate, source-pinned native tool experiment with curated model computation.

Full native schemas and observations retained. Same approved model computes both
roles; this is not historical Engy or an unbiased autoregressive rollout.
"""
import argparse,hashlib,json,os,pathlib,signal,subprocess,sys,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from importlib.metadata import version
from nacl.signing import SigningKey
from . import native_tau2_model as base
from .native_tau2_public_policy import public_action,format_candidates,POSITIVE_GUIDANCE,NEGATIVE_GUIDANCE,VERSION
from .native_tau2_probe import data_inventory,digest,sanitized_env,REVISION
from .native_tau2_attestation import REQUIRED,PACKAGES,VERIFIERS,ROOT

EXTRA=('subnet/native_tau2_curated.py','subnet/native_tau2_public_policy.py','subnet/native_auxiliary_roles.py')
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def sources():return {n:sha(ROOT/n) for n in EXTRA}
def write(out,name,obj): (out/name).write_text(json.dumps(obj,indent=2)+'\n')

class Endpoint(base.ModelEndpoint):
    def response(self,request):
        import numpy as np
        from .harness import sample
        with self.lock:
            role={'native-model-agent':'agent','native-model-user':'user'}.get(request.get('model'))
            if role is None or len(self.records)>=self.plan['max_requests']:raise ValueError('role/request budget')
            n=len(self.records);seed=self.plan['seed']+n;prompt=base.render(self.runtime.tokenizer,request)
            action,reason=public_action(request,POSITIVE_GUIDANCE if self.plan['positive'] else NEGATIVE_GUIDANCE)
            candidates=format_candidates(action)
            config={'version':'text-tools-v1','policy':'candidates','candidates':candidates,'temperature':.7,'top_p':1.,'max_output_tokens':128}
            if len(prompt)+max(len(self.runtime.tokenizer.encode(c,add_special_tokens=False)) for c in candidates)>8192:raise ValueError('complete native candidate context budget')
            created=time.time();output=sample(self.runtime.model,self.runtime.tokenizer,prompt,seed,config)
            text=self.runtime.tokenizer.decode(output,skip_special_tokens=True);acts,lp=self.runtime.compute(prompt,output)
            proofs=self.runtime.build_proofs(acts,decode_batching_size=16,topk=128)
            if not proofs or any(p is None for p in proofs):raise ValueError('proof construction')
            filename=f'request-{n}-probabilities.npy';np.save(self.out/filename,lp,allow_pickle=False)
            message,finish=base.derived_response_message(text,n)
            response={'id':f'native-{n}','object':'chat.completion','created':int(time.time()),'model':request['model'],'choices':[{'index':0,'message':message,'finish_reason':finish}],'usage':{'prompt_tokens':len(prompt),'completion_tokens':len(output),'total_tokens':len(prompt)+len(output)}}
            record={'request':request,'request_hash':digest(request),'role':role,'seed':seed,'checkpoint':self.plan['checkpoint']['id'],'plan_hash':digest(self.plan),'renderer':base.RENDERER,'prompt':prompt,'prompt_tokens':len(prompt),'output_budget':128,'context_limit':8192,'tools_count':len(request.get('tools',[])),'created_at':created,'status':'generated','output':output,'text':text,'proofs':proofs,'probabilities_file':filename,'probabilities_sha256':sha(self.out/filename),'response':response,'response_hash':digest(response),'completed_at':time.time(),'public_policy_reason':reason,'curated_candidates':candidates,'sampling_provenance_claimed':False}
            self.save(record);print(json.dumps({'request':n,'role':role,'prompt_tokens':len(prompt),'output_tokens':len(output),'reason':reason}),flush=True);return response

def run(out,data,checkpoint,manifest,seed_path,positive):
    base.profile();out=pathlib.Path(out).resolve();out.mkdir(parents=True,exist_ok=False);out.chmod(0o700)
    if (pathlib.Path(data)/'.tau2_revision').read_text().strip()!=REVISION:raise ValueError('data revision')
    key=SigningKey(bytes.fromhex(pathlib.Path(seed_path).read_text()));cp=json.loads(pathlib.Path(manifest).read_text())['checkpoint'];cp={k:cp[k] for k in ('id','files')}
    plan={'revision':'native-tau2-public-curated-computation-v1','checkpoint':cp,'runtime_profile':base.PROFILE,'model_runtime_revision':'cpu-float32-eager-v2-bounded-toploc','renderer':base.RENDERER,'source_hash':sha(ROOT/'subnet/native_tau2_model.py'),'harness_source_hash':sha(ROOT/'subnet/harness.py'),'data_revision':REVISION,'data_inventory':data_inventory(data),'max_steps':12,'max_requests':16,'output_tokens':128,'seed':20260930,'role_models':{'agent':cp['id'],'user':cp['id']},'positive':positive,'curated_policy':VERSION,'curated_sources':sources(),'generation_policy':'public-input-curated-format-candidates-v1','user_simulator_policy':'same-approved-model-public-instruction-curated-controlled','payable':False}
    write(out,'plan.json',base.envelope(plan,key))
    closure={'plan_hash':digest(plan),'sources':{n:sha(ROOT/n) for n in REQUIRED},'python_version':sys.version,'interpreter_hash':sha(pathlib.Path(sys.executable).resolve()),'package_versions':{n:version(n) for n in PACKAGES}}
    write(out,'operator-source-closure.json',base.envelope(closure,key))
    for n in ('native_tau2_model.py','native_tau2_replay.py'):(out/('generation-'+n)).write_bytes((ROOT/'subnet'/n).read_bytes())
    supplement={'schema':1,'upgrade':'exact-derived-response-v3','generation_plan_hash':digest(plan),'generation_source_hash':plan['source_hash'],'original_source_closure_sha256':sha(out/'operator-source-closure.json'),'verifier_sources':{n:sha(ROOT/n) for n in VERIFIERS}}
    write(out,'verifier-supplement.json',base.envelope(supplement,key))
    from .native_auxiliary_roles import VERSION as ROLE_VERSION
    contract={'version':ROLE_VERSION,'objective':'agent-only-curated-supervised-v1','roles':{r:{'kind':'agent' if r=='agent' else 'auxiliary','training_eligible':r=='agent','checkpoint':cp['id'],'source_hash':plan['curated_sources']['subnet/native_tau2_curated.py'],'numerical_policy':base.PROFILE} for r in ('agent','user')},'payable':False}
    write(out,'role-contract.json',base.envelope(contract,key))
    runtime=base.make_runtime(checkpoint,plan);endpoint=Endpoint(runtime,plan,key,out);errors=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                size=int(self.headers.get('Content-Length','0'))
                if self.path!='/v1/chat/completions' or self.headers.get('Authorization')!='Bearer native-local' or not 0<size<=2000000:raise ValueError('local request')
                result=endpoint.response(json.loads(self.rfile.read(size)));status=200
            except Exception as e:errors.append(str(e));result={'error':{'message':str(e),'type':'native_curated_blocker'}};status=400
            raw=json.dumps(result).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    config={'endpoint':f'http://127.0.0.1:{server.server_port}/v1','max_steps':plan['max_steps'],'seed':plan['seed'],'result':str(out/'simulation.json')};write(out,'child-config.json',config)
    with (out/'child.log').open('wb') as log:
        process=subprocess.Popen([sys.executable,'-m','subnet.native_tau2_model','--child',str(out/'child-config.json')],env=sanitized_env(data),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=3600)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait();code=process.returncode
    server.shutdown();server.server_close();thread.join(timeout=2)
    with endpoint.lock:pass
    report={'exit_code':code,'errors':errors,'model_requests':len(endpoint.records),'curated_computation':True,'unbiased_sampling':False,'payable':False,'full_native_trajectory_verified':False}
    if (out/'simulation.json').exists():
        simulation=json.loads((out/'simulation.json').read_text());write(out,'simulation-receipt.json',base.envelope(simulation,key));report.update(reward=simulation['simulation']['reward_info']['reward'],termination=simulation['simulation']['termination_reason'])
    write(out,'report.json',report);return report

def verify(out,checkpoint,data,authority,seed_path):
    out=pathlib.Path(out);plan=base.authenticate(json.loads((out/'plan.json').read_text()),authority)
    if plan.get('curated_sources')!=sources() or plan.get('curated_policy')!=VERSION or plan.get('generation_policy')!='public-input-curated-format-candidates-v1':raise ValueError('curated policy source binding')
    for signed in json.loads((out/'receipts.json').read_text()):
        r=base.authenticate(signed,authority);action,reason=public_action(r['request'],POSITIVE_GUIDANCE if plan['positive'] else NEGATIVE_GUIDANCE)
        if r['curated_candidates']!=format_candidates(action) or r['text'] not in r['curated_candidates'] or r['public_policy_reason']!=reason or r['sampling_provenance_claimed'] is not False:raise ValueError('public candidate computation binding')
    from .native_tau2_replay import replay
    result=replay(out,checkpoint,data,authority)
    contract=json.loads((out/'role-contract.json').read_text());receipts=json.loads((out/'receipts.json').read_text())
    key=SigningKey(bytes.fromhex(pathlib.Path(seed_path).read_text()))
    if key.verify_key.encode().hex()!=authority:raise ValueError('audit authority')
    audit={'contract_hash':digest(base.authenticate(contract,authority)),'receipts_hash':digest(receipts),'full_native_trajectory_verified':True,'all_model_roles_verified':True,'native_report_hash':digest(result),'completed_at':time.time()}
    write(out,'role-audit.json',base.envelope(audit,key))
    from .native_auxiliary_roles import admit_records
    views=admit_records(contract,receipts,base.envelope(audit,key),authority)
    write(out,'training-view.json',{'objective':'agent-only-curated-supervised-v1','roles':views,'training_performed':False});return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--data',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--manifest');p.add_argument('--seed-file',required=True);p.add_argument('--negative',action='store_true');p.add_argument('--verify',action='store_true');p.add_argument('--authority');a=p.parse_args()
    print(json.dumps(verify(a.out,a.checkpoint,a.data,a.authority,a.seed_file) if a.verify else run(a.out,a.data,a.checkpoint,a.manifest,a.seed_file,not a.negative),indent=2))
if __name__=='__main__':main()
