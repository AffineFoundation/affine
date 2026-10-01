"""Independent original Tau2 replay using model-verified agent/user receipts."""
from __future__ import annotations
import argparse,hashlib,json,os,pathlib,signal,subprocess,sys,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from .native_tau2_model import authenticate,verify_receipts
from .native_tau2_probe import digest,sanitized_env,data_inventory

def message_view(simulation):
    # Wall-clock timestamps do not affect native tools, model context or rewards.
    return [{k:v for k,v in m.items() if k!='timestamp'} for m in simulation['messages']]

def replay(out,checkpoint,data,authority):
    out=pathlib.Path(out);plan=authenticate(json.loads((out/'plan.json').read_text()),authority)
    from .native_tau2_attestation import require_source_closure
    closure=require_source_closure(out,plan,authority)
    if data_inventory(data)!=plan['data_inventory']:raise ValueError('data inventory')
    original=authenticate(json.loads((out/'simulation-receipt.json').read_text()),authority)
    receipts=[authenticate(r,authority) for r in json.loads((out/'receipts.json').read_text())]
    if any(r['status']!='generated' for r in receipts):raise ValueError('incomplete model role receipts')
    proof_result=verify_receipts(out,checkpoint,authority)
    if len(proof_result['verified'])!=len(receipts):raise ValueError('inference verification count')
    cursor=[0];errors=[];lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                size=int(self.headers.get('Content-Length','0'))
                if self.path!='/v1/chat/completions' or not 0<size<=2_000_000:raise ValueError('replay request')
                request=json.loads(self.rfile.read(size))
                with lock:
                    n=cursor[0]
                    if n>=len(receipts) or digest(request)!=receipts[n]['request_hash']:raise ValueError('native replay request/context mismatch')
                    response=receipts[n]['response'];cursor[0]+=1
                status=200
            except Exception as e:errors.append(str(e));response={'error':{'message':str(e),'type':'native_replay_rejection'}};status=400
            raw=json.dumps(response).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    config={'endpoint':f'http://127.0.0.1:{server.server_port}/v1','max_steps':plan['max_steps'],'seed':plan['seed'],'result':str((out/'independent-native-simulation.json').resolve())};config_path=out/'independent-native-config.json';config_path.write_text(json.dumps(config))
    with (out/'independent-native.log').open('wb') as log:
        process=subprocess.Popen([sys.executable,'-m','subnet.native_tau2_model','--child',str(config_path.resolve())],env=sanitized_env(data),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=90)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait();code=process.returncode
    server.shutdown();server.server_close();thread.join(timeout=2)
    if code!=0 or errors or cursor[0]!=len(receipts):raise ValueError('native replay failed or incomplete')
    actual=json.loads((out/'independent-native-simulation.json').read_text())
    if actual['task_hash']!=original['task_hash'] or message_view(actual['simulation'])!=message_view(original['simulation']):raise ValueError('native observation/tool trajectory mismatch')
    for field in ['termination_reason','reward_info']:
        if actual['simulation'][field]!=original['simulation'][field]:raise ValueError('native termination/original grader mismatch')
    result={'verified':True,'full_native_trajectory_verified':True,'model_roles_verified':proof_result['verified'],'user_observations_model_authenticated':True,'native_tool_and_grader_replayed':True,'task_hash':actual['task_hash'],'termination':actual['simulation']['termination_reason'],'reward':actual['simulation']['reward_info']['reward'],'checkpoint':plan['checkpoint']['id'],'completed_at':time.time(),'payable':False,'chain_transactions':False,'source_closure':closure,'replay_source_hash':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()}
    (out/'independent-full-verification.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--authority',required=True);args=p.parse_args();print(json.dumps(replay(args.out,args.checkpoint,args.data,args.authority),indent=2))
if __name__=='__main__':main()
