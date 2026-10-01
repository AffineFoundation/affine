#!/usr/bin/env python3
"""Actual native public-action conformance; HTTP outputs are explicitly mocked."""
import argparse,json,pathlib,signal,subprocess,sys,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
from subnet.native_tau2_probe import sanitized_env,digest
from subnet.native_tau2_public_policy import public_action,POSITIVE_GUIDANCE,NEGATIVE_GUIDANCE,VERSION

def run(data,out,positive):
    out=pathlib.Path(out).resolve();out.mkdir(parents=True,exist_ok=True);calls=[];errors=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            request=json.loads(self.rfile.read(int(self.headers['Content-Length'])));action,reason=public_action(request,POSITIVE_GUIDANCE if positive else NEGATIVE_GUIDANCE)
            from subnet.harness import action as parse
            result=parse(action,{'version':'text-tools-v1','policy':'autoregressive','temperature':.7,'top_p':1.,'max_output_tokens':128})
            message={'role':'assistant','content':action}
            if result['tool_calls']:message={'role':'assistant','content':None,'tool_calls':[{'id':'public-'+str(len(calls)),'type':'function','function':{'name':t['name'],'arguments':json.dumps(t['arguments'])}} for t in result['tool_calls']]}
            response={'id':'public-'+str(len(calls)),'object':'chat.completion','created':int(time.time()),'model':request['model'],'choices':[{'index':0,'message':message,'finish_reason':'tool_calls' if result['tool_calls'] else 'stop'}],'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}}
            calls.append({'request':request,'request_hash':digest(request),'action':action,'reason':reason,'response':response});(out/'requests.json').write_text(json.dumps(calls,indent=2)+'\n')
            raw=json.dumps(response).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    config={'endpoint':f'http://127.0.0.1:{server.server_port}/v1','max_steps':12,'seed':20260930,'result':str(out/'simulation.json')};cp=out/'child-config.json';cp.write_text(json.dumps(config))
    with (out/'child.log').open('wb') as log:
        process=subprocess.Popen([sys.executable,'-m','subnet.native_tau2_model','--child',str(cp)],env=sanitized_env(data),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=90)
        except subprocess.TimeoutExpired:__import__('os').killpg(process.pid,signal.SIGKILL);process.wait();code=process.returncode
    server.shutdown();server.server_close();thread.join(timeout=2)
    report={'exit_code':code,'policy':VERSION,'guidance':'positive' if positive else 'negative','mock_transport':True,'model_proofs':False,'tool_calls':sum(bool(c['response']['choices'][0]['message'].get('tool_calls')) for c in calls),'model_requests':len(calls),'chain_transactions':False}
    if (out/'simulation.json').exists():
        s=json.loads((out/'simulation.json').read_text());report.update(reward=s['simulation']['reward_info']['reward'],termination=s['simulation']['termination_reason'],task_hash=s['task_hash'],reward_info=s['simulation']['reward_info'])
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);args=p.parse_args();print(json.dumps({'positive':run(args.data,pathlib.Path(args.out)/'positive',True),'negative':run(args.data,pathlib.Path(args.out)/'negative',False)},indent=2))
if __name__=='__main__':main()
