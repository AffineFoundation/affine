"""Trusted operator runner for fresh fixed-user Tau2 role/native admission.

Generation and verification are separate commands/processes. This is a controlled
CPU prerequisite; it does not open a storage epoch, train or submit chain weights.
"""
import argparse,base64,json,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from nacl.signing import SigningKey
from .native_tau2_common_contract import canonical,digest
from .native_tau2_common_search_contract import validate_epoch,admit_sample,AUDIT_VERSION
from .native_tau2_common_search_endpoint import CommonRoleEndpoint,verify_receipt
from .native_tau2_common_cpu import CPURoleRuntime,FirstTaskPublicPolicy
from .native_tau2_common_simulation import prepare,run
from .native_tau2_common_replay import replay,checked_records


def sign(value,key):return {'payload':value,'signer':key.verify_key.encode().hex(),'signature':base64.b64encode(key.sign(canonical(value)).signature).decode()}
def save(path,value):
    path=Path(path);path.write_bytes(canonical(value));path.chmod(0o600)

def dependencies(epoch,authority,fixed_user,checkpoint_paths):
    manifest=validate_epoch(epoch,authority,fixed_user)
    if set(checkpoint_paths)!={'agent','user'}:raise ValueError('exact two role checkpoint paths')
    runtimes={name:CPURoleRuntime(path,manifest['roles'][name]) for name,path in checkpoint_paths.items()}
    policies={name:FirstTaskPublicPolicy(role['candidate_policy'],name,role['request_model']) if role.get('candidate_policy') else None for name,role in manifest['roles'].items()}
    return manifest,runtimes,policies

def generate(epoch,authority,fixed_user,checkpoint_paths,key,index,attempt,data,public_tasks,private_tasks,out):
    manifest,runtimes,policies=dependencies(epoch,authority,fixed_user,checkpoint_paths)
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);out.chmod(0o700)
    endpoint=CommonRoleEndpoint(epoch,authority,fixed_user,runtimes,key,index,trajectory_attempt=attempt,artifact_dir=out/'roles',candidate_policy=policies['agent'],user_candidate_policy=policies['user'])
    failures=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            try:
                size=int(self.headers.get('Content-Length','0'))
                if self.path!='/v1/chat/completions' or self.headers.get('Authorization')!='Bearer native-local' or not 0<size<=2_000_000:raise ValueError('local native endpoint request')
                response,_,_=endpoint.response(json.loads(self.rfile.read(size)));status=200
            except Exception as e:
                failures.append(type(e).__name__);response={'error':{'message':'Native role generation rejected','type':'fixed_user_role_rejection'}};status=400
            raw=canonical(response);self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        config=prepare(epoch,authority,fixed_user,data,public_tasks,private_tasks,index,f'http://127.0.0.1:{server.server_port}/v1',out/'simulation.json',attempt)
        original=run(config,data,out)
    finally:server.shutdown();server.server_close();thread.join(timeout=2)
    save(out/'signed-receipts.json',endpoint.records);save(out/'signed-native-simulation.json',sign(original,key));save(out/'signed-epoch.json',epoch)
    if failures:raise ValueError('original simulation had rejected role requests; inspect private logs')
    report={'epoch':manifest['epoch'],'environment_index':index,'trajectory_attempt':attempt,'role_count':len(endpoint.records),'reward':original['simulation']['reward_info']['reward'],'independent_admission_performed':False,'training_performed':False,'payable':False,'chain_transactions':False}
    save(out/'generation-report.json',report);return report

def verify(epoch,authority,fixed_user,checkpoint_paths,key,index,attempt,data,public_tasks,private_tasks,out):
    out=Path(out).resolve();manifest,runtimes,policies=dependencies(epoch,authority,fixed_user,checkpoint_paths)
    if json.loads((out/'signed-epoch.json').read_bytes())!=epoch:raise ValueError('artifact original signed epoch')
    receipts=json.loads((out/'signed-receipts.json').read_bytes())
    checked_records(epoch,receipts,authority,fixed_user,index,attempt)
    checks=[]
    for ordinal,envelope in enumerate(receipts):
        record=envelope['payload']
        if record.get('probabilities_file')!=f'role-{ordinal}.npy':raise ValueError('exact role array path')
        array=(out/'roles'/record['probabilities_file']).read_bytes()
        checks.append(verify_receipt(epoch,envelope,array,authority,fixed_user,runtimes[record['role']],index,trajectory_attempt=attempt,candidate_policy=policies[record['role']]))
    native=replay(epoch,receipts,json.loads((out/'signed-native-simulation.json').read_bytes()),authority,fixed_user,index,data,public_tasks,private_tasks,out/'independent-native-replay',attempt)
    report={**native,'role_checks':checks,'all_model_roles_verified':True,'derived_responses_verified':True,'source_closure_verified':True,'source_scope':'operator-trusted-exact-declared-source-interpreter-package-versions-v1','full_transitive_binary_dependency_closure_claimed':False}
    audit={'version':AUDIT_VERSION,'manifest_sha256':digest(manifest),'signed_receipts_sha256':digest(receipts),'verification_report_sha256':digest(report),'trajectory_attempt':attempt,'epoch':manifest['epoch'],'environment_id':manifest['environment']['id'],'environment_version':manifest['environment']['version'],'environment_index':index,'task_hash':report['task_hash'],'reward':report['reward'],'originally_sampled':False,'payable':False,'sampler_provenance':manifest['sampler_provenance']}
    for name in ('full_native_trajectory_verified','all_model_roles_verified','derived_responses_verified','source_closure_verified'):audit[name]=report[name]
    signed_audit=sign(audit,key);sample=admit_sample(epoch,receipts,signed_audit,report,authority,fixed_user)
    save(out/'independent-full-verification.json',report);save(out/'role-audit.json',signed_audit);save(out/'admitted-sample.json',sample)
    return {'epoch':manifest['epoch'],'environment_index':index,'trajectory_attempt':attempt,'role_count':len(checks),'reward':report['reward'],'independent_model_and_original_native_replay':True,'full_transitive_binary_dependency_closure_claimed':False,'common_epoch_completed':False,'training_performed':False,'payable':False,'chain_transactions':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['generate','verify'])
    for name in ('epoch','fixed-user','checkpoint-paths','seed-file','data','public-tasks','private-tasks','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--authority',required=True);parser.add_argument('--index',type=int,required=True);parser.add_argument('--attempt',type=int,required=True);args=parser.parse_args()
    key=SigningKey(bytes.fromhex(args.seed_file.read_text().strip()))
    if key.verify_key.encode().hex()!=args.authority:raise ValueError('operator signer')
    result=(generate if args.command=='generate' else verify)(json.loads(args.epoch.read_bytes()),args.authority,json.loads(args.fixed_user.read_bytes()),json.loads(args.checkpoint_paths.read_bytes()),key,args.index,args.attempt,args.data,args.public_tasks,args.private_tasks,args.out)
    print(json.dumps(result))
if __name__=='__main__':main()
