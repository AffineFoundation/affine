#!/usr/bin/env python3
"""Native tool-bearing semantic attacks with operator-resigned artifacts.

Environment-only controls reuse byte-exact independently audited computation;
there is no production option to skip model checks.
"""
import argparse,copy,hashlib,json,pathlib,shutil,sys,time
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
from nacl.signing import SigningKey
from subnet.native_tau2_model import authenticate,envelope,verify_receipts
from subnet.native_tau2_probe import digest
from subnet import native_tau2_replay

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--data',required=True);p.add_argument('--seed-file',required=True);a=p.parse_args()
    base=pathlib.Path(a.base);out=pathlib.Path(a.out);out.mkdir(exist_ok=False);out.chmod(0o700)
    key=SigningKey(bytes.fromhex(pathlib.Path(a.seed_file).read_text()));authority=key.verify_key.encode().hex()
    signed_audit=json.loads((base/'role-audit.json').read_text());audit=authenticate(signed_audit,authority)
    report=json.loads((base/'independent-full-verification.json').read_text());raw=(base/'receipts.json').read_bytes();records=json.loads(raw)
    if audit['native_report_hash']!=digest(report) or audit['receipts_hash']!=digest(records) or not report['verified'] or not report['full_native_trajectory_verified']:raise ValueError('completed exact source audit required')
    simulation=authenticate(json.loads((base/'simulation-receipt.json').read_text()),authority);results=[]
    for name in ('mutating_tool_call','native_tool_observation','native_reward'):
        folder=out/name;folder.mkdir()
        for file in base.iterdir():
            if file.is_file() and (file.name in ('plan.json','receipts.json','simulation-receipt.json','operator-source-closure.json','verifier-supplement.json') or file.name.startswith('generation-') or file.suffix=='.npy'):shutil.copyfile(file,folder/file.name)
        cache=False
        try:
            if name=='mutating_tool_call':
                changed=[copy.deepcopy(authenticate(r,authority)) for r in records]
                target=next(r for r in changed if any(t['function']['name']=='toggle_roaming' for t in r['response']['choices'][0]['message'].get('tool_calls',[])))
                target['response']['choices'][0]['message']['tool_calls'][0]['function']['name']='run_speed_test';target['response_hash']=digest(target['response'])
                (folder/'receipts.json').write_text(json.dumps([envelope(r,key) for r in changed]));verify_receipts(folder,a.checkpoint,authority)
            else:
                forged=copy.deepcopy(simulation)
                if name=='native_tool_observation':
                    message=next(m for m in forged['simulation']['messages'] if m['role']=='tool' and 'Data Roaming Enabled: No' in str(m.get('content')));message['content']=message['content'].replace('Data Roaming Enabled: No','Data Roaming Enabled: Yes')
                else:forged['simulation']['reward_info']['reward']=0. if report['reward']==1. else 1.
                (folder/'simulation-receipt.json').write_text(json.dumps(envelope(forged,key)))
                if (folder/'receipts.json').read_bytes()!=raw or (folder/'plan.json').read_bytes()!=(base/'plan.json').read_bytes():raise ValueError('invalid inference reuse')
                cache=True
                with patch.object(native_tau2_replay,'verify_receipts',return_value={'verified':report['model_roles_verified']}):native_tau2_replay.replay(folder,a.checkpoint,a.data,authority)
            rejected=False;reason='unexpected accepted'
        except Exception as e:rejected=True;reason=str(e)
        expected={'mutating_tool_call':'derived response content/tool calls/usage/model binding','native_tool_observation':'native observation/tool trajectory mismatch','native_reward':'native termination/original grader mismatch'}[name]
        if reason!=expected:raise RuntimeError(f'control did not reach intended verifier: {name}: {reason}')
        results.append({'mutation':name,'rejected':rejected,'reason':reason,'operator_resigned':True,'byte_exact_prior_model_audit_reused':cache,'completed_at':time.time()})
        (out/'results.json').write_text(json.dumps({'controls':results,'audited_receipts_sha256':hashlib.sha256(raw).hexdigest(),'signed_role_audit_sha256':hashlib.sha256((base/'role-audit.json').read_bytes()).hexdigest(),'chain_transactions':False},indent=2)+'\n')
    if not all(r['rejected'] for r in results):raise RuntimeError('native mutation accepted')
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
