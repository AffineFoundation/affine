#!/usr/bin/env python3
"""Operator-run authentic-proof and native-replay adversarial controls.

Environment-only controls reuse the exact unchanged model receipts from the
independent positive audit; they still rerun the original native orchestrator and
grader. No production verifier option skips model checks.
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
    base=pathlib.Path(a.base);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True);out.chmod(0o700)
    key=SigningKey(bytes.fromhex(pathlib.Path(a.seed_file).read_text()));authority=key.verify_key.encode().hex()
    positive=json.loads((base/'independent-full-verification.json').read_text())
    if not positive.get('verified') or not positive.get('full_native_trajectory_verified'):raise ValueError('requires completed independent positive audit')
    original_receipts=(base/'receipts.json').read_bytes();original_plan=(base/'plan.json').read_bytes();receipts=json.loads(original_receipts);simulation=authenticate(json.loads((base/'simulation-receipt.json').read_text()),authority)
    result=[]
    for name in ['last_token','last_response','forged_tool_call','user_observation','reward']:
        folder=out/name;folder.mkdir(exist_ok=True)
        for q in base.iterdir():
            if q.name in ('plan.json','receipts.json','simulation-receipt.json','operator-source-closure.json','verifier-supplement.json','generation-native_tau2_model.py','generation-native_tau2_replay.py') or q.suffix=='.npy':shutil.copyfile(q,folder/q.name)
        if name=='last_token':
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(a.checkpoint,local_files_only=True,trust_remote_code=False)
            record=copy.deepcopy(authenticate(receipts[0],authority));vocab=json.loads((pathlib.Path(a.checkpoint)/'config.json').read_text())['vocab_size'];record['output'][-1]=(record['output'][-1]+1)%vocab;record['text']=tokenizer.decode(record['output'],skip_special_tokens=True);record['response']['choices'][0]['message']['content']=record['text'];record['response_hash']=digest(record['response'])
            (folder/'receipts.json').write_text(json.dumps([envelope(record,key)]))
            try:verify_receipts(folder,a.checkpoint,authority);rejected=False;reason='unexpected accepted'
            except Exception as e:rejected=True;reason=str(e)
            cache=False
        elif name in ('last_response','forged_tool_call'):
            changed=[copy.deepcopy(authenticate(r,authority)) for r in receipts]
            if name=='last_response':changed[-1]['response']['choices'][0]['message']['content']='Forged last model-user observation'
            else:changed[-1]['response']['choices'][0]['message']={'role':'assistant','content':None,'tool_calls':[{'id':f'native-{len(changed)-1}-0','type':'function','function':{'name':'forged_tool','arguments':'{}'}}]}
            changed[-1]['response_hash']=digest(changed[-1]['response'])
            (folder/'receipts.json').write_text(json.dumps([envelope(r,key) for r in changed]))
            forged=copy.deepcopy(simulation)
            if name=='last_response':next(m for m in reversed(forged['simulation']['messages']) if m['role']=='user')['content']='Forged last model-user observation'
            (folder/'simulation-receipt.json').write_text(json.dumps(envelope(forged,key)))
            try:verify_receipts(folder,a.checkpoint,authority);rejected=False;reason='unexpected accepted'
            except Exception as e:rejected=True;reason=str(e)
            cache=False
        else:
            forged=copy.deepcopy(simulation)
            if name=='user_observation':next(m for m in forged['simulation']['messages'] if m['role']=='user')['content']='Fabricated customer observation'
            else:forged['simulation']['reward_info']['reward']=1.
            (folder/'simulation-receipt.json').write_text(json.dumps(envelope(forged,key)))
            # Computational receipts and approved model contract remain byte-exact.
            if (folder/'receipts.json').read_bytes()!=original_receipts or (folder/'plan.json').read_bytes()!=original_plan:raise ValueError('negative control invalid inference reuse')
            cache=True;model_result={'verified':positive['model_roles_verified']}
            with patch.object(native_tau2_replay,'verify_receipts',return_value=model_result):
                try:native_tau2_replay.replay(folder,a.checkpoint,a.data,authority);rejected=False;reason='unexpected accepted'
                except Exception as e:rejected=True;reason=str(e)
        result.append({'mutation':name,'rejected':rejected,'reason':reason,'operator_resigned_for_computation_test':True,'unchanged_model_proof_audit_reused':cache,'completed_at':time.time()})
        (out/'results.json').write_text(json.dumps({'controls':result,'positive_audit_sha256':hashlib.sha256((base/'independent-full-verification.json').read_bytes()).hexdigest(),'original_receipts_sha256':hashlib.sha256(original_receipts).hexdigest(),'chain_transactions':False},indent=2)+'\n')
    if not all(r['rejected'] for r in result):raise RuntimeError('tampering accepted')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
