"""Separate genuine inference and fresh replay of a controlled original Agent.

Requires an independently approved plan and preinstalled actor/grader images.
This standalone pilot does not register an adapter or write network weights.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from subnet.model import Runtime, NUMERICAL_RUNTIME_REVISION
from subnet.native_agent_isolation import NativeAgentSession, canonical, validate_descriptor
from subnet import harness
from subnet.proofs import validate_framing

def execute(session, text):
    action=harness.action(text)
    observations=[]
    for i,call in enumerate(action['tool_calls']):
        content=json.dumps(session.call(call['name'],call['arguments']),sort_keys=True,separators=(',',':'))
        observations.append(dict(role='tool',tool_call_id='native-'+str(i),name=call['name'],content=content))
    return observations

def chosen_text(session, turn, positive):
    # Only public instruction and native model-visible tool observations are
    # consulted. The gold DB/hidden fields are never accessed by this policy.
    if not positive or turn==3:return 'done'
    if turn<2:
        return json.dumps(dict(tool_call=dict(name=('list_printers','list_materials')[turn],arguments={})),separators=(',',':'))
    printers=session.events[0]['response']['result'];materials=session.events[1]['response']['result']
    material=next(m for m in materials if m['material_type']=='PLA' and m['color'].lower()=='red' and m['stock_grams']>=50)
    printer=next(p for p in printers if p['status']=='idle' and 'PLA' in p['supported_materials'])
    return json.dumps(dict(tool_call=dict(name='submit_print_job',arguments=dict(job_id='public-alex-phone-stand-1',customer='Alex',model_name='Phone Stand',material_id=material['id'],printer_id=printer['id'],estimated_grams=50))),separators=(',',':'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['generate','verify']);parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    descriptor=validate_descriptor(plan['descriptor']);policy=harness.normalize(plan['harness'])
    if hashlib.sha256(canonical(plan['checkpoint_files'])).hexdigest()!=plan['checkpoint_id']:
        raise ValueError('approved checkpoint content identity')
    if plan['runtime_revision']!=NUMERICAL_RUNTIME_REVISION or plan['harness_source_sha256']!=harness.source_hash():
        raise ValueError('approved inference/harness runtime')
    if plan['probe_sha256']!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('approved probe implementation')
    source_root=Path(__file__).resolve().parents[1]
    for name,digest in plan['source_files'].items():
        if not name.startswith('subnet/') or '..' in Path(name).parts or hashlib.sha256((source_root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('approved native model implementation')
    # Runtime's constructor requires an existing environment configuration.
    # This initialization-only spec is not executed or claimed as the Agent
    # environment. All Agent execution/task identity comes from the approved
    # native image descriptor, not Runtime.rollout/verify.
    runtime=Runtime(plan['checkpoint_path'],plan['checkpoint_files'],harness=policy,
                    environment=plan['model_initializer_environment'])
    planhash=hashlib.sha256(canonical(plan)).hexdigest()
    records=[]
    for positive in [False,True]:
        name='positive' if positive else 'negative';session=NativeAgentSession(descriptor,plan['instruction'])
        try:
            initial=session.start();messages=initial['messages'];tools=initial['tools'];turns=[]
            if args.mode=='verify':
                claimed=json.loads((out/(name+'.json')).read_text())
                if claimed['approved_plan_sha256']!=planhash or claimed['task_hash']!=initial['task_hash']:
                    raise ValueError('native proof task/plan binding')
                if len(claimed['turns'])!=(4 if positive else 1):raise ValueError('native complete trajectory length')
            for i in range(4 if positive else 1):
                prompt=runtime.prompt(messages,tools)
                if args.mode=='generate':
                    text=chosen_text(session,i,positive);tokens=runtime.tokenizer.encode(text,add_special_tokens=False)
                else:
                    turn=claimed['turns'][i];tokens=turn['output']
                    if not 0<len(tokens)<=512 or any(type(t) is not int or not 0<=t<runtime.model.config.vocab_size for t in tokens):raise ValueError('native output token schema')
                    text=runtime.tokenizer.decode(tokens,skip_special_tokens=True)
                    if turn['text']!=text or turn['prompt']!=prompt:raise ValueError('native token/context alignment')
                if len(prompt)+len(tokens)>8192:raise ValueError('native complete context budget')
                acts,lp=runtime.compute(prompt,tokens)
                if args.mode=='generate':
                    proofs=runtime.build_proofs(acts,decode_batching_size=16,topk=128)
                    np.save(out/(name+'-'+str(i)+'.npy'),lp)
                else:
                    proofs=turn['proofs'];validate_framing(proofs,1+math.ceil(len(tokens)/16))
                    claimed_lp=np.load(out/(name+'-'+str(i)+'.npy'),allow_pickle=False)
                    if claimed_lp.shape!=lp.shape or claimed_lp.dtype!=np.float32 or not np.isfinite(claimed_lp).all() or not np.allclose(lp,claimed_lp,atol=1e-5,rtol=0):raise ValueError('native full logprobs')
                    checks=runtime.verify_proofs(acts,proofs,decode_batching_size=16,topk=128)
                    if len(checks)!=1+math.ceil(len(tokens)/16) or any(c.exp_mismatches or c.mant_err_mean or c.mant_err_median for c in checks):raise ValueError('native strict TOPLOC')
                observations=execute(session,text)
                if args.mode=='verify' and turn['observations']!=observations:raise ValueError('original native tool replay')
                turns.append(dict(prompt=prompt,output=tokens,text=text,proofs=proofs,observations=observations))
                messages += [dict(role='assistant',content=text)]+harness.observations(observations)
            outcome=session.grade()
            if outcome['grade']['reward']!=int(positive):raise ValueError('original native outcome control')
            record=dict(approved_plan_sha256=planhash,checkpoint=plan['checkpoint_id'],task_hash=initial['task_hash'],policy_kind='curated-public-tool-computation',turns=turns,outcome=outcome)
            if args.mode=='generate':(out/(name+'.json')).write_bytes(canonical(record))
            elif claimed!=record:raise ValueError('native full artifact equality')
            records.append(dict(kind=name,reward=outcome['grade']['reward'],turns=len(turns),tool_calls=len(session.events),full_proof_verified=args.mode=='verify'))
        finally:session.close()
    summary=dict(mode=args.mode,approved_plan_sha256=planhash,records=records,numerical_tolerances=dict(logprobs_atol=1e-5,logprobs_rtol=0,TOPLOC_errors=0),payable=False,chain_transactions=False,production_adapter_registered=False,optimizer_updates=0,full_verifiers_orchestrator=False)
    (out/(args.mode+'-summary.json')).write_bytes(canonical(summary));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
