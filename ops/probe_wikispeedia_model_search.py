"""Prospective original Wikispeedia GPU proof search; never scores or trains."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import base64
import hashlib
import json
import sys
import time
from pathlib import Path
from nacl.signing import VerifyKey
from subnet.storage import canonical
from ops.probe_rcore_model_search import hydrate_checkpoint, gpu_wait, source_membership, validated_batch
from subnet.harness import normalize

ARTIFACT_POLICY = dict(max_compressed_bytes=100000000,max_uncompressed_bytes=500000000,
                       array_dtype='float32',max_turn_tokens=512,max_vocab_size=200000)


def windowed_config(candidate):
    """Keep public route choices intact; change only the signed visible context."""
    value=dict(candidate['harness'],version='text-tools-window-v1',
               history_prefix_messages=2,history_window_messages=2)
    return normalize(value)


def validate_contract(plan):
    if (plan.get('schema')!=1 or plan.get('experiment')!='original-wikispeedia-window-model-search-v1'
        or plan.get('payable') is not False or plan.get('chain_transactions') is not False):
        raise ValueError('nonpayable scoped probe')
    if type(plan.get('search_budget')) is not int or not 1<=plan['search_budget']<=32:
        raise ValueError('bounded search')
    if plan.get('indices')!=[0,1,2,3] or any(type(i)is not int for i in plan['indices']):
        raise ValueError('four original mining indices; heldouts excluded')
    env=plan.get('environment',{})
    if (env.get('id')!='affine_wikispeedia' or env.get('adapter')!='prime_v1'
        or type(env.get('num_samples'))is not int or env['num_samples']!=20
        or type(env.get('max_turns'))is not int or env['max_turns']!=30
        or type(env.get('max_output_tokens'))is not int or env['max_output_tokens']!=256
        or type(env.get('success_reward'))not in (int,float) or env['success_reward']!=1.):
        raise ValueError('original twenty-task scope and budgets')
    if plan.get('artifact_policy')!=ARTIFACT_POLICY:
        raise ValueError('bounded full-probability artifact policy')
    if plan.get('shared_GPU_idle_required')is not True or plan.get('normal_Numina_Pydantic_recovery_has_priority')is not True:
        raise ValueError('retained GPU priority')
    rows=plan.get('tasks')
    if not isinstance(rows,list) or [r.get('index') for r in rows]!=plan['indices']:
        raise ValueError('exact mining task population')
    for row in rows:
        if (type(row.get('index'))is not int
            or type(row.get('max_turns'))is not int or not 2<=row['max_turns']<=9
            or row.get('harness')!=normalize(row.get('harness',{}))
            or row['harness'].get('version')!='text-tools-window-v1'
            or row['harness'].get('policy')!='candidates'
            or row['harness'].get('history_prefix_messages')!=2
            or row['harness'].get('history_window_messages')!=2
            or row['harness'].get('max_output_tokens')!=256):
            raise ValueError('bounded windowed public route')
    return plan


def approved(document,authority):
    if document.get('signer')!=authority:raise ValueError('approval authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']),base64.b64decode(document['signature'],validate=True))
    plan=validate_contract(document['payload'])
    from subnet.backend_jobs import BACKEND_PROFILE,NUMERICAL_POLICY
    if plan.get('backend_profile')!=BACKEND_PROFILE or plan.get('numerical_policy')!=NUMERICAL_POLICY:
        raise ValueError('unchanged numerical policy')
    source_membership(plan.get('source_files'))
    for name,digest in plan['source_files'].items():
        path=Path(name)
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('source pin')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=plan.get('probe_sha256'):
        raise ValueError('probe source pin')
    return plan


def native_preflight(plan):
    """Resource verification must precede the provider's import-time cache choice."""
    if any(n=='wikispeedia_v1' or n.startswith('wikispeedia_v1.') for n in sys.modules):
        raise ValueError('original provider imported before resource admission')
    from subnet.native_wikispeedia import verify_public_resources,public_candidate_harness
    cache=Path(plan['resource_cache']).resolve()
    if os.environ.get('WIKISPEEDIA_CACHE_DIR')!=str(cache):
        raise ValueError('approved private cache must be set before provider import')
    if canonical(verify_public_resources(cache))!=canonical(plan['resources']):
        raise ValueError('original resource archive and extracted bytes')
    snapshot=Path(plan['environment']['config']['task_snapshot'])
    if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=plan['original_task_snapshot_sha256']:
        raise ValueError('original twenty-task snapshot bytes')
    tasks=json.loads(snapshot.read_bytes())
    if len(tasks)!=20:raise ValueError('original snapshot population')
    from subnet.environments import create_session
    session=create_session(plan['environment'])
    try:
        # Resolve and authenticate the original provider through the common adapter.
        initial=session.reset(0,0)
        from wikispeedia_v1.graph import WikiGraph
        graph=WikiGraph.load(include_text=False)
        for row in plan['tasks']:
            index=row['index'];initial=session.reset(index,0)
            if initial['task_hash']!=row['task_hash'] or canonical(initial['messages'])!=canonical(row['public_messages']):
                raise ValueError('original reset/task binding')
            data=tasks[index]['data']
            candidate=public_candidate_harness(data['source'],data['target'],initial['tools'],graph.links,30)
            if row['max_turns']!=candidate['max_turns'] or canonical(row['harness'])!=canonical(windowed_config(candidate)):
                raise ValueError('public graph choices cannot be replaced by hidden answers')
    finally:session.close()


def execute(plan,out,verify=False):
    native_preflight(plan)
    hydrate_checkpoint(plan)
    gpu_wait()
    from subnet.gpu_runtime import GPURuntime
    from subnet.batches import pack,unpack
    out.mkdir(parents=True,exist_ok=True)
    runtime=None;rows=[]
    if verify:
        records=json.loads((out/'search.json').read_bytes())
        if records.get('checkpoint')!=plan['checkpoint']['id'] or [r.get('index')for r in records.get('rows',[])]!=plan['indices']:
            raise ValueError('frozen search identity/population')
    for task in plan['tasks']:
        env=dict(plan['environment'],max_turns=task['max_turns'])
        if runtime is None:runtime=GPURuntime(plan['checkpoint_path'],plan['checkpoint']['files'],env,task['harness'])
        else:runtime.configure(env,task['harness'])
        index=task['index'];name=f'index-{index}.zip'
        if verify:
            row=next(r for r in records['rows']if r['index']==index)
            if row.get('harness')!=task['harness'] or row.get('artifact')!=name:raise ValueError('frozen per-task harness')
            data=(out/name).read_bytes()
            if len(data)!=row['artifact_size'] or hashlib.sha256(data).hexdigest()!=row['artifact_sha256']:raise ValueError('frozen artifact bytes')
            batch,arrays,positive,negative=validated_batch(plan,row,unpack(data))
            for rollout,tensors in zip(batch['rollouts'],arrays):
                if not runtime.verify(rollout,tensors):raise ValueError('fresh model/TOPLOC/native verification')
            rows.append(dict(index=index,positive=positive,negative=negative,qualifying_K1L1=positive==negative==1,rollouts_verified=len(arrays),artifact_sha256=row['artifact_sha256']))
        else:
            found={};attempts=[]
            for attempt in range(plan['search_budget']):
                seed=100+index*1000+attempt;rollout,arrays=runtime.rollout(index,seed);label=rollout['classification']
                attempts.append(dict(seed=seed,reward=rollout['reward'],classification=label,rollout_sha256=hashlib.sha256(canonical(rollout)).hexdigest()))
                if label in ('positive','negative')and label not in found:found[label]=(rollout,arrays)
                if len(found)==2:break
            chosen=[found[k]for k in ('positive','negative')if k in found]
            if not chosen:raise ValueError('no complete native rollout')
            raw_bytes=sum(t.nbytes for _,arrays in chosen for t in arrays)
            if raw_bytes>=ARTIFACT_POLICY['max_uncompressed_bytes']-2000000:raise ValueError('raw probability budget')
            batch=dict(env_id=env['id'],index=index,checkpoint=plan['checkpoint']['id'],rollouts=[r for r,_ in chosen])
            data=pack([(batch,[a for _,a in chosen])]);(out/name).write_bytes(data)
            rows.append(dict(index=index,harness=task['harness'],attempts=attempts,positive=int('positive'in found),negative=int('negative'in found),qualifying_K1L1=len(found)==2,artifact=name,artifact_size=len(data),artifact_sha256=hashlib.sha256(data).hexdigest(),probability_array_raw_bytes=raw_bytes))
            (out/'search.json').write_bytes(canonical(dict(checkpoint=plan['checkpoint']['id'],rows=rows,training=False,public_starter_control_not_autonomous_search=True,chain_transactions=False,payable=False)))
    if verify:
        report=dict(rows=rows,checkpoint=plan['checkpoint']['id'],independent_model_reload=True,full_logits_verified=True,TOPLOC_verified=True,original_native_replay=True,optimizer_ran=False,chain_transactions=False,payable=False,completed_at=time.time())
        (out/'fresh-verification.json').write_bytes(canonical(report))
    print(json.dumps(dict(tasks=len(rows),verification=verify,optimizer_ran=False,chain_transactions=False)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True,type=Path);p.add_argument('--authority',required=True);p.add_argument('--out',required=True,type=Path);p.add_argument('--verify',action='store_true');a=p.parse_args()
    execute(approved(json.loads(a.plan.read_bytes()),a.authority),a.out,a.verify)

if __name__=='__main__':main()
