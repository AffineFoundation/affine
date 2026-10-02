"""Prepare an undeployed original MATH-only pilot; never load model weights."""
import argparse,collections,hashlib,json
from pathlib import Path

def population(rows):
    if len(rows)!=7496:raise ValueError('qualified original MATH loader count changed')
    problems=[r['data']['problem'] for r in rows]
    hashes=[hashlib.sha256(v.encode()).hexdigest() for v in problems]
    if len(set(hashes))!=len(rows):raise ValueError('duplicate original public problems')
    heldout=sorted(sorted(range(len(rows)),key=lambda i:hashlib.sha256(('math-pilot-heldout-v1:'+hashes[i]).encode()).hexdigest())[:750])
    rank={'Level 1':1,'Level 2':2,'Level 3':3,'Level 4':4,'Level 5':5,'Level ?':6}
    training=sorted(set(range(len(rows)))-set(heldout),key=lambda i:(rank[rows[i]['data']['level']],hashes[i]))
    strata=collections.defaultdict(list)
    for i in heldout:strata[(rows[i]['data']['subject'],rows[i]['data']['level'])].append(i)
    evaluation=[min(strata[k],key=lambda i:hashes[i]) for k in sorted(strata)][:32]
    if len(evaluation)!=32 or set(training)&set(heldout):raise ValueError('disjoint fixed cohort')
    return training,heldout,evaluation

def configuration(spec,rows,output,base=None):
    train,reserve,evaluation=population(rows)
    harness=dict(version='text-tools-v1',policy='autoregressive',max_output_tokens=256,temperature=.8,top_p=1.)
    return dict(preparation_only=True,epoch_prefix='nonpayable-single-original-math-base-v1',payable_epochs=False,
      state=str((output/'controller-state').resolve()),evaluation_state=str((output/'evaluations').resolve()),
      evaluation_experiment_id='original-math-base-fixed32-256-v1',duration=1800,
      model_id='HuggingFaceTB/SmolLM2-1.7B-Instruct',model_revision='31b70e2e869a7173562077fd711b654946d38674',
      initial_checkpoint=base['checkpoint'] if base else None,initial_checkpoint_path=base['path'] if base else None,
      source_bundle=None,environments=[dict(spec=spec,indices=train,harness=harness)],
      owned_mining_schedule=[{'affine_math':train[i:i+16]} for i in range(0,len(train),16)],reserved_heldout_indices={'affine_math':reserve},
      training_groups=[['affine_math']],heldout=[dict(env_id='affine_math',indices=evaluation,seed=20261002,harness=dict(harness,temperature=.7))],
      max_batches=3,search_budget=32,training_steps=8,registration_allowlist=[],
      remote=dict(job_ttl_seconds_by_role=dict(mine=3600,verify=3600,train=7200,evaluate=10800)))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--base-receipt',type=Path);p.add_argument('--tokenizer',type=Path)
    args=p.parse_args();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('new empty preparation namespace required')
    out.mkdir(parents=True,exist_ok=True);out.chmod(0o700)
    from subnet.environments import build_spec,snapshot_spec
    from subnet.storage import canonical
    spec=build_spec('affine_math',{},num_samples=7496,max_turns=1,max_output_tokens=512)
    spec=snapshot_spec(spec,out/'original7496.tasks.json')
    if hashlib.sha256((out/'original7496.tasks.json').read_bytes()).hexdigest()!='77a4524abc279d0e6e95ec87d0e5604f501c8e409ac5ab656ecabf060ab50fe3':raise ValueError('pinned original provider snapshot bytes changed')
    rows=json.loads((out/'original7496.tasks.json').read_text())
    base=json.loads(args.base_receipt.read_text()) if args.base_receipt else None
    if base and (base['checkpoint']['id']!='9eeeddec648cfcc725237073dfa1ce48c480669602d8809bd77d6cf5d70c00c4' or base.get('success') is not True or base.get('all_primary_metadata_checked') is not True or base.get('revision')!='31b70e2e869a7173562077fd711b654946d38674' or hashlib.sha256(canonical(base['checkpoint']['files'])).hexdigest()!=base['checkpoint']['id']):raise ValueError('actual pinned upstream base completion required')
    config=configuration(spec.to_dict(),rows,out,base)
    from subnet.service import definitions
    definitions(config)
    public=[dict(index=i,task_name=r['data']['name'],problem_sha256=hashlib.sha256(r['data']['problem'].encode()).hexdigest(),messages=[dict(role='system',content=r['data']['system_prompt']),dict(role='user',content=r['data']['prompt'])]) for i,r in enumerate(rows)]
    budget=None
    if args.tokenizer:
      if hashlib.sha256((args.tokenizer/'tokenizer.json').read_bytes()).hexdigest()!='9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c':raise ValueError('actual pinned upstream tokenizer required')
      from transformers import AutoTokenizer
      from subnet.harness import render
      tok=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
      counts=[len(render(tok,r['messages'],[],config['environments'][0]['harness'])) for r in public]
      budget=dict(count=len(counts),max_prompt_tokens=max(counts),output_budget=256,overflow=sum(n+256>8192 for n in counts),model_loaded=False)
      if budget['overflow']:raise ValueError('complete original context exceeds signed budget')
    for name,value in [('config.template.json',config),('public-problems.json',public),('preparation.json',dict(original_count=7496,train_count=6746,reserved_heldout_count=750,evaluation_count=32,budget=budget,no_deployment=True,no_model_weights_loaded=True,no_chain_writes=True,public_reference_contamination_limitation=True))]:
      (out/name).write_bytes(canonical(value));(out/name).chmod(0o600)
    print(json.dumps(dict(output=str(out),original_count=7496,preparation_only=True)))
if __name__=='__main__':main()
