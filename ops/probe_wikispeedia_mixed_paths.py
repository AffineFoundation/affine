"""Exhaustively audit bounded original tool-choice paths and harness contexts; no model inference."""
import argparse,json,hashlib,time,itertools,gzip
from pathlib import Path
from transformers import AutoTokenizer
from subnet.environments import EnvironmentSpec,create_session
from subnet.backend_jobs import canonical
from subnet.harness import normalize,turn_config,action,render,observations

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualified-state',required=True,type=Path)
    parser.add_argument('--tokenizer',required=True,type=Path)
    parser.add_argument('--state',required=True,type=Path)
    args=parser.parse_args();p=args.qualified_state;root=args.tokenizer;out=args.state
    if out.exists():raise ValueError('fresh owned audit directory required')
    q=json.loads((p/'qualification.json').read_bytes());spec=EnvironmentSpec.from_dict(q['environment'])
    if not 1<=len(q['records'])<=32:raise ValueError('bounded original task population')
    if any(not 1<=len(record['candidate_config']['route'])<=8 for record in q['records']):raise ValueError('bounded exhaustive navigation depth')
    if sum(2**len(record['candidate_config']['route']) for record in q['records'])>4096:raise ValueError('bounded exhaustive path population')
    hydration=json.loads((root/'root-tokenizer-hydration-check.json').read_bytes());cp=hydration['checkpoint']
    if hydration['model_weights_downloaded'] is not False or hydration['model_execution'] is not False:raise ValueError('tokenizer-only preflight required')
    out.mkdir(parents=True)
    for name,b in hydration['objects'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==b['sha256']
    tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False);model=json.loads((root/'config.json').read_bytes());limit=min(model['max_position_embeddings'],8192);rows=[];started=time.time();num_traces=0
    tracefile=out/'native-choice-traces.jsonl.gz'
    assert not tracefile.exists(),'retain existing completed/failed audit rather than overwrite'
    with gzip.open(tracefile,'wb') as log:
     for record in q['records']:
      index=record['index'];hops=len(record['candidate_config']['route']);base=record['candidate_config']['harness'];stats={version:dict(max_prompt_tokens=0,max_candidate_tokens=0,turns=0,overflow_turns=0,first_overflow=None)for version in ['text-tools-v1','plain-transcript-v1','text-tools-window-v1']};positive=0;negative=0
      for choices in itertools.product([0,1],repeat=hops):
       s=create_session(spec)
       try:
        initial=s.reset(index,20261001+index);assert initial['task_hash']==record['task_hash'];messages=initial['messages'];tools=initial['tools'];turns=[]
        for turn,choice in enumerate(list(choices)+[0]):
         c=turn_config(base,turn);selected=c['candidates'][choice];candidate_lengths=[len(tokenizer.encode(text,add_special_tokens=False))for text in c['candidates']];assert min(candidate_lengths)>0 and max(candidate_lengths)<=c['max_output_tokens']
         for version,st in stats.items():
          h=normalize(dict(base,version=version));prompt=render(tokenizer,messages,tools,h);fits=len(prompt)+h['max_output_tokens']<=limit
          if not fits:
           st['overflow_turns']+=1
           if st['first_overflow'] is None:st['first_overflow']=dict(choices=list(choices),turn=turn,prompt_tokens=len(prompt),reserved_output_tokens=h['max_output_tokens'])
          if version=='text-tools-window-v1':assert fits,'bounded window still overflows'
          st['max_prompt_tokens']=max(st['max_prompt_tokens'],len(prompt));st['max_candidate_tokens']=max(st['max_candidate_tokens'],max(candidate_lengths));st['turns']+=1
         result=s.step(action(selected,base));turns.append({'text':selected,'result':result});messages=messages+[dict(role='assistant',content=selected)]+observations(result['observations'],base)
         if result['done']:break
        assert result['done'] and result['reward'] in (0.,1.);positive+=result['reward']==1.;negative+=result['reward']==0.
        log.write(canonical(dict(index=index,choices=list(choices),task_hash=initial['task_hash'],turns=turns,reward=result['reward']))+b'\n');num_traces+=1
       finally:s.close()
      rows.append(dict(index=index,choice_paths=2**hops,positive=positive,negative=negative,harnesses=stats));print(json.dumps(dict(index=index,choice_paths=2**hops,positive=positive,negative=negative)),flush=True)
    result=dict(checked_at=time.time(),elapsed_seconds=time.time()-started,checkpoint=cp,environment_definition_sha256=hashlib.sha256(canonical(spec.to_dict())).hexdigest(),candidate_qualification_sha256=hashlib.sha256((p/'qualification.json').read_bytes()).hexdigest(),tokenizer_objects=hydration['objects'],context_limit=limit,rows=rows,native_choice_paths=num_traces,all_proposed_tool_choice_paths_checked=True,window_model_contexts_fit=True,legacy_contexts_fit=not any(st['overflow_turns']for row in rows for version,st in row['harnesses'].items()if version!='text-tools-window-v1'),harness_source_sha256=hashlib.sha256(Path('subnet/harness.py').read_bytes()).hexdigest(),final_non_tool_reply='Done.; both terminal candidates token-budget checked',trace_sha256=hashlib.sha256(tracefile.read_bytes()).hexdigest(),model_execution=False,TOPLOC_generated=False,optimizer_ran=False,chain_transactions=False)
    (out/'qualification.json').write_bytes(canonical(result));print(json.dumps({k:v for k,v in result.items()if k not in ['rows','tokenizer_objects']}))

if __name__=="__main__":main()
