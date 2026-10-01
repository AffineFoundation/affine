"""Fresh original NQueens common-session conformance, no model or epoch."""
import argparse,hashlib,json
from pathlib import Path
from subnet.environments import build_spec,_taskset
from subnet.native_prolog_session import NativePrologSession,VERSION
from subnet.native_prolog_public_policy import candidates
from subnet.native_prolog_actor import public_task,build

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 config={'taskset':{'tasks':['prolog-nqueens-0005','prolog-nqueens-0014','prolog-nqueens-0023'],'num_examples':32,'difficulty':'medium'}}
 preliminary=build_spec('affine_prolog',config=config,num_samples=3,max_turns=2,max_output_tokens=1024)
 list(_taskset(preliminary))
 from affine_prolog_v1.taskset import SWIPL_SHIM
 config.update(prolog_session_revision=VERSION,prolog_runtime=build(args.out/'image',SWIPL_SHIM),prolog_source_files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('subnet/native_prolog_actor.py'),Path('subnet/native_prolog_session.py')]})
 spec=build_spec('affine_prolog',config=config,num_samples=3,max_turns=2,max_output_tokens=1024);tasks=list(_taskset(spec));rows=[]
 for i,task in enumerate(tasks):
  actions=candidates(public_task(task));results=[]
  for j,command in enumerate(actions):
   session=NativePrologSession(spec)
   try:
    reset=session.reset(i,100);step=session.step({'tool_calls':[{'name':'bash','arguments':{'command':command}}]});assert not step['done'];terminal=session.step({'text':'Done'});assert terminal['done'] and terminal['reward']==(1. if j==0 else 0.)
    results.append({'reset':reset,'tool_observation':step,'terminal':terminal,'command_sha256':hashlib.sha256(command.encode()).hexdigest(),'command_characters':len(command)})
   finally:session.close()
  session=NativePrologSession(spec)
  try:
   assert session.reset(i,100)==results[0]['reset'];assert session.step({'tool_calls':[{'name':'bash','arguments':{'command':actions[0]}}]})==results[0]['tool_observation'];assert session.step({'text':'Done'})==results[0]['terminal']
  finally:session.close()
  rows.append({'index':i,'original_index':task.data.idx,'controls':results,'fresh_exact_replay':True})
 report={'revision':VERSION,'environment':spec.to_dict(),'rows':rows,'shared_full_public_program_single_constraint_mutation':True,'model_execution':False,'common_epoch':False,'optimizer_ran':False,'train_heldout_split_defined':False}
 (args.out/'controls.json').write_text(json.dumps(report,indent=2));print(json.dumps({'passed':True,'rows':len(rows),'native_grades':[[x['terminal']['reward'] for x in r['controls']]for r in rows]}))
if __name__=='__main__':main()
