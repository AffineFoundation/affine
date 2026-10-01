"""Original NQueens native actor/grader controls; no model or shared epoch."""
import argparse,asyncio,hashlib,json,time
from pathlib import Path
from subnet.environments import build_spec,_taskset
from subnet.native_prolog_actor import public_task,build,PublicActor,nqueens_command,grade_original

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    spec=build_spec('affine_prolog',config={'taskset':{'tasks':['prolog-nqueens-0005','prolog-nqueens-0014','prolog-nqueens-0023'],'num_examples':32,'difficulty':'medium'}},num_samples=3,max_turns=2,max_output_tokens=512)
    tasks=list(_taskset(spec));assert len(tasks)==3
    from affine_prolog_v1.taskset import SWIPL_SHIM
    runtime=build(args.out/'image',SWIPL_SHIM);rows=[];geometry=[]
    for task in tasks:
        public=public_task(task);geometry.append(public['starter_sha256']);(args.out/(task.data.name+'-public.json')).write_text(json.dumps(public,indent=2));controls=[]
        for negative in (False,True):
            actor=PublicActor(runtime,public)
            try:
                actor.start();tool=actor.shell(nqueens_command(public,negative),65);grade=asyncio.run(grade_original(task,actor))
                controls.append({'negative_control':negative,'observation':tool,'native_grade':grade})
            finally:actor.close()
        assert controls[0]['native_grade']['reward']==1. and controls[1]['native_grade']['reward']==0.
        # Fresh actor replay of the exact public positive tool action/observation.
        actor=PublicActor(runtime,public)
        try:
            actor.start();replay=actor.shell(nqueens_command(public),65);grade=asyncio.run(grade_original(task,actor));assert replay==controls[0]['observation'] and grade==controls[0]['native_grade']
        finally:actor.close()
        rows.append({'original_index':task.data.idx,'task_name':task.data.name,'original_task_hash':task.hash,'public_descriptor_sha256':hashlib.sha256(json.dumps(public,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'controls':controls,'fresh_exact_native_replay':True})
    from subnet import native_prolog_actor as actor_module
    # Fresh, exact owned actors exercise hostile tool bounds without touching peers.
    attack_results=[]
    for command in ['yes AFFINE_BUDGET','while :; do :; done']:
        actor=PublicActor(runtime,public_task(tasks[0]));actor.start();started=time.monotonic()
        try:
            actor.shell(command,timeout=2);raise AssertionError('hostile command was not rejected')
        except (ValueError,__import__('subprocess').TimeoutExpired) as error:
            assert not actor.started
            attack_results.append({'kind':'output_budget' if command.startswith('yes') else 'walltime','rejected':True,'owned_container_closed':True,'seconds':time.monotonic()-started,'error_type':type(error).__name__})
        finally:actor.close()
    report={'revision':'original-prolog-public-actor-native-grader-v1','runtime':runtime,'rows':rows,'model_execution':False,'proofs_generated':False,'common_epoch':False,'optimizer_ran':False,'hidden_reference_used_by_public_policy':False,'scope':'three original medium NQueens indices; repeated problem geometries explicitly disclosed; other eight Prolog CSP kinds unqualified','unique_problem_geometries':len(set(geometry)),'problem_geometry_hashes':geometry,'train_heldout_split_defined':False,'hostile_command_controls':attack_results,'actor_source_sha256':hashlib.sha256(Path(actor_module.__file__).read_bytes()).hexdigest(),'probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'completed_at':time.time()}
    (args.out/'controls.json').write_text(json.dumps(report,indent=2));print(json.dumps({'passed':True,'rows':len(rows),'rewards':[[c['native_grade']['reward'] for c in r['controls']] for r in rows]}))
if __name__=='__main__':main()
