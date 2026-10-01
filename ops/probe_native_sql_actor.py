"""Original public Spider bash/replay controls, without model claims."""
import json
import subprocess
import time
from pathlib import Path
from subnet.native_sql_actor import build,PublicSQLActor
from subnet.native_sql_isolation import grade

def execute(runtime,public,actions):
    actor=PublicSQLActor(runtime,public)
    try:
        if actor.start()!=public:raise ValueError('public original descriptor')
        observations=[actor.call('bash',{'command':text}) for text in actions]
        inspection=json.loads(subprocess.check_output(['docker','inspect',actor.name]))[0]
        host=inspection['HostConfig']
        if inspection['Mounts'] or host['NetworkMode']!='none' or not host['ReadonlyRootfs'] or inspection['Config']['User']!='65534:65534':raise ValueError('public actor isolation')
        return {'observations':observations,'isolation':{'network':host['NetworkMode'],'read_only':host['ReadonlyRootfs'],'host_mounts':inspection['Mounts'],'user':inspection['Config']['User'],'memory':host['Memory']}}
    finally:actor.close()

def main():
    p=Path('state/native-sql-isolation');private=json.loads((p/'private-task.json').read_text())
    grader=json.loads((p/'runtime.json').read_text());runtime,public=build(private,p/'actor-image')
    (p/'actor-runtime.json').write_text(json.dumps(runtime,indent=2));(p/'actor-public.json').write_text(json.dumps(public,indent=2))
    # A public-question/schema-derived native control, not model generation.
    actions=['sqlite3 /workspace/department_management.sqlite ".tables"',
             'sqlite3 /workspace/department_management.sqlite ".schema head"',
             'sqlite3 /workspace/department_management.sqlite "SELECT COUNT(*) FROM head WHERE age > 56;"']
    first=execute(runtime,public,actions);second=execute(runtime,public,actions)
    if first!=second:raise ValueError('fresh original shell observations')
    if any(v['exit_code'] for v in first['observations']):raise ValueError('original shell control')
    positive=grade(private,'```sql\nSELECT COUNT(*) FROM head WHERE age > 56\n```',grader)
    negative=grade(private,'```sql\nSELECT COUNT(*) FROM head WHERE age > 999\n```',grader)
    if positive['reward']!=1 or negative['reward']!=0:raise ValueError('original SQL outcome')
    boundary=execute(runtime,public,['test ! -e /opt/affine-sql/taskset.py && test ! -e /opt/affine-sql/grade.py && test ! -e /home/const/subnet120/.env',
                                     "python -c 'print(\"x\"*40000)'",'id -u'])
    if boundary['observations'][0]['exit_code'] or len(boundary['observations'][1]['stdout'])!=32768 or boundary['observations'][2]['stdout']!='65534\n':raise ValueError('actor privilege/output boundary')
    report={'completed':True,'runtime':runtime,'original_index':0,'public_descriptor':public,
            'actions':actions,'observations':first['observations'],'isolation':first['isolation'],
            'fresh_exact_replay':True,'positive_reward':1,'negative_reward':0,
            'private_grader_paths_absent':True,'output_truncation_matches_original_32768':True,
            'model_proofs_generated':False,'common_epoch_admitted':False,'training_performed':False,
            'chain_transactions':False,'completed_at':time.time()}
    (p/'actor-controls.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('public_descriptor','observations','actions')}))

if __name__=='__main__':main()
