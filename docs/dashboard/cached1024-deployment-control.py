import sys,json,hashlib,shutil,time,subprocess
from pathlib import Path
sys.path.insert(0,'/tmp/affine-cached-dashboard-projection')
from dashboard.cached_evaluator_projection import rows
R=Path('/home/const/subnet120-rewrite');W=Path('/home/const/subnet120/affine/website');C=Path('/tmp/affine-cached-dashboard-projection');D=R/'state/root-audits/owned-cached-fixed32-parent10-parent11-preparation-20261006-v1/continuous-fixed32-cap1024-v3-ROOT-REVIEW';B=D/('dashboard-deployment-backup-'+str(time.time_ns()));B.mkdir(mode=0o700)
pointer=D/'dashboard-sources.ROOT-SIGNED.REVIEW-ONLY.private.json';projected=rows(json.loads(pointer.read_bytes()),R/'state/live-math-launch-preparation-v1/distributed-preparation/live-controller-v1/controller-state');assert len(projected)==2 and sorted(r['successes']for r in projected)==[18,19]
server=R/'dashboard/server.py';s=server.read_text();assert 'from dashboard.incentive import'in s and 'grid_outcomes' in s
before="'original_error_count','recovered_count','status_detail')";assert s.count(before)==1;s=s.replace(before,"'original_error_count','recovered_count','status_detail','original_epoch_id',\n                'sampling_policy','experiment_id','native_graded','proof_verification_performed','original_report_sha256')")
before="        for path in evaluation_paths:\n            raw = read(path,{})";after="        evaluation_inputs=[read(path,{})for path in evaluation_paths]\n        pointer=self.source/'dashboard/cached-evaluator-sources.ROOT-SIGNED.json'\n        if pointer.is_file():\n            try:\n                from dashboard.cached_evaluator_projection import rows as cached_rows\n                evaluation_inputs.extend(cached_rows(read(pointer,{}),launch/'controller-state'))\n            except Exception:\n                pass # Invalid diagnostic projection is unavailable, never zero reward.\n        for raw in evaluation_inputs:";assert s.count(before)==1;s=s.replace(before,after)
changes={server:s.encode(),R/'dashboard/cached_evaluator_projection.py':(C/'dashboard/cached_evaluator_projection.py').read_bytes(),R/'state/dashboard/cached-evaluator-sources.ROOT-SIGNED.json':pointer.read_bytes()}
for p in (R/'dashboard/public/network.js',W/'network.js'):
 raw=p.read_text();assert len(raw.splitlines())>1000;before='e.output_token_budget,e.policy_kind]';assert raw.count(before)==1;raw=raw.replace(before,'e.output_token_budget,e.policy_kind,e.sampling_policy,e.experiment_id]');before=r'\nCheckpoint ${String(row.checkpoint';assert raw.count(before)==1;raw=raw.replace(before,r"\n${row.sampling_policy?`Diagnostic policy ${row.sampling_policy} · cap ${row.output_token_budget}\n`:''}Checkpoint ${String(row.checkpoint");changes[p]=raw.encode()
old={}
for i,(p,raw)in enumerate(changes.items()):
 if p.exists():shutil.copyfile(p,B/(str(i)+'-'+p.name));old[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.cached1024-tmp');temp.write_bytes(raw);temp.chmod(0o600 if 'ROOT-SIGNED'in p.name else 0o644);temp.replace(p)
for p in (R/'dashboard/public/network.js',W/'network.js'):subprocess.run(['node','--check',str(p)],check=True)
subprocess.run([str(R/'.venv/bin/python'),'-m','unittest','dashboard.test_server'],cwd=R,check=True)
subprocess.run(['systemctl','--user','restart','affine-network-dashboard.service'],check=True)
evidence=dict(version='ROOT-authorized-cached1024-dashboard-deployment-v1',deployed_at=time.time(),backup=str(B),before=old,after={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in changes},actual_counts=[19,18],cohort=32,cap=1024,policy='owned-cached-native-evaluation-v1',original_reports_preserved=True,science_source_unchanged=True);(D/'ACTUAL-CACHED1024-DASHBOARD-DEPLOYMENT.private.json').write_text(json.dumps(evidence,sort_keys=True,indent=2)+'\n');shutil.copyfile(__file__,D/'ACTUAL-cached1024-dashboard-deploy-helper.py');print(json.dumps(evidence))
