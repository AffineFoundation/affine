"""ROOT-only install metadata recovery. Never stops/restarts API or auditor."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from unittest.mock import patch
def main():
 p=argparse.ArgumentParser();p.add_argument('--activate',action='store_true',required=True);p.add_argument('--package',required=True);p.add_argument('--authority',required=True);a=p.parse_args();P=Path(a.package);AUTH=a.authority;sys.path.insert(0,str(P/'operator-package'));from ops import trusted_cpu_service_lifetime as m;from ops.running_api_validation import canonical,authenticate,file_bytes,write_exclusive;document=json.loads((P/'scope.TRUSTED-CPU-SERVICE-LIFETIME.ROOT-SIGNED.private.json').read_bytes());s=authenticate(document,AUTH);prepared=json.loads((P/'prepared-service-files.private.json').read_bytes());evidence=P/'ACTUAL-lifetime-supervisor-activation.private.json';assert not evidence.exists()
 for role in ('api','auditor'):assert m.current(s['services'][role])==s['services'][role]['initial_instance'],'preserve exact original services during metadata handoff'
 original=m.file_bytes
 with patch.object(m,'file_bytes',side_effect=lambda path:original(prepared.get(str(path),path))):m.validate(document,AUTH,require_execution=True)
 staged=[]
 for target,source in prepared.items():
  dest=Path(target);raw=original(source);assert hashlib.sha256(raw).hexdigest()==s['files'][target];assert str(dest).startswith('/home/const/.config/systemd/user/')and dest==dest.resolve();dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert dest.read_bytes()==raw
  else:
   fd=os.open(dest,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
   with os.fdopen(fd,'wb')as f:f.write(raw)
  staged.append(target)
 subprocess.run(['systemctl','--user','daemon-reload'],check=True,timeout=30);m.validate(document,AUTH,require_execution=True)
 # Original service processes remain alive. Only the two new CPU supervisors start.
 for role in ('api','auditor'):assert m.current(s['services'][role])==s['services'][role]['initial_instance'];subprocess.run(['systemctl','--user','start',s['services'][role]['supervisor_unit']],check=True,timeout=45)
 deadline=time.monotonic()+60;ready={}
 while time.monotonic()<deadline:
  for role in ('api','auditor'):
   name=s['services'][role]['supervisor_unit'];observed=m.current(dict(unit=name));expected=list(s['services'][role]['expected_argv']);expected[4]='supervise'
   if not observed['process']or observed['process']['argv']!=expected or observed['systemd']['ActiveState']!='active':continue
   records=[]
   for path in Path(s['record_directory']).glob(role+'-*.json'):
    row=authenticate(json.loads(original(path)),AUTH)
    if row['policy_sha256']==m.digest(s)and row['event']in('instance_observed','restart_ready','restart_ready_after_start_timeout'):records.append(str(path))
   if records:ready[role]=dict(supervisor=observed,signed_instance_records=records)
  if len(ready)==2:break
  time.sleep(.25)
 assert len(ready)==2,'both CPU supervisors and signed instance observations required';write_exclusive(evidence,dict(at=time.time(),supervisors=ready,original_services={role:m.current(s['services'][role])for role in ('api','auditor')},API_or_auditor_stopped=False,job_permissions_unchanged=True,staged_metadata=staged));print(str(evidence))
if __name__=='__main__':main()
