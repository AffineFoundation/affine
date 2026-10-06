"""ROOT-only explicit lifetime signing. No service or historical scope mutations."""
import argparse,importlib.util,json,os,sys,time
from pathlib import Path
from unittest.mock import patch
def main():
 p=argparse.ArgumentParser();p.add_argument('--sign',action='store_true',required=True);p.add_argument('--package',required=True);p.add_argument('--authority',required=True);a=p.parse_args();P=Path(a.package);AUTH=a.authority;sys.path.insert(0,str(P/'operator-package'));from ops import trusted_cpu_service_lifetime as m;from ops.running_api_validation import canonical,digest,sign,write_exclusive;s=json.loads((P/'scope.TRUSTED-CPU-SERVICE-LIFETIME.UNSIGNED.private.json').read_bytes())['payload'];assert s['execute_allowed']is False
 for role in ('api','auditor'):assert m.current(s['services'][role])==s['services'][role]['initial_instance'],'fresh exact original process required'
 output=Path(s['policy_path']);status=Path(s['authorization_status_path']);assert not output.exists()and not status.exists();s.update(created_at=time.time(),execute_allowed=True);key=m.key_for(s['authority_seed_path'],AUTH);document=sign(key,s);authorization=sign(key,dict(version=m.STATUS,active_policy_sha256=digest(s)));prepared=json.loads((P/'prepared-service-files.private.json').read_bytes());original=m.file_bytes
 def virtual(path):
  if str(path)==s['authorization_status_path']:return canonical(authorization)
  return original(prepared.get(str(path),path))
 with patch.object(m,'file_bytes',side_effect=virtual):m.validate(document,AUTH,require_execution=True)
 write_exclusive(output,document);write_exclusive(status,authorization);print(json.dumps(dict(policy=str(output),status=str(status),services_changed=False,explicit_lifetime_authorization=True)))
if __name__=='__main__':main()
