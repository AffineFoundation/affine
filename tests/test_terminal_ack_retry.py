import base64,hashlib,json,os,shlex,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet.remote_backend import RemoteJobs
from ops.trainer_lifecycle import terminal_ack_retry as retry


class AckTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  (self.root/'runner-status').mkdir();(self.root/'.optimizer-state-cache').mkdir()
  key=SigningKey.generate();authority=key.verify_key.encode().hex();payload=dict(version='durable-original-trainer-cache-ACK-v1',job_id='original',authority_state_committed=True)
  self.ack=dict(payload=payload,signer=authority,signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())
  self.key=key
  class Client(RemoteJobs):
   def __init__(self):pass
   def command(self,command,timeout=60):return subprocess.check_output(command,shell=True,text=True,timeout=timeout)
  self.Client=Client;self.client=Client();self.client.workspace=str(self.root);self.client.python=sys.executable
  self.client.controller=types.SimpleNamespace(authority=types.SimpleNamespace(id=authority))
  retry.install(Client,retry_delay_seconds=0)
 def script(self):
  return 'from pathlib import Path;import json,time\np=Path('+repr(str(self.root/'runs'))+')\nn=int(p.read_text())+1 if p.exists()else 1\np.write_text(str(n));time.sleep(.05)\nprint(json.dumps(dict(status="deferred",reason="workspace-role-in-flight",removed_checkpoints=[])if n==1 else dict(status="complete",optimizer_cache_promotion=dict(promoted=True))))'
 def test_actual_detached_ACK_retry_uses_same_payload_preserves_original(self):
  result=self.client._durable_cache_ack(self.script(),self.ack,'original')
  self.assertEqual(result['status'],'complete');self.assertEqual((self.root/'runs').read_text(),'2')
  rows=list(self.root.glob('*-attempt.json'));self.assertEqual(len(rows),2)
  for path in rows:self.assertEqual(json.loads(path.read_bytes()),dict(ack=self.ack))
  results=[json.loads(p.read_bytes())for p in self.root.glob('*-result.json')]
  self.assertEqual(sorted(v['receipt']['status']for v in results),['complete','deferred'])
  self.assertEqual(self.client._durable_cache_ack(self.script(),self.ack,'original')['status'],'complete')
  self.assertEqual((self.root/'runs').read_text(),'2')
 def test_live_workspace_defers_without_consuming_retry_handle(self):
  ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]
  path=self.root/'runner-status/calibration.json';path.write_text(json.dumps(dict(child_pid=os.getpid(),child_pid_ticks=ticks)));path.chmod(0o600)
  result=self.client._durable_cache_ack(self.script(),self.ack,'original')
  self.assertEqual(result['reason'],'workspace-role-in-flight');self.assertEqual((self.root/'runs').read_text(),'1')
  self.assertEqual(len(list(self.root.glob('*-attempt.json'))),1)
  path.unlink()
  self.assertEqual(self.client._durable_cache_ack(self.script(),self.ack,'original')['status'],'complete')
  self.assertEqual((self.root/'runs').read_text(),'2')
 def test_unknown_or_failed_ack_never_retries(self):
  for reason in ('original-cache-ACK-pending','original-cache-ACK-failed','unrecognized'):
   calls=[]
   class C:
    def _durable_cache_ack(s,script,ack,jobid):calls.append(jobid);return dict(status='deferred',reason=reason)
   c=C();c.workspace=str(self.root);c.controller=self.client.controller
   retry.install(C,retry_delay_seconds=0)
   self.assertEqual(c._durable_cache_ack('',self.ack,'original')['reason'],reason);self.assertEqual(calls,['original'])
 def test_changed_payload_or_identity_rejected(self):
  with self.assertRaises(ValueError):self.client._durable_cache_ack(self.script(),self.ack,'different')
  self.ack['payload']['job_id']='changed'
  with self.assertRaises(Exception):self.client._durable_cache_ack(self.script(),self.ack,'changed')
  self.assertFalse((self.root/'runs').exists())


if __name__=='__main__':unittest.main()
