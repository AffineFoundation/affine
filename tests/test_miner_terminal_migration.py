import base64,hashlib,json,tempfile,types,unittest
from pathlib import Path
from nacl.signing import SigningKey
from subnet.remote_backend import RemoteJobs,RemoteMinerReserved
from subnet.storage import canonical
class TerminalMigration(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.state=Path(self.tmp.name);self.key=SigningKey.generate();self.public=self.key.verify_key.encode().hex()
  def signed(v):return {'payload':v,'signer':self.public,'signature':base64.b64encode(self.key.sign(canonical(v)).signature).decode()}
  manifest={'epoch':'old','hourly_execution_policy':{'version':'bounded-hourly-phases-v1'}};job={'job_id':'old-mine-job','role':'mine','manifest':signed(manifest)};self.sha=hashlib.sha256(canonical(job)).hexdigest()
  self.write('old-mine-job-job.json',signed(job));self.write('old.json',dict(role='mine',job_id='old-mine-job',manifest_sha256='fixed',job_sha256=self.sha,physical_workspace='/old/workspace'))
  self.calls=[];self.jobs=types.SimpleNamespace(state=self.state,workspace='/new/workspace',controller=types.SimpleNamespace(authority=types.SimpleNamespace(id=self.public)),remote_status=lambda *a,**k:self.calls.append(a))
 def write(self,n,v): (self.state/n).write_bytes(canonical(v))
 def terminal(self,**changes):self.write('old-mine-job-physical-terminal.json',dict(job_sha256=self.sha,phase='complete',**changes))
 def test_actual_terminal_allows_changed_workspace_without_remote_probe(self):
  self.terminal();RemoteJobs.mine_reservation(self.jobs,'new');self.assertEqual(self.calls,[])
 def test_failed_terminal_also_releases(self):
  self.write('old-mine-job-physical-terminal.json',dict(job_sha256=self.sha,phase='failed'));RemoteJobs.mine_reservation(self.jobs,'new');self.assertEqual(self.calls,[])
 def test_missing_terminal_changed_workspace_blocks_no_new_machine_probe(self):
  with self.assertRaises(RemoteMinerReserved):RemoteJobs.mine_reservation(self.jobs,'new')
  self.assertEqual(self.calls,[])
 def test_changed_terminal_job_digest_rejected(self):
  self.write('old-mine-job-physical-terminal.json',dict(job_sha256='0'*64,phase='complete'))
  with self.assertRaises(ValueError):RemoteJobs.mine_reservation(self.jobs,'new')
 def test_running_evidence_does_not_release(self):
  self.write('old-mine-job-physical-terminal.json',dict(job_sha256=self.sha,phase='running'))
  with self.assertRaises(ValueError):RemoteJobs.mine_reservation(self.jobs,'new')
if __name__=='__main__':unittest.main()
