import base64, contextlib, hashlib, io, json, sqlite3, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from nacl.signing import SigningKey
import ops.automatic_submission_retention as m
from ops.submission_retention import canonical,digest

def sign(key,v):return dict(signer=key.verify_key.encode().hex(),payload=v,signature=base64.b64encode(key.sign(canonical(v)).signature).decode())

class Enumeration(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.root=Path(self.t.name);self.authority_key=SigningKey.generate();self.auth=self.authority_key.verify_key.encode().hex();self.worker=SigningKey.generate();self.other=SigningKey.generate();self.who=self.worker.verify_key.encode().hex();self.unapproved=self.other.verify_key.encode().hex();self.source='a'*64;self.files={'subnet/storage.py':'c'*64};self.calls=[]
  (self.root/'roles').mkdir();self.db=self.root/'roles/verifier-queue.sqlite3';self.cx=sqlite3.connect(self.db)
  self.cx.execute('create table jobs (id text,status text,role text,attempt integer,worker text,token text,digest text,envelope text,report text,report_digest text,report_request text)');self.addCleanup(self.cx.close)
  self.config=dict(source_bundle={'sha256':self.source},state=str(self.root),bucket={'name':'durable'},remote={'roles':{'verify':[dict(worker_identity=self.who,workspace=str(self.root/'good')),dict(worker_identity=self.unapproved,workspace=str(self.root/'old'))]}})
  self.writer=dict(queue_database=str(self.db),verifier_identities=[self.who],_verifier_workforce={})
  self.cp=self.root/'config.json';self.cp.write_text(json.dumps(self.config));self.pp=self.root/'pointer.json';self.pp.write_text('{}');self.output=self.root/'out'
 def row(self,job_id,worker=None,malform=None):
  worker=worker or self.worker;who=worker.verify_key.encode().hex();work=self.root/('good'if who==self.who else'old');hashed=hashlib.sha256(b'archive').hexdigest()
  manifest=dict(epoch='nonpayable-test',payable=False,checkpoint={'id':'model'},source_bundle={'sha256':self.source},backend_profile={'profile':'test'},numerical_policy={'tolerance':0},audit_frozen_receipts={'miner':dict(sha256=hashed,size=7,frozen_key='public/nonpayable-test/submissions/'+'b'*64+'.zip')})
  job=dict(job_id=job_id,role='verify',created_at=10,expires_at=20,manifest=sign(self.authority_key,manifest),source_files=self.files,runtime_versions={'torch':'pinned'},submissions=[{'sha256':hashed,'url':'private capability'}])
  report=dict(job_id=job_id,job_sha256=digest(job),operator=self.auth,role='verify',epoch=manifest['epoch'],checkpoint='model',source_files=self.files,runtime_versions=job['runtime_versions'],backend_profile=manifest['backend_profile'],numerical_policy=manifest['numerical_policy'],chain_transactions=False,success=True,completed_at=15,audits=[dict(epoch=manifest['epoch'],submission_sha256=hashed,accepted=[])])
  request=sign(worker,dict(action='report',job_id=job_id,token='winning lease',report=report));envelope=sign(self.authority_key,job)
  if malform=='job_signature':envelope['signature']=base64.b64encode(b'x'*64).decode()
  if malform=='report_signature':request['signature']=base64.b64encode(b'x'*64).decode()
  row=dict(id=job_id,status='complete',role='verify',attempt=1,worker=who,token='winning lease',digest=digest(job),envelope=json.dumps(envelope),report=json.dumps(report),report_digest=digest(report),report_request=json.dumps(request))
  self.cx.execute('insert into jobs values (?,?,?,?,?,?,?,?,?,?,?)',tuple(row.values()));self.cx.commit()
  path=work/'backend/jobs'/job_id;path.mkdir(parents=True);(path/'submission-0.zip').write_bytes(b'archive');(path/'report.json').write_text(json.dumps(report));return row,path
 def remote(self,endpoint,code,timeout=90):
  self.calls.append(endpoint['worker_identity']);output=io.StringIO()
  # The remote apply script is executed only against temporary fixture files.
  if 'namespace["remove_verified_replica"]'in code:
   code=code.replace('plans=', 'namespace["process_directories"]=lambda:[__import__("pathlib").Path("/proc",str(__import__("os").getpid()))]\nplans=',1)
  with contextlib.redirect_stdout(output):exec(compile(code,'test-remote','exec'),{})
  return json.loads(output.getvalue())
 def run_sweep(self,apply=True):
  client=SimpleNamespace(get_object=lambda **kw:{'Body':io.BytesIO(b'archive')})
  with patch.object(m,'retention_writer',return_value=self.writer),patch.object(m,'approved_source_members',return_value={self.source:self.files}),patch.object(m,'Bucket',return_value=SimpleNamespace(client=client,name='durable')),patch.object(m,'remote',side_effect=self.remote):
   return m.run(str(self.cp),str(self.pp),self.auth,str(self.output),apply=apply)
 def test_obsolete_identity_does_not_starve_signed_current_job(self):
  good,gp=self.row('good');bad,bp=self.row('old',self.other)
  before=self.db.read_bytes();r=self.run_sweep()
  self.assertEqual(r['removed_bytes'],7);self.assertEqual(len(r['deferred_jobs']),1);self.assertEqual(r['deferred_jobs'][0]['reason'],'approved verifier identity');self.assertEqual(self.calls,[self.who,self.who]);self.assertFalse((gp/'submission-0.zip').exists());self.assertTrue((gp/'report.json').exists());self.assertEqual((bp/'submission-0.zip').read_bytes(),b'archive');self.assertEqual(self.db.read_bytes(),before)
  ledger=json.loads((self.output/'completed-cache-retention.json').read_bytes());self.assertEqual([x['job_id']for x in ledger.values()],['good'])
  r2=self.run_sweep();self.assertEqual(r2['removed_bytes'],0);self.assertEqual(len(r2['deferred_jobs']),1)
 def test_forged_job_and_worker_report_stay_unmodified(self):
  self.row('good');_,bp=self.row('badjob',malform='job_signature');_,rp=self.row('badreport',malform='report_signature');r=self.run_sweep();self.assertEqual(r['removed_bytes'],7);self.assertEqual(len(r['deferred_jobs']),2)
  for p in (bp,rp):self.assertEqual((p/'submission-0.zip').read_bytes(),b'archive');self.assertTrue((p/'report.json').exists())
 def test_only_rejections_issue_no_remote_or_archive_calls(self):
  _,bp=self.row('old',self.other);r=self.run_sweep();self.assertEqual(self.calls,[]);self.assertEqual(r['workers'],{});self.assertEqual(r['removed_bytes'],0);self.assertEqual(json.loads((self.output/'completed-cache-retention.json').read_bytes()),{});self.assertTrue((bp/'submission-0.zip').exists())
 def test_dry_run_preserves_all_files_and_never_ledgers(self):
  _,gp=self.row('good');self.row('old',self.other);r=self.run_sweep(False);self.assertEqual(r['removed_bytes'],0);self.assertTrue((gp/'submission-0.zip').exists());self.assertFalse((self.output/'completed-cache-retention.json').exists());self.assertEqual(self.calls,[self.who])
if __name__=='__main__':unittest.main()
