"""Historical contracts keep their approved pins; live admission stays strict."""
import json,sys,tempfile,unittest
from pathlib import Path
from nacl.signing import SigningKey
from ops import live_reward_writer as writer
from ops.live_reward_exporter import sign
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_live_reward_writer import lineage_fixture
from subnet.backend_jobs import SOURCE_FILES


class HistoricalSourceLineageTests(unittest.TestCase):
 def test_legacy_authenticated_job_does_not_require_future_module(self):
  with tempfile.TemporaryDirectory() as folder:
   state,manifest,audit,authority,config,db,files,key,job,remote=lineage_fixture(folder)
   original=tuple(name for name in SOURCE_FILES if name!='subnet/forced_sampling.py')
   pins={name:files[name] for name in original}
   job['source_files']=pins;remote['source_files']=pins;remote['job_sha256']=writer.sha(job)
   worker=SigningKey.generate();config['verifier_identities']=[worker.verify_key.encode().hex()]
   request=sign(dict(action='report',job_id=job['job_id'],report=remote),worker)
   envelope=sign(job,key);(state/'roles'/'verify-CPU-job.json').write_text(json.dumps(envelope))
   db.execute('UPDATE jobs SET envelope=?,worker=?,report_request=?,report=?,report_digest=?',
              (json.dumps(envelope),config['verifier_identities'][0],json.dumps(request),json.dumps(remote),writer.sha(remote)))
   with self.assertRaisesRegex(ValueError,'missing worker source pins'):
    writer.verify_audit_lineage(state,manifest,audit,authority,config,db,160.,files)
   result=writer.verify_audit_lineage(state,manifest,audit,authority,config,db,160.,files,required_source_files=original)
   self.assertEqual(result['submission_sha256'],audit['submission_sha256'])
   with self.assertRaisesRegex(ValueError,'missing worker source pins'):
    writer.verify_audit_lineage(state,manifest,audit,authority,config,db,160.,files,required_source_files=())
   with self.assertRaisesRegex(ValueError,'archive module pins'):
    writer.verify_audit_lineage(state,manifest,audit,authority,config,db,160.,dict(files,**{'subnet/model.py':'f'*64}),required_source_files=original)

 def test_declaration_is_read_without_executing_archive_code(self):
  literal="SOURCE_FILES = ('subnet/model.py','subnet/backend_jobs.py')"
  generator="SOURCE_FILES = tuple('subnet/'+n+'.py' for n in ('model','backend_jobs'))"
  expected=('subnet/model.py','subnet/backend_jobs.py')
  self.assertEqual(writer.original_required_source_files(literal),expected)
  self.assertEqual(writer.original_required_source_files(generator),expected)
  for invalid in ["SOURCE_FILES=()",literal+'\nSOURCE_FILES=()',literal+'\nSOURCE_FILES+=()',
                  "SOURCE_FILES = tuple(__import__('os').system(n) for n in ('model',))",
                  "SOURCE_FILES = tuple('subnet/'+n+'.py' for n in ('model',) if True)",
                  "SOURCE_FILES = ('../model.py',)","def change():\n SOURCE_FILES=('subnet/model.py',)",
                  "SOURCE_FILES = tuple('subnet/'+n+'.py' for n in (__import__('os'),))",
                  literal+'\nOTHER,SOURCE_FILES=(1,())',literal+'\ndel SOURCE_FILES',
                  literal+'\nfor SOURCE_FILES in (): pass']:
   with self.subTest(declaration=invalid),self.assertRaises((ValueError,TypeError)):
    writer.original_required_source_files(invalid)


if __name__=='__main__':unittest.main()
