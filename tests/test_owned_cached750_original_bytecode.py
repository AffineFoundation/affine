"""Actual CPU subprocesses only; no GPU, science deployment or live state."""
import json,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import test_owned_cached750_group_operator as fixtures
from ops.owned_cached750_group_operator import LocalOriginalTransport,digest,private_json

class OriginalBytecodeTests(unittest.TestCase):
 setUp=fixtures.OperatorTests.setUp
 digest=fixtures.OperatorTests.digest
 sign=fixtures.OperatorTests.sign
 def actual_transport(self):
  t=LocalOriginalTransport.__new__(LocalOriginalTransport);t.root=self.root;t.authority=self.authority;t.code=self.root/'CPU-stub-source';(t.code/'subnet').mkdir(parents=True);(t.code/'subnet/__init__.py').write_text('')
  (t.code/'subnet/remote_runner.py').write_text('''import json,os,pathlib,subprocess,sys
job=pathlib.Path(sys.argv[1])
code="import json,os,sys;print(json.dumps(dict(dont_write=sys.dont_write_bytecode,prefix=sys.pycache_prefix,env_no_write=os.environ.get('PYTHONDONTWRITEBYTECODE'))))"
child=json.loads(subprocess.check_output([sys.executable,'-c',code],text=True))
job.with_suffix('.CPU-evidence').write_text(json.dumps(dict(parent_dont_write=sys.dont_write_bytecode,parent_prefix=sys.pycache_prefix,child=child)))
''');return t
 def prefix(self,envelope):return self.root/('.scientific-bytecode-unused-'+digest(envelope))
 def test_two_originals_have_unique_empty_prefixes_and_descendants_inherit_no_write(self):
  t=self.actual_transport();legacy=self.root/'.scientific-bytecode-unused';legacy.mkdir();stale=legacy/'preserve.pyc';stale.write_bytes(b'stale original14 evidence')
  for envelope in self.jobs[:2]:
   jid=envelope['payload']['job_id'];t.launch(envelope);self.assertEqual(t.children[jid].wait(timeout=5),0);e=json.loads((self.root/(jid+'.CPU-evidence')).read_text());prefix=self.prefix(envelope)
   self.assertTrue(e['parent_dont_write']);self.assertEqual(e['parent_prefix'],str(prefix));self.assertTrue(e['child']['dont_write']);self.assertEqual(e['child']['prefix'],str(prefix));self.assertEqual(e['child']['env_no_write'],'1');self.assertEqual(list(prefix.rglob('*')),[]);self.assertEqual(private_json(self.root/(jid+'.json')),envelope)
  self.assertNotEqual(self.prefix(self.jobs[0]),self.prefix(self.jobs[1]));self.assertEqual(stale.read_bytes(),b'stale original14 evidence')
 def test_existing_empty_or_populated_prefix_refused_before_request_and_Popen(self):
  t=self.actual_transport();p=self.prefix(self.jobs[0]);p.mkdir()
  for populated in (False,True):
   if populated:(p/'old.pyc').write_bytes(b'evidence')
   with patch('ops.owned_cached750_group_operator.subprocess.Popen')as popen:
    with self.assertRaises(ValueError):t.launch(self.jobs[0])
    popen.assert_not_called()
   self.assertFalse((self.root/'original-group-0.json').exists());self.assertEqual((p/'old.pyc').exists(),populated)
 def test_symlink_prefix_refused_before_request_preserving_target(self):
  t=self.actual_transport();target=self.root/'preserve';target.mkdir();self.prefix(self.jobs[0]).symlink_to(target,target_is_directory=True)
  with patch('ops.owned_cached750_group_operator.subprocess.Popen')as popen:
   with self.assertRaises(ValueError):t.launch(self.jobs[0])
   popen.assert_not_called()
  self.assertFalse((self.root/'original-group-0.json').exists());self.assertTrue(self.prefix(self.jobs[0]).is_symlink())
 def test_old_request_refused_without_creating_new_prefix_or_mutating_old_journal(self):
  t=self.actual_transport();p=self.root/'original-group-0.json';p.write_bytes(b'preserve failed original request');meta=self.root/'.owned-cached-group-retention';meta.mkdir();journal=meta/'operator.json';journal.write_bytes(b'preserve dispatch_attempted')
  with patch('ops.owned_cached750_group_operator.subprocess.Popen')as popen:
   with self.assertRaises(ValueError):t.launch(self.jobs[0])
   popen.assert_not_called()
  self.assertFalse(self.prefix(self.jobs[0]).exists());self.assertEqual(p.read_bytes(),b'preserve failed original request');self.assertEqual(journal.read_bytes(),b'preserve dispatch_attempted')

if __name__=='__main__':unittest.main()
