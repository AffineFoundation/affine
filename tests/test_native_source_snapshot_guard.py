import hashlib,importlib.util,pathlib,tarfile,tempfile,unittest
p=pathlib.Path(__file__).resolve().parents[1]/'ops/native_source_snapshot_guard.py';s=importlib.util.spec_from_file_location('guard',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Controls(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.root=pathlib.Path(self.t.name);self.old=self.root/'approved';self.new=self.root/'candidate';self.new.mkdir();(self.old/'assets').mkdir(parents=True);self.data=b'exact approved problems';(self.old/'assets/tasks.json').write_bytes(self.data);self.files={'assets/tasks.json':hashlib.sha256(self.data).hexdigest()};self.env=[{'spec':{'config':{'task_snapshot':'assets/tasks.json'}}}]
 def test_ignored_missing_data_automatically_included_in_full_inventory_and_archive(self):
  (self.new/'code.py').write_text('code')
  r=m.seal_source_archive(self.new,self.root/'source.tar.gz',self.env,self.old,self.files)
  self.assertEqual(r['inventory']['assets/tasks.json'],self.files['assets/tasks.json']);self.assertEqual(m.validate_archive_snapshots(self.root/'source.tar.gz',self.env,self.files),self.files)
  installed=self.root/'client';installed.mkdir()
  with tarfile.open(self.root/'source.tar.gz')as t:t.extractall(installed,filter='data')
  self.assertEqual((installed/'assets/tasks.json').read_bytes(),self.data)
 def test_original_FE_style_asset_omission_rejected_before_publication(self):
  archive=self.root/'missing.tar.gz'
  with tarfile.open(archive,'w:gz'):pass
  with self.assertRaises(ValueError):m.validate_archive_snapshots(archive,self.env,self.files)
 def test_unapproved_or_corrupt_data_not_included(self):
  with self.assertRaises(ValueError):m.include_snapshots(self.new,self.env,self.old,{})
  (self.old/'assets/tasks.json').write_bytes(b'corrupt')
  with self.assertRaises(ValueError):m.include_snapshots(self.new,self.env,self.old,self.files)
 def test_existing_different_candidate_preserved(self):
  (self.new/'assets').mkdir();p=self.new/'assets/tasks.json';p.write_bytes(b'wrong')
  with self.assertRaises(ValueError):m.include_snapshots(self.new,self.env,self.old,self.files)
  self.assertEqual(p.read_bytes(),b'wrong')
 def test_absolute_bridge_not_misrepresented_as_self_contained(self):
  with self.assertRaises(ValueError):m.snapshot_references([{'spec':{'config':{'task_snapshot':'/var/tmp/external.json'}}}])
 def test_symlink_or_escape_not_packaged(self):
  (self.new/'assets').symlink_to(self.old/'assets',target_is_directory=True)
  with self.assertRaises(ValueError):m.include_snapshots(self.new,self.env,self.old,self.files)
  with self.assertRaises(ValueError):m.snapshot_references([{'spec':{'config':{'task_snapshot':'../elsewhere'}}}])

 def test_publication_guard_rejects_before_any_bucket_operation(self):
  from subnet.publication import publish_source_bundle
  from types import SimpleNamespace
  class ForbiddenBucket:
   def get(self,*args):raise AssertionError('missing dependency reached bucket')
   def put(self,*args):raise AssertionError('missing dependency reached bucket')
  archive=self.root/'missing.tar.gz'
  with tarfile.open(archive,'w:gz'):pass
  with self.assertRaisesRegex(ValueError,'missing exact native snapshot'):
   publish_source_bundle(SimpleNamespace(bucket=ForbiddenBucket()),archive,environments=self.env,approved_files=self.files)

 def test_only_exact_public_FE_metadata_paths_are_bootstrap_admitted(self):
  from subnet.source_bootstrap import public_path
  for name in ('LIVE_LAUNCH_PLAN.md','configs/bounded-audit-policy.json'):
   self.assertEqual(public_path(name),name)
  for name in ('other-launch-plan.md','configs/secret.json','configs/bounded-audit-policy.json/secret','configs/../wallets/key','wallets/key','credentials.json'):
   with self.assertRaises(ValueError):public_path(name)
