import hashlib,importlib.util,io,pathlib,tarfile,tempfile,unittest
p=pathlib.Path(__file__).resolve().parents[1]/'ops/native_task_asset_deployment.py';s=importlib.util.spec_from_file_location('asset',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Controls(unittest.TestCase):
 def setUp(self):
  self.old=(m.SHA,m.SIZE);self.data=b'exact task bytes';m.SHA=hashlib.sha256(self.data).hexdigest();m.SIZE=len(self.data);self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.addCleanup(self.restore)
 def restore(self):m.SHA,m.SIZE=self.old
 def archive(self,link=False):
  b=io.BytesIO()
  with tarfile.open(fileobj=b,mode='w:gz')as t:
   r=tarfile.TarInfo(m.SNAPSHOT);r.size=len(self.data)
   if link:r.type=tarfile.SYMTYPE;r.linkname='/tmp/arbitrary'
   t.addfile(r,None if link else io.BytesIO(self.data))
  return b.getvalue()
 def test_exact_archive_then_atomic_idempotent_outside_code(self):
  raw=self.archive();data=m.approved_snapshot(raw,hashlib.sha256(raw).hexdigest(),len(raw));root=pathlib.Path(self.tmp.name)/'assets';p=m.install_snapshot(data,root);self.assertEqual(p.read_bytes(),self.data);self.assertEqual(p,m.install_snapshot(data,root));self.assertEqual(p.stat().st_mode&0o777,0o600)
 def test_archive_corruption_rejected_before_install(self):
  raw=self.archive()
  with self.assertRaises(ValueError):m.approved_snapshot(raw+b'!',hashlib.sha256(raw).hexdigest(),len(raw))
 def test_symlink_archive_rejected(self):
  raw=self.archive(True)
  with self.assertRaises(ValueError):m.approved_snapshot(raw,hashlib.sha256(raw).hexdigest(),len(raw))
 def test_existing_corrupt_never_overwritten(self):
  root=pathlib.Path(self.tmp.name);p=m.install_snapshot(self.data,root);p.write_bytes(b'corrupt')
  with self.assertRaises(ValueError):m.install_snapshot(self.data,root)
  self.assertEqual(p.read_bytes(),b'corrupt')
 def test_external_symlink_rejected(self):
  root=pathlib.Path(self.tmp.name);(root/m.SHA).symlink_to(root,target_is_directory=True)
  with self.assertRaises(ValueError):m.install_snapshot(self.data,root)
 def test_path_only_original_binding_preserved(self):
  c={'source_bundle':{'sha256':'original'},'environments':[{'spec':{'id':'affine_math','source_hash':'same','config':{'task_snapshot':m.SNAPSHOT}},'indices':[1,2]}]};v=m.path_only_config(c,'/external/exact.json');self.assertEqual(c['environments'][0]['spec']['config']['task_snapshot'],m.SNAPSHOT);v['environments'][0]['spec']['config']['task_snapshot']=m.SNAPSHOT;self.assertEqual(v,c)
 def test_wrong_snapshot_does_not_change_contract(self):
  with self.assertRaises(ValueError):m.path_only_config({'environments':[{'spec':{'id':'affine_math','config':{'task_snapshot':'different'}}}]},'/external/file')
 def test_warm_hydration_never_downloads_and_still_hashes(self):
  root=pathlib.Path(self.tmp.name);p=m.install_snapshot(self.data,root)
  class Never:
   def open(self,*args,**kwargs):raise AssertionError('warm cache should not GET')
  self.assertEqual(m.hydrate_snapshot({},root,Never()),p)
  p.write_bytes(b'corrupt')
  with self.assertRaises(ValueError):m.hydrate_snapshot({},root,Never())
 def test_bounded_download_checks_actual_bytes(self):
  raw=self.archive()
  class Response:
   status=200
   def __enter__(self):return self
   def __exit__(self,*args):pass
   def read(self,size):return raw[:size]
  class Opener:
   def open(self,*args,**kwargs):return Response()
  self.assertEqual(m.download_snapshot('https://storage.invalid/approved',hashlib.sha256(raw).hexdigest(),len(raw),Opener()),self.data)
  with self.assertRaises(ValueError):m.download_snapshot('https://storage.invalid/approved',hashlib.sha256(raw).hexdigest(),len(raw)-1,Opener())
