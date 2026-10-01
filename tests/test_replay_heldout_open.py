import tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import importlib.util, subprocess, shutil
_root=Path(__file__).resolve().parents[1]
_copy=Path(tempfile.mkdtemp());(_copy/'subnet').mkdir()
shutil.copy2(_root/'subnet/remote_backend.py',_copy/'subnet/remote_backend.py')
subprocess.run(['git','apply','--include=subnet/remote_backend.py',str(_root/'ops/replay_source_patches/remote_backend.py.patch')],cwd=_copy,check=True)
_spec=importlib.util.spec_from_file_location('subnet._qualified_replay_remote',_copy/'subnet/remote_backend.py')
_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
RemoteController=_module.RemoteController
from subnet.controller import Controller
class Bucket:
 def __init__(self):self.writes=[]
 def json(self,key,value):self.writes.append((key,value))
class Tests(unittest.TestCase):
 def controller(self):
  c=object.__new__(RemoteController);c.bucket=Bucket();c.state=Path(tempfile.mkdtemp());c.signed=lambda value:dict(payload=value);return c
 def base(self,c,*args,**kwargs):
  self.assertNotIn('heldout_indices',kwargs);manifest=dict(epoch='test-open',environments=kwargs['environments']);c.bucket.json('public/test-open/manifest.json',c.signed(manifest));return manifest
 def test_first_publication_contains_registry(self):
  c=self.controller();envs=[dict(spec=dict(id='actual',num_samples=32),indices=[0])]
  with patch.object(Controller,'open',lambda c,*a,**k:self.base(c,*a,**k)):
   manifest=c.open('test-open',{}, {},environments=envs,heldout_indices={'actual':[16,17]},max_batches=3)
  self.assertEqual(manifest['heldout_indices'],{'actual':[16,17]});self.assertEqual(len(c.bucket.writes),1);self.assertEqual(c.bucket.writes[0][1]['payload']['heldout_indices'],{'actual':[16,17]})
 def test_invalid_registry_rejects_before_base_open(self):
  for registry in ({},{'actual':[0]},{'actual':[True]},{'actual':[16,16]}):
   c=self.controller()
   with patch.object(Controller,'open') as base:
    with self.assertRaises(ValueError):c.open('test-open',{}, {},environments=[dict(spec=dict(id='actual',num_samples=32),indices=[0])],heldout_indices=registry)
    base.assert_not_called();self.assertEqual(c.bucket.writes,[])
if __name__=='__main__':unittest.main()
