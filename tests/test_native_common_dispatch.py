import copy,json,unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from subnet.environments import EnvironmentSpec,_source_hash,create_session
from subnet.native_common_dispatch import validate_prolog_binding
FIXTURE=Path(__file__).parent/'fixtures/prolog_dispatch_public.json'
class DispatchTests(unittest.TestCase):
 def setUp(self):
  raw=json.loads(FIXTURE.read_text());self.spec=EnvironmentSpec.from_dict(raw);self.spec=replace(self.spec,version='prime-native-prolog-dispatch-v2',source_hash='pending');self.spec=replace(self.spec,source_hash=_source_hash(self.spec))
 def test_dispatch_uses_pinned_native_session(self):
  with patch('subnet.native_common_dispatch.NativePrologSession')as factory:
   create_session(self.spec);factory.assert_called_once_with(self.spec)
 def test_wrong_environment_version_never_dispatches(self):
  with patch('subnet.native_common_dispatch.NativePrologSession')as factory,self.assertRaises(ValueError):create_session(replace(self.spec,version='prime-v1-1'))
  factory.assert_not_called()
 def test_unknown_marker_never_falls_back_to_prime(self):
  spec=replace(self.spec,config={**self.spec.config,'prolog_session_revision':'unknown'})
  with patch('subnet.environments.EnvironmentSession')as fallback,self.assertRaises(ValueError):create_session(spec)
  fallback.assert_not_called()
 def test_wrong_source_geometry_and_image_refused(self):
  for spec in [replace(self.spec,source_hash='0'*64),replace(self.spec,max_turns=1),replace(self.spec,success_reward=True),replace(self.spec,config={**self.spec.config,'prolog_runtime':{**self.spec.config['prolog_runtime'],'image':'latest'}})]:
   with self.assertRaises(ValueError):create_session(spec)
 def test_pin_paths_cannot_select_uploaded_code(self):
  spec=replace(self.spec,config={**self.spec.config,'prolog_source_files':{'../../untrusted.py':'0'*64}})
  with self.assertRaisesRegex(ValueError,'membership'):validate_prolog_binding(spec)
 def test_ordinary_prime_dispatch_remains_exact(self):
  plain=replace(self.spec,id='affine_verbatim',config={})
  with patch('subnet.environments.EnvironmentSession')as factory:create_session(plain)
  factory.assert_called_once_with(plain)
if __name__=='__main__':unittest.main()
