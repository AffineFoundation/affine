import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from subnet.environments import EnvironmentSpec,build_spec,_taskset
from subnet.math_corpus_assets import admit_bytes
from test_math_corpus_assets import fixture

class Tests(unittest.TestCase):
 def test_actual_original_taskset_alias_and_private_origin_refusal(self):
  body,b,s=fixture();spec=EnvironmentSpec.from_dict(dict(id=s.id,version=s.version,adapter=s.adapter,config=s.config,num_samples=1,max_turns=1,max_output_tokens=256,success_reward=1.,source_hash='a'*64))
  taskset=_taskset(spec);self.assertEqual(taskset.task_type().__name__,'MathTask');self.assertEqual(type(taskset).__name__,'MathTaskset')
  for path in ('../operator/private.json','/tmp/unapproved'):
   bad=copy.deepcopy(spec.to_dict());bad['config']['task_snapshot']=path
   with self.assertRaises(ValueError):EnvironmentSpec.from_dict(bad)
 def test_asset_required_before_source_hash_and_runtime(self):
  body,b,s=fixture()
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'subnet').mkdir()
   import subnet.environments as env
   for name in ('math_corpus.py','math_corpus_provider.py','math_corpus_assets.py'):(root/'subnet'/name).write_bytes((env.PACKAGE_ROOT/name).read_bytes())
   with patch.object(env,'PACKAGE_ROOT',root/'subnet'):
    with self.assertRaises(ValueError):build_spec(s.id,config=s.config,num_samples=1,max_turns=1,max_output_tokens=256)
    path=root/b['path'];path.parent.mkdir(parents=True);path.write_bytes(admit_bytes(body,b));spec=build_spec(s.id,config=s.config,num_samples=1,max_turns=1,max_output_tokens=256);self.assertEqual(spec.version,s.version)
    bad=copy.deepcopy(spec.to_dict());bad['id']='math_corpus_deepmath103k_heldout_000'
    with self.assertRaises(ValueError):EnvironmentSpec.from_dict(bad)
 def test_unknown_alias_and_legacy_contract(self):
  with self.assertRaises(ValueError):EnvironmentSpec.from_dict(dict(id='math_corpus_other_train_000',source_hash='a'*64))
  spec=build_spec('affine_math',num_samples=1,max_turns=1,max_output_tokens=256);self.assertEqual(spec.version,'prime-v1-1');self.assertEqual(spec.id,'affine_math')
if __name__=='__main__':unittest.main()
