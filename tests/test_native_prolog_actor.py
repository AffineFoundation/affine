import unittest,hashlib,subprocess
from unittest.mock import patch
from subnet.native_prolog_actor import isolation_command,BASE,REVISION,nqueens_command,PublicActor,SHIM_SHA
class Tests(unittest.TestCase):
 def runtime(self):return {'revision':REVISION,'base_image':BASE,'image':'sha256:'+'a'*64,'shim_sha256':SHIM_SHA}
 def test_strict_container_boundary(self):
  args=isolation_command('affine-prolog-native-'+'a'*16,self.runtime())
  for v in ('--read-only','--cap-drop','ALL','--network','none','65534:65534','--pids-limit','64'):self.assertIn(v,args)
  self.assertNotIn('--privileged',args);self.assertNotIn('-v',args)
 def test_public_policy_preserves_problem_facts(self):
  public={'kind':'nqueens','starter_file':'board_size(11).\nsolve(_) :- fail.  %% TODO: replace this\n'}
  for negative in (False,True):self.assertIn('board_size(11).',nqueens_command(public,negative))
  self.assertIn('all_distinct',nqueens_command(public));self.assertIn('maplist(=(1)',nqueens_command(public,True))
 def test_private_descriptor_fields_rejected(self):
  public={'revision':REVISION,'task_name':'nqueens','original_index':5,'kind':'nqueens','messages':[],'starter_file':'board_size(11).','starter_sha256':hashlib.sha256(b'board_size(11).').hexdigest(),'source_files':{},'tools':[],'expected_answer':'private'}
  with self.assertRaises(ValueError):PublicActor(self.runtime(),public)
 def test_unqualified_runtime_or_unknown_kind_rejected(self):
  with self.assertRaises(ValueError):isolation_command('unowned',self.runtime())
  with self.assertRaises(ValueError):nqueens_command({'kind':'sudoku'})
 def test_failed_cleanup_keeps_actor_live(self):
  actor=object.__new__(PublicActor);actor.started=True;actor.name='affine-prolog-native-'+'a'*16
  with patch('subnet.native_prolog_actor.subprocess.run',return_value=subprocess.CompletedProcess([],1,b'',b'daemon failure')):
   with self.assertRaises(RuntimeError):actor.close()
  self.assertTrue(actor.started)
 def test_confirmed_removal_closes_actor(self):
  actor=object.__new__(PublicActor);actor.started=True;actor.name='affine-prolog-native-'+'a'*16
  with patch('subnet.native_prolog_actor.subprocess.run',return_value=subprocess.CompletedProcess([],0,b'',b'')) as run:
   actor.close();actor.close()
  self.assertFalse(actor.started);self.assertEqual(run.call_count,1)
if __name__=='__main__':unittest.main()
