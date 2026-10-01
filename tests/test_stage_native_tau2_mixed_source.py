"""Pure staging controls; not native/model qualification."""
import copy,pathlib,tempfile,unittest
from unittest.mock import patch
from ops.stage_native_tau2_mixed_source import bind_roles,source_files,USER_CHECKPOINT,AGENT_SEED_POLICY,SEED_POLICY
class TestMixedSourceStage(unittest.TestCase):
    def fixtures(self):
        policy={'source_files':{'subnet/policy.py':'a'*64}}
        base={k:{'candidate_policy':copy.deepcopy(policy),'seed_start':0,'generation_policy':{'temperature':.7,'top_p':1.},'request_model':'role-'+k} for k in ('agent','user')}
        agent={'checkpoint':{'id':'b'*64},'training_eligible':True};user={'checkpoint':{'id':USER_CHECKPOINT},'training_eligible':False}
        return agent,user,base,{'subnet/policy.py':'a'*64},{'interpreter_sha256':'c'*64,'packages':{'torch':'test'}}
    def test_roles_keep_policy_weights_and_independent_profile_seed(self):
        values=self.fixtures();roles=bind_roles(*values)
        self.assertEqual(roles['user']['checkpoint']['id'],USER_CHECKPOINT)
        self.assertFalse(roles['user']['training_eligible']);self.assertEqual(roles['user']['seed_policy'],SEED_POLICY)
        self.assertEqual(roles['agent']['seed_policy'],AGENT_SEED_POLICY)
        self.assertNotEqual(roles['agent']['model_runtime_revision'],roles['user']['model_runtime_revision'])
        self.assertEqual(roles['user']['candidate_policy'],values[2]['user']['candidate_policy'])
        self.assertNotIn('source_files',values[0])
    def test_policy_change_and_auxiliary_weight_mask_rejected(self):
        a,u,b,s,e=self.fixtures();s['subnet/policy.py']='d'*64
        with self.assertRaises(ValueError):bind_roles(a,u,b,s,e)
        a,u,b,s,e=self.fixtures();u['training_eligible']=True
        with self.assertRaises(ValueError):bind_roles(a,u,b,s,e)
        a,u,b,s,e=self.fixtures();u['checkpoint']['id']='e'*64
        with self.assertRaises(ValueError):bind_roles(a,u,b,s,e)
    def test_source_inventory_regular_bytes_and_symlink_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);(root/'subnet').mkdir();file=root/'subnet/a.py';file.write_text('x=1')
            with patch('ops.stage_native_tau2_mixed_source.FILES',('subnet/a.py',)):
                self.assertEqual(list(source_files(root)),['subnet/a.py'])
                file.unlink();file.symlink_to('/etc/hosts')
                with self.assertRaises(ValueError):source_files(root)
