import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from subnet import cli
from subnet.gpu_service import owned_dispatch_allowed

class MinerTaskSelection(unittest.TestCase):
    rows=[dict(env_id='affine_math',indices=[0,2,4]),dict(env_id='other',indices=[7])]

    def test_preferences_preserve_public_pool(self):
        original=copy.deepcopy(self.rows)
        with patch.object(cli,'entries',return_value=self.rows):
            self.assertEqual(cli.selected_tasks({}),[('affine_math',0),('affine_math',2),('affine_math',4),('other',7)])
            self.assertEqual(cli.selected_tasks({},'affine_math'),[('affine_math',0),('affine_math',2),('affine_math',4)])
            self.assertEqual(cli.selected_tasks({},'affine_math',[4,0]),[('affine_math',4),('affine_math',0)])
        self.assertEqual(original,self.rows)

    def test_unknown_heldout_duplicate_and_malformed_indices_refuse(self):
        for env,indices in [('absent',None),(None,[0]),('affine_math',[1]),('affine_math',[0,0]),('affine_math',[False]),('affine_math',[]),('affine_math',[7]),('affine_math',[-1])]:
            with self.subTest(env=env,indices=indices),patch.object(cli,'entries',return_value=self.rows):
                with self.assertRaises(ValueError):cli.selected_tasks({},env,indices)

    def arguments(self,state,indices,budget=7):
        return SimpleNamespace(search_budget=budget,env_id='affine_math',indices=indices,cap_file=None,key='owned',state=state,
            manifest_url='https://manifest.invalid',current_url=None,gateway='https://unused.invalid',authority='trusted',max_batches=1,once=True)

    def test_heldout_refuses_before_checkpoint_download_or_model(self):
        with tempfile.TemporaryDirectory() as temp:
            args=self.arguments(temp,[1]);manifest=dict(epoch='test',checkpoint={'id':'approved'},capabilities={'owned':{}},deadline=90)
            with patch.object(cli,'identity',return_value=SimpleNamespace(id='owned')),patch.object(cli,'fetch_signed',return_value=manifest),patch.object(cli,'entries',return_value=self.rows),patch.object(cli,'checkpoint_download') as download,patch.object(cli,'Miner') as miner:
                with self.assertRaisesRegex(ValueError,'authorized'):cli.run(args)
            download.assert_not_called();miner.assert_not_called()

    def test_authorized_selection_reaches_normal_search_and_upload(self):
        with tempfile.TemporaryDirectory() as temp:
            args=self.arguments(temp,[2]);manifest=dict(epoch='test',checkpoint={'id':'approved'},capabilities={'owned':{}},deadline=90,max_batches=3)
            miner=SimpleNamespace(batches=[],search=Mock(),upload=Mock())
            with patch.object(cli,'identity',return_value=SimpleNamespace(id='owned')),patch.object(cli,'fetch_signed',return_value=manifest),patch.object(cli,'entries',return_value=self.rows),patch.object(cli,'checkpoint_download',return_value=Path(temp)),patch.object(cli,'Miner',return_value=miner),patch.object(cli,'check_runtime_profile'),patch.object(cli.time,'time',return_value=10),patch.object(cli.time,'time_ns',return_value=123):cli.run(args)
            miner.search.assert_called_once_with(2,seed=123,max_attempts=7,env_id='affine_math');miner.upload.assert_called_once()

    def test_unbounded_attempt_budget_refuses_before_identity(self):
        for budget in [0,129,True,7.0]:
            with self.subTest(budget=budget),patch.object(cli,'identity') as identity:
                with self.assertRaises(ValueError):cli.run(SimpleNamespace(search_budget=budget))
                identity.assert_not_called()

class OwnedDispatch(unittest.TestCase):
    def test_external_only_trial_does_not_mutate_admission_or_mark_empty(self):
        manifest=dict(epoch='nonpayable-external',payable=False);original=copy.deepcopy(manifest)
        self.assertTrue(owned_dispatch_allowed({},manifest));self.assertFalse(owned_dispatch_allowed({'owned_miner_dispatch':False},manifest));self.assertEqual(manifest,original)
        from subnet.empty_epoch_policy import validate_empty_completion
        validate_empty_completion(manifest,{'points':{'miner':1},'weights':{'miner':1}},{'miner':{'accepted':[{}]}})

    def test_strict_dispatch_flag_and_signed_empty_rules(self):
        for value in [0,1,'false',None]:
            with self.subTest(value=value),self.assertRaises(ValueError):owned_dispatch_allowed({'owned_miner_dispatch':value},{})
        with self.assertRaises(ValueError):owned_dispatch_allowed({'owned_miner_dispatch':False},{'operator_test_policy':{}})

if __name__=='__main__':unittest.main()
