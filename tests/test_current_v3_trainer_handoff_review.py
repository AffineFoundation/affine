"""Synthetic operational handoff guards; no network, keys or live state writes."""
import copy
import json
from pathlib import Path
import runpy
import tempfile
import unittest

HELPER = Path(__file__).resolve().parents[1] / 'state/root-audits/prepare_current_v3_large_disk_trainer_handoff_20261004_v1.py'
MODULE = runpy.run_path(str(HELPER), run_name='trainer_handoff_review_definitions') if HELPER.is_file() else None


@unittest.skipIf(MODULE is None, 'operator-local handoff helper is absent')
class CurrentV3TrainerHandoffTests(unittest.TestCase):
    def fixtures(self):
        checkpoint = MODULE['CHECKPOINT']
        endpoint = dict(host='new-trainer', port=22, checkpoint_caches={checkpoint:'/new/current'}, code='/exact-source')
        config = dict(source_bundle={'sha256':'same'}, remote=dict(roles=dict(
            train=dict(host='old-trainer', port=22, checkpoint_caches={checkpoint:'/old/current'}, code='/exact-source'),
            mine=dict(host='miner', port=22), evaluate=dict(host='evaluator', port=22),
            verify=[dict(host='verifier-a', port=22),dict(host='verifier-b', port=22)])))
        caches = dict(train={checkpoint:'/old/current','historical':'/old/history'},
                      mine={'initial':'/mine/current'}, evaluate={checkpoint:'/eval/current'})
        return checkpoint, endpoint, config, caches

    def test_exact_two_scopes_only_preserve_other_roles_and_history(self):
        checkpoint, endpoint, config, caches = self.fixtures()
        originals = copy.deepcopy((config, caches, endpoint))
        new_config, new_caches, changes = MODULE['candidates'](config,caches,endpoint)
        self.assertTrue(all(path[:3]==('remote','roles','train') for path in changes))
        self.assertEqual(MODULE['changed_leaves'](caches,new_caches),[('train',checkpoint)])
        self.assertEqual(new_caches['train']['historical'],'/old/history')
        self.assertEqual(new_config['source_bundle'],config['source_bundle'])
        self.assertEqual((config,caches,endpoint),originals)
        new_config['remote']['roles']['train']['checkpoint_caches'][checkpoint]='mutated candidate'
        self.assertEqual((config,caches,endpoint),originals)

    def test_noncurrent_cache_or_shared_endpoint_refused(self):
        checkpoint, endpoint, config, caches = self.fixtures()
        for replacement in (dict(endpoint, checkpoint_caches={'other':'/cache'}),dict(endpoint, host='miner')):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                MODULE['candidates'](config,caches,replacement)

    def test_persisted_collect_with_partial_audits_permits_only_absent_train(self):
        with tempfile.TemporaryDirectory() as directory:
            state=Path(directory); (state/'roles').mkdir()
            status=dict(active=dict(epoch=MODULE['EPOCH'],phase='collect'), round=9,
                        training_steps=15,checkpoint=dict(id=MODULE['CHECKPOINT']))
            (state/(MODULE['EPOCH']+'-audit-challenge.json')).write_text('{"already_frozen":true}')
            guard=MODULE['training_guard'](state,status,authenticate=lambda envelope:envelope['payload'])
            self.assertEqual(guard['original_E9_signed_train_jobs'],0)
            for name in (state/'roles'/(MODULE['EPOCH']+'-train.json'),
                         state/(MODULE['EPOCH']+'-training-metrics.json'),
                         state/(MODULE['EPOCH']+'-checkpoint-publication.json')):
                name.write_text('{}')
                with self.assertRaisesRegex(ValueError,'already exists'):
                    MODULE['training_guard'](state,status,authenticate=lambda envelope:envelope['payload'])
                name.unlink()

    def test_hidden_signed_train_job_and_wrong_parent_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            state=Path(directory); (state/'roles').mkdir()
            status=dict(active=dict(epoch=MODULE['EPOCH'],phase='collect'), round=9,
                        training_steps=15,checkpoint=dict(id=MODULE['CHECKPOINT']))
            job=dict(role='train',manifest=dict(payload=dict(epoch=MODULE['EPOCH'])))
            (state/'roles'/'unlinked-job.json').write_text(json.dumps(dict(payload=job)))
            with self.assertRaisesRegex(ValueError,'signed E9 training job'):
                MODULE['training_guard'](state,status,authenticate=lambda envelope:envelope['payload'])
            (state/'roles'/'unlinked-job.json').unlink()
            for changed in (dict(status,training_steps=18),dict(status,round=10),
                            dict(status,checkpoint=dict(id='different')),
                            dict(status,active=dict(epoch=MODULE['EPOCH'],phase='after'))):
                with self.subTest(status=changed), self.assertRaises(ValueError):
                    MODULE['training_guard'](state,changed,authenticate=lambda envelope:envelope['payload'])


if __name__=='__main__':unittest.main()
