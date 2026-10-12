import json
from pathlib import Path
import tempfile
import unittest

import train_contract as c
from serial_worker import save
from test_training_eval import AUTH,fixture,sign
import training_exposure as exposure


class ExposureTests(unittest.TestCase):
    def test_upstream_reference_zero_current_run_updates_is_explicit(self):
        p,rows=fixture()
        result=exposure.collect({},dict(phase='baseline'),p['training_cohort']['payload'],sign=sign,save=save)
        self.assertEqual(result['summary']['untrained_tasks'],128)
        self.assertEqual(result['summary']['task_exposures'],0)
        self.assertIn('upstream pretraining unknown',result['assurance'])

    def test_missing_attribution_does_not_label_tasks_untrained(self):
        p,rows=fixture()
        result=exposure.collect({},dict(phase='latest'),p['training_cohort']['payload'],sign=sign,save=save)
        self.assertIsNone(result['summary']['untrained_tasks'])
        self.assertEqual(result['summary']['unknown_tasks'],128)
        self.assertIsNone(result['task_counts'])

    def test_authenticated_initial_checkpoint_exposure_and_future_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p,rows=fixture();cohort=p['training_cohort']['payload']
            provider=root/'provider.py';provider.write_text('# no historical replay needed at frozen checkpoint\n')
            counts={str(i):int(i<55)*2 for i in range(128)}
            cache=dict(training_run_id=p['training_run_id'],cohort_sha256=c.digest(cohort),last_round=157,last_step=67,
                       checkpoint=c.digest('closed67'),counts=counts,records=[])
            path=root/'initial.json';path.write_bytes(c.canonical(sign(cache)))
            config=dict(exposure_provider=dict(path=str(provider),sha256=c.file_hash(provider)),
                        production_state=str(root),local_root=str(root),authority=AUTH,training_run_id=p['training_run_id'],
                        initial_exposure_cache=dict(path=str(path),sha256=c.file_hash(path)))
            latest=dict(phase='latest',step=67,checkpoint=dict(id=cache['checkpoint']),closure=dict(round=157))
            result=exposure.collect(config,latest,cohort,sign=sign,save=save)
            self.assertEqual(result['summary']['trained_tasks'],55)
            self.assertEqual(result['summary']['task_exposures'],110)
            latest['step']=66
            result=exposure.collect(config,latest,cohort,sign=sign,save=save)
            self.assertEqual(result['summary']['unknown_tasks'],128)
            self.assertIsNone(result['summary']['untrained_tasks'])


if __name__=='__main__':unittest.main()
