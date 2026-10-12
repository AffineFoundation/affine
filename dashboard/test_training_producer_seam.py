"""Synthetic output of the separate TRAIN worker; no model or ROOT signature."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dashboard.training_diagnostic_projection import project
from dashboard.server import Database


class ProducerSeamTests(unittest.TestCase):
    def test_actual_producer_aggregate_to_public_database(self):
        fixture = Path(__file__).parent/'test_fixtures/fresh-training-aggregate.synthetic.json'
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),
                         'b3191bcbc3452b8dbb9836887bab2724b7f357e78af44664d2ed34584fd514dc')
        envelope = json.loads(fixture.read_bytes())
        row = project(envelope, '03a107bff3ce10be1d70dd18e74bc09967e4d6309ba50d5f1ddc8664125531b8')
        self.assertEqual(row['count'], 128)
        self.assertEqual(row['evaluation_split'], 'train')
        self.assertEqual(sum(v['count'] for v in row['exposure_groups'].values()), 128)
        self.assertEqual(row['mean_reward'], envelope['payload']['solved']/128)
        self.assertEqual(row['fixed_initial_trained'], envelope['payload']['fixed_initial_trained'])
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'state';source.mkdir()
            db=Database(Path(tmp)/'db.sqlite',source)
            with patch('dashboard.training_diagnostic_projection.rows',return_value=[row]):db.refresh()
            self.assertEqual(db.snapshot()['training_evaluations'],[row])
        self.assertNotIn('raw_turns',json.dumps(row))
        self.assertNotIn('durable_result',row)

if __name__=='__main__':unittest.main()
