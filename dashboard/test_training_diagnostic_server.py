import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dashboard.server import Database
from dashboard.training_diagnostic_projection import project
from dashboard.test_training_diagnostic_projection import TrainingProjectionTests


class TrainingServerTests(unittest.TestCase):
    def test_empty_series_is_empty_and_invalid_next_snapshot_drops_stale_rows(self):
        fixture = TrainingProjectionTests(); fixture.setUp()
        row = project(fixture.signed(), fixture.authority)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'state'; source.mkdir()
            db = Database(Path(tmp)/'db.sqlite', source)
            db.refresh(); self.assertEqual(db.snapshot()['training_evaluations'], [])
            with patch('dashboard.training_diagnostic_projection.rows', return_value=[row]): db.refresh()
            self.assertEqual(db.snapshot()['training_evaluations'], [row])
            db.refresh(); self.assertEqual(db.snapshot()['training_evaluations'], [])

    def test_current_run_scope_and_heldout_storage_are_independent(self):
        fixture = TrainingProjectionTests(); fixture.setUp()
        row = project(fixture.signed(), fixture.authority)
        epoch = dict(id=row['epoch_id'], source='live-reward-math', start=100.,
            accepted=1, rejected=0, unchecked=1, training=None)
        heldout = dict(run_id='original-heldout', epoch_id=epoch['id'], timestamp=900.)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'state'; source.mkdir(); db = Database(Path(tmp)/'db.sqlite', source)
            db.refresh()
            with db.connect() as conn:
                conn.execute('INSERT INTO epochs VALUES(?,?)',(epoch['id'],json.dumps(epoch)))
                conn.execute('INSERT INTO evaluations VALUES(?,?)',(heldout['run_id'],json.dumps(heldout)))
                conn.execute('INSERT INTO training_evaluations VALUES(?,?)',(row['run_id'],json.dumps(row)))
            boundary = dict(run_id=row['training_run_id'])
            with patch('dashboard.run_projection.read_boundary',return_value=boundary), \
                 patch('dashboard.run_projection.read_retirements',return_value=set()), \
                 patch('dashboard.run_projection.project',side_effect=lambda e,v,*_: (e,v)):
                snapshot = db.snapshot(current_only=True)
                self.assertEqual(snapshot['evaluations'], [heldout])
                self.assertEqual(snapshot['training_evaluations'], [row])
                boundary['run_id'] = 'other-training-run'
                self.assertEqual(db.snapshot(current_only=True)['training_evaluations'], [])


if __name__ == '__main__': unittest.main()
