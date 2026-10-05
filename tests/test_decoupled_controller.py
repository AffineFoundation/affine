"""Controller integration: absent audit/score files cannot block learning."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
from subnet.gpu_service import run

class DecoupledController(unittest.TestCase):
 def test_collect_train_publish_advance_without_any_audit_receipts(self):
  with tempfile.TemporaryDirectory() as folder:
   state=Path(folder);epoch='nonpayable-decoupled-test';old={'id':'old'};new={'id':'new'}
   manifest=dict(epoch=epoch,checkpoint=old,deadline=0,training_input_policy='committed-unaudited-training-v1')
   training=dict(manifest,training_coverage={'assurance':'unaudited'})
   inputs=[{'sha256':'a'*64}];population=[{'miner':'m','assurance':'unaudited'}]
   status=dict(active=dict(epoch=epoch,phase='collect',identities={},started_at=0),round=1,training_steps=3,checkpoint=old,checkpoint_path='old-path',initial_published=True)
   for name,value in [('controller.json',status),(epoch+'-manifest.json',manifest)]:
    (state/name).write_text(json.dumps(value))
   controller=SimpleNamespace(collect_learner_inputs=Mock(return_value=(training,inputs,population)),finalize=Mock(side_effect=AssertionError('audit barrier')),train=Mock(return_value=(new,dict(checkpoint_path='new-path',steps=1))),signed=lambda v:v)
   config=dict(state=folder,bucket={},remote={},source_bundle={},epoch_prefix='nonpayable-decoupled',registration_allowlist=[])
   with patch('subnet.gpu_service.Bucket') as bucket,patch('subnet.gpu_service.Gateway'),patch('subnet.gpu_service.RemoteController',return_value=controller),patch('subnet.gpu_service.ChainAdapter'),patch('subnet.gpu_service.evaluate'),patch('subnet.gpu_service.publish_history',side_effect=AssertionError('audit history barrier')),patch('subnet.reward_publication.emit',side_effect=AssertionError('reward barrier')):
    run(config,once=True)
   controller.finalize.assert_not_called();controller.train.assert_called_once_with(training,{},'old-path',steps=1,replay=None)
   final=json.loads((state/'controller.json').read_text());self.assertIsNone(final['active']);self.assertEqual(final['checkpoint'],new);self.assertEqual(final['training_steps'],4)
   self.assertFalse((state/(epoch+'-verified.json')).exists());self.assertFalse((state/(epoch+'-scores.json')).exists())
   saved=json.loads((state/(epoch+'-learner-population.json')).read_text());self.assertEqual(saved['submissions'],inputs)

if __name__=='__main__':unittest.main()
