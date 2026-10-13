import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_mimo_coordinator import CoordinatorIntegrationTests
from subnet.complete_group_transport import paths
from subnet.storage import canonical
from ops.mimo_host_capture_cache import CapturePass


class CacheTests(CoordinatorIntegrationTests):
    def setUp(self):
        super().setUp()
        self.cache = CapturePass(workers=4)
        self.listed = []
        self.lifecycle = SimpleNamespace(services=SimpleNamespace(committed_miners=lambda m:list(self.listed)))

    def capture(self, now=None):
        return self.cache(self.lifecycle, self.coordinator, self.opened,
                          self.bucket.clock+1 if now is None else now)

    def prepare(self):
        actor = self.actor(self.miners[0]);self.upload(actor,0)
        self.listed = [actor.identity.id]
        self.capture()
        return actor

    def test_unchanged_only_reads_commitment_and_keeps_original_signed_capture(self):
        actor=self.prepare();p=self.coordinator.root/'captured'/(actor.identity.id+'.json')
        original=p.read_bytes();self.bucket.reads.clear();self.bucket.puts.clear()
        with patch.object(self.coordinator,'collect',side_effect=AssertionError('no repeat work')):
            self.assertEqual(self.capture(),{})
        self.assertEqual(p.read_bytes(),original)
        self.assertEqual(self.bucket.reads,[paths(self.opened,actor.identity.id)])
        self.assertEqual(self.bucket.puts,[])
        self.assertEqual(self.cache.last_metrics['reused'],1)
        # Original seal still authenticates and grades the same durable input.
        self.assertEqual(self.seal()['selected_task_count'],1)

    def test_changed_cumulative_commitment_uses_original_collect(self):
        actor=self.prepare();self.upload(actor,1)
        with patch.object(self.coordinator,'collect',wraps=self.coordinator.collect) as call:
            self.capture();self.assertEqual(call.call_count,1)
        self.assertEqual(self.seal()['selected_task_count'],2)

    def test_restart_does_full_original_collect(self):
        self.prepare();self.cache=CapturePass()
        with patch.object(self.coordinator,'collect',wraps=self.coordinator.collect) as call:
            self.capture();self.assertEqual(call.call_count,1)

    def test_changed_local_record_cannot_supply_cache_credit(self):
        actor=self.prepare();p=self.coordinator.root/'captured'/(actor.identity.id+'.json')
        p.write_bytes(b'{}')
        with patch.object(self.coordinator,'collect',wraps=self.coordinator.collect) as call:
            self.capture();self.assertEqual(call.call_count,1)
        self.assertEqual(self.seal()['selected_task_count'],1)

    def test_late_same_bytes_reupload_is_rejected_and_old_capture_preserved(self):
        actor=self.prepare();key=paths(self.opened,actor.identity.id)
        data,_=self.bucket.objects[key]
        self.bucket.objects[key]=(data,self.opened['deadline']+1)
        result=self.capture(now=self.opened['deadline']+2)
        self.assertIn('outside signed epoch window',result[actor.identity.id])
        self.assertEqual(self.cache.last_metrics['reused'],0)

    def test_read_outage_does_not_finalize_or_penalize(self):
        self.prepare()
        with patch.object(self.bucket,'get_object',side_effect=OSError('outage')):
            with self.assertRaises(OSError):self.capture()
        self.assertFalse((self.coordinator.root/'POPULATION.json').exists())

    def test_malformed_commitment_uses_original_rejection(self):
        actor=self.prepare();key=paths(self.opened,actor.identity.id)
        self.bucket.store(key,b'{"invalid":true}')
        self.assertIn(actor.identity.id,self.capture())
        self.assertEqual(self.cache.last_metrics['reused'],0)

    def test_rejected_change_then_restore_requires_fresh_original_collect(self):
        actor=self.prepare();key=paths(self.opened,actor.identity.id)
        original=self.bucket.objects[key]
        self.bucket.store(key,b'{"invalid":true}')
        self.assertIn(actor.identity.id,self.capture())
        self.assertNotIn(actor.identity.id,self.cache.accepted)
        self.bucket.objects[key]=original
        with patch.object(self.coordinator,'collect',wraps=self.coordinator.collect) as call:
            self.capture();self.assertEqual(call.call_count,1)

    def test_cross_epoch_scope_clears_cache(self):
        self.prepare();other=copy.deepcopy(self.opened);other['epoch']+='-different'
        self.listed=[]
        self.cache(self.lifecycle,self.coordinator,other,self.bucket.clock+1)
        self.assertEqual(self.cache.accepted,{})

    def test_complete_listing_and_per_miner_rejections_preserved(self):
        actor=self.prepare();second=self.actor(self.miners[1]);self.upload(second,1)
        self.listed.append(second.identity.id)
        with patch.object(self.coordinator,'collect',wraps=self.coordinator.collect) as call:
            self.capture();self.assertEqual(call.call_count,1)
        self.assertEqual(self.seal()['selected_task_count'],2)

    def test_frozen_population_is_not_mutated_by_reuse(self):
        self.prepare();self.seal()
        with self.assertRaisesRegex(ValueError,'sealed'):self.capture()


if __name__=='__main__':unittest.main()
