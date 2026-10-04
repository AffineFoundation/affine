"""Adversarial metadata/state publication controls; no live job or GPU effects."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch

from subnet.persistent_cpu_adamw import (
    HYPERPARAMETERS, PersistentCPUAdamW, genesis, parameter_inventory, sha,
)
from subnet.persistent_training_state import (
    HEADER_RESERVE, admit_resources, export_state, resource_plan, restore_state,
    validate_descriptor,
)


spec = importlib.util.spec_from_file_location(
    'persistent_security_storage_fixture',
    Path(__file__).with_name('test_persistent_training_policy.py'),
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class PersistentTrainingSecurityReviewTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.workspace = Path(self.folder.name)
        self.parameters = [('weight', torch.nn.Parameter(torch.full((12,), .02, dtype=torch.bfloat16)))]
        _, self.inventory = parameter_inventory(self.parameters)
        self.input = '11' * 32
        self.output = '22' * 32
        self.cap = HEADER_RESERVE + 32
        plan = resource_plan(self.inventory, bf16_export_bytes=100,
            transfer_bytes=self.cap, disk_reserve_bytes=0, ram_reserve_bytes=0)
        self.admission = admit_resources(self.workspace, plan)
        self.contract_before = copy.deepcopy(HYPERPARAMETERS)
        self.addCleanup(self.restore_contract)

    def restore_contract(self):
        HYPERPARAMETERS.clear()
        HYPERPARAMETERS.update(copy.deepcopy(self.contract_before))

    def optimizer(self):
        start = genesis(self.inventory, self.input)
        return PersistentCPUAdamW(self.parameters, self.input, approved_genesis=start,
            approved_genesis_sha256=sha(start), resource_admission=self.admission)

    def completed(self):
        optimizer = self.optimizer()
        self.parameters[0][1].grad = torch.ones_like(self.parameters[0][1])
        optimizer.step()
        return optimizer

    def publish(self, optimizer, store=None, commit=None):
        store = store or fixture.MemoryStorage()
        result = export_state(optimizer, epoch='security-control-epoch',
            inference_checkpoint=self.output, workspace=self.workspace,
            publish_shard=store.publish, readback_shard=store.readback,
            commit_descriptor=commit or store.commit, resource_admission=self.admission,
            shard_bytes=self.cap)
        return result, store

    def test_genesis_mutation_does_not_change_the_global_optimizer_contract(self):
        document = genesis(self.inventory, self.input)
        document['hyperparameters']['betas'][0] = .125
        self.assertEqual(HYPERPARAMETERS, self.contract_before)
        self.assertEqual(genesis(self.inventory, self.input)['hyperparameters'], self.contract_before)

    def test_exported_descriptor_mutation_does_not_change_global_contract(self):
        (descriptor, _), _ = self.publish(self.completed())
        descriptor['hyperparameters']['betas'][0] = .125
        self.assertEqual(HYPERPARAMETERS, self.contract_before)

    def test_descriptor_hash_blocks_parent_counter_model_and_policy_tampering(self):
        (original, _), _ = self.publish(self.completed())
        approved = sha(original)
        changes = {
            'input_checkpoint': '33' * 32,
            'inference_checkpoint': '44' * 32,
            'genesis_sha256': '55' * 32,
            'parent_state_sha256': '66' * 32,
            'optimizer_steps': original['optimizer_steps'] + 1,
            'epoch': 'substitute-epoch',
            'hyperparameters': {**self.contract_before, 'lr': .5},
        }
        for field, replacement in changes.items():
            changed = copy.deepcopy(original)
            changed[field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'descriptor digest'):
                validate_descriptor(changed, approved, self.output, self.inventory)

    def test_self_consistent_hash_does_not_allow_path_escape_or_bad_counters(self):
        (original, _), _ = self.publish(self.completed())
        for name in ('../outside.safetensors', 'state-../outside.safetensors',
                     'state-..\\outside.safetensors', '/state-escape.safetensors'):
            changed = copy.deepcopy(original)
            changed['shards'][0]['name'] = name
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'path'):
                validate_descriptor(changed, sha(changed), self.output, self.inventory)
        for value in (True, -1, original['optimizer_steps'] + 1):
            changed = copy.deepcopy(original)
            changed['parameter_steps']['weight'] = value
            with self.subTest(counter=value), self.assertRaisesRegex(ValueError, 'counter'):
                validate_descriptor(changed, sha(changed), self.output, self.inventory)

    def test_corrupted_shard_cannot_return_a_restored_optimizer(self):
        (descriptor, _), store = self.publish(self.completed())
        name = descriptor['shards'][-1]['name']
        data = bytearray(store.objects[name])
        data[-1] ^= 1
        store.objects[name] = bytes(data)
        with self.assertRaisesRegex(ValueError, 'digest/size'):
            restore_state(descriptor, sha(descriptor), self.output, self.inventory,
                workspace=self.workspace, fetch_shard=store.fetch,
                resource_admission=self.admission)

    def test_bad_final_acknowledgement_is_not_a_completed_publication(self):
        optimizer = self.completed()
        store = fixture.MemoryStorage()
        with self.assertRaisesRegex(ValueError, 'publication/readback'):
            self.publish(optimizer, store=store,
                commit=lambda descriptor: dict(descriptor_sha256='00' * 32,
                                               durable_readback_verified=True))
        self.assertEqual(store.committed, [])
        self.assertEqual(optimizer.global_step, 1)
        self.assertTrue(all(event[0] in ('publish', 'readback') for event in store.events))

    def test_same_bf16_model_can_have_distinct_persistent_successors(self):
        optimizer = self.completed()
        first_bf16 = self.parameters[0][1].detach().clone()
        (first, _), store = self.publish(optimizer)
        restored, _ = restore_state(first, sha(first), self.output, self.inventory,
            workspace=self.workspace, fetch_shard=store.fetch,
            resource_admission=self.admission)
        next_parameters = [('weight', torch.nn.Parameter(first_bf16.clone()))]
        next_optimizer = PersistentCPUAdamW(next_parameters, self.output, restored=restored)
        next_parameters[0][1].grad = torch.ones_like(next_parameters[0][1])
        next_optimizer.step()
        (second, _), _ = self.publish(next_optimizer)
        self.assertTrue(torch.equal(first_bf16, next_parameters[0][1].detach()))
        self.assertEqual(first['inference_checkpoint'], second['inference_checkpoint'])
        self.assertNotEqual(sha(first), sha(second))
        self.assertEqual(second['parent_state_sha256'], sha(first))
        self.assertEqual(second['optimizer_steps'], first['optimizer_steps'] + 1)
        # Checkpoint-only recovery would conflate these two valid states; the
        # signed original job must select the exact latest descriptor digest.


if __name__ == '__main__':
    unittest.main()
