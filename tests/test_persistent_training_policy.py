import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from subnet.persistent_cpu_adamw import (POLICY, HYPERPARAMETERS, PersistentCPUAdamW,
                                       genesis, parameter_inventory, sha)
from subnet.persistent_training_state import (HEADER_RESERVE, MAX_SHARD_BYTES,
    admit_resources, resource_plan, restore_state, export_state, validate_descriptor)
from subnet.task_normalized_training import task_groups, accumulate_tasks


class MemoryStorage:
    def __init__(self):
        self.objects = {}; self.events = []; self.committed = []

    def publish(self, name, path):
        self.events.append(('publish', name)); self.objects[name] = path.read_bytes()

    def readback(self, name):
        self.events.append(('readback', name)); value = self.objects[name]
        for start in range(0, len(value), 31): yield value[start:start+31]

    def fetch(self, name, path):
        # Streaming restore must have retired every preceding successful shard.
        self.events.append(('fetch', name))
        if list(path.parent.iterdir()): raise AssertionError('multiple local restore shards')
        path.write_bytes(self.objects[name])

    def commit(self, descriptor):
        self.events.append(('commit', 'descriptor'))
        self.committed.append(copy.deepcopy(descriptor))
        return dict(descriptor_sha256=sha(descriptor), durable_readback_verified=True,authority_committed=True)


class PersistentTrainingPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.parameters = [('weight', torch.nn.Parameter(torch.tensor([.02]*12, dtype=torch.bfloat16)))]
        _, self.inventory = parameter_inventory(self.parameters)
        self.input = '11'*32; self.successor = '22'*32
        self.cap = HEADER_RESERVE + 32
        plan = resource_plan(self.inventory, bf16_export_bytes=100,
            transfer_bytes=self.cap, disk_reserve_bytes=0, ram_reserve_bytes=0)
        self.admission = admit_resources(self.workspace, plan)

    def tearDown(self):
        self.temporary.cleanup()

    def optimizer(self, parameters=None, input_checkpoint=None, restored=None):
        parameters = self.parameters if parameters is None else parameters
        checkpoint = self.input if input_checkpoint is None else input_checkpoint
        _, inventory = parameter_inventory(parameters)
        if restored is not None:
            return PersistentCPUAdamW(parameters, checkpoint, restored=restored)
        approved = genesis(inventory, checkpoint)
        return PersistentCPUAdamW(parameters, checkpoint, approved_genesis=approved,
            approved_genesis_sha256=sha(approved), resource_admission=self.admission)

    def steps(self, optimizer, count):
        for _ in range(count):
            for _, parameter in optimizer.parameters: parameter.grad = torch.ones_like(parameter)
            optimizer.step()

    def publish(self, optimizer, store=None, checkpoint=None):
        store = store or MemoryStorage()
        descriptor, evidence = export_state(optimizer, epoch='test-epoch',
            inference_checkpoint=checkpoint or self.successor, workspace=self.workspace,
            publish_shard=store.publish, readback_shard=store.readback,
            commit_descriptor=store.commit, resource_admission=self.admission,
            shard_bytes=self.cap)
        return descriptor, evidence, store

    def restore(self, descriptor, store, checkpoint=None):
        return restore_state(descriptor, sha(descriptor), checkpoint or self.successor,
            self.inventory, workspace=self.workspace, fetch_shard=store.fetch,
            resource_admission=self.admission)

    def test_three_updates_preserve_small_master_changes_despite_bf16_stall(self):
        optimizer = self.optimizer(); original = self.parameters[0][1].detach().clone()
        master_before = optimizer.rows['weight']['master'].clone()
        self.steps(optimizer, 3)
        self.assertTrue(torch.equal(self.parameters[0][1].detach(), original))
        self.assertTrue(bool(torch.all(optimizer.rows['weight']['master'] < master_before)))
        self.assertEqual(optimizer.global_step, 3)
        self.assertEqual(optimizer.last_update['bf16_changed_elements'], 0)
        self.assertEqual(optimizer.last_update['master_changed_elements'], 12)
        self.assertEqual(optimizer.rows['weight']['exp_avg'].dtype, torch.float32)

    def test_cpu_adam_equations_match_torch_fp32_adamw(self):
        optimizer = self.optimizer()
        reference = torch.nn.Parameter(self.parameters[0][1].detach().float())
        expected = torch.optim.AdamW([reference], lr=1e-5, foreach=False,
            betas=tuple(HYPERPARAMETERS['betas']), eps=HYPERPARAMETERS['eps'], weight_decay=.01)
        for _ in range(7):
            self.parameters[0][1].grad = torch.full_like(self.parameters[0][1], .125)
            reference.grad = torch.full_like(reference, .125)
            optimizer.step(); expected.step()
        torch.testing.assert_close(optimizer.rows['weight']['master'], reference.detach(), rtol=0, atol=0)
        torch.testing.assert_close(optimizer.rows['weight']['exp_avg'], expected.state[reference]['exp_avg'], rtol=0, atol=0)
        torch.testing.assert_close(optimizer.rows['weight']['exp_avg_sq'], expected.state[reference]['exp_avg_sq'], rtol=0, atol=0)

    def test_streamed_round_trip_accepts_shuffled_shard_order(self):
        optimizer = self.optimizer(); self.steps(optimizer, 3)
        descriptor, evidence, store = self.publish(optimizer)
        self.assertGreater(len(descriptor['shards']), 1)
        self.assertEqual(store.events[-1], ('commit', 'descriptor'))
        self.assertTrue(evidence['descriptor_committed_last'])
        self.assertFalse(list(self.workspace.iterdir()))
        descriptor['shards'].reverse()
        restored, restored_evidence = self.restore(descriptor, store)
        clone = [('weight', torch.nn.Parameter(self.parameters[0][1].detach().clone()))]
        resumed = self.optimizer(clone, self.successor, restored)
        self.assertEqual(resumed.global_step, 3)
        for slot in ('master', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(resumed.rows['weight'][slot], optimizer.rows['weight'][slot], rtol=0, atol=0)
        self.assertTrue(all(r['verified_materialization'] for r in restored_evidence))
        self.assertFalse(list(self.workspace.iterdir()))

    def test_small_updates_survive_multiple_bf16_export_reload_epochs(self):
        optimizer = self.optimizer(); original = self.parameters[0][1].detach().clone()
        parents = []
        for epoch in range(4):
            self.steps(optimizer, 3)
            checkpoint = self.input if epoch == 0 else format(epoch, '064x')
            descriptor, _, store = self.publish(optimizer, checkpoint=checkpoint)
            parents.append(sha(descriptor))
            if epoch: self.assertEqual(descriptor['parent_state_sha256'], parents[-2])
            restored, _ = self.restore(descriptor, store, checkpoint)
            clone = [('weight', torch.nn.Parameter(optimizer.parameters[0][1].detach().clone()))]
            optimizer = self.optimizer(clone, checkpoint, restored)
        self.assertEqual(optimizer.global_step, 12)
        self.assertTrue(bool(torch.all(optimizer.parameters[0][1].detach() < original)))
        broken = torch.nn.Parameter(original.clone())
        for _ in range(4):
            low = torch.optim.AdamW([broken], lr=1e-5, foreach=False)
            for _ in range(3): broken.grad = torch.ones_like(broken); low.step()
        self.assertTrue(torch.equal(broken.detach(), original))

    def test_missing_state_never_silently_resets_and_genesis_is_bound(self):
        with self.assertRaises(ValueError): PersistentCPUAdamW(self.parameters, self.input)
        approved = genesis(self.inventory, self.input)
        with self.assertRaises(ValueError):
            PersistentCPUAdamW(self.parameters, self.successor, approved_genesis=approved,
                approved_genesis_sha256=sha(approved), resource_admission=self.admission)
        with self.assertRaises(ValueError):
            PersistentCPUAdamW(self.parameters, self.input, approved_genesis=approved,
                approved_genesis_sha256='aa'*32, resource_admission=self.admission)
        with self.assertRaises(ValueError):
            PersistentCPUAdamW(self.parameters, self.input, approved_genesis=approved,
                approved_genesis_sha256=sha(approved))

    def test_parent_descriptor_policy_name_shape_hash_and_counters_refused(self):
        optimizer = self.optimizer(); self.steps(optimizer, 1)
        descriptor, _, store = self.publish(optimizer)
        mutations = []
        for field, value in [('policy', 'other'), ('inference_checkpoint', self.input),
                             ('parent_state_sha256', 'bad'), ('optimizer_steps', True)]:
            bad = copy.deepcopy(descriptor); bad[field] = value; mutations.append(bad)
        bad = copy.deepcopy(descriptor); bad['hyperparameters']['max_grad_norm'] = True; mutations.append(bad)
        bad = copy.deepcopy(descriptor); bad['parameter_steps']['weight'] = True; mutations.append(bad)
        bad = copy.deepcopy(descriptor); bad['parameters'][0]['name'] = 'other'; mutations.append(bad)
        bad = copy.deepcopy(descriptor); bad['parameters'][0]['shape'] = [6, 2]; mutations.append(bad)
        bad = copy.deepcopy(descriptor); bad['shards'][0]['sha256'] = 'bb'*32; mutations.append(bad)
        bad = copy.deepcopy(descriptor); bad['shards'][0]['name'] = '../state.safetensors'; mutations.append(bad)
        bad = copy.deepcopy(descriptor); bad['shards'][0]['tensors'][0]['start'] = 1; mutations.append(bad)
        bad = copy.deepcopy(descriptor); bad['shards'].append(copy.deepcopy(bad['shards'][0])); mutations.append(bad)
        for bad in mutations:
            with self.subTest(change=bad), self.assertRaises(ValueError):
                # Hash changes do not waive semantic contracts. A forged object
                # digest is detected during restore if metadata is otherwise valid.
                self.restore(bad, store)
        with self.assertRaises(ValueError):
            validate_descriptor(descriptor, 'cc'*32, self.successor, self.inventory)

    def test_model_projection_and_nonfinite_gradient_refused(self):
        optimizer = self.optimizer(); self.steps(optimizer, 1)
        descriptor, _, store = self.publish(optimizer)
        restored, _ = self.restore(descriptor, store)
        other = [('weight', torch.nn.Parameter(torch.full((12,), .3, dtype=torch.bfloat16)))]
        with self.assertRaises(ValueError): self.optimizer(other, self.successor, restored)
        self.parameters[0][1].grad = torch.full_like(self.parameters[0][1], float('nan'))
        with self.assertRaises(ValueError): optimizer.step()
        self.parameters[0][1].grad = None
        with self.assertRaises(ValueError): optimizer.step()

    def test_failed_durable_readback_keeps_shard_and_never_commits_descriptor(self):
        optimizer = self.optimizer(); self.steps(optimizer, 1); store = MemoryStorage()
        with self.assertRaises(ValueError):
            export_state(optimizer, epoch='test', inference_checkpoint=self.successor,
                workspace=self.workspace, publish_shard=store.publish,
                readback_shard=lambda name: iter([b'corrupt']), commit_descriptor=store.commit,
                resource_admission=self.admission, shard_bytes=self.cap)
        self.assertFalse(store.committed)
        self.assertEqual(len(list(self.workspace.glob('*/state-*.safetensors'))), 1)

    def test_corrupt_restore_keeps_failed_transfer_and_returns_no_state(self):
        optimizer = self.optimizer(); self.steps(optimizer, 1)
        descriptor, _, store = self.publish(optimizer)
        store.objects[descriptor['shards'][0]['name']] = b'corrupt'
        with self.assertRaises(ValueError): self.restore(descriptor, store)
        self.assertEqual(len(list(self.workspace.glob('*/state-*.safetensors'))), 1)

    def test_actual_tensor_dtype_and_shape_are_checked_after_hash_verification(self):
        from safetensors.torch import load, save
        optimizer = self.optimizer(); self.steps(optimizer, 1)
        descriptor, _, store = self.publish(optimizer)
        for mutation in ('dtype', 'shape'):
            bad = copy.deepcopy(descriptor); changed_store = MemoryStorage()
            changed_store.objects = dict(store.objects)
            first = bad['shards'][0]; tensors = load(changed_store.objects[first['name']])
            key = first['tensors'][0]['key']
            tensors[key] = tensors[key].double() if mutation == 'dtype' else tensors[key][:-1]
            payload = save(tensors); changed_store.objects[first['name']] = payload
            first['size'] = len(payload); first['sha256'] = hashlib.sha256(payload).hexdigest()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.restore(bad, changed_store)

    def test_second_moment_cannot_be_negative_even_with_matching_object_hash(self):
        from safetensors.torch import load, save
        optimizer = self.optimizer(); self.steps(optimizer, 1)
        descriptor, _, store = self.publish(optimizer)
        for shard in descriptor['shards']:
            metadata = next((row for row in shard['tensors'] if row['slot'] == 'exp_avg_sq'), None)
            if metadata is None: continue
            tensors = load(store.objects[shard['name']]); tensors[metadata['key']].fill_(-1)
            payload = save(tensors); store.objects[shard['name']] = payload
            shard['size'] = len(payload); shard['sha256'] = hashlib.sha256(payload).hexdigest()
            break
        with self.assertRaises(ValueError): self.restore(descriptor, store)

    def test_state_publication_seals_optimizer_mutations(self):
        optimizer = self.optimizer(); self.steps(optimizer, 1); store = MemoryStorage()
        def invalid_callback(name, path):
            store.publish(name, path)
            optimizer.step()
        with self.assertRaises(ValueError):
            export_state(optimizer, epoch='test', inference_checkpoint=self.successor,
                workspace=self.workspace, publish_shard=invalid_callback,
                readback_shard=store.readback, commit_descriptor=store.commit,
                resource_admission=self.admission, shard_bytes=self.cap)
        self.assertEqual(optimizer.global_step, 1)
        self.assertFalse(store.committed)
        self.assertFalse(optimizer._publishing)

    def test_budget_requires_ram_and_only_bounded_disk_state_transfer(self):
        inventory = [dict(name='large', shape=[7_615_616_512], numel=7_615_616_512)]
        plan = resource_plan(inventory, bf16_export_bytes=15_242_726_234)
        self.assertEqual(plan['cpu_state_bytes'], 91_387_398_144)
        self.assertLess(plan['additional_disk_required_bytes'], 30_000_000_000)
        self.assertFalse(plan['full_state_disk_hydration'])
        with self.assertRaises(ValueError):
            admit_resources(self.workspace, plan, ram_available=1, disk_available=200_000_000_000)
        with self.assertRaises(ValueError):
            admit_resources(self.workspace, plan, ram_available=500_000_000_000, disk_available=1)
        with self.assertRaises(ValueError): resource_plan(inventory, bf16_export_bytes=1, transfer_bytes=MAX_SHARD_BYTES+1)


def pair(index, chosen, rejected, task_hash=None):
    task_hash = task_hash or format(index+1, '064x')
    return (dict(env_id='math'),
        dict(env_id='math', index=index, task_hash=task_hash, classification='positive', turns=[dict(output=[chosen])]),
        dict(env_id='math', index=index, task_hash=task_hash, classification='negative', turns=[dict(output=[rejected])]))


class TaskNormalizationTests(unittest.TestCase):
    def test_more_pairs_for_one_task_cannot_multiply_its_gradient_weight(self):
        population = [pair(0, 1, 2), pair(0, 3, 4), pair(0, 5, 6), pair(1, 7, 8)]
        pairs, tasks, groups, _ = task_groups(population, 1, 'ab'*32)
        values = torch.nn.Parameter(torch.zeros(2))
        references = [0.]*len(pairs)
        observations = accumulate_tasks(torch, lambda i: values[pairs[i][1]['index']],
            references, tasks, groups[0])
        torch.testing.assert_close(values.grad, torch.tensor([-.025, -.025]))
        for task in tasks:
            self.assertAlmostEqual(sum(r['gradient_weight'] for r in observations
                if r['task_sha256'] == task['task_sha256']), .5)

    def test_exact_pair_clones_and_input_order_do_not_change_task_schedule(self):
        population = [pair(0, 1, 2), pair(0, 3, 4), pair(1, 5, 6), pair(2, 7, 8)]
        a = task_groups(population, 2, 'ab'*32)
        b = task_groups(population[::-1] + [copy.deepcopy(population[0])], 2, 'ab'*32)
        self.assertEqual(len(a[0]), len(b[0]))
        self.assertEqual([[a[1][i]['task_sha256'] for i in group] for group in a[2]],
                         [[b[1][i]['task_sha256'] for i in group] for group in b[2]])

    def test_task_hash_binding_and_nonfinite_objective_refused(self):
        with self.assertRaises(ValueError):
            task_groups([pair(0, 1, 2), pair(0, 3, 4, task_hash='ab'*32)], 1, 'ab'*32)
        bad = pair(0, 1, 2); bad[2]['task_hash'] = 'cd'*32
        with self.assertRaises(ValueError): task_groups([bad], 1, 'ab'*32)
        pairs, tasks, groups, _ = task_groups([pair(0, 1, 2)], 1, 'ab'*32)
        value = torch.nn.Parameter(torch.tensor(float('nan')))
        with self.assertRaises(ValueError):
            accumulate_tasks(torch, lambda i: value, [0.], tasks, groups[0])


if __name__ == '__main__':
    unittest.main()
