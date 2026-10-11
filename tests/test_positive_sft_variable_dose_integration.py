"""CPU binding/routing tests with ephemeral authorities; no GPU qualification."""
import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO/'tests')]
from test_positive_sft_execution_route import PositiveSFTRouteTests
from test_positive_only_training import pairs_for_tasks, EOS, SEED
from subnet import unaudited_training_execution as execution
from subnet.training_receipts import sha
from ops import positive_sft_variable_dose as f
from ops import positive_sft_variable_dose_lifecycle as l

CAP = dict(target_tasks_per_step=64, max_tasks=512, max_steps=8)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fx = PositiveSFTRouteTests(); self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        self.sign, self.authority = self.fx.sign, self.fx.authority
        r = copy.deepcopy(self.fx.release['payload']); r.pop('genesis_document')
        r.update(version=execution.SFT_RELEASE, method=execution.SFT_METHOD,
                 minimum_optimizer_step=8)
        self.releases = [self.sign(dict(r, steps=k)) for k in range(1, 9)]
        self.bank = self.sign(dict(version=l.BANK_VERSION, capacity=CAP, releases=self.releases))
        self.now = r['created_at'] + 1

    def test_bank_authenticates_all_exact_releases(self):
        value, common = l.bank(self.bank, self.authority, now=self.now)
        self.assertEqual([v['payload']['steps'] for v in value['releases']], list(range(1, 9)))
        self.assertNotIn('steps', common)
        for field, changed in [('releases', self.releases[:-1]), ('capacity', dict(CAP, max_steps=7))]:
            bad = copy.deepcopy(self.bank['payload']); bad[field] = changed
            with self.assertRaises(ValueError): l.bank(self.sign(bad), self.authority, now=self.now)

    def test_bank_rejects_mixed_lineage_runtime_and_expiry(self):
        for field, val in [('minimum_optimizer_step', 9), ('epoch_prefix', 'other-'),
                           ('expires_at', self.bank['payload']['releases'][0]['payload']['expires_at']-1)]:
            bad = copy.deepcopy(self.bank['payload'])
            r = bad['releases'][3]['payload']; r[field] = val
            bad['releases'][3] = self.sign(r)
            with self.assertRaises(ValueError): l.bank(self.sign(bad), self.authority, now=self.now)
        bad = copy.deepcopy(self.bank['payload']); bad['releases'][0] = self.fx.release
        with self.assertRaises(ValueError): l.bank(self.sign(bad), self.authority, now=self.now)

    def test_step_count_is_bound_in_actual_job_lr_and_parent(self):
        for k in range(1, 9):
            job = self.fx.job(100+k, 8); job['steps'] = k
            amended = self.fx.amend(job, self.releases[k-1])
            value = execution.validate(self.sign(amended), self.authority)
            self.assertEqual(value['steps'], k)
            self.assertEqual(value['learning_rate_authorization']['payload']['steps'], k)
            self.assertEqual(value['optimizer_step_before'], 8)
            self.assertEqual(value['execution_release_sha256'], sha(self.releases[k-1]))
            bad = copy.deepcopy(job); bad['steps'] = k+1
            with self.assertRaises(ValueError):
                execution.derive_preparation(self.releases[k-1], bad,
                    json.loads((self.fx.fx.state/(job['manifest']['payload']['epoch']+'-first-signed-manifest.json')).read_bytes()),
                    {name: json.loads((self.fx.fx.state/'native-outcome-eligibility'/job['manifest']['payload']['epoch']/(name+'.ROOT-SIGNED.json')).read_bytes())
                     for name in ('context','grades','subset')}, self.authority, created_at=job['created_at'])

    def test_expired_release_can_finish_original_dispatch_but_not_issue_new_job(self):
        job = self.fx.job(115, 8); job['steps'] = 2
        amended = self.fx.amend(job, self.releases[1])
        c = self.fx.fx.controller; c.signed = self.sign
        roles = c.state/'roles'; roles.mkdir(exist_ok=True)
        (roles/(job['job_id']+'-job.json')).write_text(json.dumps(self.sign(amended)))
        record = dict(job_id=job['job_id'], job_sha256=sha(amended))
        record_path = roles/(job['manifest']['payload']['epoch']+'-train.json')
        record_path.write_text(json.dumps(record))
        actual = l.original_dispatch(c, job['manifest']['payload'], job['submissions'])
        self.assertEqual(actual, dict(created_at=job['created_at'], steps=2,
                                    release_sha256=sha(self.releases[1])))
        expired = self.releases[1]['payload']['expires_at']+1
        with self.assertRaises(ValueError): l.bank(self.bank, self.authority, now=expired)
        l.bank(self.bank, self.authority, now=actual['created_at'])
        # Merely prepared or unissued requests never receive this historical path.
        record_path.write_text(json.dumps(dict(record, prepared_only=True)))
        self.assertIsNone(l.original_dispatch(c, job['manifest']['payload'], job['submissions']))
        record_path.unlink()
        self.assertIsNone(l.original_dispatch(c, job['manifest']['payload'], job['submissions']))
        record_path.write_text(json.dumps(dict(record, job_sha256='0'*64)))
        with self.assertRaises(ValueError): l.original_dispatch(c, job['manifest']['payload'], job['submissions'])

    def test_actual_population_selects_small_steps_without_repetition(self):
        for n in (1, 63, 64, 65, 128, 129, 511, 512):
            rows = pairs_for_tasks(n)
            selected = f.selection(rows, seed=SEED, eos_token_id=EOS,
                                    required_pairs_per_task=4, capacity=CAP)
            self.assertEqual(selected['steps'], (n+63)//64)
            self.assertEqual([i for group in selected['groups'] for i in group], list(range(n)))
            self.assertEqual(len(selected['selection']['retained']), 4*n)
            self.assertLessEqual(max(map(len, selected['groups'])), 64)

    def test_snapshot_uses_full_input_triples_and_failure_does_not_become_empty(self):
        job = self.fx.job(110, 8); manifest = job['manifest']['payload']
        rows = pairs_for_tasks(1)
        controller = self.fx.fx.controller; controller.signed = self.sign
        source = job['submissions']
        public = json.loads((self.fx.fx.state/(manifest['epoch']+'-first-signed-manifest.json')).read_bytes())
        docs = {name: json.loads((self.fx.fx.state/'native-outcome-eligibility'/manifest['epoch']/(name+'.ROOT-SIGNED.json')).read_bytes())
                for name in ('context','grades','subset')}
        with patch.object(l, 'admitted_submission', return_value=({}, rows)) as read:
            pairs, signed_manifest, signed_snapshot, native = l.pair_snapshot(controller,
                manifest, source, public, docs, [Path('fixture') for _ in source])
        self.assertEqual(read.call_count, len(source))
        self.assertTrue(all(c.kwargs['retire'] is False for c in read.call_args_list))
        self.assertEqual(signed_snapshot['payload']['admitted_pairs_sha256'], sha(pairs))
        changed = copy.deepcopy(pairs); changed[0][0]['changed_metadata'] = True
        self.assertNotEqual(sha(changed), signed_snapshot['payload']['admitted_pairs_sha256'])
        self.assertFalse(signed_snapshot['payload']['proof_verification_performed'])
        with patch.object(l, 'admitted_submission', side_effect=FileNotFoundError('missing native copy')):
            with self.assertRaises(FileNotFoundError):
                l.pair_snapshot(controller, manifest, source, public, docs, [Path('fixture') for _ in source])
        with self.assertRaises(ValueError): l.pair_snapshot(controller, manifest, source, public, docs, [])

    def test_no_genesis_bank_even_when_individual_release_is_genesis_qualified(self):
        job = self.fx.job(111, 0)
        with patch.object(l, 'admitted_submission', return_value=({}, pairs_for_tasks(1))):
            with self.assertRaisesRegex(ValueError, 'retained'):
                l.pair_snapshot(self.fx.fx.controller, job['manifest']['payload'], job['submissions'], {}, {}, [])

    def test_installed_dispatch_uses_exact_count_and_original_completion_abi(self):
        controller = self.fx.fx.controller
        controller.signed = self.sign
        common = self.bank['payload']['releases'][0]['payload']
        controller.jobs = SimpleNamespace(config={}, metadata=dict(
            source_files=common['execution_source_files'], runtime_versions=common['runtime_versions']))
        epoch = 'fixture-146'
        parent_status = {'active': {'epoch': epoch, 'phase': 'train'}}
        (controller.state/'controller.json').write_text(json.dumps(parent_status))
        with patch.object(l.time, 'time', return_value=self.now):
            l.install(controller, self.bank)
        selected = {'payload': {'steps': 2}, 'selected_release': self.releases[1]}
        with patch.object(l, 'prepare_route', return_value=selected):
            actual = controller.positive_sft_dispatch_route({}, [], 8)
        self.assertEqual(actual, {'steps': 2, 'no_update': None})
        self.assertEqual(controller.jobs.config['unaudited_training_execution_release'], self.releases[1])
        selected = dict(payload={'steps': 0}, pairs=[], manifest={}, admission={}, scope={})
        no_update = ({'id': 'unchanged'}, {'steps': 0, 'native_no_update': {'status': 'closed'}})
        with patch.object(l, 'prepare_route', return_value=selected), patch.object(f, 'close_no_update', return_value=no_update) as close:
            actual = controller.positive_sft_dispatch_route({}, [], 8)
        self.assertEqual(actual, {'steps': 0, 'no_update': no_update})
        self.assertEqual(close.call_args.args[-1], parent_status)
        self.assertTrue(callable(controller.training_completion_fields))
        # Source/runtime mismatch is refused before installation or job issue.
        del controller.positive_sft_dispatch_route
        controller.jobs.metadata['source_files'] = {}
        with patch.object(l.time, 'time', return_value=self.now), self.assertRaisesRegex(ValueError, 'installed'):
            l.install(controller, self.bank)

    def test_routed_jobs_selects_training_peer_not_verifier_metadata(self):
        controller = self.fx.fx.controller; controller.signed = self.sign
        common = self.releases[0]['payload']
        trainer = SimpleNamespace(config={}, metadata=dict(
            source_files=common['execution_source_files'], runtime_versions=common['runtime_versions']))
        router_config = {'unrelated': 'unchanged'}
        verifier = SimpleNamespace(config={'role': 'audit'}, metadata={'source_files': {'different.py': '0'*64}})
        controller.jobs = SimpleNamespace(config=router_config, metadata=verifier.metadata,
            roles={'train': trainer, 'verify': verifier})
        with patch.object(l.time, 'time', return_value=self.now): l.install(controller, self.bank)
        selected = dict(payload={'steps': 2}, selected_release=self.releases[1])
        with patch.object(l, 'prepare_route', return_value=selected):
            result = controller.positive_sft_dispatch_route({}, [], 8)
        self.assertEqual(result['steps'], 2)
        self.assertEqual(controller.jobs.config, {'unrelated': 'unchanged'})
        self.assertEqual(verifier.config, {'role': 'audit'})
        self.assertEqual(trainer.config['unaudited_training_execution_release'], self.releases[1])
        job = self.fx.job(113, 8); job['steps'] = result['steps']
        amended = self.fx.amend(job, trainer.config['unaudited_training_execution_release'])
        value = execution.validate(self.sign(amended), self.authority)
        self.assertEqual(value['steps'], 2)
        self.assertEqual(value['learning_rate_authorization']['payload']['steps'], 2)
        self.assertEqual(value['execution_release_sha256'], sha(self.releases[1]))

    def test_lifecycle_failure_propagates_and_does_not_dispatch(self):
        controller = self.fx.fx.controller; controller.signed = self.sign
        common = self.releases[0]['payload']
        controller.jobs = SimpleNamespace(config={}, metadata=dict(
            source_files=common['execution_source_files'], runtime_versions=common['runtime_versions']))
        with patch.object(l.time, 'time', return_value=self.now): l.install(controller, self.bank)
        with patch.object(l, 'prepare_route', side_effect=OSError('native copy unavailable')):
            with self.assertRaises(OSError): controller.positive_sft_dispatch_route({}, [], 8)
        self.assertEqual(controller.jobs.config, {})

    def test_source_hook_is_before_any_capability_and_propagates_zero_result(self):
        import ast
        source = (REPO/'subnet/persistent_training_controller.py').read_text()
        ast.parse(source)
        hook = source.index("sft_route=getattr(controller,'positive_sft_dispatch_route',None)")
        self.assertLess(hook, source.index('controller.jobs.run(', hook))
        self.assertLess(hook, source.index("cached=controller.state/(epoch+'-training-metrics.json')", hook))
        self.assertIn("if routed['no_update'] is not None:return routed['no_update']", source)
        self.assertIn("if not learner or recovery is not None:", source)


class CommittedZeroReentryTests(unittest.TestCase):
    def setUp(self):
        from test_positive_sft_variable_dose import RouteTests
        self.fx = RouteTests(); self.fx.setUp(); self.addCleanup(self.fx.doCleanups)
        self.c = self.fx.controller; self.m = self.fx.manifest
        self.bank = self.fx.sign({'unit_fixture': 'bank-identity'})
        self.submissions = []
        (self.c.state/'controller.json').write_text(json.dumps(self.fx.status))
        self.fx.close()
        self.root = self.c.state/(self.m['epoch']+'-positive-sft-fallback')
        route = json.loads((self.root/'route.ROOT-SIGNED.json').read_bytes())['payload']
        dispatch = dict(version=l.ROUTE_VERSION, epoch=self.m['epoch'], steps=0,
            selected_release_sha256=None, bank_sha256=f.sha(self.bank),
            manifest_sha256=f.sha(self.fx.sign(self.m)),
            input_inventory_sha256=f.sha(l.receipt_inventory(self.submissions)), route=route)
        (self.root/'dispatch.ROOT-SIGNED.json').write_text(json.dumps(self.fx.sign(dispatch)))

    def test_only_full_published_zero_closure_gets_historical_bank_admission(self):
        existing = l.original_zero_closure(self.c, self.m, self.submissions, self.bank)
        self.assertEqual(existing['no_update_receipt_sha256'],
            f.sha(json.loads((self.root/'no-update.ROOT-SIGNED.json').read_bytes())))
        # Stop at bank admission: its real signature/expiry controls are tested
        # above; this checks only the prior-closure choice of admission time.
        with patch.object(l, 'bank', side_effect=RuntimeError('admission-boundary')) as bank:
            with self.assertRaisesRegex(RuntimeError, 'admission-boundary'):
                l.prepare_route(self.c, self.m, self.submissions, self.bank, now=999999)
        self.assertIsNone(bank.call_args.kwargs['now'])
        self.fx.bucket.objects.clear()  # A prepared local receipt is not durable.
        self.assertIsNone(l.original_zero_closure(self.c, self.m, self.submissions, self.bank))
        with patch.object(l, 'bank', side_effect=RuntimeError('admission-boundary')) as bank:
            with self.assertRaisesRegex(RuntimeError, 'admission-boundary'):
                l.prepare_route(self.c, self.m, self.submissions, self.bank, now=999999)
        self.assertEqual(bank.call_args.kwargs['now'], 999999)

    def test_zero_reentry_refuses_changed_bank_inventory_parent_and_missing_readback(self):
        changed_bank = self.fx.sign({'unit_fixture': 'different-bank'})
        with self.assertRaisesRegex(ValueError, 'bindings'):
            l.original_zero_closure(self.c, self.m, self.submissions, changed_bank)
        with patch.object(l, 'receipt_inventory', return_value=['changed-inventory']):
            with self.assertRaisesRegex(ValueError, 'bindings'):
                l.original_zero_closure(self.c, self.m, self.submissions, self.bank)
        status = copy.deepcopy(self.fx.status); status['training_steps'] += 1
        (self.c.state/'controller.json').write_text(json.dumps(status))
        with self.assertRaisesRegex(ValueError, 'parent'):
            l.original_zero_closure(self.c, self.m, self.submissions, self.bank)
        (self.c.state/'controller.json').write_text(json.dumps(self.fx.status))
        with patch.object(self.fx.bucket, 'get_bounded', side_effect=TimeoutError('readback')):
            with self.assertRaises(TimeoutError):
                l.original_zero_closure(self.c, self.m, self.submissions, self.bank)
        key = next(iter(self.fx.bucket.objects)); self.fx.bucket.objects[key] = b'x'*len(self.fx.bucket.objects[key])
        with self.assertRaisesRegex(ValueError, 'readback'):
            l.original_zero_closure(self.c, self.m, self.submissions, self.bank)


if __name__ == '__main__':
    unittest.main()
