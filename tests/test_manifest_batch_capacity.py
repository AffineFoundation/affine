"""CPU-only prospective cap controls: synthetic keys; no network or GPU work."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from subnet import forced_sampling as forced, cli
from subnet import commitment_transport as commitments
from subnet import committed_training_inputs as learner
from subnet.batches import pack
from subnet.batch_quotas import configured_quotas
from subnet.miner import Miner
from subnet.sampling_uniqueness import validate_batch
from subnet.storage import canonical, Identity
from subnet.training_receipts import sha
from subnet.backend_jobs import COVERED_POLICY
from subnet.backend_profiles import HOPPER_FP32_REVISION, profile
from training_receipt_fixtures import sign
import test_fast_prefill_audit as fast_fixture
import test_class_quota_opening as opening_fixture
import test_committed_training_inputs as learner_fixture
import test_v5_successor_calibration as calibration_fixture
from subnet import successor_calibration


def sampling_contract():
    _, m = fast_fixture.Controls().support_runtime()
    c = copy.deepcopy(m['sampling_contract'])
    c.update(version=forced.MINER_VERSION, max_attempts=1000)
    return c


class BatchCapacity(unittest.TestCase):
    def fixture(self, cap):
        c=learner_fixture.LearnerAdmissionTests();c.setUp();self.addCleanup(c.doCleanups)
        m=c.manifest;m.update(K=4,L=4,samples_per_batch=8,max_batches=cap,
            sampling_source_hash=forced.source_hash(),sampling_contract=sampling_contract(),
            submission_transport_policy=commitments.VERSION2)
        definition=m['environments'][0];definition['indices']=list(range(cap+1))
        definition['spec']['num_samples']=cap+1
        if 'sample_harness_registry' in m:
            m['sample_harness_registry'][definition['env_id']]['indices']=list(range(cap+1))
        context=forced.binding(m,c.identity);batches=[]
        for index in range(cap):
            b=copy.deepcopy(c.batch);b.update(index=index,sample_index=index)
            b['rollouts']=[]
            for nonce in range(8):
                row=copy.deepcopy(c.batch['rollouts'][0 if nonce<4 else 1])
                row.update(index=index,sample_index=index,env_id=b['env_id'],environment_version=b['environment_version'],
                    task_hash=hashlib.sha256(f'task-{index}'.encode()).hexdigest(),seed=nonce,
                    sampling=forced.receipt(context,nonce),classification='positive' if nonce<4 else 'negative')
                row['turns'][0]['output']=[100+index,20+nonce]
                b['rollouts'].append(row)
            validate_batch(b,m,c.identity);batches.append(b)
        return c,m,batches

    def test_signed_cap_bounds_and_draws_are_unchanged(self):
        c,m,_=self.fixture(3);baseline=forced.binding(m,c.identity)
        for cap in (1,3,9,256):
            changed=dict(m,max_batches=cap)
            self.assertEqual(forced.binding(changed,c.identity),baseline)
            self.assertEqual(forced.uniform(forced.binding(changed,c.identity),'math','c'*64,0,999,0,3),
                             forced.uniform(baseline,'math','c'*64,0,999,0,3))
        for cap in (None,True,8.,'8',0,-1,257):
            with self.subTest(cap=cap),self.assertRaises(ValueError):forced.binding(dict(m,max_batches=cap),c.identity)
        for nonce in (-1,1000,True):
            with self.assertRaises(ValueError):forced.receipt(baseline,nonce)

    def test_larger_cap_keeps_four_four_and_no_duplicate_or_old_nonce_credit(self):
        c,m,batches=self.fixture(9)
        for mutate in (
            lambda b:b['rollouts'].pop(),
            lambda b:b['rollouts'][7].update(classification='positive'),
            lambda b:b['rollouts'][7]['turns'][0].update(output=b['rollouts'][0]['turns'][0]['output']),
            lambda b:b['rollouts'][7].update(seed=0,sampling=forced.receipt(forced.binding(m,c.identity),0)),
            lambda b:b['rollouts'][7].update(seed=1000),
        ):
            bad=copy.deepcopy(batches[-1]);mutate(bad)
            with self.assertRaises(ValueError):validate_batch(bad,m,c.identity)
        with self.assertRaises(ValueError):validate_batch(batches[-1],m,'f'*64)
        with self.assertRaises(ValueError):validate_batch(batches[-1],dict(m,epoch='another'),c.identity)

    def test_real_prepared_state_restores_signed_higher_caps_and_preserves_slot_caps(self):
        for cap in (3,9):
            with self.subTest(cap=cap):
                c,m,batches=self.fixture(cap)
                packed=[(b,pack([(b,[[] for _ in b['rollouts']])])) for b in batches]
                path=c.root/'prepared.json';commitments.write_prepared_state(path,m,packed)
                with patch('subnet.miner.check_runtime_profile'):
                    restored=Miner(SimpleNamespace(id=c.identity),m,'unused',capability={'put_url':'unused'},state_path=path)
                self.assertEqual(len(restored.batches),cap)
                self.assertEqual([b for b,_ in restored.batches],batches)
                with self.assertRaises(ValueError):
                    commitments.read_prepared_state(path,dict(m,max_batches=cap-1))
                # Legacy ZIP restoration is also bounded by the signed cap.
                legacy=c.root/'legacy.zip';legacy.write_bytes(pack([(b,[[]for _ in b['rollouts']])for b in batches]))
                with patch('subnet.miner.check_runtime_profile'):
                    self.assertEqual(len(Miner(SimpleNamespace(id=c.identity),m,'unused',capability={'put_url':'unused'},state_path=legacy).batches),cap)
                    with self.assertRaisesRegex(ValueError,'signed per-UID limit'):
                        Miner(SimpleNamespace(id=c.identity),dict(m,max_batches=cap-1),'unused',capability={'put_url':'unused'},state_path=legacy)

    def test_last_slot_real_signed_commitment_and_unaudited_admission(self):
        for cap in (9,):
            with self.subTest(cap=cap):
                c,m,batches=self.fixture(cap)
                identity=SimpleNamespace(id=c.identity,key=c.miner)
                packed=[(b,pack([(b,[[]for _ in b['rollouts']])]))for b in batches]
                commitment=commitments.make(identity,m,packed)
                commitments.validate(canonical(commitment),m['epoch'],c.identity,cap)
                with self.assertRaises(ValueError):commitments.validate(canonical(commitment),m['epoch'],c.identity,cap-1)
                b=batches[-1];slot=cap-1;entry=commitment['payload']['batches'][slot]
                data=canonical(dict(version=learner.ARTIFACT_VERSION,epoch=m['epoch'],checkpoint=m['checkpoint']['id'],miner=c.identity,slot=slot,batch=b))
                path=c.root/'last-slot.json';path.write_bytes(data)
                admission=dict(version=learner.VERSION,epoch=m['epoch'],checkpoint=m['checkpoint']['id'],source_sha256=m['source_bundle']['sha256'],miner_identity=c.identity,slot=slot,original_commitment=commitment,commitment_sha256=sha(commitment),proof_sha256=entry['sha256'],batch_sha256=entry['batch_sha256'],document_sha256=entry['training_sha256'],document_size=len(data),captured_at=m['deadline']+1,assurance='unaudited')
                obj=dict(sha256=entry['training_sha256'],size=len(data),learner_admission=sign(c.operator,admission))
                with patch('subnet.batches.unpack',side_effect=AssertionError('no proof replay')), patch('subnet.model.Runtime.compute',side_effect=AssertionError('no GPU/model')):
                    summary,pairs=learner.admitted_submission(path,obj,m,c.authority)
                self.assertEqual(len(pairs),4);self.assertEqual(summary['slot'],slot)
                self.assertFalse(summary['trainer_verification_performed'])
                with self.assertRaises(ValueError):learner.admitted_submission(path,obj,dict(m,max_batches=cap-1),c.authority)
                tampered=copy.deepcopy(commitment['payload']);tampered['batches'][slot]['index']=tampered['batches'][0]['index']
                with self.assertRaises(ValueError):commitments.validate(canonical(sign(c.miner,tampered)),m['epoch'],c.identity,cap)


class OpeningCapacity(unittest.TestCase):
    def test_first_signed_opening_issues_exact_signed_slots(self):
        for cap in (3,9):
            with self.subTest(cap=cap):
                case=opening_fixture.QuotaOpeningTests();case.setUp();self.addCleanup(case.doCleanups)
                identity=Identity();case.miner=identity.id
                c=sampling_contract();policy={k:v for k,v in c.items()if k not in('randomness','verification','generation')}
                manifest=case.opening(K=4,L=4,max_batches=cap,sampling_policy=policy,
                    source_bundle={'sha256':'a'*64,'size':1},submission_transport_policy=commitments.VERSION2,
                    training_input_policy=learner.VERSION,training_policy=COVERED_POLICY,
                    model_runtime_revision=HOPPER_FP32_REVISION,backend_profile=profile(HOPPER_FP32_REVISION)[1],
                    numerical_policy=profile(HOPPER_FP32_REVISION)[2])
                original=json.loads(case.bucket.objects['public/nonpayable-quota/manifest.json'])
                self.assertEqual(original['payload'],manifest)
                capdoc=identity.decrypt(manifest['capabilities'][identity.id])
                self.assertEqual(manifest['max_batches'],cap)
                self.assertEqual(configured_quotas(manifest),(4,4))
                if 'samples_per_batch' in manifest:self.assertEqual(manifest['samples_per_batch'],8)
                self.assertEqual(len(capdoc['batch_put_urls']),cap)
                self.assertEqual(len(capdoc['training_put_urls']),cap)
                self.assertEqual(case.gateway.epochs[manifest['epoch']]['max_batches'],cap)


    def test_successor_calibration_keeps_exact_draws_for_higher_signed_caps(self):
        for cap in (3,9):
            with self.subTest(cap=cap):
                case=calibration_fixture.Controls();case.setUp();self.addCleanup(case.doCleanups)
                case.opening.update(K=4,L=4,samples_per_batch=8,max_batches=cap)
                result=case.next_opening()
                self.assertEqual(result['max_batches'],cap)
                self.assertEqual(len(case.seen),2)
                _,manifest,request=case.seen[0]
                context=successor_calibration.draw_context(manifest,request)
                self.assertEqual(context,successor_calibration.draw_context(dict(manifest,max_batches=3),request))
                for invalid in (0,257,True,'6'):
                    with self.assertRaises(ValueError):successor_calibration.draw_context(dict(manifest,max_batches=invalid),request)

    def test_invalid_cap_refuses_before_upload_capabilities(self):
        case=opening_fixture.QuotaOpeningTests();case.setUp();self.addCleanup(case.doCleanups)
        c=sampling_contract();policy={k:v for k,v in c.items()if k not in('randomness','verification','generation')}
        for cap in (True,8.,'8',0,-1,257):
            with self.subTest(cap=cap),patch.object(case.gateway,'open')as opening:
                with self.assertRaises(ValueError):case.opening(K=4,L=4,max_batches=cap,sampling_policy=policy)
                opening.assert_not_called()


class ClientCapacity(unittest.TestCase):
    def args(self,temp,local):
        return SimpleNamespace(search_budget=8,env_id='math',indices=None,cap_file=None,key='synthetic',state=temp,
            manifest_url='https://manifest.invalid',current_url=None,gateway='https://unused.invalid',authority='trusted',max_batches=local,once=True)

    def test_direct_client_fills_manifest_cap_but_respects_explicit_lower_cap(self):
        for cap,local,expected in ((3,None,3),(9,None,9),(9,3,3),(9,12,9)):
            with self.subTest(cap=cap,local=local),tempfile.TemporaryDirectory()as temp:
                manifest=dict(epoch='test',checkpoint={'id':'approved'},capabilities={'miner':{}},deadline=100,max_batches=cap)
                fake=SimpleNamespace(batches=[],upload=lambda:None,close=lambda:None)
                def search(index,**kwargs):fake.batches.append((dict(env_id='math',index=index,checkpoint='approved'),[]))
                fake.search=search
                with patch.object(cli,'identity',return_value=SimpleNamespace(id='miner')),patch.object(cli,'fetch_signed',return_value=manifest),patch.object(cli,'entries',return_value=[dict(env_id='math',indices=list(range(20)))]),patch.object(cli,'checkpoint_download',return_value=Path(temp)),patch.object(cli,'Miner',return_value=fake),patch.object(cli,'check_runtime_profile'),patch.object(cli.time,'time',return_value=10),patch.object(cli.time,'time_ns',return_value=123):
                    cli.run(self.args(temp,local))
                self.assertEqual(len(fake.batches),expected)

    def test_supervisor_default_is_manifest_driven_and_explicit_three_survives(self):
        from subnet import miner_supervisor as supervisor
        class Observed(Exception):pass
        for explicit,expected in (([],None),(['--max-batches','3'],3),(['--max-batches','9'],9)):
            with self.subTest(explicit=explicit),tempfile.TemporaryDirectory()as temp:
                root=Path(temp);key=root/'fixture-key';key.write_text('fixture-not-a-real-key');key.chmod(0o600)
                seen=[]
                def cycle(args,*rest):seen.append(args);raise Observed
                argv=['--authority','a'*64,'--key',str(key),'--state',str(root/'state'),'--source-cache',str(root/'sources'),*explicit]
                with patch.object(supervisor,'cycle',side_effect=cycle),self.assertRaises(Observed):supervisor.main(argv)
                self.assertEqual(seen[0].max_batches,expected)
                args=supervisor.launch_arguments(seen[0],{'source_bundle':{'sha256':'b'*64}},'https://manifest.invalid')
                if expected is None:self.assertNotIn('--max-batches',args)
                else:self.assertEqual(args[args.index('--max-batches')+1],str(expected))


if __name__=='__main__':unittest.main()
