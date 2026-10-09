import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from subnet import weight_submission_transaction as t
from subnet.weight_submission_reconciliation import collect_evidence, reconcile


class Reader:
    def __init__(self, observation, *, committed=True, pending=True, wrong_hash=False, failure=False):
        self.observation = observation
        self.committed, self.pending, self.wrong_hash, self.failure = committed, pending, wrong_hash, failure
    def finalized_head(self): return dict(number=102, hash='head102')
    def owner_state(self, intent, head):
        return dict(block=102, block_hash='head102', last_update=101,
                    owner_pending=self.pending, weights=[[85, 44444]])
    def block(self, number):
        commits = []
        if number == 101 and self.committed:
            commits.append(dict(signer='owner', netuid=120, mecid=0, nonce=self.observation['nonce'],
                extrinsic_hash='wrong' if self.wrong_hash else self.observation['extrinsic_hash'],
                extrinsic_index=2, call_function='commit_timelocked_mechanism_weights',
                success=not self.failure))
        return dict(number=number, hash='block'+str(number), commits=commits)


class JournalControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cursor = self.root/'current-assessment-v1/submission.json'
        self.chainstate = self.root/'chain'; self.chainstate.mkdir()
        self.journal=t.SubmissionJournal(self.cursor, self.chainstate,
                                         assessment_sha256='a'*64, policy_sha256='b'*64)
        self.signed=NS(data=b'actual signed call', extrinsic_hash='0x'+hashlib.blake2b(b'actual signed call',digest_size=32).hexdigest())
    def begin(self):
        return self.journal.begin(owner='owner', owner_uid=0, netuid=120, window_end=7200,
            vector=[[85,65535],[86,32000]], registrations={'key':{'uid':85}},
            attempt_start_block=100, signed=self.signed, nonce=42)
    def observation(self):
        cursor=json.loads(self.cursor.read_text())
        return json.loads((Path(cursor['attempt_directory'])/'signed-observation.json').read_text())
    def test_begin_binds_exact_signed_hash_and_vector(self):
        cursor=self.begin()
        doc=json.loads((Path(cursor['attempt_directory'])/'intent.json').read_text())
        self.assertEqual(doc['payload']['expected_weights'],[[85,65535],[86,32000]])
        self.assertEqual(self.observation()['extrinsic_hash'],self.signed.extrinsic_hash)
        self.assertEqual(json.loads(self.cursor.read_text())['status'],'submitting')
    def test_journal_never_stores_signed_transaction_bytes(self):
        self.begin()
        self.assertNotIn(self.signed.data.hex(), ''.join(p.read_text() for p in self.root.rglob('*.json')))
    def test_owner_never_enters_intent(self):
        with self.assertRaises(ValueError):
            self.journal.begin(owner='owner',owner_uid=85,netuid=120,window_end=7200,
                vector=[[85,65535]],registrations={},attempt_start_block=100,signed=self.signed,nonce=42)
        self.assertFalse(self.cursor.exists())
    def test_finalized_actual_hash_resolves_before_reveal(self):
        self.begin(); decision=self.journal.recover(None, reader=Reader(self.observation()))
        self.assertEqual(decision['status'],'submitted_finalized')
        self.assertTrue(decision['owner_timelock_pending'])
        self.assertFalse(decision['revealed_weights_asserted'])
        self.assertEqual(json.loads((self.chainstate/'weights.json').read_text())['last_submitted_window'],7200)
        self.assertEqual(json.loads(self.cursor.read_text())['status'],'submitted')
    def test_different_signed_hash_keeps_fence(self):
        self.begin(); decision=self.journal.recover(None,reader=Reader(self.observation(),wrong_hash=True))
        self.assertTrue(decision['preserve_fence']);self.assertEqual(json.loads(self.cursor.read_text())['status'],'submitting')
        self.assertFalse((self.chainstate/'weights.json').exists())
    def test_dispatch_failure_keeps_fence(self):
        self.begin(); self.assertTrue(self.journal.recover(None,reader=Reader(self.observation(),failure=True))['preserve_fence'])
    def test_absent_commit_never_authorizes_retry(self):
        self.begin(); decision=self.journal.recover(None,reader=Reader(self.observation(),committed=False))
        self.assertTrue(decision['preserve_fence']);self.assertFalse(decision['old_window_replay_allowed'])
    def test_later_hour_cursor_is_never_regressed(self):
        t.durable_atomic(self.chainstate/'weights.json',dict(last_submitted_window=10800,latest={'original':True}))
        self.begin(); self.journal.recover(None,reader=Reader(self.observation()))
        self.assertEqual(json.loads((self.chainstate/'weights.json').read_text()),dict(last_submitted_window=10800,latest={'original':True}))
    def test_existing_legacy_fence_requires_manual_proof(self):
        t.durable_atomic(self.cursor,dict(status='submitting',window_end=3600))
        with self.assertRaisesRegex(RuntimeError,'legacy uncertain'):self.journal.recover(None)
        with self.assertRaisesRegex(RuntimeError,'uncertain'):self.begin()
    def test_tampered_signed_observation_cannot_recover(self):
        cursor=self.begin(); path=Path(cursor['attempt_directory'])/'signed-observation.json'
        doc=json.loads(path.read_text());doc['nonce']=43;path.write_text(json.dumps(doc))
        with self.assertRaises(ValueError):self.journal.recover(None,reader=Reader(doc))
    def test_failed_durable_fence_prevents_begin_return(self):
        with patch.object(t,'durable_atomic',side_effect=OSError('disk full')):
            with self.assertRaises(OSError):self.begin()
        self.assertFalse(self.cursor.exists())
    def test_crash_after_weights_before_cursor_is_idempotent(self):
        self.begin(); orig=t.durable_atomic
        def fail_cursor(path,value):
            if Path(path)==self.cursor:raise OSError('process crash')
            return orig(path,value)
        with patch.object(t,'durable_atomic',side_effect=fail_cursor):
            with self.assertRaises(OSError):self.journal.recover(None,reader=Reader(self.observation()))
        self.assertEqual(json.loads((self.chainstate/'weights.json').read_text())['last_submitted_window'],7200)
        decision=self.journal.recover(None,reader=Reader(self.observation()))
        self.assertEqual(decision['status'],'submitted_finalized')


class PreparationControls(unittest.TestCase):
    def setUp(self):
        self.events=[]
        self.call=NS(data=b'one exact plan ciphertext')
        self.plan=NS(ok=True, signer='hotkey', signer_address='owner', call=self.call)
        self.intent=NS(journal_call_sha256=hashlib.sha256(self.call.data).hexdigest())
        self.wallet=NS(hotkey=NS(ss58_address='owner',crypto_type=1,sign=lambda payload: b'sig'))
        self.unsigned=NS(address='owner',call_data=self.call.data,payload=b'signing payload',nonce=42,era={'period':64,'current':100},era_block_hash='0x'+format(100,'064x'))
        self.signed=NS(data=b'exact signed tx',extrinsic_hash='0x'+hashlib.blake2b(b'exact signed tx',digest_size=32).hexdigest())
        async def attach(unsigned,sig):
            self.events.append('assemble');return self.signed
        async def submit(signed,key,**kwargs):
            self.events.append('broadcast');self.assertIs(signed,self.signed)
            self.assertTrue(kwargs['wait_for_finalization']);return NS(success=True)
        self.chain=NS(prepare_call=lambda *a,**k:self.unsigned, _call=asyncio.run,
            _client=NS(_substrate=NS(raw=NS(attach_signature=attach),submit_signed=submit)))
    def test_preparation_cannot_broadcast_and_sole_send_uses_exact_object(self):
        signed,nonce=t.prepare_exact_plan(self.chain,self.plan,self.intent,self.wallet)
        self.assertEqual(nonce,42);self.assertEqual(self.events,['assemble'])
        t.submit_prepared(self.chain,signed,self.wallet)
        self.assertEqual(self.events,['assemble','broadcast'])
    def test_unrelated_plan_refuses_before_preparing(self):
        self.intent.journal_call_sha256='0'*64
        with self.assertRaises(ValueError):t.prepare_exact_plan(self.chain,self.plan,self.intent,self.wallet)
        self.assertEqual(self.events,[])
    def test_nonce_rpc_error_is_prebroadcast(self):
        def fail(*a,**kw):raise TimeoutError()
        self.chain.prepare_call=fail
        with self.assertRaises(TimeoutError):t.prepare_exact_plan(self.chain,self.plan,self.intent,self.wallet)
        self.assertEqual(self.events,[])
    def test_signing_error_is_prebroadcast(self):
        def fail(*a):raise ValueError('cannot sign')
        self.wallet.hotkey.sign=fail
        with self.assertRaises(ValueError):t.prepare_exact_plan(self.chain,self.plan,self.intent,self.wallet)
        self.assertEqual(self.events,[])
    def test_wrong_sdk_extrinsic_hash_refuses(self):
        self.signed.extrinsic_hash='0x'+'f'*64
        with self.assertRaisesRegex(ValueError,'hash mismatch'):t.prepare_exact_plan(self.chain,self.plan,self.intent,self.wallet)
        self.assertEqual(self.events,['assemble'])
    def test_wrong_prepared_owner_or_call_refuses(self):
        for field,value in [('address','attacker'),('call_data',b'different')]:
            original=getattr(self.unsigned,field);setattr(self.unsigned,field,value)
            with self.assertRaises(ValueError):t.prepare_exact_plan(self.chain,self.plan,self.intent,self.wallet)
            setattr(self.unsigned,field,original)
        self.assertEqual(self.events,[])
    def test_sdk_seams_are_concrete_regular_local_files(self):
        self.assertEqual(len(t.sdk_seam_paths()),7)
        self.assertTrue(all(Path(p).is_file() for p in t.sdk_seam_paths()))


class ExactVectorControls(unittest.IsolatedAsyncioTestCase):
    async def test_real_sdk_conforming_vector_is_retained_from_actual_build(self):
        import bittensor as bt
        import bittensor.intents.weights as w
        built=NS(call=NS(data=b'timelock encrypted normalized values'))
        preflight=NS(uid=0,commit_reveal=True,min_allowed_weights=2,max_weight_limit=32767)
        with patch.object(w,'_preflight',new=AsyncMock(return_value=preflight)), \
             patch.object(w,'_build_timelocked',new=AsyncMock(return_value=built)) as compile_call:
            intent=t.make_recorded_intent(bt,netuid=120,uids=[85,86],weights=[.99,.01],version_key=0)
            intent.hotkey_address=lambda wallet:'owner';intent.hotkey_public_key=lambda wallet:b'public'
            result=await intent.build(None,None)
            self.assertIs(result,built)
            self.assertEqual(intent.journal_vector,[[85,65535],[86,65535]])
            self.assertEqual(compile_call.await_args.args[4:6],([85,86],[65535,65535]))
    async def test_owner_or_plaintext_branch_is_never_built(self):
        import bittensor as bt
        import bittensor.intents.weights as w
        for uid,enabled in [(85,True),(0,False)]:
            preflight=NS(uid=uid,commit_reveal=enabled,min_allowed_weights=1,max_weight_limit=65535)
            with patch.object(w,'_preflight',new=AsyncMock(return_value=preflight)), \
                 patch.object(w,'_build_timelocked',new=AsyncMock()) as compile_call:
                intent=t.make_recorded_intent(bt,netuid=120,uids=[85],weights=[1.],version_key=0)
                intent.hotkey_address=lambda wallet:'owner'
                with self.assertRaises(ValueError):await intent.build(None,None)
                compile_call.assert_not_called()





class AdapterBoundaryControls(unittest.TestCase):
    def setUp(self):
        import subnet.chain as chain_module
        self.chain_module=chain_module
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.state=self.root/'state';self.state.mkdir()
        marker=self.root/'.local/state/affine-transition/active';marker.parent.mkdir(parents=True);marker.touch()
        (self.state/'writer.enabled').touch()
        self.cursor=self.root/'current-assessment-v1/submission.json'
        self.journal=t.SubmissionJournal(self.cursor,self.state,assessment_sha256='a'*64,policy_sha256='b'*64)
        self.events=[];self.regs={'miner':dict(uid=85,public_key='public',snapshot_block=100)}
        self.a=chain_module.ChainAdapter.__new__(chain_module.ChainAdapter)
        self.a.state_dir=self.state;self.a.netuid=120;self.a.owner='owner'
        self.a.bt=NS(Wallet=lambda **kw:NS(hotkey=NS(ss58_address='owner')),SetWeights=lambda **kw:kw)
        self.a.chain=NS(block=100,plan=lambda *a:NS(ok=True),execute=lambda *a,**k:self.fail('execute replans'))
        self.a.registrations=lambda:self.regs
        self.a.query=lambda name,params,block:{'SubnetOwnerHotkey':'owner','Uids':0,'Keys':'miner',
            'LastUpdate':[0],'WeightsSetRateLimit':0,'WeightsVersionKey':0}[name]
        self.intent=NS(journal_vector=[[85,65535]],journal_owner_uid=0,journal_era={'period':64,'birth':100,'death':164,'block_hash':'0x'+format(100,'064x')})
        self.signed=NS(data=b'transaction',extrinsic_hash='0x'+hashlib.blake2b(b'transaction',digest_size=32).hexdigest())
        self.error=None
    def invoke(self,prepare_error=None,submit_error=None,begin_error=None,clock=None):
        def prepare(*a):
            self.events.append('prepare')
            if prepare_error:raise prepare_error
            self.assertFalse(self.cursor.exists())
            return self.signed,42
        def submit(*a):
            self.events.append('broadcast')
            self.assertEqual(json.loads(self.cursor.read_text())['status'],'submitting')
            if submit_error:raise submit_error
            return NS(block_hash='actual block',raise_for_failure=lambda:None)
        with patch.object(Path,'home',return_value=self.root), \
             patch.object(self.chain_module.time,'time',side_effect=clock or (lambda:7300)), \
             patch.object(self.chain_module.subprocess,'run',return_value=NS(stdout='inactive')), \
             patch.object(t,'make_recorded_intent',return_value=self.intent), \
             patch.object(t,'prepare_exact_plan',side_effect=prepare), \
             patch.object(t,'submit_prepared',side_effect=submit):
            if begin_error:
                with patch.object(self.journal,'begin',side_effect=begin_error):
                    return self.a.submit_hour({'miner':1},self.regs,7200,execute=True,
                        zero_total_policy='no-owner-retain-v1',registration_change_policy='current-hotkey-snapshot-v1',submission_journal=self.journal)
            return self.a.submit_hour({'miner':1},self.regs,7200,execute=True,
                zero_total_policy='no-owner-retain-v1',registration_change_policy='current-hotkey-snapshot-v1',submission_journal=self.journal)
    def test_preflight_network_failure_never_creates_fence(self):
        self.a.registrations=lambda:(_ for _ in ()).throw(TimeoutError())
        with self.assertRaises(TimeoutError):self.invoke()
        self.assertFalse(self.cursor.exists());self.assertEqual(self.events,[])
    def test_plan_policy_refusal_never_creates_fence(self):
        self.a.chain.plan=lambda *a:NS(ok=False)
        self.assertEqual(self.invoke()['status'],'chain_policy_denied')
        self.assertFalse(self.cursor.exists());self.assertEqual(self.events,[])
    def test_preparation_failure_never_creates_fence(self):
        with self.assertRaises(TimeoutError):self.invoke(prepare_error=TimeoutError())
        self.assertFalse(self.cursor.exists());self.assertEqual(self.events,['prepare'])
    def test_journal_disk_failure_prevents_send(self):
        with self.assertRaises(OSError):self.invoke(begin_error=OSError('disk full'))
        self.assertFalse(self.cursor.exists());self.assertEqual(self.events,['prepare'])
    def test_lost_ack_keeps_fence_and_sends_once(self):
        with self.assertRaises(TimeoutError):self.invoke(submit_error=TimeoutError())
        self.assertEqual(json.loads(self.cursor.read_text())['status'],'submitting')
        self.assertFalse((self.state/'weights.json').exists());self.assertEqual(self.events,['prepare','broadcast'])
    def test_success_then_repeat_does_not_rebroadcast(self):
        self.assertEqual(self.invoke()['status'],'submitted')
        self.assertEqual(self.invoke()['status'],'already_submitted')
        self.assertEqual(self.events,['prepare','broadcast'])
    def test_owner_uid_churn_after_plan_is_prebroadcast_failure(self):
        self.intent.journal_owner_uid=5
        with self.assertRaisesRegex(RuntimeError,'owner UID changed'):self.invoke()
        self.assertFalse(self.cursor.exists());self.assertEqual(self.events,[])


    def test_hour_changes_before_preparation_never_fences_or_broadcasts(self):
        with self.assertRaisesRegex(RuntimeError,'hour advanced'):
            self.invoke(clock=[7300,10800])
        self.assertFalse(self.cursor.exists());self.assertEqual(self.events,[])
    def test_hour_changes_during_preparation_never_fences_or_broadcasts(self):
        with self.assertRaisesRegex(RuntimeError,'hour advanced'):
            self.invoke(clock=[7300,7300,10800])
        self.assertFalse(self.cursor.exists());self.assertEqual(self.events,['prepare'])


class EvidenceBudgetControls(unittest.TestCase):
    def test_measured_reader_has_360_seconds_without_stealing_finality_reserve(self):
        from ops import current_assessment_writer as w
        self.assertEqual(w.EVIDENCE_TIMEOUT_SECONDS,360)
        with patch.object(w.signal,'getsignal',return_value='old'), \
             patch.object(w.signal,'getitimer',return_value=(720,0)), \
             patch.object(w.signal,'signal'),patch.object(w.signal,'setitimer') as timers, \
             patch.object(w.time,'monotonic',side_effect=[100,460]):
            with w.evidence_budget():pass
        self.assertEqual(timers.call_args_list[0].args[1],360)
        self.assertEqual(timers.call_args_list[-1].args[1],360)




class EraReader(Reader):
    def __init__(self,observation,*,height=164,found=None,missing_inventory=False):
        super().__init__(observation,committed=False,pending=True)
        self.height=height;self.found=found;self.calls=[];self.missing_inventory=missing_inventory
    def finalized_head(self):return dict(number=self.height,hash='head'+str(self.height))
    def owner_state(self,intent,head):
        return dict(block=self.height,block_hash='head'+str(self.height),last_update=50,
                    owner_pending=True,weights=[[85,11111]])
    def block(self,number):
        self.calls.append(number)
        hashes=[];commits=[]
        if number==105 and self.found:
            hashes=[self.observation['extrinsic_hash']]
            if self.found in ('failed','success','unknown'):
                commits=[dict(signer='owner',netuid=120,mecid=0,nonce=self.observation['nonce'],
                    extrinsic_hash=self.observation['extrinsic_hash'],extrinsic_index=4,
                    call_function='commit_timelocked_mechanism_weights',
                    success=self.found=='success',dispatch_failed=self.found=='failed')]
        doc=dict(number=number,hash='0x'+format(number,'064x'),commits=commits)
        if not self.missing_inventory:doc['extrinsic_hashes']=hashes
        return doc


class MortalEraControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.cursor=self.root/'current-assessment-v1/submission.json'
        self.state=self.root/'chain';self.state.mkdir()
        self.journal=t.SubmissionJournal(self.cursor,self.state,assessment_sha256='a'*64,policy_sha256='b'*64)
        raw=b'real object shape dummy data';signed=NS(data=raw,extrinsic_hash='0x'+hashlib.blake2b(raw,digest_size=32).hexdigest())
        cursor=self.journal.begin(owner='owner',owner_uid=0,netuid=120,window_end=7200,vector=[[85,65535]],
            registrations={},attempt_start_block=101,signed=signed,nonce=42,
            era=dict(period=64,birth=100,death=164,block_hash='0x'+format(100,'064x')))
        self.directory=Path(cursor['attempt_directory'])
        self.document=json.loads((self.directory/'intent.json').read_text())
        self.observation=json.loads((self.directory/'signed-observation.json').read_text())
    def decision(self,reader=None,**kwargs):
        reader=reader or EraReader(self.observation)
        evidence=collect_evidence(reader,self.document,actual_signed_observation=self.observation,**kwargs)
        return reconcile(self.document,evidence,actual_signed_observation=self.observation),evidence
    def test_complete_finalized_era_absence_resolves_without_advancing_paid_hour(self):
        reader=EraReader(self.observation)
        decision=self.journal.recover(None,reader=reader)
        self.assertEqual(decision['status'],'expired_unincluded')
        self.assertEqual(reader.calls,list(range(100,164)))
        self.assertFalse((self.state/'weights.json').exists())
        self.assertEqual(json.loads(self.cursor.read_text())['status'],'expired_unincluded')
        self.assertEqual(json.loads((self.directory/'terminal-cursor.json').read_text())['status'],'expired_unincluded')
    def test_finalized_dispatch_failure_resolves_without_advancing_paid_hour(self):
        t.durable_atomic(self.state/'weights.json',dict(last_submitted_window=3600,latest={'old':True}))
        reader=EraReader(self.observation,found='failed')
        decision=self.journal.recover(None,reader=reader)
        self.assertEqual(decision['status'],'failed_finalized')
        self.assertEqual(reader.calls,list(range(100,106)))
        self.assertEqual(json.loads((self.state/'weights.json').read_text()),dict(last_submitted_window=3600,latest={'old':True}))
        self.assertFalse(decision['old_window_replay_allowed'])
    def test_successful_exact_hash_resolves_before_expiry_and_reveal(self):
        decision,evidence=self.decision(EraReader(self.observation,height=110,found='success'))
        self.assertEqual(decision['status'],'submitted_finalized');self.assertEqual(evidence['scan_stop'],105)
    def test_unexpired_absence_keeps_fence(self):
        decision,_=self.decision(EraReader(self.observation,height=163))
        self.assertTrue(decision['preserve_fence'])
    def test_partial_era_cannot_prove_absence(self):
        decision,_=self.decision(max_blocks=32)
        self.assertTrue(decision['preserve_fence']);self.assertEqual(decision['reason'],'incomplete_finalized_era_absence')
    def test_timeout_mid_era_cannot_prove_absence(self):
        reader=EraReader(self.observation)
        orig=reader.block
        def block(number):
            if number==120:raise TimeoutError()
            return orig(number)
        reader.block=block
        decision,evidence=self.decision(reader)
        self.assertTrue(decision['preserve_fence']);self.assertEqual(evidence['read_error'],'TimeoutError')
    def test_missing_block_hash_inventory_cannot_prove_absence(self):
        decision,_=self.decision(EraReader(self.observation,missing_inventory=True))
        self.assertTrue(decision['preserve_fence']);self.assertEqual(decision['reason'],'incomplete_block_extrinsic_inventory')
    def test_hash_present_but_dispatch_unrecognized_keeps_fence(self):
        for kind in ('hash-only','unknown'):
            decision,_=self.decision(EraReader(self.observation,found=kind))
            self.assertTrue(decision['preserve_fence'])
    def test_mismatched_prepared_era_anchor_keeps_fence(self):
        self.observation['era']['block_hash']='0x'+'f'*64
        decision,_=self.decision()
        self.assertEqual(decision['reason'],'prepared_era_anchor_mismatch')
    def test_gapped_finalized_blocks_cannot_prove_absence(self):
        _,evidence=self.decision();del evidence['blocks'][3]
        decision=reconcile(self.document,evidence,actual_signed_observation=self.observation)
        self.assertTrue(decision['preserve_fence']);self.assertEqual(decision['reason'],'incomplete_scan')
    def test_unsupported_or_corrupt_actual_era_is_rejected(self):
        for change in ({'period':128},{'birth':True},{'death':165},{'block_hash':'unknown'}):
            bad=dict(self.observation['era'],**change)
            with self.assertRaises(ValueError):t.validate_era(bad)
    def test_terminal_failure_allows_new_current_intent_but_preserves_evidence(self):
        self.journal.recover(None,reader=EraReader(self.observation,found='failed'))
        original=(self.directory/'terminal-cursor.json').read_bytes()
        # A NEW current-hour intent, not rebroadcasting this old signed object.
        raw=b'new current call';signed=NS(data=raw,extrinsic_hash='0x'+hashlib.blake2b(raw,digest_size=32).hexdigest())
        result=self.journal.begin(owner='owner',owner_uid=0,netuid=120,window_end=10800,
            vector=[[85,65535]],registrations={},attempt_start_block=170,signed=signed,nonce=43,
            era=dict(period=64,birth=170,death=234,block_hash='0x'+format(170,'064x')))
        self.assertEqual(result['window_end'],10800)
        self.assertEqual((self.directory/'terminal-cursor.json').read_bytes(),original)




class WeightLockUpgradeControls(unittest.TestCase):
    def test_legacy0664_owned_lock_is_normalized_without_replacement(self):
        import os, stat
        from subnet.chain import weights_lock
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'weights.lock';path.write_text('preserved');path.chmod(0o664)
            before=path.stat()
            with weights_lock(path):
                after=path.stat()
                self.assertEqual((before.st_dev,before.st_ino),(after.st_dev,after.st_ino))
                self.assertEqual(stat.S_IMODE(after.st_mode),0o600)
                self.assertEqual(path.read_text(),'preserved')
    def test_symlink_lock_is_rejected_without_touching_target(self):
        from subnet.chain import weights_lock
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'target';target.write_text('keep');target.chmod(0o664)
            path=Path(d)/'weights.lock';path.symlink_to(target)
            with self.assertRaises(OSError):
                with weights_lock(path):pass
            self.assertEqual(target.stat().st_mode&0o777,0o664)

if __name__=='__main__':unittest.main()
