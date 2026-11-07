"""CPU/native-schema adapter qualification. Not a substitute for rented-GPU replay."""
import copy
import tempfile
import unittest
from pathlib import Path
from .durable import Stage, digest, read
from .inbox import Inbox
from .outbox import Producer, EffectWorker
from .k1 import K1Join
from .test_protocol import Replay


class OutboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.r = Replay(Path(self.tmp.name) / 'stage')

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_signed_producer_claim_before_job(self):
        inbox = Inbox(self.r.stage.root, self.r.protocol, self.r.operator.id)
        p = Producer(self.r.stage.root, self.r.operator, inbox)
        body = dict(epoch=8, training_head=self.r.genesis, turn_sha256=digest('native-turn'))
        eid = p.emit('claim', body)
        self.assertEqual(p.emit('claim', body), eid)
        self.assertFalse(p.acknowledged(eid))
        self.assertEqual(inbox.tick(), [])
        self.assertTrue(p.acknowledged(eid))
        self.assertEqual(self.r.stage.load()['claims']['8'], body)

    def test_crash_after_receipt_reuses_effect(self):
        self.r.protocol.request_restore(9)
        calls = []
        def effect(c, heartbeat):
            calls.append(c['id'])
            return self.r.execute(c)
        w = EffectWorker(self.r.stage.root, self.r.protocol, 'worker-one', {'restore': effect})
        def die(c): raise SystemExit('injected after receipt fsync')
        with self.assertRaises(SystemExit):
            w.tick(now=100, ttl=10, after_receipt=die)
        self.assertEqual(w.tick(now=105, ttl=10), None)
        w2 = EffectWorker(self.r.stage.root, self.r.protocol, 'worker-two', {'restore': effect})
        self.assertEqual(w2.tick(now=111)['state'], 'acked')
        self.assertEqual(len(calls), 1)

    def test_lease_renewal_is_fenced(self):
        self.r.protocol.request_restore(12)
        c = self.r.protocol.lease({'restore'}, 'one', now=100, ttl=10)
        self.assertEqual(self.r.protocol.renew(c, now=105, ttl=10), 115)
        self.assertIsNone(self.r.protocol.lease({'restore'}, 'two', now=111))
        c2 = self.r.protocol.lease({'restore'}, 'two', now=116)
        with self.assertRaises(ValueError): self.r.protocol.renew(c, now=117)
        self.r.protocol.ack(c2, self.r.execute(c2))
        with self.assertRaises(ValueError): self.r.protocol.renew(c2, now=118)

    def test_expired_lease_cannot_be_revived(self):
        self.r.protocol.request_restore(13)
        c = self.r.protocol.lease({'restore'}, 'one', now=10, ttl=5)
        with self.assertRaises(ValueError): self.r.protocol.renew(c, now=15)


class K1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s = Stage.create(Path(self.tmp.name) / 'stage')
        self.k = K1Join(self.s.root)
        self.k.initialize([], dict(enabled=True, grants=[]))
        self.public = dict(status='turn', epoch_index=8, hotkey='fixture-worker', file_sha256=digest('file'),
                           head=digest('bf16'), optimizer_step=0, at=100)
        self.row = {k:self.public[k] for k in ('epoch_index', 'hotkey', 'file_sha256', 'head', 'at')}
        self.sha = self.k.capture(8, self.public, self.row)

    def tearDown(self): self.tmp.cleanup()

    def payload(self, verdict):
        return dict(epoch=8, verdict=verdict, turn_sha256=self.sha,
                    consume_turn=verdict!='no_decision', retry_grant=verdict=='no_decision')

    def test_red_once_preserves_native_row(self):
        p=self.payload('red'); r=self.k.join(p)
        self.assertEqual(self.k.join(p),r)
        self.assertEqual(self.k._rows(),[self.row])
        self.assertEqual(read(self.k.grants)['grants'],[])

    def test_no_decision_never_spends_turn(self):
        p=self.payload('no_decision');r=self.k.join(p)
        self.assertFalse(r['consume_turn']); self.assertTrue(r['retry_grant'])
        self.assertEqual(self.k._rows(),[])
        self.assertEqual(len(read(self.k.grants)['grants']),1)
        self.assertEqual(self.k.join(p),r)

    def test_native_turn_timestamp_frozen(self):
        changed=dict(self.public,at=101)
        with self.assertRaises(ValueError): self.k.capture(8,changed,self.row)

    def test_wrong_claim_digest_rejected(self):
        p=self.payload('red');p['turn_sha256']=digest('wrong')
        with self.assertRaises(ValueError):self.k.join(p)
        self.assertEqual(self.k._rows(),[])

    def test_crash_after_commit_does_not_duplicate(self):
        def die(phase):raise SystemExit('injected after durable native write')
        with self.assertRaises(SystemExit):self.k.join(self.payload('green'),fault=die)
        r=self.k.join(self.payload('green'))
        self.assertTrue(r['consume_turn']);self.assertEqual(self.k._rows(),[self.row])

    def test_crash_after_grant_does_not_duplicate(self):
        def die(phase):
            if phase=='grant-durable':raise SystemExit('injected after durable grant')
        with self.assertRaises(SystemExit):self.k.join(self.payload('no_decision'),fault=die)
        self.k.join(self.payload('no_decision'))
        self.assertEqual(len(read(self.k.grants)['grants']),1)
        self.assertEqual(self.k._rows(),[])

    def test_infrastructure_cannot_consume(self):
        p=self.payload('no_decision');p['consume_turn']=True
        with self.assertRaises(ValueError):self.k.join(p)


    def test_pending_reserves_fifo_once_then_no_decision_releases(self):
        self.assertEqual(self.k.queue_rows(),[self.row])
        self.assertEqual(self.k.queue_rows(before_epoch=8),[])
        self.k.join(self.payload('no_decision'))
        self.assertEqual(self.k.queue_rows(),[])
        self.assertEqual(self.k._rows(),[])

    def test_committed_pending_not_double_counted(self):
        self.k.join(self.payload('red'))
        self.assertEqual(self.k.queue_rows(),[self.row])


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        from .lifecycle import LifecycleConsumers
        from .durable import atomic
        self.tmp=tempfile.TemporaryDirectory();self.r=Replay(Path(self.tmp.name)/'stage')
        self.inbox=Inbox(self.r.stage.root,self.r.protocol,self.r.operator.id)
        self.producer=Producer(self.r.stage.root,self.r.operator,self.inbox)
        self.consumer=LifecycleConsumers(self.r.stage.root,self.r.protocol,self.producer)
        self.current=read(self.r.stage.root/'CURRENT.json')
        self.turn=self.r.stage.root/'FROZEN-TURN.json';atomic(self.turn,dict(epoch=0,hotkey='fixture'))

    def tearDown(self):self.tmp.cleanup()

    def claimed(self):
        from .lifecycle import Pending
        self.assertIsInstance(self.consumer.prepare_job(0,self.current,self.turn),Pending)
        self.assertEqual(self.inbox.tick(),[])
        return self.consumer.prepare_job(0,self.current,self.turn)

    def trained(self):
        claim=self.claimed();model,master=self.r.weights('native-adapter-candidate',1.2)
        row=self.r.publisher.register(model,master)
        binding=dict(epoch=0,candidate=row['id'],training_head=claim['training_head'])
        artifact=self.r.seal('trained',binding,dict(checkpoint=row['checkpoint'],master_digest=row['master_digest']))
        self.consumer.trained(0,row['id'],artifact);self.assertEqual(self.inbox.tick(),[])
        return row

    def test_claim_ack_precedes_native_job(self):
        claim=self.claimed();self.assertEqual(claim,self.r.stage.load()['claims']['0'])

    def test_keep_then_epoch_advance_without_waiting_for_eval(self):
        from .lifecycle import Pending
        from .durable import atomic
        from subnet.mimo_release_policy import sign
        from subnet.mimo_lifecycle import save_manifest
        self.trained();self.assertIsInstance(self.consumer.result(0,self.current),Pending)
        self.r.drain({'keep','pin','dispatch'})
        result=self.consumer.result(0,self.current)
        self.assertEqual(result['pe_state'],'pending')
        self.assertEqual(self.r.stage.load()['jobs']['0']['state'],'evaluating')
        finalized=sign(self.r.operator,dict(input_checkpoint=self.current['checkpoint']['id'],result=result))
        save_manifest(self.r.stage.root/'NATIVE-FINALIZED.json',finalized)
        save_manifest(self.r.stage.root/'CURRENT.json',dict(self.current,epoch_index=1))
        proof=self.consumer.observe_epoch_current(0,self.current,self.r.stage.root/'CURRENT.json',self.r.stage.root/'NATIVE-FINALIZED.json',self.r.operator.id)
        self.assertEqual(proof['pe_state'],'pending')
        # Actual next native slot can claim the restored head while epoch 0's
        # independent GPU-eval state is STILL evaluating.
        self.r.protocol.request_restore(1);self.r.drain({'restore'})
        now=read(self.r.stage.root/'CURRENT.json');turn=self.r.stage.root/'TURN-1.json';atomic(turn,dict(epoch=1))
        self.assertIsInstance(self.consumer.prepare_job(1,now,turn),Pending)
        self.assertEqual(self.inbox.tick(),[])
        self.assertEqual(self.consumer.prepare_job(1,now,turn)['epoch'],1)
        self.assertEqual(self.r.stage.load()['jobs']['0']['state'],'evaluating')

    def test_same_bf16_wrong_generation_rejected(self):
        changed=copy.deepcopy(self.current);changed['pe_head']['generation_id']=digest('different-generation')
        changed['pe_head_id']=digest(changed['pe_head'])
        with self.assertRaises(ValueError):self.consumer.prepare_job(0,changed,self.turn)

    def test_other_miner_epoch_cannot_be_consumed(self):
        self.trained();self.r.drain({'keep','pin','dispatch'})
        with self.assertRaises(ValueError):self.consumer.result(0,dict(self.current,epoch_index=1))

    def test_green_is_not_adopted_as_this_epoch_result(self):
        self.trained();self.r.drain({'keep','pin','dispatch'});self.r.score(0,'green')
        self.r.protocol.reconcile();result=self.consumer.result(0,self.current)
        self.assertEqual(result['status'],'no-update');self.assertEqual(result['checkpoint'],self.current['checkpoint'])
        with self.assertRaises(ValueError):self.consumer.promotion_result(self.current)


class PublicationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.r=Replay(Path(self.tmp.name)/'stage')
        self.row=self.r.train(7,1.3);self.command=digest('crash-safe-publication')
        self.request=self.r.publisher.request(self.command,self.r.genesis,self.row['id'],self.r.genesis,1)
        self.gd=self.r.stage.root/'generations'/self.command

    def tearDown(self):self.tmp.cleanup()

    def test_full_projection_is_not_repeated_when_recording_same_install(self):
        from unittest.mock import patch
        store=self.r.publisher.masters
        with patch.object(store,'install',wraps=store.install) as install:
            pub=self.r.publisher.publish(self.request)
        self.assertEqual(install.call_count,1)
        self.assertTrue(self.r.publisher.restore(pub,self.r.genesis))

    def test_same_base_and_training_head_is_deep_verified_once(self):
        from unittest.mock import patch
        store=self.r.publisher.masters
        entry=self.r.publisher.registry[self.r.genesis]['master_entry']
        with patch.object(store,'verify',wraps=store.verify) as verify:
            self.r.publisher.publish(self.request)
        self.assertEqual(sum(c.args[0]==entry for c in verify.call_args_list),1)

    def test_signed_request_persisted_before_any_effect(self):
        def die(phase):raise SystemExit('request persisted')
        with self.assertRaises(SystemExit):self.r.publisher.publish(self.request,fault=die)
        self.assertEqual(read(self.gd/'REQUEST.json'),self.request)
        self.assertFalse((self.gd/'model').exists())
        other=self.r.publisher.request(self.command,self.r.genesis,self.row['id'],self.r.genesis,2)
        with self.assertRaises(ValueError):self.r.publisher.publish(other)

    def test_incomplete_master_copy_never_admitted_or_stuck(self):
        self.gd.mkdir();master=self.gd/'master';master.mkdir();(master/'master.safetensors').write_bytes(b'incomplete')
        pub=self.r.publisher.publish(self.request)
        self.assertTrue(self.r.publisher.verify(pub,self.r.genesis))
        self.assertEqual(len(list(self.gd.glob('.incomplete-master-*'))),1)
        self.assertTrue(self.r.publisher.restore(pub,self.r.genesis))

    def test_death_after_master_copy_reuses_exact_bytes(self):
        def die(phase):
            if phase=='model-and-master-durable':raise SystemExit('master durable')
        with self.assertRaises(SystemExit):self.r.publisher.publish(self.request,fault=die)
        f=self.gd/'master/master.safetensors';inode=f.stat().st_ino
        pub=self.r.publisher.publish(self.request)
        self.assertEqual(f.stat().st_ino,inode)
        self.assertEqual(self.r.publisher.publish(self.request),pub)


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        from .ownership import GateOwnership
        from subnet.storage import Identity
        from subnet.backend_jobs import signed
        self.tmp=tempfile.TemporaryDirectory();self.s=Stage.create(Path(self.tmp.name)/'stage')
        self.identity=Identity()
        def verify(binding,proof):
            try:return signed(proof,self.identity.id)==dict(binding=binding,drained=True)
            except Exception:return False
        self.g=GateOwnership(self.s.root,verify)

    def tearDown(self):self.tmp.cleanup()

    def test_missing_owner_fails_closed(self):
        with self.assertRaises(FileNotFoundError):
            with self.g.guard('legacy'):pass

    def test_drained_transfer_fences_all_legacy_writes_not_only_claimed_epoch(self):
        from subnet.mimo_release_policy import sign
        self.g.initialize('legacy',819,digest('generation-head'))
        before=read(self.g.path)
        binding=dict(before=before,owner='pe',epoch=819,head=digest('generation-head'),drained_required=True)
        proof=sign(self.identity,dict(binding=binding,drained=True))
        self.g.transfer('legacy','pe',819,digest('generation-head'),proof)
        # A late old-epoch result is still forbidden: ownership is not decided
        # by whether that old epoch has a PE claim.
        with self.assertRaises(ValueError):
            with self.g.guard('legacy',generation=0):pass
        with self.g.guard('pe',generation=1) as value:self.assertEqual(value['owner'],'pe')

    def test_unverified_transfer_does_not_change_owner(self):
        self.g.initialize('legacy',819,digest('head'))
        with self.assertRaises(ValueError):self.g.transfer('legacy','pe',819,digest('head'),{})
        with self.g.guard('legacy') as value:self.assertEqual(value['generation'],0)


if __name__=='__main__':unittest.main()
