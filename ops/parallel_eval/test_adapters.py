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


if __name__=='__main__':unittest.main()
