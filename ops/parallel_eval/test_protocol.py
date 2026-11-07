"""Real signatures and native master compatibility; CPU fixtures, NOT GPU qualification."""
import ast
import copy
import hashlib
import json
import os
import tempfile
import time
import types
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file
from subnet.storage import Identity
from subnet.backend_jobs import signed
from subnet.mimo_release_policy import ReleasePolicy, build_policy, sign
from .durable import Stage, atomic, digest, read
from .masters import MasterStore, install_gc_guard, install_generation_hooks
from .publication import PrivatePublisher, checkpoint, copy_immutable
from .protocol import Protocol
from .inbox import Inbox
from .activation import Activation, StageServices


class Replay:
    def __init__(self, root):
        self.stage = Stage.create(root)
        self.operator, self.worker, self.eval_worker = Identity(), Identity(), Identity()
        self.paths = {}
        self.original = self.weights('head', 1.0)
        self.genesis = checkpoint(self.original[0])['id']
        atomic(self.stage.root / 'CURRENT.json', dict(epoch_index=0, checkpoint=checkpoint(self.original[0]), optimizer_step=0))
        manifest = dict(run_id='pe-cpu-shadow', payable=False, checkpoint=checkpoint(self.original[0]),
                        sampling_contract=dict(randomness=digest('nonpayable-fixture')),
                        epoch='test-pe', start=1, deadline=2)
        policy = build_policy(self.operator, manifest, task_assets={'assets/fixture.json':dict(sha256=digest('tasks'), bytes=2)}, enabled=True)
        self.policy = ReleasePolicy(policy, self.operator.id)
        self.publisher = PrivatePublisher(root, self.operator, self.worker, self.policy, lambda:read(self.stage.root/'CURRENT.json')['pe_head_id'])
        row = self.publisher.register(*self.original)
        self.genesis = row['id']
        atomic(self.stage.root / 'CURRENT.json', dict(epoch_index=0, checkpoint=row['checkpoint'], optimizer_step=0, pe_head=row['head'], pe_head_id=row['id']))
        self.protocol = Protocol(root, self.verify)
        self.protocol.initialize(self.genesis, 0, self.round(self.genesis))
        self.dispatches = {}
        self.drain({'baseline','pin-head','pin','unpin'})
        self.protocol.request_restore(0)
        self.drain({'restore'})

    def weights(self, name, value):
        d = self.stage.root/'inputs'/name
        model, master = d/'model', d/'master'
        model.mkdir(parents=True); master.mkdir()
        # Nonpayable fixture gradient update; no claim to be host GRPO replay.
        w = torch.tensor([value, 2.0], dtype=torch.float32)
        save_file({'w':w.to(torch.bfloat16)},str(model/'model.safetensors'),metadata={'format':'pt'})
        (model/'config.json').write_text('{"model_type":"nonpayable-cpu-fixture"}')
        save_file({'w':w},str(master/'master.safetensors'),metadata={'format':'pt'})
        return model,master

    @staticmethod
    def round(head):
        return dict(draw_sha256=digest(['fixture-draw', head]), settings_sha256=digest('live-settings-fixture'),
                    n_drawn=500, k_cand=8, head_rollouts=12, max_output_tokens=65536, max_num_seqs=512, dtype='bfloat16')

    def seal(self, kind, binding, value=True):
        identity = self.eval_worker if kind=='score' else self.worker
        return sign(identity, dict(kind=kind, request_sha256=digest(binding), value=value))

    def verify(self, kind, binding, receipt):
        try:
            if kind=='durable-current':
                actual=read(self.stage.root/'CURRENT.json')
                if receipt['current'] != actual:return False
                f=signed(receipt['finalized'], self.operator.id)
                pub=signed(binding['publication'],self.worker.id)['value']
                self.publisher.verify(pub,binding['base_head'])
                expected={k:pub[k] for k in ('checkpoint','optimizer_step','checkpoint_authorization','publication_receipt_sha256')}
                return (f['input_checkpoint']==binding['base_head'] and f['result']==expected
                        and actual['checkpoint']==pub['checkpoint'] and actual['optimizer_step']==pub['optimizer_step']
                        and actual['epoch_index']==f['epoch_index']+1 and actual['pe_head']==pub['head'] and actual['pe_head_id']==pub['head_id'])
            identity = self.eval_worker if kind=='score' else self.worker
            r=signed(receipt,identity.id)
            if r['kind']!=kind or r['request_sha256']!=digest(binding):return False
            if kind=='publish':
                return self.publisher.verify(r['value'],binding['base_head']) and r['value']['optimizer_step']==binding['optimizer_step']
            if kind=='abandon':return r['value']==dict(quiesced=True)
            if kind=='trained':
                row=self.publisher.registry[binding['candidate']]
                return r['value']==dict(checkpoint=row['checkpoint'],master_digest=row['master_digest'])
            return r['value'] is not False
        except Exception:
            return False

    def execute(self, command):
        kind,p=command['kind'],command['payload']
        value=True
        if kind=='baseline':
            value=dict(head=p['head'],baseline_sha256=digest(p))
        elif kind=='keep':
            row=self.publisher.registry[p['candidate']]
            dst=self.stage.root/'keeps'/p['candidate'];dst.parent.mkdir(exist_ok=True)
            copy_immutable(row['model'],dst)
        elif kind in ('pin','unpin'):
            for head in p['heads']:
                entry=self.publisher.registry[head]['master_entry']
                getattr(self.publisher.masters,kind)(entry,'eval:'+p['owner'])
        elif kind in ('pin-head','unpin-head'):
            entry=self.publisher.registry[p['head']]['master_entry']
            if kind=='pin-head':self.publisher.masters.pin(entry,'head:'+p['head'])
            else:
                self.publisher.masters.unpin(entry,'head:'+p['head'])
                if p['publication']:
                    pub=signed(p['publication'],self.worker.id)['value']
                    self.publisher.masters.unpin(entry,'generation:'+pub['generation'])
        elif kind=='dispatch':
            self.dispatches.setdefault(command['id'],p)
            value=dict(job=command['id'],attempt=0)
        elif kind=='publish':
            request=self.publisher.request(command['id'],p['base_head'],p['candidate'],p['training_head'],p['optimizer_step'])
            value=self.publisher.publish(request)
            row=self.publisher.registry[value['head_id']]
            assert row['checkpoint']==value['checkpoint']
        elif kind in ('ledger','k1','offer'):
            d=self.stage.root/kind;d.mkdir(exist_ok=True);path=d/(command['key']+'.json')
            if path.exists() and read(path)!=p:raise ValueError('duplicate effect conflict')
            atomic(path,p)
        elif kind!='restore':
            raise ValueError('unknown worker effect')
        return self.seal(kind,p,value)

    def drain(self,kinds=None):
        kinds=kinds or {'baseline','keep','pin','unpin','restore','dispatch','publish','offer','ledger','k1','pin-head','unpin-head'}
        for _ in range(200):
            self.protocol.reconcile()
            c=self.protocol.lease(kinds,'fixture-worker')
            if not c:return
            self.protocol.ack(c,self.execute(c))
        raise AssertionError('drain failed to quiesce')

    def train(self,epoch,value):
        head=self.stage.load()['head']
        self.protocol.request_restore(epoch)
        self.drain({'restore'})
        self.protocol.claim(epoch,head,digest(['fixture-turn',epoch]))
        model,master=self.weights('candidate-'+str(epoch),value)
        row=self.publisher.register(model,master,optimizer_step=self.stage.load()['optimizer_step']+1)
        binding=dict(epoch=epoch,candidate=row['id'],training_head=head)
        artifact=self.seal('trained',binding,dict(checkpoint=row['checkpoint'],master_digest=row['master_digest']))
        self.protocol.trained(**binding,artifact=artifact)
        return row

    def score(self,epoch,verdict,at=1):
        s=self.stage.load();j=s['jobs'][str(epoch)];rd=s['rounds'][j['round_head']]
        binding=dict(epoch=epoch,candidate=j['candidate'],round_head=j['round_head'],verdict=verdict,spec=rd['spec'],baseline=rd['baseline'],green_at=at if verdict=='green' else None)
        self.protocol.outcome(epoch,verdict,self.seal('score',binding),green_at=at if verdict=='green' else None)

    def apply(self,crash_test=False,drain_after=True):
        self.drain()
        s=self.stage.load();a=s['apply'];self.assert_apply(a)
        pub=signed(s['commands'][a['publish']]['receipt'],self.worker.id)['value']
        self.publisher.restore(pub,a['base_head'])
        current=read(self.stage.root/'CURRENT.json')
        final_payload=dict(epoch_index=current['epoch_index'],input_checkpoint=a['base_head'],
                           result={k:pub[k] for k in ('checkpoint','optimizer_step','checkpoint_authorization','publication_receipt_sha256')})
        finalized=sign(self.operator,final_payload)
        atomic(self.stage.root/'FINALIZED.json',finalized)
        if crash_test:
            try:self.protocol.confirm_current(finalized,current,self.round(pub['head_id']))
            except ValueError:pass
            else:raise AssertionError('consumed before durable CURRENT')
        new=dict(epoch_index=current['epoch_index']+1,checkpoint=pub['checkpoint'],optimizer_step=pub['optimizer_step'],pe_head=pub['head'],pe_head_id=pub['head_id'])
        atomic(self.stage.root/'CURRENT.json',new)
        # Recreate controller after the simulated crash-after-CURRENT gap.
        self.protocol=Protocol(self.stage.root,self.verify)
        assert self.protocol.confirm_current(finalized,new,self.round(pub['head_id']))
        assert not self.protocol.confirm_current(finalized,new,self.round(pub['head_id']))
        if drain_after:self.drain()
        return pub

    @staticmethod
    def assert_apply(a):
        assert a and a['state']=='offered'


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.r=Replay(Path(self.tmp.name)/'nonpayable')

    def test_probe_contract_rejects_fp8_or_smaller_sample_serving_settings(self):
        for key,value in [('dtype','fp8'),('max_num_seqs',64),('k_cand',4),('max_output_tokens',32768),('n_drawn',600)]:
            spec=self.r.round(self.r.genesis);spec[key]=value
            with self.assertRaises(ValueError):self.r.protocol.initialize(self.r.genesis,0,spec)

    def test_signed_fresh_stale_apply_and_current_crash_recovery(self):
        a=self.r.train(10,1.01);self.r.drain()
        b=self.r.train(11,1.02);self.r.drain()
        self.r.score(11,'green',20);self.r.score(10,'green',10)
        first=self.r.apply(crash_test=True)
        self.assertEqual(first['checkpoint'],a['checkpoint'])
        second=self.r.apply()
        model=self.r.stage.root/'generations'/second['generation']/'model'/'model.safetensors'
        expected=torch.tensor([1.01,2.])+(torch.tensor([1.02,2.])-torch.tensor([1.,2.]))
        self.assertTrue(torch.equal(load_file(str(model))['w'],expected.to(torch.bfloat16)))
        s=self.r.stage.load()
        self.assertEqual([j['epoch'] for j in s['chain']],[10,11])
        self.assertEqual(len(s['rounds']),3)
        self.assertEqual(read(self.r.stage.root/'k1/10.json')['consume_turn'],True)
        self.assertEqual(read(self.r.stage.root/'ledger/11.json')['verdict'],'green')

    def test_stale_overflow_gets_no_decision_and_k1_retry(self):
        for e,value in zip(range(20,24),(1.02,1.05,1.11,1.2)):
            self.r.train(e,value);self.r.drain()
        for e in range(20,23):self.r.score(e,'green',e);self.r.apply()
        self.r.score(23,'green',23);self.r.drain()
        s=self.r.stage.load();j=s['jobs']['23']
        self.assertEqual(j['final'],'no_decision');self.assertEqual(j['reason'],'stale-overflow')
        k=read(self.r.stage.root/'k1/23.json');self.assertFalse(k['consume_turn']);self.assertTrue(k['retry_grant'])
        self.assertEqual(len(s['chain']),3)

    def test_round_master_pin_survives_until_baseline_receipt(self):
        self.r.train(24,1.02);self.r.drain();self.r.score(24,'green',1)
        pub=self.r.apply(drain_after=False)
        self.r.drain({'pin','pin-head','unpin-head','ledger','k1'})
        head=pub['head_id'];pin=self.r.publisher.masters.root/pub['master_entry']/'PE-PINS.json'
        self.assertIn('eval:round:'+head,read(pin)['owners'])
        self.assertEqual(self.r.stage.load()['rounds'][head]['state'],'preparing')
        self.r.drain({'baseline','unpin'})
        self.assertNotIn('eval:round:'+head,read(pin)['owners'])

    def test_current_head_pin_and_old_generation_retire_after_next_head_pin(self):
        self.r.train(25,1.02);self.r.drain();self.r.score(25,'green',1)
        first=self.r.apply()
        self.r.train(26,1.06);self.r.drain();self.r.score(26,'green',2)
        second=self.r.apply()
        old=self.r.publisher.masters.root/first['master_entry']/'PE-PINS.json'
        self.assertFalse(old.exists())
        new=read(self.r.publisher.masters.root/second['master_entry']/'PE-PINS.json')
        self.assertIn('head:'+second['head_id'],new['owners'])

    def test_old_round_can_score_after_head_move(self):
        self.r.train(30,1.01);self.r.drain();self.r.train(31,1.02);self.r.drain()
        old=self.r.stage.load()['jobs']['31']['round_head']
        self.r.score(30,'green',1);self.r.apply()
        self.assertTrue(self.r.stage.load()['rounds'][old]['closed'])
        self.r.score(31,'red');self.r.drain()
        self.assertEqual(self.r.stage.load()['jobs']['31']['round_head'],old)

    def test_guard_and_ledger_k1_crash_keep_pins_until_both_acks(self):
        row=self.r.train(40,1.03);self.r.drain();self.r.score(40,'no_decision')
        self.r.drain({'ledger'})
        pin=self.r.publisher.masters.root/row['master_entry']/'PE-PINS.json'
        self.assertTrue(pin.exists())
        self.r.protocol=Protocol(self.r.stage.root,self.r.verify)
        self.r.drain()
        self.assertFalse(pin.exists())
        k=read(self.r.stage.root/'k1/40.json');self.assertTrue(k['retry_grant']);self.assertFalse(k['consume_turn'])

    def test_node_loss_lease_expiry_rejects_late_ack(self):
        self.r.train(50,1.04);self.r.drain({'keep','pin'})
        old=self.r.protocol.lease({'dispatch'},'dead-worker',now=100,ttl=10)
        self.assertIsNotNone(old)
        self.assertIsNone(self.r.protocol.lease({'dispatch'},'other',now=105))
        new=self.r.protocol.lease({'dispatch'},'replacement',now=111)
        self.assertEqual(old['id'],new['id'])
        with self.assertRaisesRegex(ValueError,'lost lease'):self.r.protocol.ack(old,self.r.execute(old))
        self.r.protocol.ack(new,self.r.execute(new));self.r.protocol.reconcile()
        self.assertEqual(len(self.r.dispatches),1)
        self.assertEqual(self.r.stage.load()['jobs']['50']['state'],'evaluating')

    def test_no_turn_count_refresh_after_nine_reds(self):
        for e in range(60,69):
            self.r.train(e,1.+(e-59)*.01);self.r.drain();self.r.score(e,'red');self.r.drain()
        self.assertEqual(len(self.r.stage.load()['rounds']),1)

    def test_claim_and_keep_restore_guards(self):
        self.r.train(70,1.03)
        with self.assertRaisesRegex(ValueError,'keep'):self.r.protocol.request_restore(71)
        with self.assertRaisesRegex(ValueError,'restore'):self.r.protocol.claim(71,self.r.genesis,digest(['fixture-turn',71]))
        self.r.drain()
        with self.assertRaisesRegex(ValueError,'restore'):self.r.protocol.claim(71,self.r.genesis,digest(['fixture-turn',71]))
        self.r.protocol.request_restore(71);self.r.drain({'restore'})
        self.r.protocol.claim(71,self.r.genesis,digest(['fixture-turn',71]))
        with self.assertRaises(ValueError):self.r.protocol.claim(71,'f'*64,digest(['fixture-turn',71]))

    def test_signed_inbox_idempotent_and_bad_event_does_not_block(self):
        inbox=Inbox(self.r.stage.root,self.r.protocol,self.r.operator.id)
        event=sign(self.r.operator,dict(kind='claim',body=dict(epoch=80,training_head=self.r.genesis,turn_sha256=digest(['fixture-turn',80]))))
        p=inbox.pending/'a.json';atomic(p,event)
        first=inbox.consume(p);self.assertEqual(first,inbox.consume(p))
        event['payload']['body']['epoch']=81;atomic(inbox.pending/'bad.json',event)
        errors=inbox.tick();self.assertEqual(len(errors),1)
        self.assertTrue(p.exists())

    def test_missing_master_abandon_grants_retry_after_quiescence(self):
        self.r.train(85,1.06)
        s=self.r.stage.load();claim=s['claims']['85']
        binding=dict(epoch=85,claim=claim,reason='missing-master',quiesced_required=True)
        self.r.protocol.abandon(85,'missing-master',self.r.seal('abandon',binding,dict(quiesced=True)))
        self.r.drain()
        self.assertEqual(self.r.stage.load()['jobs']['85']['final'],'no_decision')
        self.assertTrue(read(self.r.stage.root/'k1/85.json')['retry_grant'])

    def test_restore_timeout_before_candidate_is_not_a_consumed_turn(self):
        self.r.protocol.claim(86,self.r.genesis,digest(['fixture-turn',86]))
        self.r.protocol.request_restore(86)
        claim=self.r.stage.load()['claims']['86']
        b=dict(epoch=86,claim=claim,reason='restore-timeout',quiesced_required=True)
        self.r.protocol.abandon(86,'restore-timeout',self.r.seal('abandon',b,dict(quiesced=True)))
        self.r.drain()
        k=read(self.r.stage.root/'k1/86.json')
        self.assertFalse(k['consume_turn']);self.assertTrue(k['retry_grant'])
        self.assertEqual(k['turn_sha256'],claim['turn_sha256'])

    def test_tampered_publication_and_wrong_current_rejected(self):
        self.r.train(90,1.02);self.r.drain();self.r.score(90,'green');self.r.drain()
        s=self.r.stage.load();a=s['apply'];p=signed(s['commands'][a['publish']]['receipt'],self.r.worker.id)['value']
        bad=copy.deepcopy(p);bad['checkpoint_authorization']['payload']['optimizer_step']+=1
        with self.assertRaises(Exception):self.r.publisher.verify(bad,a['base_head'])
        path=self.r.stage.root/'objects'/p['checkpoint']['id']/'config.json';path.write_text('{}')
        with self.assertRaises(ValueError):self.r.publisher.verify(p,a['base_head'])


class ObjectStoreTests(unittest.TestCase):
    def test_real_adapter_full_get_and_tamper_rejection(self):
        import io
        from .object_store import R2Objects
        class S3:
            def __init__(self): self.objects={}
            def upload_file(self,path,bucket,key): self.objects[key]=Path(path).read_bytes()
            def get_object(self,**kw): return {'Body':io.BytesIO(self.objects[kw['Key']])}
        with tempfile.TemporaryDirectory() as td:
            r=Replay(Path(td)/'r2')
            client=S3()
            store=R2Objects(r.stage.root,types.SimpleNamespace(client=client,name='private-test'))
            r.publisher.object_store=store
            r.train(93,1.0001);r.drain();r.score(93,'green');pub=r.apply()
            self.assertEqual(len(client.objects),4)
            r.publisher.restore(pub,r.genesis)
            key=next(k for k in client.objects if '/models/' in k)
            client.objects[key]+=b'tamper'
            with self.assertRaisesRegex(ValueError,'full GET integrity'):r.publisher.restore(pub,r.genesis)
            with self.assertRaises(ValueError):store._key('../live')


class ActivationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.s=Stage.create(Path(self.tmp.name)/'activation')
        self.image=dict(current=dict(epoch=1,head='H'),units=dict(lifecycle='active',gate='active'),
                        flag=dict(other=True,trainer='host-v834d',parallel_eval=dict(enabled=False)),
                        caps=dict(output=65536,batch=69632),dropins=dict(other='keep'),training_job=False,
                        recovery_required=False,gate_busy=False,unacked_apply=False)
        atomic(self.s.root/'SERVICE-IMAGE.json',self.image)

    def test_drained_surgical_success_is_not_live_qualification(self):
        j=Activation(self.s.root).simulate(1,'H',dict(enabled=True),'pe-wrapper')
        after=StageServices(self.s.root).snapshot()
        self.assertEqual(j['state'],'verified');self.assertFalse(j['ready_for_live'])
        self.assertEqual(after['caps'],self.image['caps']);self.assertEqual(after['flag']['trainer'],'host-v834d')

    def test_busy_and_changed_epoch_rollback_without_overwriting_concurrent_state(self):
        def move(phase):
            if phase=='configured':
                svc=StageServices(self.s.root);v=svc.snapshot();v['current']['epoch']=2;v['caps']['chunk']=1024;svc.save(v)
        with self.assertRaisesRegex(ValueError,'changed'):Activation(self.s.root).simulate(1,'H',dict(enabled=True),'pe',move)
        v=StageServices(self.s.root).snapshot();self.assertEqual(v['current']['epoch'],2);self.assertEqual(v['caps']['chunk'],1024)
        self.assertEqual(v['flag']['parallel_eval'],self.image['flag']['parallel_eval'])
        self.assertEqual(v['units'],self.image['units'])

    def test_crash_recovery_preserves_64k_and_foreign_dropins(self):
        j=dict(state='configured',before=self.image,live=False,ready_for_live=False)
        atomic(self.s.root/'ACTIVATION-SIMULATION.json',j)
        altered=copy.deepcopy(self.image);altered['flag']['parallel_eval']=dict(enabled=True);altered['dropins']['parallel_eval']='PE';altered['units']['lifecycle']='stopped'
        atomic(self.s.root/'SERVICE-IMAGE.json',altered)
        self.assertEqual(Activation(self.s.root).recover()['state'],'rolled-back')
        self.assertEqual(StageServices(self.s.root).snapshot(),self.image)


class NativeMasterTests(unittest.TestCase):
    def test_native_digest_load_and_gc_after_expired_legacy_pin(self):
        source=os.environ.get('PE_NATIVE_TRAINER_SOURCE')
        if not source:self.skipTest('native source not supplied; not a qualification pass')
        with tempfile.TemporaryDirectory() as td:
            r=Replay(Path(td)/'native');store=r.publisher.masters
            names={'_sha_tensor','_digest_map','_fp32_load','_fp32_gc','_fp32_store_entries'}
            tree=ast.parse(Path(source).read_text())
            nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
            self.assertEqual({n.name for n in nodes},names)
            mod=types.ModuleType('native_master_fixture')
            mod.__dict__.update(Path=Path,hashlib=hashlib,json=json,FP32_STORE=str(store.root),FP32_KEEP=2,FP32_VERSION='v833f-fp32-head-master-v1')
            exec(compile(ast.Module(body=nodes,type_ignores=[]),source,'exec'),mod.__dict__)
            row=r.publisher.registry[r.genesis];entry=row['master_entry'];meta=read(store.root/entry/'meta.json')
            w=load_file(str(r.original[0]/'model.safetensors'))['w']
            opt=types.SimpleNamespace(parameters=[('w',w)],masters={'w':torch.zeros_like(w,dtype=torch.float32)})
            self.assertEqual(mod._digest_map(opt.parameters)[0],meta['bf16_sha256'])
            install_generation_hooks(mod,store,lambda opt,live:row)
            self.assertEqual(mod._fp32_prepare(opt,None,lambda x:None)['input_master_digest'],meta['master_digest'])
            store.unpin(entry,'head:'+r.genesis)
            store.pin(entry,'eval:pending');(store.root/entry/'PIN').touch()
            os.utime(store.root/entry/'PIN',(time.time()-20000,)*2)
            for i in range(5):
                d=store.root/digest(['unused',i]);d.mkdir();atomic(d/'meta.json',{})
            install_gc_guard(mod,store);mod._fp32_gc((),lambda x:None)
            self.assertTrue((store.root/entry).exists(),'durable pin survives old 3h expiry')
            store.unpin(entry,'eval:pending')
            self.assertFalse((store.root/entry/'PE-PINS.json').exists())

    def test_same_bf16_different_master_generations_remain_distinct(self):
        with tempfile.TemporaryDirectory() as td:
            r=Replay(Path(td)/'collision')
            model,master=r.weights('collision',1.0001)
            self.assertEqual(checkpoint(model),checkpoint(r.original[0]))
            row=r.publisher.register(model,master)
            original=r.publisher.registry[r.genesis]
            self.assertNotEqual(row['master_entry'],original['master_entry'])
            self.assertNotEqual(row['id'],original['id'])
            for item in (row,original):
                r.publisher.masters.verify(item['master_entry'],item['model'])
            r.train(91,1.0002);r.drain();r.score(91,'green')
            pub=r.apply()
            self.assertEqual(pub['checkpoint'],original['checkpoint'])
            self.assertNotEqual(pub['head_id'],r.genesis)
            self.assertEqual(len(r.stage.load()['rounds']),2)

    def test_equal_master_distinct_generation_ids_do_not_alias(self):
        with tempfile.TemporaryDirectory() as td:
            r=Replay(Path(td)/'same-master')
            a=r.publisher.register(*r.original,generation_id=digest('generation-a'))
            b=r.publisher.register(*r.original,generation_id=digest('generation-b'))
            self.assertEqual(a['master_digest'],b['master_digest'])
            self.assertEqual(a['checkpoint'],b['checkpoint'])
            self.assertNotEqual(a['master_entry'],b['master_entry'])
            self.assertNotEqual(a['id'],b['id'])

    def test_generation_hook_invalidates_same_bf16_memory_cache(self):
        with tempfile.TemporaryDirectory() as td:
            r=Replay(Path(td)/'cache')
            model,master=r.weights('residual',1.0001)
            old=r.publisher.registry[r.genesis]
            new=r.publisher.register(model,master)
            chosen=[old]
            def digest_map(items):
                from .masters import tensor_sha
                values={n:tensor_sha(v) for n,v in items}
                return digest(values),values
            native=types.SimpleNamespace(FP32_STORE=str(r.publisher.masters.root),_digest_map=digest_map,_fp32_gc=lambda protect,emit:None)
            install_generation_hooks(native,r.publisher.masters,lambda opt,live:chosen[0])
            w=load_file(str(model/'model.safetensors'))['w']
            opt=types.SimpleNamespace(parameters=[('w',w)],masters={'w':torch.zeros_like(w,dtype=torch.float32)})
            native._fp32_prepare(opt,None,lambda x:None)
            chosen[0]=new
            self.assertEqual(native._fp32_prepare(opt,None,lambda x:None)['input_master_digest'],new['master_digest'])
            self.assertTrue(torch.equal(opt.masters['w'],load_file(str(master/'master.safetensors'))['w']))
            self.assertEqual(opt._pe_generation,new['generation_id'])
            chosen[0]=dict(new,master_digest='0'*64)
            with self.assertRaisesRegex(ValueError,'selection'):native._fp32_prepare(opt,None,lambda x:None)



if __name__=='__main__':unittest.main()
