"""CPU-only transport fixtures with real fsynced capture journals; no R2 calls."""
import copy
import datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
PACKAGE = '_capture64_fixture'
package = types.ModuleType(PACKAGE)
package.__path__ = [str(HERE/'subnet')]
sys.modules[PACKAGE] = package
td = __import__(PACKAGE+'.training_documents', fromlist=['capture'])
storage = __import__(PACKAGE+'.storage', fromlist=['Bucket'])
wal = __import__(PACKAGE+'.capture_journal', fromlist=['CaptureJournal'])
commitments = __import__(PACKAGE+'.commitment_transport', fromlist=['freeze'])
spec = importlib.util.spec_from_file_location('capture_capacity', HERE/'ops/trainer_lifecycle/capture_capacity.py')
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


def policy(version=td.CAPACITY_CAPTURE_VERSION, workers=64):
    result = dict(version=version, workers=workers, max_document_bytes=2_000_000,
        max_inflight_bytes=workers*2_000_000, completion_order='first-completed')
    if version != td.CAPTURE_VERSION:
        result.update(journal_version=wal.VERSION, state_checkpoint_documents=128)
    return result


class Client:
    def __init__(self, bucket):
        self.bucket=bucket
        self.closed=False

    def get_object(self, *, Bucket, Key):
        b=self.bucket
        with b.lock:
            b.active+=1; b.maximum=max(b.maximum,b.active); b.gets.append(Key)
        try:
            if b.gate is not None:b.gate.wait(timeout=5)
            if b.delay:time.sleep(b.delay)
            if Key in b.fail_get:
                if b.fail_get[Key] > 0:
                    b.fail_get[Key]-=1
                    raise TimeoutError('synthetic network timeout')
            data=b.data[Key]
            return {'Body':io.BytesIO(data),'LastModified':datetime.datetime.fromtimestamp(b.upload_at,datetime.timezone.utc),'ETag':'synthetic-etag'}
        finally:
            with b.lock:b.active-=1

    def put_object(self, *, Bucket, Key, Body, ContentType):
        b=self.bucket
        with b.lock:
            b.capacity_puts.append(Key)
            b.data[Key]=Body
        if b.uncertain_put:
            b.uncertain_put=False
            raise TimeoutError('PUT committed, reply lost')

    def close(self):self.closed=True


class Bucket:
    name='local-fixture'
    def __init__(self):
        self.client=object(); self.data={}; self.lock=threading.Lock()
        self.active=0; self.maximum=0; self.gets=[]; self.capacity_puts=[]
        self.legacy_puts=[]; self.readers=[]; self.clients=[]; self.fail_get={}
        self.delay=0; self.gate=None; self.uncertain_put=False

    def capture_io_client(self, *, parallel_workers):
        assert parallel_workers==64
        self.readers.append(('v3',parallel_workers))
        client=Client(self); self.clients.append(client); return client

    def commitment_read_client(self, *, parallel_workers=None):
        self.readers.append(('legacy',parallel_workers))
        client=Client(self); self.clients.append(client); return client

    def put(self,key,data):self.legacy_puts.append(key); self.data[key]=data
    def json(self,key,value):self.put(key,td.canonical(value))
    def presign(self,key,method='get_object',expiry=60):return 'https://fixture.invalid/'+key
    def complete_commitment_listing(self,epoch,miners,cutoff):
        return sorted(m for m in miners if f'private/{epoch}/commitments/{m}.json'in self.data)


class Gateway:
    def __init__(self, root, count=144, capture_policy=None, seconds=5):
        self.state_path=Path(root)/'gateway.json'; self.epoch='fixture-151'; self.bucket=Bucket(); self.persists=0
        now=time.time(); self.bucket.upload_at=now-2
        state=dict(start=now-100,deadline=now-1,rejections={},commitment_capture_complete=True,
            commitment_binding=dict(checkpoint='a'*64,freeze_until=now+seconds,
                learner_capture_policy=capture_policy or policy()),commitment_pending={})
        for index in range(count):
            miner=f'miner-{index//9:03d}'; slot=index%9
            batch=dict(env_id='math',index=index,tokens=[1,2,3])
            data=td.document(batch,dict(epoch=self.epoch,checkpoint={'id':'a'*64}),miner,slot)
            entry=dict(slot=slot,training_size=len(data),training_sha256=td.sha(data),
                batch_sha256=td.sha(td.canonical(batch)),env_id='math',index=index)
            pending=state['commitment_pending'].setdefault(miner,dict(sha256=td.sha(miner.encode()),root='frozen/'+miner,document={'payload':{'batches':[]}}))
            pending['document']['payload']['batches'].append(entry)
            self.bucket.data['private/'+self.epoch+'/training/'+miner+'/'+str(slot)+'.json']=data
        self.epochs={self.epoch:state}; self.persist()

    def persist(self):
        self.persists+=1
        path=self.state_path.with_suffix('.new'); path.write_bytes(td.canonical(self.epochs)); path.replace(self.state_path)

    @property
    def state(self):return self.epochs[self.epoch]


class CaptureCapacityTests(unittest.TestCase):
    def test_exact_policy_preserves_historical_limits(self):
        for version in (td.CAPTURE_VERSION,td.DURABLE_CAPTURE_VERSION):
            for workers in (4,8,16):self.assertEqual(td.capture_policy(policy(version,workers)),policy(version,workers))
            with self.assertRaises(ValueError):td.capture_policy(policy(version,64))
        self.assertEqual(td.capture_policy(policy()),policy())
        for workers in (16,32,True,128):
            with self.assertRaises(ValueError):td.capture_policy(policy(workers=workers))
        bad=policy(); bad['max_inflight_bytes']-=1
        with self.assertRaises(ValueError):td.capture_policy(bad)

    def test_real_wal_full_capture_and_64_parallel_gets(self):
        with tempfile.TemporaryDirectory() as root:
            g=Gateway(root); g.bucket.gate=threading.Barrier(64)
            # Only the first64 calls rendezvous; the final partial wave must not wait.
            original=Client.get_object; lock=threading.Lock(); counter=[0]
            def get(client,**kwargs):
                with lock:
                    counter[0]+=1
                    if counter[0]>64:g.bucket.gate=None
                return original(client,**kwargs)
            order=capacity.prepare_order(g,g.epoch)
            with patch.object(Client,'get_object',get):td.capture(g,g.epoch,ordering=order)
            stats=g.state['training_capture_runs'][-1]
            self.assertEqual(stats['published_documents'],144)
            self.assertEqual(stats['maximum_inflight'],64)
            self.assertEqual(g.bucket.maximum,64)
            self.assertEqual(stats['raw_document_byte_limit'],128_000_000)
            self.assertEqual(stats['journal_records'],289)
            self.assertEqual(stats['deferred_slots'],0)
            self.assertEqual(g.bucket.legacy_puts,[])
            self.assertEqual(len(g.bucket.capacity_puts),144)
            self.assertTrue(all(c.closed for c in g.bucket.clients))
            for miner,rows in g.state['training_document_snapshots'].items():
                for slot,receipt in rows.items():
                    self.assertEqual(g.bucket.data[receipt['frozen_key']],g.bucket.data[f'private/{g.epoch}/training/{miner}/{slot}.json'])
            self.assertEqual(stats['stage_timing_seconds']['immutable_PUT']['calls'],144)
            # Replay the original disk checkpoint, retaining the actual immutable store.
            g.epochs=json.loads(g.state_path.read_bytes()); before=len(g.bucket.gets)
            td.capture(g,g.epoch,ordering=order)
            self.assertEqual(len(g.bucket.gets),before)

    def test_cutoff_defers_without_rejection_or_put(self):
        with tempfile.TemporaryDirectory() as root:
            g=Gateway(root,count=9,seconds=-.01)
            td.capture(g,g.epoch,ordering=capacity.prepare_order(g,g.epoch))
            self.assertEqual(g.state['rejections'],{})
            self.assertEqual(g.state['training_capture_runs'][-1]['deferred_slots'],9)
            self.assertEqual(g.bucket.gets,[])
            self.assertEqual(g.bucket.capacity_puts,[])

    def test_uncertain_put_recovers_exact_immutable_bytes_after_restart(self):
        with tempfile.TemporaryDirectory() as root:
            g=Gateway(root,count=1); g.bucket.uncertain_put=True
            order=capacity.prepare_order(g,g.epoch)
            original_wait=td.time.sleep
            def abort_after_failure(seconds):raise RuntimeError('synthetic process interruption')
            with patch.object(td.time,'sleep',abort_after_failure):
                with self.assertRaisesRegex(RuntimeError,'synthetic process'):td.capture(g,g.epoch,ordering=order)
            self.assertEqual(len(g.bucket.capacity_puts),1)
            g.epochs=json.loads(g.state_path.read_bytes())
            td.capture(g,g.epoch,ordering=order)
            self.assertEqual(len(g.bucket.capacity_puts),1)
            self.assertEqual(g.state['training_capture_runs'][-1]['reconciled_publications'],1)

    def test_structural_mismatch_rejected_not_infrastructure(self):
        with tempfile.TemporaryDirectory() as root:
            g=Gateway(root,count=1)
            key=next(iter(g.bucket.data)); g.bucket.data[key]=b'{}'
            td.capture(g,g.epoch,ordering=capacity.prepare_order(g,g.epoch))
            stats=g.state['training_capture_runs'][-1]
            self.assertEqual(stats['structural_failures'],1)
            self.assertEqual(stats['transient_failures'],0)
            self.assertEqual(g.bucket.capacity_puts,[])

    def test_transient_get_retries_without_miner_penalty(self):
        with tempfile.TemporaryDirectory() as root:
            g=Gateway(root,count=1); g.bucket.fail_get[next(iter(g.bucket.data))]=1
            td.capture(g,g.epoch,ordering=capacity.prepare_order(g,g.epoch))
            stats=g.state['training_capture_runs'][-1]
            self.assertEqual(stats['transient_failures'],1)
            self.assertEqual(stats['published_documents'],1)
            self.assertEqual(g.state['rejections'],{})

    def test_historical_policy_uses_legacy_put(self):
        with tempfile.TemporaryDirectory() as root:
            g=Gateway(root,count=10,capture_policy=policy(td.DURABLE_CAPTURE_VERSION,16))
            td.capture(g,g.epoch,ordering=capacity.prepare_order(g,g.epoch))
            self.assertEqual(g.bucket.readers,[('legacy',16)])
            self.assertEqual(len(g.bucket.legacy_puts),10)
            self.assertEqual(g.bucket.capacity_puts,[])
            self.assertNotIn('stage_timing_seconds',g.state['training_capture_runs'][-1])

    def test_pool_configuration_shared_get_put_is_bounded(self):
        from botocore.config import Config
        b=storage.Bucket.__new__(storage.Bucket)
        b.client=types.SimpleNamespace(meta=types.SimpleNamespace(config=Config()))
        b._client_options={'endpoint_url':'https://fixture.invalid','region_name':'auto'}
        with patch.object(storage.boto3,'client') as create:
            b.capture_io_client(parallel_workers=64)
        c=create.call_args.kwargs['config']
        self.assertEqual(c.max_pool_connections,64)
        self.assertEqual(c.retries['total_max_attempts'],1)
        self.assertEqual((c.connect_timeout,c.read_timeout),(5,10))
        with self.assertRaises(ValueError):b.commitment_read_client(parallel_workers=64)

    def test_prospective_wrapper_preserves_original_round_and_config(self):
        old=policy(td.DURABLE_CAPTURE_VERSION,16); new=policy()
        capacity.validate_config({'learner_capture_policy':old},{'learner_capture_policy':new})
        calls=[]
        module=types.SimpleNamespace(capture=lambda g,e:calls.append(('historical',e)),freeze_receipts=lambda g,e:'historical')
        contract=types.SimpleNamespace(contract=lambda config,round_number:dict(learner_capture_policy=dict(old),other='unchanged'))
        # Wrapper's private module import name requires the real subnet relative imports.
        with patch.dict(sys.modules,{'subnet':package,'subnet.commitment_transport':sys.modules[PACKAGE+'.commitment_transport']}):
            capacity.install(module,HERE/'subnet/training_documents.py',td.sha((HERE/'subnet/training_documents.py').read_bytes()),152,contract_module=contract,previous_policy=old)
        self.assertEqual(contract.contract({},151)['learner_capture_policy'],old)
        self.assertEqual(contract.contract({},152)['learner_capture_policy'],new)
        module.capture(None,'fixture-151')
        self.assertEqual(calls,[('historical','fixture-151')])
        g=types.SimpleNamespace(epochs={'fixture-152':{'commitment_binding':{'learner_capture_policy':old}}})
        with self.assertRaisesRegex(ValueError,'signed opening policy'):module.capture(g,'fixture-152')

    def test_order_is_same_hash_rank_and_reentry_seed(self):
        with tempfile.TemporaryDirectory() as root:
            g=Gateway(root,count=100)
            first=capacity.prepare_order(g,g.epoch)
            self.assertEqual(first,capacity.prepare_order(g,g.epoch))
            g.state['commitment_pending']['miner-000']['sha256']='f'*64
            with self.assertRaises(ValueError):capacity.prepare_order(g,g.epoch)

    def test_first_wave_contains_exact_original_random_rank_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            g=Gateway(root,count=144); g.bucket.gate=threading.Barrier(64)
            order=capacity.prepare_order(g,g.epoch)
            pending=g.state['commitment_pending']
            ranked=sorted([(miner,b)for miner,p in pending.items()for b in p['document']['payload']['batches']],
                key=lambda row:td.sha(td.canonical(dict(domain='postcommit-token-capture-order-v1',seed=order['seed'],
                    epoch=g.epoch,miner=row[0],commitment_sha256=pending[row[0]]['sha256'],slot=row[1]['slot']))))
            expected={f'private/{g.epoch}/training/{miner}/{b["slot"]}.json'for miner,b in ranked[:64]}
            first=[]; lock=threading.Lock(); original=Client.get_object
            def get(client,**kwargs):
                with lock:
                    first.append(kwargs['Key'])
                    if len(first)>64:g.bucket.gate=None
                return original(client,**kwargs)
            with patch.object(Client,'get_object',get):td.capture(g,g.epoch,ordering=order)
            self.assertEqual(set(first[:64]),expected)

    def test_document_finishing_get_after_cutoff_is_not_published(self):
        with tempfile.TemporaryDirectory() as root:
            g=Gateway(root,count=1,seconds=.10); g.bucket.delay=.15
            td.capture(g,g.epoch,ordering=capacity.prepare_order(g,g.epoch))
            stats=g.state['training_capture_runs'][-1]
            self.assertEqual(stats['GET_attempts'],1)
            self.assertEqual(stats['deferred_slots'],1)
            self.assertEqual(stats['transient_failures'],1)
            self.assertEqual(g.state['rejections'],{})
            self.assertEqual(g.bucket.capacity_puts,[])

    def test_actual_gateway_open_signed_commitments_freeze_and_both_wals(self):
        with tempfile.TemporaryDirectory() as root:
            g=storage.Gateway.__new__(storage.Gateway)
            g.bucket=Bucket();g.bucket.upload_at=1005;g.direct_r2=True
            g.epochs={};g.lock=threading.RLock();g.secret=bytes(range(32));g.state_path=Path(root)/'gateway.json'
            identities=[storage.Identity(bytes([i])*32)for i in (1,2)]
            manifest=dict(epoch='fixture-151',checkpoint={'id':'a'*64},source_bundle={'sha256':'b'*64},
                submission_transport_policy=commitments.VERSION2)
            binding=dict(version=commitments.VERSION2,checkpoint='a'*64,source='b'*64,max_batches=9,
                freeze_until=1080,learner_capture_policy=policy(),training_input_policy='committed-unaudited-training-v1')
            with patch.object(time,'time',return_value=1000):
                capabilities=g.open(manifest['epoch'],[i.id for i in identities],1020,commitment_binding=binding)
            for identity in identities:
                decoded=identity.decrypt(capabilities[identity.id])
                self.assertEqual(len(decoded['training_put_urls']),9)
                batches=[(dict(env_id='math',index=i,tokens=[1,2,3]),bytes([i+1]))for i in range(9)]
                envelope=commitments.make(identity,manifest,batches)
                g.bucket.data[f'private/{manifest["epoch"]}/commitments/{identity.id}.json']=td.canonical(envelope)
                for i,(batch,_)in enumerate(batches):
                    g.bucket.data[f'private/{manifest["epoch"]}/training/{identity.id}/{i}.json']=td.document(batch,manifest,identity.id,i)
            original=td.capture
            def ordered(gateway,epoch):return original(gateway,epoch,ordering=capacity.prepare_order(gateway,epoch))
            with patch.object(time,'time',return_value=1021),patch.object(td,'capture',ordered):
                receipts=g.capture_learner(manifest['epoch'])
            self.assertEqual(len(receipts),2)
            self.assertEqual(sum(len(v['training_documents'])for v in receipts.values()),18)
            journals=list((Path(root)/'capture-journals').glob('*.jsonl'))
            self.assertEqual(len(journals),2)
            records=[[json.loads(line)['payload']['kind']for line in p.read_bytes().splitlines()]for p in journals]
            self.assertIn(['header','commitment','commitment'],records)
            self.assertEqual(sum(k=='commit'for kinds in records for k in kinds),18)
            self.assertEqual(g.epochs[manifest['epoch']]['training_capture_runs'][-1]['policy'],policy())
            self.assertEqual(len(g.bucket.capacity_puts),18)


if __name__=='__main__':unittest.main()
