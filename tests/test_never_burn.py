import copy, importlib.util, json, math, signal, tempfile, time, unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from nacl.signing import SigningKey
from nacl.exceptions import BadSignatureError
from ops import current_assessment_writer as w
from ops import live_reward_writer as legacy
from ops.live_reward_exporter import sign
from subnet.current_assessment import calculate, recipients
from subnet.chain import ChainAdapter, OWNER

class Writer(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        self.key=SigningKey.generate();self.auth=self.key.verify_key.encode().hex()
        seed=self.path/'seed';seed.write_text(self.key.encode().hex());seed.chmod(0o600)
        self.c=dict(global_lock_path=str(self.path/'lock'),reward_state=str(self.path),chain_state=str(self.path),authority_seed_file=str(seed))
        self.cutover=sign({'original':True},self.key);self.anchor=sign({'original':True},self.key)
        modules=['ops.current_assessment_writer','subnet.numerical_resolution','subnet.continuous_audit_policy','ops.current_assessment_evidence','subnet.current_assessment','subnet.chain','ops.live_reward_writer']
        pins={str(Path(importlib.util.find_spec(m).origin).resolve()):None for m in modules}
        pins={p:w.file_hash(p) for p in pins}
        self.body=dict(version=w.NEVER_BURN_VERSION,half_life_hours=6,first_window=3600,netuid=120,owner_hotkey=OWNER,audit_config='/not-training',source_admission_sha256='a'*64,numerical_resolution_policy_sha256='b'*64,verifiers=['v'],module_hashes=pins,cutover_sha256=w.sha(self.cutover),anchor_sha256=w.sha(self.anchor),execute_enabled=True,zero_total_policy='no-owner-retain-v1',registration_change_policy='current-hotkey-snapshot-v1',fallback_assessments=[])
        self.policy=sign(self.body,self.key);self.calls=[];self.status='planned';self.regs={'hotkey':{'public_key':'miner','uid':85}};controls=self
        class Adapter:
            def __init__(self,*a,**kw):pass
            def registrations(self):return controls.regs
            def submit_hour(self,points,regs,end,execute=False,**kw):
                controls.calls.append((points,regs,end,execute,kw))
                if isinstance(controls.status,Exception):raise controls.status
                return dict(status=controls.status,window_end=end)
        self.adapter=Adapter
    def evidence(self,cutoff=7200,miners=None,empty=False):
        miners=miners if miners is not None else {'miner':dict(unique_eligible_batches=3,validity_probability=.5,reward_multiplier=1.,blacklisted=False)}
        return dict(snapshots=[] if empty else [dict(epoch='epoch',round=13,cutoff=cutoff,miners=miners)],committed_at_by_epoch={} if empty else {'epoch':7000},evidence_hashes={'source_admission_sha256':'a'*64,'numerical_resolution_policy_sha256':'b'*64})
    def invoke(self,evidence=None,now=7300,execute=False):
        def load(*a,**k):
            if isinstance(evidence,Exception):raise evidence
            return self.evidence(cutoff=int(now)//3600*3600) if evidence is None else evidence
        with patch.object(w,'authenticate_cutover',return_value=(self.c,{})),patch.object(w,'global_lock',return_value=nullcontext()),patch.object(w,'guard_files'),patch.object(w,'observe_units',return_value=[]),patch.object(w,'writer_gate'),patch.object(w.time,'time',return_value=now):
            return w.run_once(self.policy,self.cutover,self.anchor,self.auth,execute=execute,adapter_factory=self.adapter,evidence_loader=load)
    def saved(self,name='last-run.json'):
        return json.loads((self.path/w.ASSESSMENT_DIRECTORY/name).read_text())
    def test_fresh_current_registered_without_training(self):
        self.invoke();self.assertEqual(self.calls[0][0],{'hotkey':1_000_000_000});self.assertEqual(self.calls[0][1]['hotkey']['uid'],85)
    def test_error_fallback_keeps_last_valid(self):
        self.invoke()
        for i,error in enumerate((TimeoutError(),ConnectionError(),ValueError('bounded assessment input exceeded'),BadSignatureError('fresh invalid signature'))):
            with self.subTest(error=type(error).__name__):
                self.invoke(error,now=11000+i*3600)
                self.assertEqual(self.calls[-1][0],self.calls[0][0]);self.assertTrue(self.saved()['assessment_stale'])
                self.assertEqual(self.saved('assessment-'+str(10800+i*3600)+'.json')['payload']['evidence_error'],type(error).__name__)
    def test_malformed_fresh_uses_only_signed_last_good(self):
        self.invoke();self.invoke({'snapshots': 'not-valid'},now=11000)
        self.assertTrue(self.saved()['assessment_stale']);self.assertEqual(self.calls[0][0],self.calls[-1][0])
    def test_empty_no_evidence_fallback(self):
        self.invoke();self.invoke(self.evidence(10800,empty=True),now=11000)
        self.assertEqual(self.calls[0][0],self.calls[1][0]);self.assertTrue(self.saved()['assessment_stale'])
    def test_blacklist_never_revived_on_empty_or_error(self):
        self.invoke();bad={'miner':dict(unique_eligible_batches=3,validity_probability=.5,reward_multiplier=0.,blacklisted=True)}
        result=self.invoke(self.evidence(10800,miners=bad),now=11000)
        self.assertEqual(result['status'],'no_valid_registered_recipients');self.assertEqual(len(self.calls),1)
        self.invoke(ValueError('reader unavailable'),now=14600)
        self.assertEqual(len(self.calls),1);self.assertTrue(self.saved()['result']['retained_onchain_weights'])
    def test_new_partial_penalty_applied_to_history(self):
        self.regs['other']={'public_key':'other','uid':86}
        m={v:dict(unique_eligible_batches=3,validity_probability=1.,reward_multiplier=1.) for v in ('miner','other')}
        self.invoke(self.evidence(miners=m));m['miner']['unique_eligible_batches']=0;m['other']['unique_eligible_batches']=0;m['miner']['reward_multiplier']=.1
        self.invoke(self.evidence(10800,miners=m),now=11000)
        p=self.calls[-1][0];self.assertAlmostEqual(p['hotkey']/p['other'],.1,places=7)
    def test_invalid_fresh_assessment_not_used(self):
        self.invoke();e=self.evidence(10800);e['evidence_hashes']['source_admission_sha256']='c'*64
        self.invoke(e,now=11000);self.assertTrue(self.saved()['assessment_stale']);self.assertEqual(self.calls[0][0],self.calls[-1][0])
    def test_no_history_error_retains_chain_without_call(self):
        r=self.invoke(ValueError('bounded assessment input exceeded'));self.assertEqual(r['status'],'no_valid_assessment');self.assertTrue(r['retained_onchain_weights']);self.assertEqual(self.calls,[])
    def test_no_history_empty_retains_chain(self):
        r=self.invoke(self.evidence(empty=True),execute=True);self.assertEqual(r['status'],'no_valid_registered_recipients');self.assertEqual(self.calls,[])
    def test_tampered_history_rejected(self):
        self.invoke()
        for name in ('last-valid-assessment.json','last-positive-assessment.json'):
            p=self.path/w.ASSESSMENT_DIRECTORY/name;d=json.loads(p.read_text());d['payload']['points']['miner']=100;p.write_text(json.dumps(d))
        r=self.invoke(TimeoutError(),now=11000);self.assertEqual(r['status'],'no_valid_assessment');self.assertEqual(len(self.calls),1)
    def test_bootstrap_binding_and_corruption(self):
        a=calculate(self.evidence()['snapshots'],{'epoch':7000},7200);a.update(evidence_hashes=self.evidence()['evidence_hashes'],writer_policy_sha256='e'*64,assessment_stale=False)
        p=self.path/'bootstrap.json';p.write_text(json.dumps(sign(a,self.key)))
        self.body['fallback_assessments']=[dict(path=str(p),sha256=w.file_hash(p),writer_policy_sha256='e'*64)];self.policy=sign(self.body,self.key)
        self.invoke(TimeoutError(),now=11000);self.assertEqual(len(self.calls),1)
        p.write_text(p.read_text()+' ');r=self.invoke(TimeoutError(),now=14600);self.assertEqual(r['status'],'no_valid_assessment');self.assertEqual(len(self.calls),1)
    def test_shared_uncertain_old_cursor_blocks_new_directory(self):
        p=self.path/'current-assessment-v1';p.mkdir();(p/'submission.json').write_text(json.dumps({'status':'submitting'}))
        with self.assertRaisesRegex(RuntimeError,'uncertain'):self.invoke(execute=True)
        self.assertEqual(self.calls,[])
    def test_uncertain_chain_error_stays_fenced(self):
        self.status=OSError('after submission')
        with self.assertRaises(OSError):self.invoke(execute=True)
        with self.assertRaisesRegex(RuntimeError,'uncertain'):self.invoke(execute=True)
        self.assertEqual(len(self.calls),1)
    def test_hour_idempotency_and_safe_rate_retry(self):
        self.status='deferred_rate_limit';self.invoke(execute=True);self.invoke(ValueError('not reread'),execute=True);self.assertEqual(self.calls[0],self.calls[1])
        (self.path/'weights.json').write_text(json.dumps({'last_submitted_window':7200}));self.assertEqual(self.invoke()['status'],'already_submitted');self.assertEqual(len(self.calls),2)
    def test_departed_all_and_owner_only_never_submit(self):
        self.regs={OWNER:{'public_key':'miner','uid':0}}
        self.assertEqual(self.invoke()['status'],'no_valid_registered_recipients');self.assertEqual(self.calls,[])
    def test_old_owner_policy_rejected(self):
        self.body['zero_total_policy']='owner-sink-v1';self.policy=sign(self.body,self.key)
        with self.assertRaises(ValueError):self.invoke()
    def test_tiny_positive_normalized(self):
        p,r,e=recipients({'points':{'miner':1e-300,'other':1e-320}}, {'hk':{'public_key':'miner','uid':12},'hk2':{'public_key':'other','uid':3}})
        self.assertGreater(p['hk'],0);self.assertGreater(p['hk2'],0)

    def make_cache(self, cutoff=7200, producer='d'*64, miners=None):
        e=self.evidence(cutoff,miners=miners)
        a=calculate(e['snapshots'],e['committed_at_by_epoch'],cutoff)
        a.update(evidence_cutoff=cutoff,assessment_stale=False,evidence_hashes=e['evidence_hashes'],writer_policy_sha256=producer)
        path=self.path/'producer-cache.json';doc=sign(a,self.key);path.write_text(json.dumps(doc))
        self.body['authenticated_assessment_sources']=[dict(path=str(path),writer_policy_sha256='d'*64)]
        self.policy=sign(self.body,self.key)
        return path,doc
    def test_current_authenticated_cache_avoids_calculation(self):
        path,doc=self.make_cache()
        self.invoke(AssertionError('loader must not be called'))
        result=self.saved('assessment-7200.json')['payload']
        self.assertFalse(result['assessment_stale'])
        self.assertEqual(result['authenticated_source']['original_envelope_sha256'],w.sha(doc))
        self.assertEqual(result['writer_policy_sha256'],w.sha(self.policy))
    def test_older_cache_is_authenticated_fallback(self):
        path,doc=self.make_cache()
        self.invoke(ValueError('bounded assessment input exceeded'),now=11000)
        result=self.saved('assessment-10800.json')['payload']
        self.assertTrue(result['assessment_stale']);self.assertEqual(result['evidence_cutoff'],7200)
        self.assertEqual(result['authenticated_source']['original_envelope_sha256'],w.sha(doc))
    def test_wrong_producer_cache_is_not_used_or_a_gate(self):
        self.make_cache(producer='e'*64);self.invoke()
        a=self.saved('assessment-7200.json')['payload'];self.assertFalse(a['assessment_stale']);self.assertNotIn('authenticated_source',a)
        self.assertEqual(len(a['fallback_history_refusals']),1)
    def test_invalid_signature_cache_does_not_gate_calculation(self):
        path,doc=self.make_cache();doc['payload']['points']['miner']=99;path.write_text(json.dumps(doc))
        self.invoke();a=self.saved('assessment-7200.json')['payload']
        self.assertFalse(a['assessment_stale']);self.assertNotIn('authenticated_source',a);self.assertEqual(len(a['fallback_history_refusals']),1)
    def test_future_cache_cannot_replace_current_evidence(self):
        self.make_cache(cutoff=10800);self.invoke()
        a=self.saved('assessment-7200.json')['payload'];self.assertNotIn('authenticated_source',a);self.assertEqual(a['cutoff'],7200)
    def test_missing_optional_cache_is_not_a_gate(self):
        path,_=self.make_cache();path.unlink();self.invoke();self.assertEqual(len(self.calls),1);self.assertFalse(self.saved()['assessment_stale'])
    def test_current_cache_zero_preserves_latest_penalties(self):
        self.make_cache();self.invoke()
        bad={'miner':dict(unique_eligible_batches=3,validity_probability=.5,reward_multiplier=0.,blacklisted=True)}
        self.make_cache(10800,miners=bad)
        r=self.invoke(AssertionError('current zero cache needs no calculation'),now=11000)
        self.assertEqual(r['status'],'no_valid_registered_recipients');self.assertEqual(len(self.calls),1)
    def test_incompatible_cache_ema_is_not_used(self):
        path,doc=self.make_cache();doc['payload']['hourly_alpha']=.9;doc=sign(doc['payload'],self.key);path.write_text(json.dumps(doc))
        self.invoke();a=self.saved('assessment-7200.json')['payload'];self.assertNotIn('authenticated_source',a)

class Chain(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        a=ChainAdapter.__new__(ChainAdapter);a.state_dir=Path(self.temp.name);a.netuid=120;a.owner=OWNER
        a.bt=SimpleNamespace(Wallet=lambda **k:SimpleNamespace(hotkey=SimpleNamespace(ss58_address=OWNER)),SetWeights=lambda **k:k)
        a.chain=SimpleNamespace(block=100,plan=lambda intent,wallet:SimpleNamespace(ok=True));a.registrations=lambda:{}
        a.query=lambda n,p,b:{'SubnetOwnerHotkey':OWNER,'Uids':0,'Keys':OWNER,'LastUpdate':[0],'WeightsSetRateLimit':0,'WeightsVersionKey':0}[n]
        self.a=a;self.end=int(time.time())//3600*3600
    def submit(self,points={},before={}):return self.a.submit_hour(points,before,self.end,zero_total_policy='no-owner-retain-v1',registration_change_policy='current-hotkey-snapshot-v1')
    def test_empty_preserves_chain(self):
        r=self.submit();self.assertEqual(r['status'],'zero_points_no_submission');self.assertNotIn('uids',r);self.assertTrue(r['retained_onchain_weights'])
    def test_owner_rejected_even_forged_registration(self):
        self.a.registrations=lambda:{OWNER:{'public_key':'owner','uid':0,'snapshot_block':99}}
        with self.assertRaisesRegex(ValueError,'owner'):self.submit({OWNER:1},{OWNER:{'uid':0,'public_key':'owner'}})
    def test_current_hotkey_remapped_departed_excluded(self):
        self.a.registrations=lambda:{'live':{'public_key':'live','uid':90,'snapshot_block':99}}
        r=self.submit({'gone':100,'live':3},{'gone':{'uid':85,'public_key':'gone'},'live':{'uid':89,'public_key':'live'}})
        self.assertEqual(r['uids'],[90]);self.assertEqual(r['weights'],[1.]);self.assertEqual(r['excluded_unregistered'],['gone']);self.assertEqual(r['remapped_uids'][0]['to_uid'],90)
    def test_foreign_identity_rejected(self):
        self.a.registrations=lambda:{'live':{'public_key':'wrong','uid':90,'snapshot_block':99}}
        with self.assertRaises(ValueError):self.submit({'live':3},{'live':{'uid':90,'public_key':'live'}})
    def test_rate_limit_preserved(self):
        self.a.registrations=lambda:{'live':{'public_key':'live','uid':90,'snapshot_block':99}};old=self.a.query;self.a.query=lambda n,p,b:200 if n=='WeightsSetRateLimit' else old(n,p,b)
        self.assertEqual(self.submit({'live':1},{'live':{'uid':90,'public_key':'live'}})['status'],'deferred_rate_limit')

class TimingAndGuards(unittest.TestCase):
    def test_evidence_timer_expires_and_restores_outer_deadline(self):
        old=signal.getsignal(signal.SIGALRM);signal.signal(signal.SIGALRM,lambda *_:None);signal.setitimer(signal.ITIMER_REAL,2)
        try:
            with self.assertRaises(TimeoutError):
                with w.evidence_budget(.01):time.sleep(.1)
            left=signal.getitimer(signal.ITIMER_REAL)[0];self.assertGreater(left,1.5);self.assertLess(left,2)
        finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
    def test_masked_inactive_accepted_not_found_active_refused(self):
        def result(load,active,file):return lambda *a,**k:SimpleNamespace(returncode=0,stdout=f'LoadState={load}\nActiveState={active}\nUnitFileState={file}\n')
        self.assertEqual(len(legacy.observe_units(result('masked','inactive','masked'))),4)
        for fields in [('not-found','inactive','masked'),('masked','active','masked'),('masked','inactive','disabled')]:
            with self.assertRaises(Exception):legacy.observe_units(result(*fields))

if __name__=='__main__':unittest.main()
