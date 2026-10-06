import copy,io,json,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import test_forced_sampling as fixture
import test_compact_threeway_sampling as compact
from subnet import token_only_protocol as p,token_only_runtime as v
from subnet import forced_sampling as f
from subnet.audit_policy import InvalidSample
from subnet.fast_prefill_audit import NumericalAmbiguity
from subnet.artifact_budget import LEGACY

class TokenProductionControls(unittest.TestCase):
    def setUp(self):
        self.r,self.m=compact.CompactThreeway().runtime()
        self.m.pop('probability_artifact_policy');self.r.probability_artifact_policy=None
        self.m.update(token_artifact_policy=dict(p.POLICY),submission_transport_policy=p.TRANSPORT)
        p.bind_runtime(self.r,self.m)
    def generate(self,seed=0):
        with patch('subnet.model.create_session',return_value=fixture.Session()),patch.object(self.r,'compute',side_effect=AssertionError('no extra LP forward')),patch.object(self.r,'build_proofs',side_effect=AssertionError('no TOPLOC construction')):
            return self.r.rollout(2,seed)
    def verify(self,rollout):
        with patch('subnet.environments.create_session',return_value=fixture.Session()):return v.verify(self.r,self.m,rollout,eligible_indices={2})
    def batch(self,rollout):return dict(rollouts=[rollout],env_id='tiny',index=2)
    def test_actual_prescribed_generation_omits_extra_compute_and_upload_claims(self):
        rollout,arrays=self.generate();self.assertEqual(arrays,[]);self.assertNotIn('proofs',rollout['turns'][0]);self.assertEqual(rollout['sampling'],f.receipt(f.binding(self.m),0))
        before=self.r.model.calls;self.assertTrue(self.verify(rollout)['valid']);self.assertEqual(self.r.model.calls-before,1)
    def test_wrong_epoch_checkpoint_source_contract_rejected_before_generation(self):
        for key,value in [('epoch','wrong'),('checkpoint',{'id':'d'*64}),('sampling_source_hash','d'*64),('sampling_contract',{})]:
            changed=copy.deepcopy(self.m);changed[key]=value;self.r.token_artifact_manifest=changed;before=self.r.model.calls
            with self.assertRaises(ValueError):self.generate()
            self.assertEqual(self.r.model.calls,before)
        self.r.token_artifact_manifest=self.m
    def test_codec_has_distinct_stable_framing_and_no_legacy_tensor_entries(self):
        rollout,arrays=self.generate();batch=self.batch(rollout);data=p.pack([(batch,[arrays])],budget=LEGACY)
        self.assertEqual(data,p.pack([(batch,[arrays])],budget=LEGACY));self.assertEqual(p.unpack(data,budget=LEGACY,max_batches=1),[(batch,[[]])])
        from subnet.batches import unpack
        with self.assertRaises(KeyError):unpack(data)
        legacy=io.BytesIO()
        with zipfile.ZipFile(legacy,'w')as z:z.writestr('manifest.json','[]')
        with self.assertRaises(ValueError):p.unpack(legacy.getvalue(),budget=LEGACY,max_batches=1)
    def test_forged_toploc_or_probability_claim_refused_in_new_codec(self):
        rollout,_=self.generate()
        for key in ['proofs','probabilities','logprobs']:
            wrong=copy.deepcopy(rollout);wrong['turns'][0][key]=[]
            with self.assertRaises(ValueError):p.pack([(self.batch(wrong),[[]])],budget=LEGACY)
        with self.assertRaises(ValueError):p.pack([(self.batch(rollout),[[[1.0]]])],budget=LEGACY)
    def test_extra_zip_members_and_noncanonical_document_refused(self):
        rollout,_=self.generate();data=p.pack([(self.batch(rollout),[[]])],budget=LEGACY)
        with zipfile.ZipFile(io.BytesIO(data))as z:raw=z.read('tokens.json')
        for names in [[('tokens.json',raw),('other',b'x')],[('tokens.json',raw+b' ')]]:
            stream=io.BytesIO()
            with zipfile.ZipFile(stream,'w')as z:
                for name,body in names:z.writestr(name,body)
            with self.assertRaises(ValueError):p.unpack(stream.getvalue(),budget=LEGACY,max_batches=1)
    def test_policy_absent_preserves_default_and_v3_requires_new_policy(self):
        self.assertIsNone(p.for_manifest({}));wrong=copy.deepcopy(self.m);wrong.pop('token_artifact_policy')
        with self.assertRaises(ValueError):p.for_manifest(wrong)
        for key,value in [('probability_artifact_policy',{'version':'selected-token-logprobs-v1'}),('submission_transport_policy','small-commitment-pairs-v2')]:
            wrong=copy.deepcopy(self.m);wrong[key]=value
            with self.assertRaises(ValueError):p.for_manifest(wrong)
    def test_receipt_task_seed_binding_and_native_failure_reject(self):
        rollout,_=self.generate()
        for key,value in [('index',3),('seed',128),('task_hash','d'*64),('sampling',{})]:
            wrong=copy.deepcopy(rollout);wrong[key]=value
            with self.assertRaises(InvalidSample):self.verify(wrong)
        wrong=copy.deepcopy(rollout);wrong['reward']=1.-rollout['reward']
        with self.assertRaises(InvalidSample):self.verify(wrong)
    def test_unknown_native_continues_and_invalid_dominates(self):
        rollout,_=self.generate()
        with patch('subnet.threeway_prefill_research.verify_sampling',side_effect=NumericalAmbiguity('control')):
            with self.assertRaises(NumericalAmbiguity)as e:self.verify(rollout)
            self.assertTrue(e.exception.environment_verification_complete)
            wrong=copy.deepcopy(rollout);wrong['reward']=1.-rollout['reward']
            with self.assertRaises(InvalidSample):self.verify(wrong)
    def test_duplicate_same_task_pair_cannot_gain_quota(self):
        rollout,_=self.generate();self.m.update(K=1,L=1)
        with self.assertRaisesRegex(InvalidSample,'duplicate'):v.verify_pair(self.r,self.m,[rollout,copy.deepcopy(rollout)],eligible_indices={2})
    def test_native_scope_requires_explicit_signed_job_policy(self):
        self.assertEqual(p.prepare_native_validations({},self.m,'a'*64),{})
        with self.assertRaises(ValueError):p.prepare_native_validations({'native_source_validation_scopes':{}},self.m,'a'*64)
        self.m['native_source_validation_policy']=dict(p.NATIVE_POLICY)
        with self.assertRaises(ValueError):p.prepare_native_validations({},self.m,'a'*64)

class TokenBackendControls(unittest.TestCase):
    def setUp(self):
        import test_v4_invalid_dominates_unknown as old
        self.old=old;case=old.InvalidDominates();case.setUp();self.r,self.m=case.r,case.m
        self.m.pop('probability_artifact_policy');self.r.probability_artifact_policy=None
        self.m.update(token_artifact_policy=dict(p.POLICY),submission_transport_policy=p.TRANSPORT)
        p.bind_runtime(self.r,self.m)
        rolls=[v.document(case.pos[0]),v.document(case.neg[0])]
        self.batch=dict(schema=2,epoch=self.m['epoch'],checkpoint=self.m['checkpoint']['id'],env_id='tiny',environment_version='v1',index=2,sample_index=2,rollouts=rolls)
        self.m['audit_policy']={'mode':'full'}
    def run_batch(self,batch=None,effects=None):
        from subnet.backend_jobs import audit
        import hashlib
        batch=batch or self.batch;data=p.pack([(batch,[[],[]])],budget=LEGACY);miner='e'*64
        self.m['audit_frozen_receipts']={miner:{'artifacts':[dict(sha256=hashlib.sha256(data).hexdigest(),batch_sha256=hashlib.sha256(p.canonical(batch)).hexdigest())]}}
        with patch.object(self.r,'for_environment',return_value=self.r),patch('subnet.environments.create_session',side_effect=lambda *args,**kw:self.old.PairSession()),patch('subnet.threeway_prefill_research.verify_sampling',side_effect=effects):
            return audit(data,self.m,self.r,commitment_miner=miner)
    def test_real_backend_unknown_has_no_credit_and_checks_native_both(self):
        report,pairs=self.run_batch(effects=[NumericalAmbiguity('control'),None]);self.assertEqual(report['accepted'],[]);self.assertEqual(pairs,[]);o=report['outcomes'][0]
        self.assertIsNone(o['valid']);self.assertEqual(o['failure_kind'],'numerical_ambiguous');self.assertTrue(o['environment_verification_complete'])
    def test_real_backend_later_invalid_dominates_unknown(self):
        before=self.r.model.calls;report,pairs=self.run_batch(effects=[NumericalAmbiguity('control'),InvalidSample('CDF interval outside')]);self.assertEqual(self.r.model.calls-before,2)
        self.assertFalse(report['outcomes'][0]['valid']);self.assertEqual(report['outcomes'][0]['failure_kind'],'confirmed_invalid');self.assertEqual(report['accepted'],[]);self.assertEqual(pairs,[])
    def test_real_backend_valid_pair_and_frozen_receipt_binding(self):
        report,pairs=self.run_batch();self.assertTrue(report['outcomes'][0]['valid']);self.assertEqual(len(pairs),1)
        from subnet.backend_jobs import audit
        data=p.pack([(self.batch,[[],[]])],budget=LEGACY);self.m['audit_frozen_receipts']={}
        before=self.r.model.calls;report,pairs=audit(data,self.m,self.r,commitment_miner='e'*64);self.assertEqual(self.r.model.calls,before);self.assertEqual(report['accepted'],[])
    def test_signed_new_commitment_binds_distinct_cheap_token_document(self):
        from subnet.storage import Identity
        from subnet import commitment_transport as c,training_documents as d
        import hashlib
        self.m['source_bundle']={'sha256':'b'*64};identity=Identity();data=c.pair_artifact(self.batch,[[],[]],self.m);c.check_prepared_cumulative([(self.batch,data)],self.m,1)
        value=c.make(identity,self.m,[(self.batch,data)]);envelope=c.validate(c.canonical(value),self.m['epoch'],identity.id);self.assertEqual(envelope['payload']['version'],p.TRANSPORT)
        entry=envelope['payload']['batches'][0];body=d.document(self.batch,self.m,identity.id,0)
        document=d.validate(body,self.m['epoch'],self.m['checkpoint']['id'],identity.id,entry,transport=p.TRANSPORT);self.assertEqual(document['version'],d.TOKEN_VERSION)
        with self.assertRaises(ValueError):d.validate(body,self.m['epoch'],self.m['checkpoint']['id'],identity.id,entry)

class TokenNativeAdmissionControls(unittest.TestCase):
 def test_signed_policy_prepares_real_job_cache_and_replays_fresh_native_outcomes(self):
  import base64,hashlib
  import test_native_session_validation as native
  from subnet import environments as e,harness
  case=native.NativeValidationControls();case.setUp();self.addCleanup(case.doCleanups)
  r,m=compact.CompactThreeway().runtime();m.pop('probability_artifact_policy');m.update(token_artifact_policy=dict(p.POLICY),submission_transport_policy=p.TRANSPORT,native_source_validation_policy=dict(p.NATIVE_POLICY),harness_source_hash=harness.source_hash(),environments=[dict(env_id=case.spec.id,spec=case.spec.to_dict(),indices=[0,1],harness=r.harness)])
  payload=case.scope;signed=dict(payload=payload,signer=case.key.verify_key.encode().hex(),signature=base64.b64encode(case.key.sign(p.canonical(payload)).signature).decode())
  job=dict(job_id='CPU-control',source_files={'subnet/native_session_validation.py':hashlib.sha256(Path(native.v.__file__).read_bytes()).hexdigest()},native_source_validation_scopes={case.spec.id:signed})
  with patch.object(e,'_source_hash',wraps=e._source_hash)as hashing:
   caches=p.prepare_native_validations(job,m,signed['signer'])
   r.native_source_validations=caches;chosen=r.for_environment(case.spec.to_dict(),r.harness)
   self.assertIs(chosen.native_source_validation,caches[case.spec.id])
   for index,answer,label in [(0,'2','positive'),(1,'2','negative'),(0,'9','negative')]:
    session=e.create_session(chosen.spec,source_validation=chosen.native_source_validation)
    try:
     observation=session.reset(index,17);self.assertEqual(observation['task_name'],'control-'+str(index));self.assertEqual(session.turns,0)
     self.assertEqual(session.step({'text':r'\boxed{'+answer+'}'})['classification'],label)
    finally:session.close()
   self.assertEqual(hashing.call_count,1)
