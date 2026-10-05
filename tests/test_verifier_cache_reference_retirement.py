import json,time,unittest
from nacl.signing import SigningKey
import base64
from subnet.storage import canonical
from subnet.distributed_roles import digest
from ops.verifier_cache_reference_retirement import proof_for_replaced_reference,remaining_queue_references

class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex();self.workerkey=SigningKey.generate();self.worker=self.workerkey.verify_key.encode().hex();self.now=time.time();self.cp='a'*64
        manifest=dict(epoch='epoch-7',checkpoint=dict(id=self.cp,files={}),backend_profile={},numerical_policy={})
        common=dict(role='verify',manifest=self.sign(manifest),source_files={'subnet/gpu_runtime.py':'b'*64},runtime_versions={'torch':'2.14.0'},submissions=[dict(sha256='c'*64)])
        self.old=dict(common,job_id='old',created_at=self.now-1000,expires_at=self.now-100)
        self.new=dict(common,job_id='new',created_at=self.now-90,expires_at=self.now+1000)
        self.original=dict(id='old',envelope=json.dumps(self.sign(self.old)),digest=digest(self.old),status='failed',attempt=3,expires=self.old['expires_at'],report=None)
        report=dict(job_id='new',job_sha256=digest(self.new),operator=self.authority,role='verify',epoch='epoch-7',checkpoint=self.cp,source_files=common['source_files'],runtime_versions=common['runtime_versions'],backend_profile={},numerical_policy={},chain_transactions=False,success=True,completed_at=self.now-10,audits=[dict(epoch='epoch-7',submission_sha256='c'*64,accepted=[])])
        request=dict(action='report',job_id='new',token='d'*64,report=report)
        self.replacement=dict(id='new',envelope=json.dumps(self.sign(self.new)),digest=digest(self.new),status='complete',worker=self.worker,token='d'*64,report=json.dumps(report),report_digest=digest(report),report_request=json.dumps(self.sign(request,self.workerkey)))
        self.decision=self.sign(dict(schema='terminal-infrastructure-replacement-v1',original_job_id='old',original_job_sha256=digest(self.old),original_status='failed',original_attempts=3,replacement_job_id='new',replacement_job_sha256=digest(self.new),checkpoint=self.cp,original_expiry_unchanged=True,expired_leases_extended=False,frozen_inputs_unchanged=True))
        proof=proof_for_replaced_reference(self.original,self.replacement,self.decision,self.authority,now=self.now,max_attempts=3)
        self.documents={'private/history/old/job.json':canonical(json.loads(self.original['envelope'])),'private/history/new/job.json':canonical(json.loads(self.replacement['envelope'])),'private/history/new/report.json':canonical(report),'private/history/new/worker-report.json':canonical(json.loads(self.replacement['report_request']))}
        import hashlib
        self.entry=dict(proof,decision=self.decision,canonical_r2_documents={k:hashlib.sha256(v).hexdigest()for k,v in self.documents.items()})
    def sign(self,payload,key=None):
        key=key or self.key;return dict(payload=payload,signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())
    def ledger(self):return self.sign(dict(authority=self.authority,schema='verifier-cache-reference-retirements-v1',scope='named-terminal-queue-cache-references-only',history_prefix='private/history',entries=[self.entry]))
    def verify(self,rows=None,fetch=None):return remaining_queue_references(rows or[self.original,self.replacement],self.ledger(),self.authority,now=self.now,max_attempts=3,fetch_document=fetch or self.documents.__getitem__)
    def test_authenticates_exact_terminal_reference_without_row_mutation(self):
        before=canonical([self.original,self.replacement]);r=self.verify();self.assertEqual(r['retired_reference_ids'],{'old'});self.assertEqual(r['protected_checkpoints'],set());self.assertEqual(before,canonical([self.original,self.replacement]))
    def test_queued_leased_and_new_requests_reprotect(self):
        for status in('queued','leased'):
            a=dict(self.original,status=status);r=self.verify([a,self.replacement]);self.assertEqual(r['protected_checkpoints'],{self.cp});self.assertEqual(r['retired_reference_ids'],set())
        job=dict(self.new,job_id='another');row=dict(self.original,id='another',status='queued',envelope=json.dumps(self.sign(job)),digest=digest(job));self.assertEqual(self.verify([self.original,self.replacement,row])['protected_checkpoints'],{self.cp})
    def test_wrong_decision_or_worker_signature_refuses(self):
        self.entry['decision']=self.sign(dict(self.decision['payload'],checkpoint='f'*64))
        with self.assertRaises(ValueError):self.verify()
        self.entry['decision']=self.decision
        req=json.loads(self.replacement['report_request'])['payload'];self.replacement['report_request']=json.dumps(self.sign(req,SigningKey.generate()))
        with self.assertRaises(ValueError):self.verify()
    def test_canonical_r2_tampering_missing_history_and_scope_refuse(self):
        with self.assertRaises(ValueError):self.verify(fetch=lambda key:b'{}')
        with self.assertRaises(ValueError):self.verify([self.replacement])
        self.entry['replacement_report_sha256']='0'*64
        with self.assertRaises(ValueError):self.verify()
    def test_expiry_exhaustion_and_bool_attempt_policy_refuse(self):
        for old,maximum in[(dict(self.original,attempt=2),3),(self.original,True)]:
            with self.assertRaises(ValueError):proof_for_replaced_reference(old,self.replacement,self.decision,self.authority,now=self.now,max_attempts=maximum)
        with self.assertRaises(ValueError):proof_for_replaced_reference(self.original,self.replacement,self.decision,self.authority,now=self.old['expires_at']-1,max_attempts=3)
if __name__=='__main__':unittest.main()
