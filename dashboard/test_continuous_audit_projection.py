import base64
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from nacl.signing import SigningKey
from dashboard import continuous_audit_projection as module
from dashboard.learner_projection import canonical, digest, project as learner_project


def sign(value, key):
    return dict(payload=value, signer=key.verify_key.encode().hex(),
                signature=base64.b64encode(key.sign(canonical(value)).signature).decode())


class ContinuousProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(); self.root = Path(self.temporary.name)
        self.operator = SigningKey.generate(); self.worker = SigningKey.generate(); self.miner = SigningKey.generate()
        self.authority = self.operator.verify_key.encode().hex(); self.worker_id = self.worker.verify_key.encode().hex()
        self.miner_id = self.miner.verify_key.encode().hex()
        self.manifest = dict(epoch='epoch94', checkpoint={'id': 'checkpoint94'},
            source_bundle={'sha256': 'source94'}, start=100, deadline=200,
            backend_profile={'dtype': 'bf16'}, numerical_policy={'atol': 0})
        self.batches = [dict(epoch='epoch94', checkpoint='checkpoint94', env_id='math',
                            index=100+i, rollouts=[{'sample': j} for j in range(8)]) for i in range(3)]
        self.children = [dict(slot=i, env_id='math', index=100+i, batch_sha256=digest(batch),
            sha256=digest({'proof': i}), training_sha256=digest({'training': i}), training_size=100)
            for i, batch in enumerate(self.batches)]
        commitment = sign(dict(version='small-commitment-pairs-v2', epoch='epoch94',
            checkpoint='checkpoint94', source='source94', miner=self.miner_id, batches=self.children), self.miner)
        self.commitment_hash = digest(commitment)
        record = dict(miner=self.miner_id, commitment_document=commitment, commitment_sha256=digest(commitment),
            training_documents=[dict(slot=c['slot'], sha256=c['training_sha256'], size=100) for c in self.children],
            training_document_deferred_slots=[])
        self.population = dict(version='committed-unaudited-training-v1', population=dict(
            assurance='unaudited', epoch='epoch94', checkpoint='checkpoint94', committed_inventory=[record],
            committed_count=3, eligible_count=0, eligible_inventory=[]), submissions=[])
        self.queue = self.root/'queue.sqlite3'
        with sqlite3.connect(self.queue) as db:
            db.executescript('''CREATE TABLE jobs(id TEXT PRIMARY KEY,digest TEXT,envelope TEXT,role TEXT,
                status TEXT,worker TEXT,token TEXT,report TEXT,report_digest TEXT,report_request TEXT);
                CREATE TABLE events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT,at REAL,kind TEXT,detail TEXT);''')
        self.counter = 0; self.projection = module.Projection(self.authority)

    def tearDown(self): self.temporary.cleanup()

    def row(self, slot=0, classification='accepted', *, manifest=None):
        self.counter += 1; identifier = 'continuous-audit-group-test-' + str(self.counter)
        manifest = copy.deepcopy(manifest or self.manifest); child = self.children[slot]
        ref = dict(miner=self.miner_id, commitment_sha256=self.commitment_hash,
                   slot=slot, batch_sha256=child['batch_sha256'], env_id='math', index=child['index'])
        job = dict(job_id=identifier, role='verify', manifest=sign(manifest, self.operator),
                   source_files={'subnet/test.py': 'pinned'}, runtime_versions={'torch': 'pinned'},
                   submissions=[dict(commitment_ref=ref, sha256=child['sha256'])])
        outcome = dict(env_id='math', index=child['index'], fully_audited=True, valid=True)
        accepted = [self.batches[slot]]
        if classification != 'accepted':
            accepted = []; outcome.update(valid=False, failure_kind='confirmed_invalid')
            if classification in ('ambiguous', 'infra'):
                outcome.update(valid=None, fully_audited=False,
                    failure_kind='numerical_ambiguous' if classification == 'ambiguous' else 'infrastructure_error')
        audit = dict(epoch=manifest['epoch'], submission_sha256=child['sha256'], accepted=accepted, outcomes=[outcome])
        report = dict(success=True, role='verify', operator=self.authority, job_id=identifier, job_sha256=digest(job),
            epoch=manifest['epoch'], checkpoint=manifest['checkpoint']['id'], source_files=job['source_files'],
            runtime_versions=job['runtime_versions'], backend_profile=manifest['backend_profile'],
            numerical_policy=manifest['numerical_policy'], execution_resources_enforced=True, audits=[audit])
        row = dict(id=identifier, digest=digest(job), envelope=sign(job, self.operator), role='verify',
            status='complete', worker=self.worker_id, token='PRIVATE-LEASE-TOKEN', report=report,
            report_digest=digest(report), report_request=sign(dict(action='report', job_id=identifier,
                token='PRIVATE-LEASE-TOKEN', report=report), self.worker))
        return row

    def resign_report(self, row):
        row['report_digest'] = digest(row['report'])
        row['report_request'] = sign(dict(action='report', job_id=row['id'], token=row['token'], report=row['report']), self.worker)

    def insert(self, row, kind='completed'):
        with sqlite3.connect(self.queue) as db:
            db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?)', tuple(
                json.dumps(row[k]) if isinstance(row[k], dict) else row[k] for k in (
                'id','digest','envelope','role','status','worker','token','report','report_digest','report_request')))
            db.execute('INSERT INTO events(job,at,kind,detail) VALUES(?,?,?,?)', (row['id'], 150, kind, '{}'))

    def result(self): return self.projection.project(self.queue, self.population, self.manifest)

    def assert_counts(self, accepted, rejected, unchecked):
        r = self.result(); self.assertEqual((r['accepted'],r['rejected'],r['unchecked']), (accepted,rejected,unchecked)); return r

    def test_real_completed_audit_updates_counts_without_relabeling_training_assurance(self):
        self.insert(self.row(0)); self.insert(self.row(1, 'rejected'))
        result = self.assert_counts(1,1,1)
        self.assertEqual(result['captured'], 3)
        self.assertEqual(self.population['population']['assurance'], 'unaudited')
        self.assertNotIn('PRIVATE', json.dumps(result))
        self.assertNotIn(self.miner_id, json.dumps(result))

    def test_duplicate_jobs_and_duplicate_completed_events_count_once(self):
        row = self.row(0); self.insert(row); self.insert(self.row(0))
        with sqlite3.connect(self.queue) as db:
            db.execute('INSERT INTO events(job,at,kind,detail) VALUES(?,?,?,?)', (row['id'], 151, 'completed', '{}'))
        self.assert_counts(1,0,2)

    def test_conflicting_conclusive_outcomes_remain_unchecked(self):
        self.insert(self.row(0)); self.insert(self.row(0, 'rejected'))
        self.assertEqual(self.assert_counts(0,0,3)['conflicts'], 1)

    def test_numerical_and_infrastructure_outcomes_do_not_become_rejections(self):
        self.insert(self.row(0, 'ambiguous')); self.insert(self.row(1, 'infra'))
        self.assert_counts(0,0,3)
        self.insert(self.row(0)); self.assert_counts(1,0,2)

    def test_inprogress_failed_and_nonverifier_jobs_are_excluded(self):
        for status in ('queued','leased','failed','expired'):
            row=self.row();row['status']=status;self.insert(row)
        row=self.row();row['role']='train';self.insert(row)
        self.assert_counts(0,0,3)

    def test_foreign_epoch_checkpoint_and_source_do_not_join(self):
        for field,value in [('epoch','other'),('checkpoint',{'id':'other'}),('source_bundle',{'sha256':'other'})]:
            manifest=dict(self.manifest, **{field:value});self.insert(self.row(manifest=manifest))
        self.assert_counts(0,0,3)

    def test_invalid_signatures_report_digest_and_terminal_lease_are_excluded(self):
        row=self.row();row['envelope']['signature']='AAAA';self.insert(row)
        row=self.row();row['report_request']['signature']='AAAA';self.insert(row)
        row=self.row();row['report_digest']='0'*64;self.insert(row)
        row=self.row();row['token']='different';self.insert(row)
        self.assert_counts(0,0,3)

    def test_authenticated_but_wrong_accepted_batch_or_source_report_is_excluded(self):
        row=self.row();row['report']['audits'][0]['accepted']=[dict(self.batches[0],index=999)]
        self.resign_report(row);self.insert(row)
        row=self.row();row['report']['source_files']={'bad':'pin'};self.resign_report(row);self.insert(row)
        self.assert_counts(0,0,3)

    def standard_backend(self):
        self.manifest['model_runtime_revision']='runtime94'
        row=self.row();job=row['envelope']['payload']
        job['source_files']={'subnet/backend_jobs.py':'pinned'}
        row['envelope']=sign(job,self.operator);row['digest']=digest(job)
        row['report'].update(job_sha256=digest(job),source_files=job['source_files'],execution_resources_enforced=False)
        self.resign_report(row)
        entry=dict(backend='standard-backend-no-os-resource-enforcement-v1',backend_module_sha256='pinned',
            model_runtime_revision='runtime94',backend_profile=self.manifest['backend_profile'],
            numerical_policy=self.manifest['numerical_policy'],runtime_versions=job['runtime_versions'],
            execution_resources_enforced=False)
        payload=dict(version='continuous-audit-service-sources-v1',approved_sources={'source94':job['source_files']},
            execution_evidence_policy=dict(version='explicit-backend-execution-evidence-v1',effective_cutoff=100,
                                           sources={'source94':entry}))
        return row,payload

    def test_unenforced_resources_require_exact_existing_ROOT_backend_admission(self):
        row,policy=self.standard_backend();self.insert(row)
        self.assert_counts(0,0,3)
        self.projection.configure_source_admission(sign(policy,self.operator));self.assert_counts(1,0,2)
        changed=copy.deepcopy(policy)
        changed['execution_evidence_policy']['sources']['source94']['backend_module_sha256']='different'
        self.projection.configure_source_admission(sign(changed,self.operator));self.assert_counts(0,0,3)

    def test_bad_source_admission_signature_and_future_policy_cannot_admit(self):
        import time
        from nacl.exceptions import BadSignatureError
        row,policy=self.standard_backend();self.insert(row)
        envelope=copy.deepcopy(sign(policy,self.operator));envelope['payload']['version']='changed'
        with self.assertRaises(BadSignatureError):self.projection.configure_source_admission(envelope)
        policy['execution_evidence_policy']['effective_cutoff']=time.time()+1000
        self.projection.configure_source_admission(sign(policy,self.operator));self.assert_counts(0,0,3)

    def test_v2_standard_backend_has_its_own_prospective_source_cutoff(self):
        import time
        row,policy=self.standard_backend();self.insert(row)
        policy['execution_evidence_policy']['version']='explicit-backend-execution-evidence-v2'
        entry=policy['execution_evidence_policy']['sources']['source94'];entry['effective_cutoff']=time.time()+1000
        self.projection.configure_source_admission(sign(policy,self.operator));self.assert_counts(0,0,3)
        entry['effective_cutoff']=101
        self.projection.configure_source_admission(sign(policy,self.operator));self.assert_counts(1,0,2)

    def test_signed_foreign_commitment_proof_or_slot_does_not_join(self):
        for field,value in [('commitment_sha256','0'*64),('slot',9),('miner','0'*64)]:
            row=self.row();job=row['envelope']['payload'];job['submissions'][0]['commitment_ref'][field]=value
            row['envelope']=sign(job,self.operator);row['digest']=digest(job);row['report']['job_sha256']=digest(job)
            self.resign_report(row);self.insert(row)
        row=self.row();job=row['envelope']['payload'];job['submissions'][0]['sha256']='0'*64
        row['envelope']=sign(job,self.operator);row['digest']=digest(job);row['report']['job_sha256']=digest(job)
        row['report']['audits'][0]['submission_sha256']='0'*64;self.resign_report(row);self.insert(row)
        self.assert_counts(0,0,3)

    def test_deferred_training_documents_are_not_counted_as_captured(self):
        self.insert(self.row(2))
        record=self.population['population']['committed_inventory'][0]
        record['training_documents'].pop();record['training_document_deferred_slots']=[2]
        self.assert_counts(0,0,2)

    def test_incremental_cache_authenticates_only_new_jobs(self):
        self.insert(self.row(0))
        with patch.object(module,'observations',wraps=module.observations) as verify:
            self.assert_counts(1,0,2);self.assert_counts(1,0,2)
            self.assertEqual(verify.call_count,1)
            self.insert(self.row(1));self.assert_counts(2,0,1)
            self.assertEqual(verify.call_count,2)

    def test_refresh_budget_keeps_unprocessed_batches_unchecked(self):
        self.projection=module.Projection(self.authority,max_events=1)
        self.insert(self.row(0));self.insert(self.row(1))
        self.assertTrue(self.assert_counts(1,0,2)['backlog'])
        self.assert_counts(2,0,1);self.assertFalse(self.result()['backlog'])

    def test_missing_queue_is_unavailable_and_readonly_connection_cannot_write(self):
        self.assertIsNone(self.projection.project(self.root/'missing',self.population,self.manifest))
        before=self.queue.read_bytes();self.result();self.assertEqual(self.queue.read_bytes(),before)

    def test_database_projection_keeps_display_and_capture_counts_updates_only_audits(self):
        from dashboard import server
        state=self.root/'state'
        folder=state/'live-math-launch-preparation-v1/distributed-preparation/live-controller-v1/controller-state'
        (folder/'roles').mkdir(parents=True)
        self.insert(self.row(0)); self.queue.rename(folder/'roles/verifier-queue.sqlite3')
        self.queue=folder/'roles/verifier-queue.sqlite3'
        (folder/'epoch94-manifest.json').write_text(json.dumps(self.manifest))
        (folder/'epoch94-learner-population.json').write_text(json.dumps(self.population))
        (folder/'epoch94-registrations.json').write_text(json.dumps({'miner':dict(public_key=self.miner_id,uid=85)}))
        pointer=state/'dashboard/continuous-audit-sources.ROOT-SIGNED.json';pointer.parent.mkdir()
        source_document=sign(dict(version='continuous-audit-service-sources-v1',
            approved_sources={'source94':{'subnet/test.py':'pinned'}}),self.operator)
        pointer.write_text(json.dumps(source_document))
        with patch.object(server,'project_learner',side_effect=lambda d,m,u:learner_project(d,m,u,self.authority)):
            db=server.Database(self.root/'dashboard.sqlite',state);db.continuous_audits=self.projection
            db.refresh();snapshot=db.snapshot(current_only=True);row=snapshot['epochs'][0]
        self.assertEqual((row['batches'],row['accepted'],row['rejected'],row['unchecked']),(3,1,0,2))
        self.assertEqual(row['grid'][85],3)
        self.assertEqual(row['learner_input_assurance'],'unaudited')
        self.assertEqual((snapshot['summary']['accepted'],snapshot['summary']['unchecked']),(1,2))
        self.assertNotIn('PRIVATE',json.dumps(snapshot))
        # A missing/malformed/tampered audit-source pointer cannot freeze all
        # capture/evaluation publication or turn unaudited batches into failures.
        tampered=copy.deepcopy(source_document);tampered['payload']['version']='tampered'
        for invalid in (None,b'not-json',canonical(tampered)):
            if invalid is None:pointer.unlink(missing_ok=True)
            else:pointer.write_bytes(invalid)
            with patch.object(server,'project_learner',side_effect=lambda d,m,u:learner_project(d,m,u,self.authority)):
                db.refresh();unknown=db.snapshot(current_only=True)['epochs'][0]
            self.assertEqual((unknown['batches'],unknown['accepted'],unknown['rejected'],unknown['unchecked']),(3,0,0,3))
            self.assertEqual(unknown['audit_projection_status'],'unavailable')
            self.assertTrue(unknown['audit_projection_pending'])


if __name__ == '__main__': unittest.main(verbosity=2)
