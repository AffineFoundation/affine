import base64,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from nacl.signing import SigningKey
from dashboard.run_projection import project,read_retirements


class RunProjectionTests(unittest.TestCase):
    def test_new_run_does_not_include_old_results_or_modify_history(self):
        old = dict(id='nonpayable-live-reward-math-v1--123-76', source='live-reward-math', start=100)
        new = dict(id='nonpayable-live-reward-math-v1--456-78', source='live-reward-math', start=200)
        forged = dict(id='nonpayable-live-reward-math-v1--456-79', source='live-reward-math', start=99)
        epochs = [old, new, forged]
        results = [dict(epoch_id=old['id'], timestamp=250), dict(epoch_id=new['id'], timestamp=210),
                   dict(epoch_id=new['id'], timestamp=99)]
        selected, evaluations = project(epochs, results, dict(first_round=78, started_at=150, run_id='fresh'))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['display_epoch'], 1)
        self.assertEqual(len(evaluations), 1)
        self.assertNotIn('display_epoch', new)
        self.assertEqual(len(epochs), 3)

    def test_absent_boundary_keeps_existing_behavior(self):
        epochs, evaluations = [dict(id='old')], []
        self.assertEqual(project(epochs, evaluations, None), (epochs, evaluations))

    def test_retired_opening_cannot_claim_completed_checkpoint_evaluation(self):
        first=dict(id='nonpayable-live-reward-math-v1--200-78',source='live-reward-math',start=200,
                   phase='trained',training=dict(checkpoint='b'*64,weights_changed=True,steps=1))
        retired=dict(id='nonpayable-live-reward-math-v1--220-79',source='live-reward-math',start=220,
                     phase='verified',finalized=True,checkpoint='b'*64,training=None)
        current=dict(id='nonpayable-live-reward-math-v1--300-86',source='live-reward-math',start=300,
                     phase='train',training=None)
        failure=dict(id='nonpayable-live-reward-math-v1--400-87',source='live-reward-math',start=400,
                     phase='aborted',training=None)
        result=dict(epoch_id=retired['id'],original_epoch_id='diagnostic-b',checkpoint='b'*64,
                    timestamp=250,status='complete',count=32,successes=21,original_report_sha256='c'*64)
        error=dict(epoch_id=retired['id'],checkpoint='b'*64,timestamp=250,status='error',count=0)
        epochs=[first,retired,current,failure];results=[result,error];original=copy.deepcopy((epochs,results))
        selected,evaluations=project(epochs,results,dict(first_round=78,started_at=150,run_id='fresh'),{retired['id']})
        self.assertEqual([r['id']for r in selected],[first['id'],current['id'],failure['id']])
        self.assertEqual([r['display_epoch']for r in selected],[1,2,3])
        self.assertEqual(len(evaluations),1)
        self.assertEqual(evaluations[0]['epoch_id'],first['id'])
        self.assertEqual(evaluations[0]['successes'],21)
        self.assertEqual(selected[-1]['phase'],'aborted')
        self.assertEqual((epochs,results),original)

    def test_audit_finalization_does_not_count_as_training_completion(self):
        row=dict(id='nonpayable-live-reward-math-v1--200-78',source='live-reward-math',start=200,
                 phase='verified',finalized=True,checkpoint='b'*64,training=None)
        evaluation=dict(epoch_id='retired',original_epoch_id='diagnostic',original_report_sha256='c'*64,
                        checkpoint='b'*64,timestamp=250,status='complete')
        _,result=project([row],[evaluation],dict(first_round=78,started_at=150,run_id='fresh'),{'retired'})
        self.assertEqual(result,[])


class RetirementEvidence(unittest.TestCase):
    def test_authenticated_exact_scope_and_tamper_rejection(self):
        key=SigningKey.generate();authority=key.verify_key.encode().hex()
        def sign(body):
            raw=json.dumps(body,sort_keys=True,separators=(',',':')).encode()
            return dict(payload=body,signer=authority,signature=base64.b64encode(key.sign(raw).signature).decode())
        body=dict(version='operator-protocol-regression-retirement-v1',source_sha256='a'*64,
                  artifacts_preserved=True,miner_fault=False,penalty_evidence_eligible=False,training_completed=False,
                  epochs=[dict(epoch='nonpayable-live-reward-math-v1--220-79',round=79,manifest_sha256='c'*64)])
        with tempfile.TemporaryDirectory()as tmp,patch('dashboard.run_projection.AUTHORITY',authority):
            root=Path(tmp);path=root/'retirement.ROOT-SIGNED.json';path.write_text(json.dumps(sign(body)))
            self.assertEqual(read_retirements(root,{'source_sha256':'a'*64}),{body['epochs'][0]['epoch']})
            self.assertEqual(read_retirements(root,{'source_sha256':'b'*64}),set())
            d=json.loads(path.read_bytes());d['payload']['epochs'][0]['round']=80;path.write_text(json.dumps(d))
            with self.assertRaises(Exception):read_retirements(root,{'source_sha256':'a'*64})
            for change in ({'training_completed':True},{'miner_fault':True},{'artifacts_preserved':False}):
                path.write_text(json.dumps(sign(dict(body,**change))))
                with self.subTest(change=change),self.assertRaises(ValueError):read_retirements(root,{'source_sha256':'a'*64})


class CheckpointAssociation(unittest.TestCase):
    def test_reset_base_score_attaches_to_fresh_run_not_first_old_use(self):
        from dashboard.cached_evaluator_projection import checkpoint_epoch
        old=(100,'nonpayable-live-reward-math-v1--100-22')
        new=(220,'nonpayable-live-reward-math-v1--220-78')
        b=dict(version='dashboard-training-run-boundary-v1',first_round=78,started_at=200)
        self.assertIsNone(checkpoint_epoch([old],210,b))
        self.assertEqual(checkpoint_epoch([old,new],210,b),new[1])
        self.assertIsNone(checkpoint_epoch([old,new],190,b))
        self.assertEqual(checkpoint_epoch([old,new],210),old[1])
