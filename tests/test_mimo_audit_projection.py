import copy,json,sqlite3,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from dashboard.mimo_audit_projection import summarize,AuditProjection

class AuditCounts(unittest.TestCase):
 def snapshot(self):return dict(summary={},epochs=[dict(id='closed',checkpoint='c',batches=3,batches_available=True,deadline=2),dict(id='open',checkpoint='d',batches=0,batches_available=False,deadline=20)],incentive={'keep':True})
 def admission(self,outcome='verified_valid',batch='b',epoch='closed'):
  return dict(observations=[dict(epoch=epoch,checkpoint='c',miner='identity-secret',batch_sha256=batch,outcome=outcome,job_sha256='job',completed_at=3)],original_report_request_sha256='report')
 def test_unique_batch_not_retry_count(self):
  a=self.admission();b=copy.deepcopy(a);b['observations'][0]['job_sha256']='retry'
  r=summarize(self.snapshot(),{'a':a,'b':b},{'closed'});self.assertEqual(r['summary']['accepted'],1);self.assertEqual(r['summary']['unchecked'],2)
 def test_conflicts_are_not_accepted_or_rejected_twice(self):
  r=summarize(self.snapshot(),{'a':self.admission(),'b':self.admission('confirmed_invalid')},{'closed'});self.assertEqual((r['summary']['accepted'],r['summary']['rejected'],r['summary']['unchecked'],r['summary']['audit_conflicting_batches']),(0,0,3,1))
 def test_rejections_not_reinterpreted_by_penalty_deferral(self):
  r=summarize(self.snapshot(),{'a':self.admission('confirmed_invalid')},{'closed'});self.assertEqual(r['summary']['rejected'],1);self.assertEqual(r['summary']['accepted'],0)
 def test_missing_differs_from_true_zero(self):
  unknown=summarize(self.snapshot(),{},set());known=summarize(self.snapshot(),{},{'closed'})
  self.assertIsNone(unknown['summary']['accepted']);self.assertFalse(unknown['summary']['audit_breakdown_available']);self.assertEqual(known['summary']['accepted'],0);self.assertTrue(known['summary']['audit_breakdown_available'])
 def test_no_open_epoch_counts_or_identifiers(self):
  r=summarize(self.snapshot(),{'a':self.admission(),'b':self.admission(epoch='open')},{'closed'});self.assertIsNone(r['epochs'][1]['accepted']);self.assertNotIn('identity-secret',json.dumps(r));self.assertNotIn('batch_sha256',json.dumps(r));self.assertEqual(r['incentive'],{'keep':True})
 def test_foreign_checkpoint_and_excess_counts_fail(self):
  a=self.admission();a['observations'][0]['checkpoint']='other'
  with self.assertRaises(ValueError):summarize(self.snapshot(),{'a':a},{'closed'})
  with self.assertRaises(ValueError):summarize(self.snapshot(),{str(i):self.admission(batch=str(i))for i in range(4)},{'closed'})
 def test_input_not_mutated(self):
  s=self.snapshot();old=copy.deepcopy(s);summarize(s,{'a':self.admission()},{'closed'});self.assertEqual(s,old)

class ReadOnlySelection(AuditCounts):
 def test_closed_population_and_ro_queue(self):
  with tempfile.TemporaryDirectory()as tmp:
   root=Path(tmp);opening={'payload':{'epoch':'closed'}};dispatch={'signed':'original'}
   for name,eid in [('0','closed'),('1','open')]:
    d=root/'epochs'/name;(d/'coordinator').mkdir(parents=True);(d/'coordinator/OPENING.json').write_text(json.dumps({'payload':{'epoch':eid}}));(d/'coordinator/DISPATCH.json').write_text(json.dumps(dispatch));(d/'AUDIT-POPULATION.json').write_text(json.dumps({'epoch':eid,'payload':{'manifest_document':{'payload':{'epoch':eid}}}}))
   q=root/'q.sqlite3';db=sqlite3.connect(q);db.execute('create table jobs(status text, role text,envelope text,digest text)')
   for eid in ('closed','open'):db.execute('insert into jobs values(?,?,?,?)',('complete','verify',json.dumps({'payload':{'manifest':{'payload':{'epoch':eid}}}}),'a'))
   db.commit();db.close();before=q.read_bytes();seen=[]
   def population(doc,*a,**kw):
    seen.append(doc['epoch']);return dict(manifest_document=opening,committed_at=2,dispatch_document=dispatch)
   def admission(rows,populations,**kw):
    self.assertEqual(len(rows),1);self.assertEqual(json.loads(rows[0]['envelope'])['payload']['manifest']['payload']['epoch'],'closed');return {'a':self.admission()},[]
   with patch('dashboard.mimo_audit_projection.authenticate_population',side_effect=population),patch('dashboard.mimo_audit_projection.admit_queue_reports',side_effect=admission):
    p=AuditProjection(root,q,'auth',{'worker':['verify']},lambda m:None,{});r=p(self.snapshot(),{'closed':opening},3)
   self.assertEqual(seen,['closed']);self.assertEqual(r['summary']['accepted'],1);self.assertEqual(q.read_bytes(),before)
   with patch('dashboard.mimo_audit_projection.authenticate_population')as pa,patch('dashboard.mimo_audit_projection.admit_queue_reports')as ar:
    again=p(self.snapshot(),{'closed':opening},3);pa.assert_not_called();ar.assert_not_called();self.assertEqual(again['summary']['accepted'],1)
   with patch('dashboard.mimo_audit_projection.admit_queue_reports',return_value=({},[{'future':True}]))as ar:
    earlier=p(self.snapshot(),{'closed':opening},2.5);ar.assert_called_once();self.assertEqual(earlier['summary']['accepted'],0)
   db=sqlite3.connect(q);db.execute("UPDATE jobs SET envelope=? WHERE digest='a' AND json_extract(envelope,'$.payload.manifest.payload.epoch')='closed'",(json.dumps({'payload':{'manifest':{'payload':{'epoch':'closed'}},'tampered':True}}),));db.commit();db.close()
   with patch('dashboard.mimo_audit_projection.admit_queue_reports',return_value=({},[{'refused':True}]))as ar:
    changed=p(self.snapshot(),{'closed':opening},3);ar.assert_called_once();self.assertEqual(changed['summary']['accepted'],0);self.assertEqual(changed['summary']['audit_evidence_refused'],1)

   # A signed population from another dispatch must not be projected.
   (root/'epochs/0/coordinator/DISPATCH.json').write_text(json.dumps({'tampered':True}))
   with patch('dashboard.mimo_audit_projection.authenticate_population',return_value=dict(manifest_document=opening,committed_at=2,dispatch_document={'other':True})),patch('dashboard.mimo_audit_projection.admit_queue_reports')as admit:
    r=p(self.snapshot(),{'closed':opening},3);admit.assert_not_called();self.assertIsNone(r['summary']['accepted'])
class BackgroundView(unittest.TestCase):
 def test_http_uses_completed_snapshot_and_failed_refresh_preserves_it(self):
  from ops.publish_mimo_dashboard import database_view
  calls=[]
  class Original:
   def snapshot(self,current_only=False):return {'legacy':'unchanged'}
  class Project:
   fail=False
   def project(self,old,now):
    calls.append(old)
    if self.fail:raise ValueError('bad new evidence')
    return {'summary':{'accepted':1}}
  projection=Project();db=database_view(Original,projection)()
  with self.assertRaises(RuntimeError):db.snapshot(True)
  db.refresh();self.assertEqual(len(calls),1)
  for _ in range(4):self.assertEqual(db.snapshot(True)['summary']['accepted'],1)
  self.assertEqual(len(calls),1);self.assertEqual(db.snapshot(False),{'legacy':'unchanged'})
  projection.fail=True
  with self.assertRaises(ValueError):db.refresh()
  self.assertEqual(db.snapshot(True)['summary']['accepted'],1)
if __name__=='__main__':unittest.main()
