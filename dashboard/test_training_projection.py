import base64,copy,unittest
from nacl.signing import SigningKey
from dashboard.training_projection import project
from dashboard.learner_projection import canonical

class ClosedProjection(unittest.TestCase):
 def setUp(self):
  self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
  self.m=dict(epoch='epoch-150',checkpoint={'id':'a'*64},deadline=10)
  self.c=dict(epoch='epoch-150',checkpoint='a'*64,next_checkpoint='b'*64,round=150,completed_at=20)
  self.t=dict(source_epoch='epoch-150',input_checkpoint='a'*64,checkpoint='b'*64,steps=1,weights_changed=True,state_authority_committed=True,training_policy='task-normalized-persistent-v4',updates=[dict(loss=.69314718,training_policy='task-normalized-persistent-v4')],persistent_diagnostics=dict(complete=False,state_staged=True,input_checkpoint='a'*64,optimizer_steps=1,pair_count=2048,task_count=512),checkpoint_path='/private/path',secret='never emit',rollouts=['never emit'])
 def sign(self,v):return dict(payload=v,signer=self.authority,signature=base64.b64encode(self.key.sign(canonical(v)).signature).decode())
 def run_projection(self,m=None,c=None,t=None):return project(self.sign(m or self.m),self.sign(c or self.c),self.sign(t or self.t),authority=self.authority,now=30)
 def test_actual_nested_fields_and_allowlist(self):
  v=self.run_projection();self.assertTrue(v['finalized']);self.assertEqual(v['training']['losses'],[.69314718]);self.assertEqual(v['training']['input_pairs'],2048);self.assertEqual(v['training']['objective'],'task-normalized-persistent-v4');self.assertNotIn('private',str(v));self.assertNotIn('secret',str(v));self.assertNotIn('rollouts',str(v))
 def test_closed_without_cached_training_is_finalized(self):
  v=project(self.sign(self.m),self.sign(self.c),authority=self.authority,now=30);self.assertTrue(v['finalized']);self.assertNotIn('training',v)
 def test_open_future_wrong_checkpoint_or_wrong_epoch_refuse(self):
  for key,value in [('completed_at',31),('completed_at',9),('checkpoint','c'*64),('epoch','other')]:
   c=dict(self.c,**{key:value})
   with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.run_projection(c=c)
 def test_signed_receipt_mutation_and_scope_refuse(self):
  d=self.sign(self.t);d['payload']=dict(self.t,steps=2)
  with self.assertRaises(Exception):project(self.sign(self.m),self.sign(self.c),d,authority=self.authority,now=30)
  for key,value in [('source_epoch','other'),('input_checkpoint','c'*64),('checkpoint','c'*64)]:
   with self.subTest(key=key),self.assertRaises(ValueError):self.run_projection(t=dict(self.t,**{key:value}))
 def test_missing_incomplete_nonfinite_diagnostics_do_not_invent_metrics(self):
  for edit in (lambda t:t['updates'][0].update(loss=float('nan')),lambda t:t.update(state_authority_committed=False),lambda t:t.update(steps=2),lambda t:t['persistent_diagnostics'].pop('pair_count')):
   t=copy.deepcopy(self.t);edit(t)
   with self.assertRaises(ValueError):self.run_projection(t=t)
 def test_historical_format_and_true_zero_update(self):
  t={k:v for k,v in self.t.items()if k not in('updates','persistent_diagnostics','training_policy')};t.update(losses=[.4],objective='dpo',input_pairs=2)
  self.assertEqual(self.run_projection(t=t)['training']['losses'],[.4])
  c=dict(self.c,next_checkpoint='a'*64);t=dict(status='closed_no_eligible_batches',checkpoint='a'*64)
  self.assertEqual(self.run_projection(c=c,t=t)['training'],dict(steps=0,losses=[],weights_changed=False,checkpoint='a'*64,objective=None,input_pairs=0))
 def test_real_database_refresh_closed_receipt_and_open_epoch(self):
  import tempfile,json
  from pathlib import Path
  from unittest.mock import patch
  from dashboard.server import Database
  with tempfile.TemporaryDirectory()as td:
   root=Path(td);state=root/'state';folder=state/'live';folder.mkdir(parents=True);receipts=root/'receipts';receipts.mkdir()
   epoch=self.m['epoch']
   def write(path,value):path.write_text(json.dumps(value))
   write(folder/(epoch+'-manifest.json'),self.m)
   write(folder/(epoch+'-first-signed-manifest.json'),self.sign(self.m))
   write(folder/(epoch+'-signed-learner-completion.json'),self.sign(self.c))
   write(folder/(epoch+'-training-metrics.json'),self.t)
   write(receipts/(epoch+'.signed.json'),self.sign(self.t))
   # Legacy summary must not overwrite stronger signed current evidence.
   write(folder/'report.json',dict(epoch=epoch,training=dict(steps=99,losses=[999])))
   opened=dict(self.m,epoch='open',deadline=10**12)
   write(folder/'open-manifest.json',opened)
   write(folder/'open-first-signed-manifest.json',self.sign(opened))
   write(receipts/'open.signed.json',self.sign(dict(self.t,source_epoch='open')))
   with patch('dashboard.training_projection.AUTHORITY',self.authority):
    db=Database(root/'db.sqlite',state,receipts);db.refresh();rows={r['id']:r for r in db.snapshot()['epochs']}
   self.assertTrue(rows[epoch]['finalized']);self.assertEqual(rows[epoch]['training']['losses'],[.69314718])
   self.assertFalse(rows['open']['finalized']);self.assertIsNone(rows['open']['training'])
   self.assertNotIn('secret',json.dumps(rows));self.assertNotIn('/private/path',json.dumps(rows))
if __name__=='__main__':unittest.main()
