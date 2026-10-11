"""Prospective dispatch priorities only; original draws and evidence stay intact."""
import copy,json,unittest
from unittest.mock import patch
from subnet.continuous_audit_policy import digest,random_selection
from subnet.continuous_audit_service import audit_work_population,fair_pending_groups,in_recent_audit_window

def row(round,index=0):
 return dict(epoch=f'e{round}',round=round,checkpoint='a'*64,miner='b'*64,env_id='math',index=index,batch_sha256=f'{index+round:064x}',proof_sha256='c'*64,commitment_sha256='d'*64,verifier_contract_sha256='e'*64,committed_at=float(round))
def draw(r,at=1):return dict(row=r,seed='1'*64,selected_at=at)
def plan(r):
 key=digest(r);return(key,dict(epoch=r['epoch'],row_sha256s=[key],resolved=False))
class RecentWindow(unittest.TestCase):
 def choose(self,rows,draws=None,retry=None,pending=None):
  return audit_work_population(rows,draws or{},retry or[],pending or[],current_round=148,recent_epochs=8)
 def test_exact_signed_eight_round_boundary(self):
  self.assertTrue(in_recent_audit_window(141,148,8));self.assertFalse(in_recent_audit_window(140,148,8))
 def test_old_retry_cannot_starve_recent_draw(self):
  old,new=row(63),row(148);d={digest(old):draw(old)};pop,retry=self.choose([old,new],d,[old]);self.assertEqual(pop,[new]);self.assertEqual(retry,[])
 def test_all_eight_recent_rounds_get_priority(self):
  rows=[row(r)for r in [63,140,141,144,147,148]];pop,_=self.choose(rows);self.assertEqual([r['round']for r in pop],[141,144,147,148])
 def test_older_recent_retry_keeps_original_seed(self):
  old,new=row(63),row(141);d={digest(r):draw(r)for r in [old,new]};before=copy.deepcopy(d);pop,retry=self.choose([old,new],d,[old,new]);self.assertEqual(retry,[new]);self.assertEqual(d,before)
 def test_selected_recent_pending_blocks_historical_draw(self):
  old,new=row(63),row(141);d={digest(new):draw(new)};pop,retry=self.choose([old,new],d,[],[plan(new)]);self.assertEqual(pop,[new]);self.assertEqual(random_selection(pop,'2'*64,4,d),[])
 def test_forensic_fallback_after_recent_exhaustion(self):
  old,new=row(63),row(148);d={digest(new):draw(new)};pop,retry=self.choose([old,new],d);self.assertEqual(random_selection(pop,'2'*64,4,d),[old])
 def test_forensic_original_retry_fallback(self):
  old,new=row(63),row(148);d={digest(r):draw(r)for r in [old,new]};pop,retry=self.choose([old,new],d,[old]);self.assertEqual(retry,[old])
 def test_pending_recent_before_all_old_and_not_just_latest(self):
  rows=[row(r)for r in [63,141,144,148]];d={digest(r):draw(r,r['round'])for r in rows};p=[plan(r)for r in rows];before=copy.deepcopy((p,d));selected=fair_pending_groups(p,d,8,current_round=148,recent_epochs=8);self.assertEqual([v['epoch']for _,v in selected],['e148','e141','e144']);self.assertEqual((p,d),before)
 def test_pending_old_returns_only_when_no_recent(self):
  rows=[row(63),row(64)];d={digest(r):draw(r,r['round'])for r in rows};p=[plan(r)for r in rows];self.assertEqual({k for k,_ in fair_pending_groups(p,d,8,current_round=148,recent_epochs=8)},{k for k,_ in p})
 def test_restart_json_roundtrip_same_plans_no_duplicate_credit(self):
  rows=[row(63),row(145),row(148)];d={digest(r):draw(r,r['round'])for r in rows};p=[plan(r)for r in rows];reloaded=json.loads(json.dumps(dict(draws=d,pending=p)));args=dict(current_round=148,recent_epochs=8);self.assertEqual(fair_pending_groups(p,d,2,**args),fair_pending_groups(reloaded['pending'],reloaded['draws'],2,**args));self.assertEqual(set(d),set(reloaded['draws']))
 def test_unchanged_random_order_within_admitted_population(self):
  recent=[row(141,1),row(148,2),row(145,3)];pop,_=self.choose([row(63),*recent]);self.assertEqual(random_selection(pop,'9'*64,2),random_selection(recent,'9'*64,2))
 def test_malformed_round_and_window_fail_closed(self):
  for x in [True,-1,1.5,149,'148',None]:
   with self.assertRaises(ValueError):self.choose([dict(row(148),round=x)])
  for window in [0,129,True,1.5,None]:
   with self.assertRaises(ValueError):in_recent_audit_window(148,148,window)
 def test_pending_future_and_missing_draw_rejected(self):
  future=row(149)
  with self.assertRaises(ValueError):self.choose([row(148)],{digest(future):draw(future)},[],[plan(future)])
  with self.assertRaises(ValueError):self.choose([row(148)],{},[],[plan(row(148))])
 def test_original_group_member_identity_preserved(self):
  r=row(148);key,p=plan(r);d={key:draw(r)};out=fair_pending_groups([(key,p)],d,1,current_round=148,recent_epochs=8);self.assertIs(out[0][1],p);self.assertIs(d[key]['row'],r)
 def test_half_newest_and_rest_oldest_within_recent_window(self):
  rows=[row(r,i)for r in [63,141,145,148]for i in range(8)];d={digest(r):draw(r,r['round']+r['index']/100)for r in rows};selected=fair_pending_groups([plan(r)for r in rows],d,8,current_round=148,recent_epochs=8)
  self.assertEqual([v['epoch']for _,v in selected],['e148']*4+['e141']*4)
 def test_empty_latest_population_advances_window_and_mismatch_fails(self):
  from types import SimpleNamespace
  from subnet.continuous_audit_service import ContinuousAuditor
  fake=SimpleNamespace(state={'populations':{'old':{'payload':{'round':140,'records':[row(140)]}},'latest':{'payload':{'round':148,'records':[]}}}})
  self.assertEqual(ContinuousAuditor.current_audit_round(fake),148)
  fake.state['populations']['old']['payload']['round']=139
  with self.assertRaises(ValueError):ContinuousAuditor.current_audit_round(fake)

class ActualTickRouting(unittest.TestCase):
 def test_single_and_grouped_skip_old_retries_without_changing_them(self):
  from test_continuous_audit_service import ServiceControls
  from test_continuous_audit_policy import signed
  for group_size in (1,4):
   fixture=ServiceControls();fixture.setUp()
   try:
    service=fixture.service;service.group_size=group_size
    old=dict(fixture.row,epoch='old',round=63);new=dict(fixture.row,epoch='new',round=148,index=1,batch_sha256='7'*64)
    service.state['populations']={}
    for r in (old,new):
     p=copy.deepcopy(fixture.p);p['round']=r['round'];p['records']=[r];mf=p['manifest_document']['payload'];mf['epoch']=r['epoch'];mf['checkpoint']={'id':r['checkpoint']};p['manifest_document']=signed(fixture.key,mf);service.state['populations'][r['epoch']]=signed(fixture.key,p)
    original=draw(old);service.state['draws'][digest(old)]=copy.deepcopy(original)
    with patch.object(service,'_capture',side_effect=OSError('neutral capture outage'))as captured:
     service.tick(now=30)
    self.assertEqual([call.args[0]['epoch']for call in captured.call_args_list],['new'])
    self.assertEqual(service.state['draws'][digest(old)],original)
    self.assertIn(digest(new),service.state['draws'])
    self.assertEqual(service.state['jobs'],{})
   finally:fixture.doCleanups()
if __name__=='__main__':unittest.main()
