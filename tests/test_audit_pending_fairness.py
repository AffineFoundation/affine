"""Extract pure scheduler by AST; no live state, signatures, queues or services."""
import copy,unittest
from subnet.continuous_audit_service import fair_pending_groups as select

def fixture(history=424,current=5):
 pending=[];draws={}
 for r,n in [(51,history),(53,current)]:
  for i in range(n):
   key=f'{r}-{i:05d}';pending.append((key,{'epoch':f'E{r}','resolved':False,'row_sha256s':[key]}));draws[key]={'row':{'epoch':f'E{r}','round':r},'selected_at':100+i if r==51 else 1000+i}
 return pending,draws
class Tests(unittest.TestCase):
 def test_starvation_424_history(self):
  p,d=fixture();old=copy.deepcopy((p,d));v=select(p,d,8);self.assertEqual(sum(x[1]['epoch']=='E53'for x in v),4);self.assertEqual(sum(x[1]['epoch']=='E51'for x in v),4);self.assertEqual((p,d),old)
 def test_minimum_capacity_two(self):
  p,d=fixture();v=select(p,d,2);self.assertEqual([x[1]['epoch']for x in v],['E53','E51'])
 def test_single_slot_newest(self):
  p,d=fixture();self.assertEqual(select(p,d,1)[0][1]['epoch'],'E53')
 def test_current_insufficient_spill(self):
  p,d=fixture(current=1);v=select(p,d,8);self.assertEqual(len(v),8);self.assertEqual(sum(x[1]['epoch']=='E53'for x in v),1)
 def test_history_insufficient_spill(self):
  p,d=fixture(history=1,current=12);v=select(p,d,8);self.assertEqual(len(v),8);self.assertEqual(sum(x[1]['epoch']=='E51'for x in v),1)
 def test_no_history_full_capacity(self):
  p,d=fixture(history=0,current=12);self.assertEqual(len(select(p,d,8)),8)
 def test_total_fewer(self):
  p,d=fixture(history=1,current=1);self.assertEqual(len(select(p,d,8)),2)
 def test_no_duplicates(self):
  p,d=fixture();v=select(p,d,8);self.assertEqual(len({x[0]for x in v}),8)
 def test_oldest_not_hash_order(self):
  p,d=fixture(history=2,current=2);d['51-00001']['selected_at']=0;self.assertEqual(select(p,d,2)[1][0],'51-00001')
 def test_next_epoch_generic(self):
  p,d=fixture();k='54-next';p.append((k,{'epoch':'E54','row_sha256s':[k]}));d[k]={'row':{'epoch':'E54','round':54},'selected_at':3000};self.assertEqual(select(p,d,2)[0][0],k)
 def test_zero(self):
  p,d=fixture();self.assertEqual(select(p,d,0),[])
 def test_bad_capacity(self):
  p,d=fixture()
  for c in[-1,True,1.5]:
   with self.assertRaises(ValueError):select(p,d,c)
 def test_missing_original_draw(self):
  p,d=fixture();del d[p[0][0]]
  with self.assertRaises(ValueError):select(p,d,8)
 def test_foreign_epoch(self):
  p,d=fixture();d[p[0][0]]['row']['epoch']='foreign'
  with self.assertRaises(ValueError):select(p,d,8)
 def test_nan_age(self):
  p,d=fixture();d[p[0][0]]['selected_at']=float('nan')
  with self.assertRaises(ValueError):select(p,d,8)
 def test_group_members_preserved(self):
  p,d=fixture();k='multi';members=['a','b'];p.append((k,{'epoch':'E53','row_sha256s':members}));d.update({x:{'row':{'epoch':'E53','round':53},'selected_at':0}for x in members});self.assertIs(select(p,d,2)[0][1],p[-1][1]);self.assertEqual(select(p,d,2)[0][1]['row_sha256s'],members)
if __name__=='__main__':unittest.main()
