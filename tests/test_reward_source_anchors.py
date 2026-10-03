"""Real signed reward history survives a prospective source approval extension."""
import copy,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('source_anchor_fixture',ROOT/'tests/test_live_reward_completeness.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
from ops import live_reward_exporter as exporter,live_reward_writer as writer

class SourceAnchorTests(unittest.TestCase):
 def test_existing_signed_ledger_and_closed_hour_survive_approval_extension(self):
  with tempfile.TemporaryDirectory() as tmp:
   key,i,regs,reports,state,reward,epoch,put=fixture.population(tmp)
   put('signed-compute-scores',i['score_document'])
   for miner,doc in i['audit_documents'].items():put('signed-compute-audit-'+miner,doc)
   original=i['anchor_document'];source=i['manifest_document']['payload']['source_bundle']['sha256']
   exporter.run_once(state,reward,original,i['authority'],key,regs,7200)
   old=(reward/'signed-reward-ledger.json').read_bytes();proposal=json.loads((reward/'hour-7200-reward-units.json').read_text())
   (reward/'writer-cursor.json').write_text(json.dumps(dict(window_end=7200,status='submitted',proposal_sha256=exporter.sha(proposal))))
   extended=copy.deepcopy(original['payload']);extended['approved_compute_sources'].append('f'*64);extended=exporter.sign(extended,key)
   with self.assertRaisesRegex(ValueError,'immutable reward ledger collision'):
    exporter.run_once(state,reward,extended,i['authority'],key,regs,10800)
   anchors={source:original,'f'*64:extended}
   exporter.run_once(state,reward,extended,i['authority'],key,regs,10800,source_anchors=anchors)
   self.assertEqual(old,(reward/'signed-reward-ledger.json').read_bytes())
   c=dict(compute_state=str(state),reward_state=str(reward),approved_source_anchors=anchors)
   self.assertEqual(len(writer.finalized_reward_completeness(c,extended,i['authority'],window_end=10800)),1)
   self.assertEqual(exporter.epoch_anchor({'source_bundle':{'sha256':'f'*64}},extended,i['authority'],anchors),extended)

 def test_missing_tampered_or_changed_cutover_anchor_refused(self):
  key,i,regs,reports=fixture.fx.fixture();m=i['manifest_document']['payload'];source=m['source_bundle']['sha256'];original=i['anchor_document'];auth=i['authority']
  self.assertEqual(exporter.epoch_anchor(m,original,auth),original)
  for anchors in ({},{'f'*64:original}):
   with self.assertRaises(ValueError):exporter.epoch_anchor(m,original,auth,anchors)
  tampered=copy.deepcopy(original);tampered['payload']['effective_at']+=1
  with self.assertRaises(Exception):exporter.epoch_anchor(m,original,auth,{source:tampered})
  for field in ('effective_at','cutover_id','owner_hotkey','compute_epoch_prefix','live_epoch_prefix','netuid'):
   changed=copy.deepcopy(original['payload']);changed[field]=changed[field]+1 if type(changed[field]) is int else changed[field]+'OTHER'
   with self.assertRaisesRegex(ValueError,'cutover identity or time'):
    exporter.epoch_anchor(m,original,auth,{source:exporter.sign(changed,key)})
  changed=copy.deepcopy(original['payload']);changed['approved_compute_sources']=[]
  with self.assertRaisesRegex(ValueError,'approval must be retained'):
   exporter.epoch_anchor(m,original,auth,{source:exporter.sign(changed,key)})

if __name__=='__main__':unittest.main()
