"""CPU reproduction of delayed-sidecar hour loss and prospective refusal gate."""
import sys,importlib.util,tempfile,json,unittest,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sp=importlib.util.spec_from_file_location('bridge_fixture',ROOT/'tests/test_live_reward_bridge.py');fx=importlib.util.module_from_spec(sp);sp.loader.exec_module(fx)
from ops import live_reward_writer as writer
from ops import live_reward_exporter as exporter,live_reward_writer as baseline

def population(tmp):
 key,inputs,regs,reports=fx.fixture();state=Path(tmp)/'compute';state.mkdir();(state/'controller.json').write_text(json.dumps({'active':None}));reward=Path(tmp)/'reward';reward.mkdir();epoch=inputs['manifest_document']['payload']['epoch']
 def put(label,value):(state/(epoch+'-'+label+'.json')).write_text(json.dumps(value))
 put('manifest',inputs['manifest_document']['payload']);put('scores',inputs['score_document']['payload']);put('first-signed-manifest',inputs['manifest_document']);put('opening-attestation',inputs['opening_document']);put('signed-registrations',inputs['registrations_document'])
 return key,inputs,regs,reports,state,reward,epoch,put
class CompletenessTests(unittest.TestCase):
 def test_unguarded_exporter_can_lose_delayed_rewards_after_zero_hour(self):
  with tempfile.TemporaryDirectory() as tmp:
   key,i,regs,reports,state,reward,epoch,put=population(tmp)
   hour=exporter.run_once(state,reward,i['anchor_document'],i['authority'],key,regs,7200);self.assertEqual(hour['points'],{})
   (reward/'writer-cursor.json').write_text(json.dumps(dict(window_end=7200,status='zero_points_no_submission')))
   put('signed-compute-scores',i['score_document'])
   for miner,doc in i['audit_documents'].items():put('signed-compute-audit-'+miner,doc)
   self.assertEqual(baseline.choose_hour(reward,i['anchor_document']['payload'],12000),10800)
   next_hour=exporter.run_once(state,reward,i['anchor_document'],i['authority'],key,regs,10800);self.assertEqual(next_hour['points'],{})
   ledger=json.loads((reward/'signed-reward-ledger.json').read_text());self.assertEqual(ledger[0]['payload']['finalized_at'],4100)
 def test_gate_refuses_missing_signed_scores_first_or_audit_before_export(self):
  with tempfile.TemporaryDirectory() as tmp:
   key,i,regs,reports,state,reward,epoch,put=population(tmp);c=dict(compute_state=str(state),reward_state=str(reward))
   with self.assertRaisesRegex(ValueError,'incomplete finalized reward'):writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200)
   put('signed-compute-scores',i['score_document'])
   with self.assertRaisesRegex(ValueError,'incomplete finalized audit'):writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200)
   for miner,doc in i['audit_documents'].items():put('signed-compute-audit-'+miner,doc)
   self.assertEqual(len(writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200)),1)
   (state/(epoch+'-first-signed-manifest.json')).unlink()
   with self.assertRaisesRegex(ValueError,'incomplete finalized reward'):writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200)
 def test_late_old_unexported_record_refuses_but_exact_existing_ledger_passes(self):
  with tempfile.TemporaryDirectory() as tmp:
   key,i,regs,reports,state,reward,epoch,put=population(tmp);c=dict(compute_state=str(state),reward_state=str(reward));put('signed-compute-scores',i['score_document'])
   for miner,doc in i['audit_documents'].items():put('signed-compute-audit-'+miner,doc)
   (reward/'writer-cursor.json').write_text(json.dumps(dict(window_end=7200,status='zero_points_no_submission')))
   with self.assertRaisesRegex(ValueError,'late unexported'):writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200)
   # Existing exact signed ledger record establishes that closed-hour reward wasn't newly imported.
   report=exporter.export_epoch(state,epoch,i['anchor_document'],i['authority']);(reward/'signed-reward-ledger.json').write_text(json.dumps([fx.sign(report,key)]))
   with self.assertRaises(FileNotFoundError):writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200)
   hourly=fx.b.hourly_reward_units([fx.sign(report,key)],i['authority'],7200,fresh_registrations=regs);proposal=fx.sign(hourly,key)
   (reward/'hour-7200-reward-units.json').write_text(json.dumps(proposal));(reward/'writer-cursor.json').write_text(json.dumps(dict(window_end=7200,status='zero_points_no_submission',proposal_sha256=fx.b.sha(proposal))))
   self.assertEqual(len(writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200)),1)
   badproposal=copy.deepcopy(hourly);badproposal['source_reward_records']=[];badproposal=fx.sign(badproposal,key);(reward/'hour-7200-reward-units.json').write_text(json.dumps(badproposal))
   with self.assertRaisesRegex(ValueError,'omitted existing reward'):writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200)
   (reward/'hour-7200-reward-units.json').write_text(json.dumps(proposal))
   bad=copy.deepcopy(report);bad['raw_unique_observed_points']['hotkey-0']=99;(reward/'signed-reward-ledger.json').write_text(json.dumps([fx.sign(bad,key)]))
   with self.assertRaisesRegex(ValueError,'immutable ledger'):writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200)
 def test_active_unfinalized_epoch_ignored_and_original_score_tamper_refused(self):
  with tempfile.TemporaryDirectory() as tmp:
   key,i,regs,reports,state,reward,epoch,put=population(tmp);c=dict(compute_state=str(state),reward_state=str(reward));(state/(epoch+'-scores.json')).unlink()
   self.assertEqual(writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200),[])
   put('scores',dict(i['score_document']['payload'],total=99));put('signed-compute-scores',i['score_document'])
   for miner,doc in i['audit_documents'].items():put('signed-compute-audit-'+miner,doc)
   with self.assertRaisesRegex(ValueError,'signed versus original'):writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200)
class WatermarkTests(unittest.TestCase):
 def test_raw_write_race_past_collect_blocks_without_any_original_scores(self):
  with tempfile.TemporaryDirectory() as tmp:
   key,i,regs,reports,state,reward,epoch,put=population(tmp);c=dict(compute_state=str(state),reward_state=str(reward))
   (state/(epoch+'-scores.json')).unlink();(state/'controller.json').write_text(json.dumps({'active':{'epoch':epoch,'phase':'collect'}}))
   with self.assertRaisesRegex(ValueError,'lacks original scores'):writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200)
   put('scores',i['score_document']['payload'])
   with self.assertRaisesRegex(ValueError,'lacks signed scores'):writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200)
   put('signed-compute-scores',i['score_document'])
   for miner,doc in i['audit_documents'].items():put('signed-compute-audit-'+miner,doc)
   self.assertEqual(len(writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200)),1)
 def test_future_deadline_allowed_but_unknown_opening_refuses(self):
  with tempfile.TemporaryDirectory() as tmp:
   key,i,regs,reports,state,reward,epoch,put=population(tmp);c=dict(compute_state=str(state),reward_state=str(reward));(state/(epoch+'-scores.json')).unlink()
   m=i['manifest_document']['payload'];m['deadline']=8000;put('manifest',m);(state/'controller.json').write_text(json.dumps({'active':{'epoch':epoch,'phase':'mine'}}))
   self.assertEqual(writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200),[])
   (state/(epoch+'-manifest.json')).unlink();(state/'controller.json').write_text(json.dumps({'active':{'epoch':epoch,'phase':'opening'}}))
   with self.assertRaisesRegex(ValueError,'unresolved opening watermark'):writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200)
 def test_watermark_change_during_evidence_read_refuses(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as tmp:
   key,i,regs,reports,state,reward,epoch,put=population(tmp);c=dict(compute_state=str(state),reward_state=str(reward));put('signed-compute-scores',i['score_document'])
   for miner,doc in i['audit_documents'].items():put('signed-compute-audit-'+miner,doc)
   with patch.object(writer,'finalized_hour_watermark',side_effect=[{'active':None},{'active':{'epoch':epoch,'phase':'mine'}}]):
    with self.assertRaisesRegex(ValueError,'watermark changed'):writer.finalized_reward_completeness(c,i['anchor_document'],i['authority'],window_end=7200)
if __name__=='__main__':unittest.main()
