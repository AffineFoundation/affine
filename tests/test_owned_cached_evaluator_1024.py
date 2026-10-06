import copy,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from ops.continuous_owned_cached_evaluator import config_admission,queue_original,observe,pair_finished,FIXED32_INDICES
from subnet.owned_cached_evaluation import POLICY
from subnet.storage import Identity,canonical
import base64,hashlib

class Explicit1024Controls(unittest.TestCase):
 def config(self):
  sha='4db060697ab8303ee667e5787831a574b41e4a10afb8fe6a209dd7db40b9f373'
  return dict(version='owned-cached-fixed32-1024-pair-v2',dispatch_allowed=True,owned_evaluation_policy=POLICY,state='new1024',production_state='production',source_sha256=sha,source_bundle={'sha256':sha},heldout=[dict(indices=list(FIXED32_INDICES),seed=20261002,harness=dict(version='text-tools-long-kv-v3',policy='autoregressive',max_output_tokens=1024,temperature=.7,top_p=1.))],evaluation_mode='independent-checkpoints-v1',legacy_evaluator_scheduler_must_remain_stopped=True,evaluation_token_cap=1024,evaluation_job_ttl_seconds=1800,stop_after_pair=True,before_optimizer_steps=10,after_optimizer_steps=11,evaluation_experiment_id='fixed32-cap1024-v1')
 def test_explicit_cap_and_pair_admission(self):
  c=self.config();config_admission(c)
  for change in [dict(evaluation_token_cap=True),dict(evaluation_token_cap=128),dict(stop_after_pair=False),dict(after_optimizer_steps=12),dict(evaluation_experiment_id='old128'),dict(version='unknown')]:
   with self.subTest(change=change),self.assertRaises(ValueError):config_admission(dict(c,**change))
  c['heldout'][0]['harness']['max_output_tokens']=128
  with self.assertRaises(ValueError):config_admission(c)
 def test_legacy_cap_cannot_silently_change(self):
  c=self.config();c.update(version='continuous-owned-cached-checkpoints-v1');c['heldout'][0]['harness']['max_output_tokens']=128
  with self.assertRaises(ValueError):config_admission(c)
  c.pop('evaluation_token_cap');c.pop('stop_after_pair');config_admission(c)
 def test_distinct_public_keys_and_immutable_base(self):
  c=self.config();base=dict(checkpoint={'id':'a'*64},epoch='original',source_bundle={'sha256':'old'})
  with patch('subnet.checkpoint_evaluator.enqueue')as enqueue:
   queue_original(None,c,base,10,'before');new=enqueue.call_args.args[1]
   queue_original(None,dict(c,version='continuous-owned-cached-checkpoints-v1'),base,10,'before');old=enqueue.call_args.args[1]
  self.assertNotEqual(new['epoch'],old['epoch']);self.assertIn('cap1024',new['epoch']);self.assertEqual(base['epoch'],'original')
 def test_signed_exact_closed_pair_does_not_follow_future_checkpoint(self):
  who=Identity(bytes(range(32)));c=self.config();base=dict(checkpoint={'id':'a'*64},epoch='old')
  cp=dict(id='b'*64,files={'config.json':'c'*64,'model.safetensors':'d'*64})
  closure=dict(checkpoint='a'*64,next_checkpoint=cp['id'])
  def signed(v):return dict(payload=v,signer=who.id,signature=base64.b64encode(who.key.sign(canonical(v)).signature).decode())
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'original.json';p.write_text('{}');c.update(before_original_signed_job=str(p),before_original_signed_job_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),before_checkpoint='a'*64,after_checkpoint=cp,after_signed_completion=signed(closure),after_durable_pointer=dict(inference_checkpoint=cp['id'],optimizer_steps=11))
   controller=SimpleNamespace(authority=who)
   with patch('ops.continuous_owned_cached_evaluator.authenticated_manifest',return_value=base),patch('ops.continuous_owned_cached_evaluator.queue_original',return_value={})as queue,patch('ops.continuous_owned_cached_evaluator.durable_checkpoint',side_effect=AssertionError('must not follow newer parent')):
    observe(controller,c);self.assertEqual(queue.call_args.args[2]['checkpoint'],cp)
    c['after_signed_completion']=signed(dict(closure,next_checkpoint='future'))
    with self.assertRaisesRegex(ValueError,'closure'):observe(controller,c)
 def test_exit_waits_exact_both_jobs_and_actual_disposal(self):
  c=self.config();c.update(before_checkpoint='a',after_checkpoint={'id':'b'})
  with tempfile.TemporaryDirectory()as d:
   state=Path(d);(state/'checkpoint-evaluations').mkdir();(state/'roles').mkdir();(state/'cache-disposal').mkdir();controller=SimpleNamespace(state=state)
   for phase,cp,step in [('before','a',10),('after','b',11)]:
    row=dict(status='complete',request=dict(phase=phase,manifest=dict(checkpoint={'id':cp}),public_optimizer_steps=step,label=phase))
    (state/'checkpoint-evaluations'/(phase+'.json')).write_text(json.dumps(row));(state/'roles'/(phase+'.json')).write_text(json.dumps({'job_id':phase}))
   self.assertFalse(pair_finished(controller,c))
   for phase in ['before','after']:(state/'cache-disposal'/(phase+'.json')).write_text(json.dumps({'result':{'status':'complete'}}))
   self.assertTrue(pair_finished(controller,c))
   (state/'cache-disposal/after.json').write_text(json.dumps({'result':{'status':'deferred','reason':'active-lease'}}));self.assertFalse(pair_finished(controller,c))

if __name__=='__main__':unittest.main()
